from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffpack import rotamer, util
from diffpack.backends.pyg_runtime import PygConfigTranslator
from diffpack.device import choose_torch_device, move_to_device


@dataclass
class StageDiff:
    stage: str
    max_abs_delta: float
    mean_abs_delta: float
    same_shape: bool
    compared_values: int


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pack_protein_stage(protein) -> dict[str, Any]:
    return {
        "num_atom": int(protein.num_node),
        "num_residue": int(protein.num_residue),
        "atom2residue": _as_numpy(protein.atom2residue),
        "atom_name": _as_numpy(protein.atom_name),
        "residue_type": _as_numpy(protein.residue_type),
        "node_feature": _as_numpy(protein.node_feature),
        "chi_mask": _as_numpy(protein.chi_mask),
        "chi_1pi_periodic_mask": _as_numpy(protein.chi_1pi_periodic_mask),
        "chi_2pi_periodic_mask": _as_numpy(protein.chi_2pi_periodic_mask),
        "repack_residue_mask": _as_numpy(getattr(protein, "repack_residue_mask", torch.ones(protein.num_residue, dtype=torch.bool))),
    }


def _build_graph_stage(task_module, protein) -> dict[str, Any]:
    if getattr(task_module, "graph_construction_model", None):
        graph = task_module.graph_construction_model(protein)
    else:
        graph = protein
    edge_list = _as_numpy(graph.edge_list)
    edge_feature = None
    if hasattr(graph, "edge_feature") and graph.edge_feature is not None:
        edge_feature = _as_numpy(graph.edge_feature)
    return {
        "edge_list": edge_list,
        "num_relation": int(getattr(graph, "num_relation", 0)),
        "edge_feature": edge_feature,
    }


def _schedule_stage(task_module, device: torch.device) -> dict[str, Any]:
    schedule = task_module.schedule_1pi_periodic.reverse_t_schedule.to(device)
    if schedule.numel() < 2:
        return {"t": _as_numpy(schedule), "sigma": np.array([]), "dt": np.array([])}
    sigma = task_module.schedule_1pi_periodic.t_to_sigma(schedule[:-1])
    dt = schedule[:-1] - schedule[1:]
    return {
        "t": _as_numpy(schedule),
        "sigma": _as_numpy(sigma),
        "dt": _as_numpy(dt),
    }


@torch.no_grad()
def _generation_stage(task_module, protein, *, randomize: bool, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    work = protein.clone()
    repack_residue_mask = getattr(work, "repack_residue_mask", None)
    repack_chi_mask = None
    if repack_residue_mask is not None:
        repack_residue_mask = repack_residue_mask.to(work.device).bool()
        repack_chi_mask = repack_residue_mask.unsqueeze(-1).expand(-1, 4)

    original = work.clone()
    if randomize:
        work = rotamer.randomize(work)
        if repack_residue_mask is not None:
            keep_atom_mask = (~repack_residue_mask)[work.atom2residue]
            work.node_position[keep_atom_mask] = original.node_position[keep_atom_mask]

    schedule = task_module.schedule_1pi_periodic.reverse_t_schedule.to(work.device)
    num_step = max(int(schedule.numel()) - 1, 0)
    pred_scores: list[np.ndarray] = []
    chi_states: list[np.ndarray] = []

    for chi_id in range(4):
        per_chi_pred = []
        per_chi_chis = [_as_numpy(rotamer.get_chis(work, work.node_position))]
        for j in range(num_step):
            t = schedule[j]
            dt = schedule[j] - schedule[j + 1]
            chis = rotamer.get_chis(work, work.node_position)
            sigma = task_module.schedule_1pi_periodic.t_to_sigma(t).repeat(work.batch_size)
            chi_protein = rotamer.remove_by_chi(work, chi_id)
            pred_score, _ = task_module.predict({"graph": chi_protein, "sigma": sigma, "chi_id": chi_id})
            chi_1 = chi_protein.chi_1pi_periodic_mask
            chi_2 = chi_protein.chi_2pi_periodic_mask
            if repack_chi_mask is not None:
                chi_1 = chi_1 & repack_chi_mask
                chi_2 = chi_2 & repack_chi_mask
            chis = task_module.schedule_1pi_periodic.step(chis, pred_score, t, dt, chi_1)
            chis = task_module.schedule_2pi_periodic.step(chis, pred_score, t, dt, chi_2)
            work = rotamer.set_chis(work, chis)
            per_chi_pred.append(_as_numpy(pred_score))
            per_chi_chis.append(_as_numpy(chis))
        pred_scores.append(np.stack(per_chi_pred, axis=0) if per_chi_pred else np.zeros((0, work.num_residue, 4), dtype=np.float32))
        chi_states.append(np.stack(per_chi_chis, axis=0))

    return {
        "pred_scores": np.stack(pred_scores, axis=0),
        "chi_states": np.stack(chi_states, axis=0),
        "final_node_position": _as_numpy(work.node_position),
    }


def _summarize_delta(lhs, rhs, stage: str) -> StageDiff:
    a = _as_numpy(lhs)
    b = _as_numpy(rhs)
    if a.shape != b.shape:
        return StageDiff(stage=stage, max_abs_delta=float("inf"), mean_abs_delta=float("inf"), same_shape=False, compared_values=0)
    if a.size == 0:
        return StageDiff(stage=stage, max_abs_delta=0.0, mean_abs_delta=0.0, same_shape=True, compared_values=0)
    if stage == "graph.edge_list" and a.ndim == 2 and a.shape[1] == 3:
        order_a = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
        order_b = np.lexsort((b[:, 2], b[:, 1], b[:, 0]))
        a = a[order_a]
        b = b[order_b]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if not np.isfinite(d).all():
        return StageDiff(stage=stage, max_abs_delta=float("inf"), mean_abs_delta=float("inf"), same_shape=True, compared_values=int(d.size))
    return StageDiff(
        stage=stage,
        max_abs_delta=float(d.max()),
        mean_abs_delta=float(d.mean()),
        same_shape=True,
        compared_values=int(d.size),
    )


def _first_divergence(diffs: list[StageDiff], *, atol: float, mtol: float) -> str | None:
    for d in diffs:
        if (not d.same_shape) or d.max_abs_delta > atol or d.mean_abs_delta > mtol:
            return d.stage
    return None


def _load_torchdrug_task_and_protein(cfg, device: torch.device):
    from diffpack.torchdrug import core
    from diffpack.torchdrug import data as td_data
    from diffpack import dataset as _dataset  # noqa: F401
    from diffpack import layer as _layer  # noqa: F401
    from diffpack import schedule as _schedule  # noqa: F401
    from diffpack import task as _task  # noqa: F401

    task_module = core.Configurable.load_config_dict(cfg.task)
    if "model_checkpoint" in cfg and cfg.model_checkpoint:
        ckpt = torch.load(os.path.expanduser(cfg.model_checkpoint), map_location=torch.device("cpu"))
        state = ckpt.get("model", ckpt)
        task_module.load_state_dict(state, strict=False)
    task_module = task_module.to(device)
    test_set = core.Configurable.load_config_dict(cfg.test_set)
    item = test_set.get_item(0)
    item = move_to_device(item, device)
    protein = item["graph"]
    if not hasattr(protein, "num_cum_nodes"):
        protein = td_data.Protein.pack([protein]).to(device)
    return task_module, protein


def _load_pyg_task_and_protein(cfg, device: torch.device):
    translator = PygConfigTranslator(cfg)
    task_module = translator.build_task().to(device)
    if "model_checkpoint" in cfg and cfg.model_checkpoint:
        ckpt = torch.load(os.path.expanduser(cfg.model_checkpoint), map_location=torch.device("cpu"))
        state = ckpt.get("model", ckpt)
        task_module.load_state_dict(state, strict=False)
    dataset = translator.build_dataset()
    item = dataset.get_item(0)
    item = move_to_device(item, device)
    return task_module, item["graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stagewise parity trace between torchdrug_fork and pyg backends")
    parser.add_argument("--config", required=True, help="inference config yaml")
    parser.add_argument("--pdb_file", required=True, help="single pdb path (e.g., 1ubq.pdb)")
    parser.add_argument("--output_dir", default="benchmark_output/parity_trace", help="output directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--center_residues", nargs="*", default=[])
    parser.add_argument("--repack_radius", type=float, default=None)
    parser.add_argument("--hetero_policy", choices=["exclude", "context_only", "error"], default="exclude")
    parser.add_argument("--max_abs_tolerance", type=float, default=1e-5)
    parser.add_argument("--mean_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(out_dir / ".torch_extensions"))
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    os.environ.setdefault("TORCHDRUG_ENABLE_OPENMP", "0")
    (out_dir / ".torch_extensions").mkdir(parents=True, exist_ok=True)
    (out_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (out_dir / ".cache").mkdir(parents=True, exist_ok=True)
    device = choose_torch_device(args.device)

    cfg = util.load_config(os.path.realpath(args.config))
    cfg.test_set.pdb_files = [os.path.realpath(args.pdb_file)]
    cfg.test_set.center_residues = args.center_residues
    cfg.test_set.repack_radius = args.repack_radius
    cfg.test_set.hetero_policy = args.hetero_policy

    td_task, td_protein = _load_torchdrug_task_and_protein(cfg, device)
    pyg_task, pyg_protein = _load_pyg_task_and_protein(cfg, device)

    trace_td = {
        "dataset": _pack_protein_stage(td_protein),
        "graph": _build_graph_stage(td_task, td_protein),
        "schedule": _schedule_stage(td_task, device),
        "generation": _generation_stage(td_task, td_protein, randomize=True, seed=args.seed),
    }
    trace_pyg = {
        "dataset": _pack_protein_stage(pyg_protein),
        "graph": _build_graph_stage(pyg_task, pyg_protein),
        "schedule": _schedule_stage(pyg_task, device),
        "generation": _generation_stage(pyg_task, pyg_protein, randomize=True, seed=args.seed),
    }

    diffs = [
        _summarize_delta(trace_td["dataset"]["atom2residue"], trace_pyg["dataset"]["atom2residue"], "dataset.atom2residue"),
        _summarize_delta(trace_td["dataset"]["node_feature"], trace_pyg["dataset"]["node_feature"], "dataset.node_feature"),
        _summarize_delta(trace_td["dataset"]["chi_mask"], trace_pyg["dataset"]["chi_mask"], "dataset.chi_mask"),
        _summarize_delta(trace_td["graph"]["edge_list"], trace_pyg["graph"]["edge_list"], "graph.edge_list"),
        _summarize_delta(trace_td["schedule"]["sigma"], trace_pyg["schedule"]["sigma"], "schedule.sigma"),
        _summarize_delta(trace_td["generation"]["pred_scores"], trace_pyg["generation"]["pred_scores"], "generation.pred_scores"),
        _summarize_delta(trace_td["generation"]["chi_states"], trace_pyg["generation"]["chi_states"], "generation.chi_states"),
        _summarize_delta(trace_td["generation"]["final_node_position"], trace_pyg["generation"]["final_node_position"], "generation.final_node_position"),
    ]
    first_div = _first_divergence(diffs, atol=args.max_abs_tolerance, mtol=args.mean_tolerance)
    parity_status = "pass" if first_div is None else "fail"

    report = {
        "backend_requested": "pyg",
        "backend_effective": "pyg",
        "backend_mode": "native",
        "fallback_reason": None,
        "parity_status": parity_status,
        "parity_stage": first_div,
        "metric_delta_vs_reference": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "mean_tolerance": args.mean_tolerance,
        },
        "stage_diffs": [asdict(d) for d in diffs],
    }

    np.savez_compressed(out_dir / "trace_torchdrug_fork.npz", **{
        "dataset_atom2residue": trace_td["dataset"]["atom2residue"],
        "dataset_node_feature": trace_td["dataset"]["node_feature"],
        "dataset_chi_mask": trace_td["dataset"]["chi_mask"],
        "graph_edge_list": trace_td["graph"]["edge_list"],
        "schedule_sigma": trace_td["schedule"]["sigma"],
        "generation_pred_scores": trace_td["generation"]["pred_scores"],
        "generation_chi_states": trace_td["generation"]["chi_states"],
        "generation_final_node_position": trace_td["generation"]["final_node_position"],
    })
    np.savez_compressed(out_dir / "trace_pyg.npz", **{
        "dataset_atom2residue": trace_pyg["dataset"]["atom2residue"],
        "dataset_node_feature": trace_pyg["dataset"]["node_feature"],
        "dataset_chi_mask": trace_pyg["dataset"]["chi_mask"],
        "graph_edge_list": trace_pyg["graph"]["edge_list"],
        "schedule_sigma": trace_pyg["schedule"]["sigma"],
        "generation_pred_scores": trace_pyg["generation"]["pred_scores"],
        "generation_chi_states": trace_pyg["generation"]["chi_states"],
        "generation_final_node_position": trace_pyg["generation"]["final_node_position"],
    })
    (out_dir / "parity_trace_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
