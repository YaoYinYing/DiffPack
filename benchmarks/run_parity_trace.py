from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffpack import rotamer, util
from diffpack.device import choose_torch_device, move_to_device


@dataclass
class StageDiff:
    stage: str
    max_abs_delta: float
    mean_abs_delta: float
    same_shape: bool
    compared_values: int


def _as_numpy(x):
    if x is None:
        return np.zeros((0,), dtype=np.float32)
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
    if edge_list.ndim == 2 and edge_list.shape[1] >= 3 and edge_list.shape[0] > 0:
        order = np.lexsort((edge_list[:, 2], edge_list[:, 1], edge_list[:, 0]))
        edge_list = edge_list[order]
        if edge_feature is not None and edge_feature.shape[0] == order.shape[0]:
            edge_feature = edge_feature[order]
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
def _tensor_stats(x: Any) -> np.ndarray:
    arr = _as_numpy(x).astype(np.float64, copy=False)
    if arr.size == 0:
        return np.zeros((5,), dtype=np.float64)
    return np.array([arr.mean(), arr.std(), arr.min(), arr.max(), np.linalg.norm(arr)], dtype=np.float64)


def _edge_list_stats(edge_list: Any) -> np.ndarray:
    arr = _as_numpy(edge_list)
    if arr.size == 0:
        return np.zeros((4,), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.array([float(arr.shape[0]), 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array(
        [float(arr.shape[0]), float(arr[:, 0].sum()), float(arr[:, 1].sum()), float(arr[:, 2].sum())],
        dtype=np.float64,
    )


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
    score_norms: list[np.ndarray] = []
    step_masks_1pi: list[np.ndarray] = []
    step_masks_2pi: list[np.ndarray] = []
    step_t: list[np.ndarray] = []
    step_dt: list[np.ndarray] = []
    step_sigma: list[np.ndarray] = []
    chi_states: list[np.ndarray] = []
    debug_line_graph_stats: list[np.ndarray] = []
    debug_node_hidden_stats: list[np.ndarray] = []
    debug_edge_hidden_stats: list[np.ndarray] = []
    debug_residue_feature_stats: list[np.ndarray] = []
    debug_torsion_mlp_stats: list[np.ndarray] = []
    debug_torsion_mlp_full: list[np.ndarray] = []
    debug_predict_chi_mask: list[np.ndarray] = []
    if hasattr(task_module, "set_parity_debug"):
        task_module.set_parity_debug(True)

    for chi_id in range(4):
        per_chi_pred = []
        per_chi_norm = []
        per_chi_mask_1pi = []
        per_chi_mask_2pi = []
        per_chi_t = []
        per_chi_dt = []
        per_chi_sigma = []
        per_chi_chis = [_as_numpy(rotamer.get_chis(work, work.node_position))]
        per_chi_dbg_line = []
        per_chi_dbg_node = []
        per_chi_dbg_edge = []
        per_chi_dbg_res = []
        per_chi_dbg_mlp = []
        per_chi_dbg_mlp_full = []
        per_chi_dbg_chi_mask = []
        for j in range(num_step):
            t = schedule[j]
            dt = schedule[j] - schedule[j + 1]
            chis = rotamer.get_chis(work, work.node_position)
            sigma = task_module.schedule_1pi_periodic.t_to_sigma(t).repeat(work.batch_size)
            chi_protein = rotamer.remove_by_chi(work, chi_id)
            pred_score, score_norm = task_module.predict({"graph": chi_protein, "sigma": sigma, "chi_id": chi_id})
            dbg = task_module.pop_last_predict_debug() if hasattr(task_module, "pop_last_predict_debug") else None
            chi_1 = chi_protein.chi_1pi_periodic_mask
            chi_2 = chi_protein.chi_2pi_periodic_mask
            if repack_chi_mask is not None:
                chi_1 = chi_1 & repack_chi_mask
                chi_2 = chi_2 & repack_chi_mask
            per_chi_t.append(float(t.item()))
            per_chi_dt.append(float(dt.item()))
            per_chi_sigma.append(_as_numpy(sigma))
            per_chi_norm.append(_as_numpy(score_norm))
            per_chi_dbg_line.append(_edge_list_stats(dbg.get("model_line_graph_edge_list")) if dbg else np.zeros((4,), dtype=np.float64))
            per_chi_dbg_node.append(_tensor_stats(dbg.get("model_layer_node_hidden_last")) if dbg else np.zeros((5,), dtype=np.float64))
            per_chi_dbg_edge.append(_tensor_stats(dbg.get("model_layer_edge_hidden_last")) if dbg else np.zeros((5,), dtype=np.float64))
            per_chi_dbg_res.append(_tensor_stats(dbg.get("model_residue_feature")) if dbg else np.zeros((5,), dtype=np.float64))
            per_chi_dbg_mlp.append(_tensor_stats(dbg.get("model_torsion_mlp_output")) if dbg else np.zeros((5,), dtype=np.float64))
            per_chi_dbg_mlp_full.append(_as_numpy(dbg.get("model_torsion_mlp_output")) if dbg else np.zeros((work.num_residue, 4), dtype=np.float32))
            per_chi_dbg_chi_mask.append(_as_numpy(dbg.get("predict_graph_chi_mask")) if dbg else np.zeros((work.num_residue, 4), dtype=np.float32))
            per_chi_mask_1pi.append(_as_numpy(chi_1))
            per_chi_mask_2pi.append(_as_numpy(chi_2))
            chis = task_module.schedule_1pi_periodic.step(chis, pred_score, t, dt, chi_1)
            chis = task_module.schedule_2pi_periodic.step(chis, pred_score, t, dt, chi_2)
            work = rotamer.set_chis(work, chis)
            per_chi_pred.append(_as_numpy(pred_score))
            per_chi_chis.append(_as_numpy(chis))
        pred_scores.append(np.stack(per_chi_pred, axis=0) if per_chi_pred else np.zeros((0, work.num_residue, 4), dtype=np.float32))
        score_norms.append(np.stack(per_chi_norm, axis=0) if per_chi_norm else np.zeros((0, work.num_residue, 4), dtype=np.float32))
        step_masks_1pi.append(np.stack(per_chi_mask_1pi, axis=0) if per_chi_mask_1pi else np.zeros((0, work.num_residue, 4), dtype=bool))
        step_masks_2pi.append(np.stack(per_chi_mask_2pi, axis=0) if per_chi_mask_2pi else np.zeros((0, work.num_residue, 4), dtype=bool))
        step_t.append(np.asarray(per_chi_t, dtype=np.float32))
        step_dt.append(np.asarray(per_chi_dt, dtype=np.float32))
        step_sigma.append(np.stack(per_chi_sigma, axis=0) if per_chi_sigma else np.zeros((0, work.batch_size), dtype=np.float32))
        chi_states.append(np.stack(per_chi_chis, axis=0))
        debug_line_graph_stats.append(np.stack(per_chi_dbg_line, axis=0) if per_chi_dbg_line else np.zeros((0, 4), dtype=np.float64))
        debug_node_hidden_stats.append(np.stack(per_chi_dbg_node, axis=0) if per_chi_dbg_node else np.zeros((0, 5), dtype=np.float64))
        debug_edge_hidden_stats.append(np.stack(per_chi_dbg_edge, axis=0) if per_chi_dbg_edge else np.zeros((0, 5), dtype=np.float64))
        debug_residue_feature_stats.append(np.stack(per_chi_dbg_res, axis=0) if per_chi_dbg_res else np.zeros((0, 5), dtype=np.float64))
        debug_torsion_mlp_stats.append(np.stack(per_chi_dbg_mlp, axis=0) if per_chi_dbg_mlp else np.zeros((0, 5), dtype=np.float64))
        debug_torsion_mlp_full.append(np.stack(per_chi_dbg_mlp_full, axis=0) if per_chi_dbg_mlp_full else np.zeros((0, work.num_residue, 4), dtype=np.float32))
        debug_predict_chi_mask.append(np.stack(per_chi_dbg_chi_mask, axis=0) if per_chi_dbg_chi_mask else np.zeros((0, work.num_residue, 4), dtype=np.float32))
    if hasattr(task_module, "set_parity_debug"):
        task_module.set_parity_debug(False)

    return {
        "pred_scores": np.stack(pred_scores, axis=0),
        "score_norms": np.stack(score_norms, axis=0),
        "mask_1pi": np.stack(step_masks_1pi, axis=0),
        "mask_2pi": np.stack(step_masks_2pi, axis=0),
        "step_t": np.stack(step_t, axis=0),
        "step_dt": np.stack(step_dt, axis=0),
        "step_sigma": np.stack(step_sigma, axis=0),
        "chi_states": np.stack(chi_states, axis=0),
        "final_node_position": _as_numpy(work.node_position),
        "model_line_graph_edge_stats": np.stack(debug_line_graph_stats, axis=0),
        "model_layer_node_hidden_stats": np.stack(debug_node_hidden_stats, axis=0),
        "model_layer_edge_hidden_stats": np.stack(debug_edge_hidden_stats, axis=0),
        "model_residue_feature_stats": np.stack(debug_residue_feature_stats, axis=0),
        "model_torsion_mlp_stats": np.stack(debug_torsion_mlp_stats, axis=0),
        "model_torsion_mlp_full": np.stack(debug_torsion_mlp_full, axis=0),
        "predict_graph_chi_mask": np.stack(debug_predict_chi_mask, axis=0),
    }


def _summarize_delta(lhs, rhs, stage: str, *, equal_nan: bool = False) -> StageDiff:
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
    if equal_nan:
        both_nan = np.isnan(a) & np.isnan(b)
        a = np.where(both_nan, 0.0, a)
        b = np.where(both_nan, 0.0, b)
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


def _load_torchdrug_task_and_protein(cfg, device: torch.device, *, strict_checkpoint: bool):
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
        missing, unexpected = task_module.load_state_dict(state, strict=False)
        if strict_checkpoint and missing:
            raise RuntimeError(f"TorchDrug strict parity checkpoint load failed. missing_keys={missing[:20]}")
    task_module = task_module.to(device)
    test_set = core.Configurable.load_config_dict(cfg.test_set)
    item = test_set.get_item(0)
    item = move_to_device(item, device)
    protein = item["graph"]
    if not hasattr(protein, "num_cum_nodes"):
        protein = td_data.Protein.pack([protein]).to(device)
    return task_module, protein


def _load_framework_task_and_protein(cfg, device: torch.device, backend: str, *, strict_checkpoint: bool):
    if backend == "pyg":
        from diffpack.backends.pyg_runtime import PygConfigTranslator

        translator = PygConfigTranslator(cfg)
    elif backend == "native":
        from diffpack.backends.native_runtime import NativeConfigTranslator

        translator = NativeConfigTranslator(cfg)
    else:
        raise ValueError(f"Unsupported translated backend `{backend}` for parity trace")
    task_module = translator.build_task().to(device)
    if "model_checkpoint" in cfg and cfg.model_checkpoint:
        ckpt = torch.load(os.path.expanduser(cfg.model_checkpoint), map_location=torch.device("cpu"))
        state = ckpt.get("model", ckpt)
        missing, unexpected = task_module.load_state_dict(state, strict=False)
        if strict_checkpoint and missing:
            raise RuntimeError(
                f"{backend} strict parity checkpoint load failed. missing_keys={missing[:20]} unexpected_keys={unexpected[:20]}"
            )
    dataset = translator.build_dataset()
    item = dataset.get_item(0)
    item = move_to_device(item, device)
    return task_module, item["graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stagewise parity trace between torchdrug and native/pyg backends")
    parser.add_argument("--config", required=True, help="inference config yaml")
    parser.add_argument("--pdb_file", required=True, help="single pdb path (e.g., 1ubq.pdb)")
    parser.add_argument("--output_dir", default="benchmark_output/parity_trace", help="output directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--center_residues", nargs="*", default=[])
    parser.add_argument("--repack_radius", type=float, default=None)
    parser.add_argument("--hetero_policy", choices=["exclude", "context_only", "error"], default="exclude")
    parser.add_argument("--cache_root", default=None, help="cache root override for schedule tables")
    parser.add_argument("--backend", choices=["native", "pyg"], default="pyg")
    parser.add_argument("--reference_backend", choices=["torchdrug"], default="torchdrug")
    parser.add_argument("--max_abs_tolerance", type=float, default=1e-5)
    parser.add_argument("--mean_tolerance", type=float, default=1e-6)
    parser.add_argument("--equal_nan", action="store_true", help="treat matching NaN values as equal in diffs")
    parser.add_argument("--parity_mode", choices=["default", "strict"], default="default")
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
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    cfg.test_set.pdb_files = [os.path.realpath(args.pdb_file)]
    cfg.test_set.center_residues = args.center_residues
    cfg.test_set.repack_radius = args.repack_radius
    cfg.test_set.hetero_policy = args.hetero_policy
    if args.cache_root:
        cache_root = os.path.realpath(args.cache_root)
        cfg.cache = cfg.get("cache", {})
        cfg.cache["root"] = cache_root
        cfg.cache["mode"] = "read_only"
        for skey in ("schedule_1pi_periodic", "schedule_2pi_periodic"):
            if skey in cfg.task:
                cfg.task[skey]["cache_folder"] = cache_root
                cfg.task[skey]["cache_read_only"] = True

    strict_checkpoint = args.parity_mode == "strict"
    td_task, td_protein = _load_torchdrug_task_and_protein(cfg, device, strict_checkpoint=strict_checkpoint)
    run_task, run_protein = _load_framework_task_and_protein(cfg, device, args.backend, strict_checkpoint=strict_checkpoint)

    trace_td = {
        "dataset": _pack_protein_stage(td_protein),
        "graph": _build_graph_stage(td_task, td_protein),
        "schedule": _schedule_stage(td_task, device),
        "generation": _generation_stage(td_task, td_protein, randomize=True, seed=args.seed),
    }
    trace_run = {
        "dataset": _pack_protein_stage(run_protein),
        "graph": _build_graph_stage(run_task, run_protein),
        "schedule": _schedule_stage(run_task, device),
        "generation": _generation_stage(run_task, run_protein, randomize=True, seed=args.seed),
    }

    diffs = [
        _summarize_delta(trace_td["dataset"]["atom2residue"], trace_run["dataset"]["atom2residue"], "dataset.atom2residue"),
        _summarize_delta(trace_td["dataset"]["node_feature"], trace_run["dataset"]["node_feature"], "dataset.node_feature"),
        _summarize_delta(trace_td["dataset"]["chi_mask"], trace_run["dataset"]["chi_mask"], "dataset.chi_mask"),
        _summarize_delta(trace_td["graph"]["edge_list"], trace_run["graph"]["edge_list"], "graph.edge_list", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["graph"]["edge_feature"], trace_run["graph"]["edge_feature"], "graph.edge_feature", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["schedule"]["sigma"], trace_run["schedule"]["sigma"], "schedule.sigma", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["step_t"], trace_run["generation"]["step_t"], "generation.step_t", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["step_dt"], trace_run["generation"]["step_dt"], "generation.step_dt", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["step_sigma"], trace_run["generation"]["step_sigma"], "generation.step_sigma", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["mask_1pi"], trace_run["generation"]["mask_1pi"], "generation.mask_1pi", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["mask_2pi"], trace_run["generation"]["mask_2pi"], "generation.mask_2pi", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["score_norms"], trace_run["generation"]["score_norms"], "generation.score_norms", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["pred_scores"], trace_run["generation"]["pred_scores"], "generation.pred_scores", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["model_line_graph_edge_stats"], trace_run["generation"]["model_line_graph_edge_stats"], "model.line_graph_edge_stats", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["model_layer_node_hidden_stats"], trace_run["generation"]["model_layer_node_hidden_stats"], "model.layer_node_hidden_stats", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["model_layer_edge_hidden_stats"], trace_run["generation"]["model_layer_edge_hidden_stats"], "model.layer_edge_hidden_stats", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["model_residue_feature_stats"], trace_run["generation"]["model_residue_feature_stats"], "model.residue_feature_stats", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["model_torsion_mlp_stats"], trace_run["generation"]["model_torsion_mlp_stats"], "model.torsion_mlp_stats", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["model_torsion_mlp_full"], trace_run["generation"]["model_torsion_mlp_full"], "model.torsion_mlp_full", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["predict_graph_chi_mask"], trace_run["generation"]["predict_graph_chi_mask"], "model.predict_graph_chi_mask", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["chi_states"], trace_run["generation"]["chi_states"], "generation.chi_states", equal_nan=args.equal_nan),
        _summarize_delta(trace_td["generation"]["final_node_position"], trace_run["generation"]["final_node_position"], "generation.final_node_position", equal_nan=args.equal_nan),
    ]
    first_div = _first_divergence(diffs, atol=args.max_abs_tolerance, mtol=args.mean_tolerance)
    parity_status = "pass" if first_div is None else "fail"

    report = {
        "backend_requested": args.backend,
        "backend_effective": args.backend,
        "backend_mode": "native",
        "fallback_reason": None,
        "parity_status": parity_status,
        "parity_stage": first_div,
        "metric_delta_vs_reference": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "mean_tolerance": args.mean_tolerance,
            "equal_nan": args.equal_nan,
            "parity_mode": args.parity_mode,
            "strict_checkpoint": strict_checkpoint,
            "config_hash": cfg_hash,
        },
        "stage_diffs": [asdict(d) for d in diffs],
    }

    np.savez_compressed(out_dir / "trace_torchdrug.npz", **{
        "dataset_atom2residue": trace_td["dataset"]["atom2residue"],
        "dataset_node_feature": trace_td["dataset"]["node_feature"],
        "dataset_chi_mask": trace_td["dataset"]["chi_mask"],
        "graph_edge_list": trace_td["graph"]["edge_list"],
        "graph_edge_feature": trace_td["graph"]["edge_feature"],
        "schedule_sigma": trace_td["schedule"]["sigma"],
        "generation_step_t": trace_td["generation"]["step_t"],
        "generation_step_dt": trace_td["generation"]["step_dt"],
        "generation_step_sigma": trace_td["generation"]["step_sigma"],
        "generation_mask_1pi": trace_td["generation"]["mask_1pi"],
        "generation_mask_2pi": trace_td["generation"]["mask_2pi"],
        "generation_score_norms": trace_td["generation"]["score_norms"],
        "generation_pred_scores": trace_td["generation"]["pred_scores"],
        "model_line_graph_edge_stats": trace_td["generation"]["model_line_graph_edge_stats"],
        "model_layer_node_hidden_stats": trace_td["generation"]["model_layer_node_hidden_stats"],
        "model_layer_edge_hidden_stats": trace_td["generation"]["model_layer_edge_hidden_stats"],
        "model_residue_feature_stats": trace_td["generation"]["model_residue_feature_stats"],
        "model_torsion_mlp_stats": trace_td["generation"]["model_torsion_mlp_stats"],
        "model_torsion_mlp_full": trace_td["generation"]["model_torsion_mlp_full"],
        "predict_graph_chi_mask": trace_td["generation"]["predict_graph_chi_mask"],
        "generation_chi_states": trace_td["generation"]["chi_states"],
        "generation_final_node_position": trace_td["generation"]["final_node_position"],
    })
    np.savez_compressed(out_dir / f"trace_{args.backend}.npz", **{
        "dataset_atom2residue": trace_run["dataset"]["atom2residue"],
        "dataset_node_feature": trace_run["dataset"]["node_feature"],
        "dataset_chi_mask": trace_run["dataset"]["chi_mask"],
        "graph_edge_list": trace_run["graph"]["edge_list"],
        "graph_edge_feature": trace_run["graph"]["edge_feature"],
        "schedule_sigma": trace_run["schedule"]["sigma"],
        "generation_step_t": trace_run["generation"]["step_t"],
        "generation_step_dt": trace_run["generation"]["step_dt"],
        "generation_step_sigma": trace_run["generation"]["step_sigma"],
        "generation_mask_1pi": trace_run["generation"]["mask_1pi"],
        "generation_mask_2pi": trace_run["generation"]["mask_2pi"],
        "generation_score_norms": trace_run["generation"]["score_norms"],
        "generation_pred_scores": trace_run["generation"]["pred_scores"],
        "model_line_graph_edge_stats": trace_run["generation"]["model_line_graph_edge_stats"],
        "model_layer_node_hidden_stats": trace_run["generation"]["model_layer_node_hidden_stats"],
        "model_layer_edge_hidden_stats": trace_run["generation"]["model_layer_edge_hidden_stats"],
        "model_residue_feature_stats": trace_run["generation"]["model_residue_feature_stats"],
        "model_torsion_mlp_stats": trace_run["generation"]["model_torsion_mlp_stats"],
        "model_torsion_mlp_full": trace_run["generation"]["model_torsion_mlp_full"],
        "predict_graph_chi_mask": trace_run["generation"]["predict_graph_chi_mask"],
        "generation_chi_states": trace_run["generation"]["chi_states"],
        "generation_final_node_position": trace_run["generation"]["final_node_position"],
    })
    (out_dir / "parity_trace_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
