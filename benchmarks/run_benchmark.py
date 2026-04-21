from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from diffpack.backends import InferenceRequest, get_backend_adapter
from diffpack.checker import run_structure_checks
from diffpack.util import get_default_config_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffPack benchmark")
    parser.add_argument("--backend", default="native", choices=["native", "torchdrug", "pyg"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--config", default=get_default_config_path("inference_confidence.yaml"))
    parser.add_argument("--pdb_files", nargs="*", default=["1ubq.pdb"])
    parser.add_argument("--output_dir", default="benchmark_output")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--center_residues", nargs="*", default=[])
    parser.add_argument("--repack_radius", type=float, default=None)
    parser.add_argument("--hetero_policy", choices=["exclude", "context_only", "error"], default="exclude")
    parser.add_argument("--cache_root", default=None, help="cache root override (read-only mode)")
    parser.add_argument("--reference_backend", choices=["native", "torchdrug", "pyg"], default=None)
    parser.add_argument("--parity_max_abs_tolerance", type=float, default=10.0)
    parser.add_argument("--parity_mean_tolerance", type=float, default=2.0)
    parser.add_argument("--parity_mode", choices=["default", "strict"], default="default")
    parser.add_argument("--clash_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_clashes", type=int, default=20)
    return parser.parse_args()


def _compute_pdb_deltas(pred_pdb: str, ref_pdb: str) -> dict[str, float]:
    parser = PDBParser(QUIET=True)
    pred = parser.get_structure("pred", pred_pdb)
    ref = parser.get_structure("ref", ref_pdb)
    deltas = []
    for pred_atom, ref_atom in zip(pred.get_atoms(), ref.get_atoms()):
        deltas.append(float(np.linalg.norm(pred_atom.coord - ref_atom.coord)))
    if not deltas:
        return {"num_atoms_compared": 0, "max_abs_delta": 0.0, "mean_abs_delta": 0.0, "p95_abs_delta": 0.0}
    arr = np.array(deltas)
    return {
        "num_atoms_compared": int(arr.size),
        "max_abs_delta": float(arr.max()),
        "mean_abs_delta": float(arr.mean()),
        "p95_abs_delta": float(np.percentile(arr, 95)),
    }


def main():
    args = parse_args()
    if args.parity_mode == "strict":
        os.environ["DIFFPACK_ENABLE_CLASH_GUARD"] = "0"
    request = InferenceRequest(
        config=os.path.realpath(args.config),
        seed=args.seed,
        output_dir=os.path.realpath(args.output_dir),
        pdb_files=[os.path.realpath(p) for p in args.pdb_files],
        center_residues=args.center_residues,
        repack_radius=args.repack_radius,
        hetero_policy=args.hetero_policy,
        device=args.device,
        fast=True,
        profile=False,
        cache_root=args.cache_root,
    )
    adapter = get_backend_adapter(args.backend)
    t0 = time.perf_counter()
    result = adapter.run_inference(request)
    elapsed = time.perf_counter() - t0
    result["wall_elapsed_sec"] = elapsed
    result["benchmark_backend"] = args.backend
    result["benchmark_device"] = args.device
    result["parity_mode"] = args.parity_mode

    checker_reports = []
    output_files = result.get("output_files", [])
    for output_file, input_file in zip(output_files, request.pdb_files):
        report = run_structure_checks(
            input_pdb=input_file,
            output_pdb=output_file,
            center_residues=args.center_residues,
            repack_radius=args.repack_radius,
            metadata_path=None,
            strict_geometry=True,
            clash_threshold=args.clash_threshold,
            top_n_clashes=args.top_n_clashes,
        )
        checker_reports.append(report)
    result["checker_reports"] = checker_reports
    result["checker_status"] = "pass" if all(r["status"] == "pass" for r in checker_reports) else "fail"

    if args.reference_backend and args.reference_backend != args.backend:
        ref_output_dir = os.path.join(args.output_dir, f"reference_{args.reference_backend}")
        ref_request = InferenceRequest(
            config=request.config,
            seed=request.seed,
            output_dir=ref_output_dir,
            pdb_files=request.pdb_files,
            center_residues=request.center_residues,
            repack_radius=request.repack_radius,
            hetero_policy=request.hetero_policy,
            device=request.device,
            fast=request.fast,
            profile=request.profile,
            cache_root=request.cache_root,
        )
        ref_result = get_backend_adapter(args.reference_backend).run_inference(ref_request)
        ref_files = ref_result.get("output_files", [])
        parity_reports = []
        for pred_file, ref_file in zip(output_files, ref_files):
            report = _compute_pdb_deltas(pred_file, ref_file)
            report["prediction_file"] = pred_file
            report["reference_file"] = ref_file
            report["passes_tolerance"] = (
                report["max_abs_delta"] <= args.parity_max_abs_tolerance
                and report["mean_abs_delta"] <= args.parity_mean_tolerance
            )
            parity_reports.append(report)
        result["parity_against"] = args.reference_backend
        result["parity_reports"] = parity_reports
        result["parity_status"] = "pass" if all(r["passes_tolerance"] for r in parity_reports) else "fail"
        result["parity_stage"] = None if result["parity_status"] == "pass" else "benchmark.metric_delta"
        if parity_reports:
            result["metric_delta_vs_reference"] = {
                "max_abs_delta": max(r["max_abs_delta"] for r in parity_reports),
                "mean_abs_delta": max(r["mean_abs_delta"] for r in parity_reports),
                "p95_abs_delta": max(r["p95_abs_delta"] for r in parity_reports),
                "num_atoms_compared": sum(r["num_atoms_compared"] for r in parity_reports),
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_result.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote benchmark result: {out_path}")


if __name__ == "__main__":
    main()
