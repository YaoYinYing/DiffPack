from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from diffpack.backends import InferenceRequest, get_backend_adapter
from diffpack.checker import run_structure_checks
from diffpack.util import get_default_config_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffPack benchmark")
    parser.add_argument("--backend", default="torchdrug_fork", choices=["torchdrug_fork", "pyg"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--config", default=get_default_config_path("inference_confidence.yaml"))
    parser.add_argument("--pdb_files", nargs="*", default=["1ubq.pdb"])
    parser.add_argument("--output_dir", default="benchmark_output")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--center_residues", nargs="*", default=[])
    parser.add_argument("--repack_radius", type=float, default=None)
    parser.add_argument("--hetero_policy", choices=["exclude", "context_only", "error"], default="exclude")
    return parser.parse_args()


def main():
    args = parse_args()
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
    )
    adapter = get_backend_adapter(args.backend)
    t0 = time.perf_counter()
    result = adapter.run_inference(request)
    elapsed = time.perf_counter() - t0
    result["wall_elapsed_sec"] = elapsed
    result["benchmark_backend"] = args.backend
    result["benchmark_device"] = args.device

    checker_reports = []
    output_files = result.get("output_files", [])
    for output_file, input_file in zip(output_files, request.pdb_files):
        report = run_structure_checks(
            input_pdb=input_file,
            output_pdb=output_file,
            center_residues=args.center_residues,
            repack_radius=args.repack_radius,
            metadata_path=None,
        )
        checker_reports.append(report)
    result["checker_reports"] = checker_reports
    result["checker_status"] = "pass" if all(r["status"] == "pass" for r in checker_reports) else "fail"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_result.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote benchmark result: {out_path}")


if __name__ == "__main__":
    main()
