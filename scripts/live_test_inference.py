#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiffPack live inference test with structure checker")
    parser.add_argument("--backend", choices=["native", "torchdrug", "pyg"], default="native")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--config", default="config/inference_confidence.yaml")
    parser.add_argument("--pdb_file", default="1ubq.pdb")
    parser.add_argument("--output_dir", default="live_test_output")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--center_residues", nargs="*", default=[])
    parser.add_argument("--repack_radius", type=float, default=None)
    parser.add_argument("--hetero_policy", choices=["exclude", "context_only", "error"], default="exclude")
    parser.add_argument("--clash_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_clashes", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cmd = [
        "diffpack-infer",
        "--backend",
        args.backend,
        "--device",
        args.device,
        "-c",
        str(Path(args.config).resolve()),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
        "--pdb_files",
        str(Path(args.pdb_file).resolve()),
        "--hetero_policy",
        args.hetero_policy,
    ]
    if args.center_residues:
        infer_cmd.extend(["--center_residues", *args.center_residues])
    if args.repack_radius is not None:
        infer_cmd.extend(["--repack_radius", str(args.repack_radius)])

    subprocess.run(infer_cmd, check=True)

    output_pdb = output_dir / f"{Path(args.pdb_file).stem}.pdb"
    report_path = output_dir / "checker.json"
    check_cmd = [
        "diffpack-check-structure",
        "--input",
        str(Path(args.pdb_file).resolve()),
        "--output",
        str(output_pdb),
        "--metadata",
        str(output_dir / "run_metadata.json"),
        "--strict_geometry",
        "--clash_threshold",
        str(args.clash_threshold),
        "--top_n_clashes",
        str(args.top_n_clashes),
        "--report",
        str(report_path),
    ]
    if args.center_residues:
        check_cmd.extend(["--center_residues", *args.center_residues])
    if args.repack_radius is not None:
        check_cmd.extend(["--repack_radius", str(args.repack_radius)])

    checker = subprocess.run(check_cmd, check=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return checker.returncode


if __name__ == "__main__":
    sys.exit(main())

