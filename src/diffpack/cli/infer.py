from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from diffpack.backends import InferenceRequest, get_backend_adapter
from diffpack.device import device_diagnostics
from diffpack.util import get_default_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiffPack inference CLI")
    parser.add_argument(
        "-c",
        "--config",
        default=get_default_config_path("inference_confidence.yaml"),
        help="YAML configuration file path",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("-o", "--output_dir", default="output", help="output directory")
    parser.add_argument("-f", "--pdb_files", nargs="*", default=[], help="input pdb files")
    parser.add_argument(
        "--center_residues",
        nargs="*",
        default=[],
        help="center residues in CHAIN:RESID format, e.g. A:72 B:155",
    )
    parser.add_argument("--repack_radius", type=float, default=None, help="radius in Angstrom for local repacking")
    parser.add_argument(
        "--hetero_policy",
        choices=["exclude", "context_only", "error"],
        default="exclude",
        help="how to handle HETATM / non-canonical residues during packing",
    )
    parser.add_argument("--backend", choices=["native", "torchdrug", "pyg"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--diagnose", action="store_true", help="print runtime diagnostics and exit")
    parser.add_argument("--profile", action="store_true", help="write CPU profiler table to output directory")
    parser.add_argument("--fast", action="store_true", help="safe deterministic runtime optimizations")
    return parser


def run_diagnostics():
    diagnostics = device_diagnostics()
    diagnostics["clang"] = bool(shutil.which("clang"))
    diagnostics["clang++"] = bool(shutil.which("clang++"))
    diagnostics["git"] = bool(shutil.which("git"))
    diagnostics["openmp_clang_test"] = None
    if diagnostics["clang++"]:
        test_cmd = ["clang++", "-x", "c++", "-", "-fopenmp", "-o", os.devnull]
        try:
            completed = subprocess.run(
                test_cmd,
                input="int main(){return 0;}",
                text=True,
                capture_output=True,
                check=False,
            )
            diagnostics["openmp_clang_test"] = {
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip()[-400:],
            }
        except OSError as error:
            diagnostics["openmp_clang_test"] = {"error": str(error)}
    diagnostics["backend_resolution_preview"] = {
        "native": {
            "backend_requested": "native",
            "backend_effective": "native",
            "backend_mode": "native",
            "fallback_reason": None,
        },
        "torchdrug": {
            "backend_requested": "torchdrug",
            "backend_effective": "torchdrug",
            "backend_mode": "native",
            "fallback_reason": None,
        },
        "pyg": {
            "backend_requested": "pyg",
            "backend_effective": "pyg",
            "backend_mode": "native",
            "fallback_reason": None,
        },
    }
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.repack_radius is not None and not args.center_residues:
        parser.error("`--center_residues` must be provided when `--repack_radius` is set.")
    if args.center_residues and args.repack_radius is None:
        parser.error("`--repack_radius` must be provided when `--center_residues` is set.")
    if args.repack_radius is not None and args.repack_radius <= 0:
        parser.error("`--repack_radius` must be > 0.")

    args.output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    args.config = os.path.realpath(os.path.expanduser(args.config))
    args.pdb_files = [os.path.realpath(os.path.expanduser(p)) for p in args.pdb_files]

    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.diagnose:
        run_diagnostics()
        return 0

    backend_name = args.backend
    if backend_name is None:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                backend_name = (yaml.safe_load(f) or {}).get("backend", "native")
        except FileNotFoundError:
            backend_name = "native"

    request = InferenceRequest(
        config=args.config,
        seed=args.seed,
        output_dir=args.output_dir,
        pdb_files=args.pdb_files,
        center_residues=args.center_residues,
        repack_radius=args.repack_radius,
        hetero_policy=args.hetero_policy,
        device=args.device,
        fast=args.fast,
        profile=args.profile,
    )
    adapter = get_backend_adapter(backend_name)
    metadata = adapter.run_inference(request)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.output_dir) / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Done. Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
