from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from diffpack.backend_preflight import probe_backend_dependencies
from diffpack.backends import InferenceRequest, get_backend_adapter
from diffpack.device import device_diagnostics
from diffpack.memory import cuda_memory_probe
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
        "--mutations",
        default=None,
        help="comma-separated mutation tokens in `[chain][old][position][new]` format, e.g. `AG76A,AG65A`",
    )
    parser.add_argument(
        "--pro_remodel_window",
        choices=["residue_only", "tripeptide", "pentapeptide"],
        default="tripeptide",
        help="local backbone remodel window used for X->PRO mutation preprocessing",
    )
    parser.add_argument(
        "--pro_remodel_max_steps",
        type=int,
        default=24,
        help="max deterministic refinement steps for X->PRO mutation preprocessing",
    )
    parser.add_argument(
        "--hetero_policy",
        choices=["exclude", "context_only", "error"],
        default="exclude",
        help="how to handle HETATM / non-canonical residues during packing",
    )
    parser.add_argument("--backend", choices=["native", "torchdrug", "pyg"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--diagnose", action="store_true", help="print runtime diagnostics and exit")
    parser.add_argument("--profile", action="store_true", help="write CPU profiler table to output directory")
    parser.add_argument("--fast", action="store_true", help="safe deterministic runtime optimizations")
    parser.add_argument(
        "--memory_mode",
        choices=["quality", "balanced", "aggressive"],
        default="quality",
        help="memory optimization mode for native / pyg generation paths",
    )
    parser.add_argument("--cache_root", default=None, help="cache root override (default: platformdirs DiffPackCache)")
    parser.add_argument(
        "--cache_read_only",
        action="store_true",
        default=True,
        help="enforce read-only cache behavior during inference (always enabled)",
    )
    return parser


def run_diagnostics():
    diagnostics = device_diagnostics()
    diagnostics["numpy_version"] = np.__version__
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
    diagnostics["backend_dependency_probe"] = {
        "native": probe_backend_dependencies("native"),
        "torchdrug": probe_backend_dependencies("torchdrug"),
        "pyg": probe_backend_dependencies("pyg"),
    }
    diagnostics["cuda_memory_probe"] = cuda_memory_probe()
    diagnostics["memory_telemetry_fields"] = [
        "memory_mode",
        "device_allocated_peak_bytes",
        "device_reserved_peak_bytes",
        "memory_phase_peaks",
    ]
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.repack_radius is not None and args.repack_radius < -1:
        parser.error("`--repack_radius` must be one of -1, 0, or >0.")
    if args.repack_radius in (-1, 0) and not args.mutations:
        parser.error("`--repack_radius` values -1 and 0 are only supported when `--mutations` is provided.")
    if args.repack_radius is not None and args.repack_radius > 0 and not args.center_residues and not args.mutations:
        parser.error("`--center_residues` must be provided when `--repack_radius` is set.")
    if args.center_residues and args.repack_radius is None and not args.mutations:
        parser.error("`--repack_radius` must be provided when `--center_residues` is set.")
    if args.mutations and args.repack_radius is None:
        parser.error(
            "`--repack_radius` must be provided when `--mutations` is set. "
            "Use -1 (mutated residues only), 0 (full repack), or >0 (local repack)."
        )
    if args.pro_remodel_max_steps < 1:
        parser.error("`--pro_remodel_max_steps` must be >= 1.")

    args.output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    args.config = os.path.realpath(os.path.expanduser(args.config))
    args.pdb_files = [os.path.realpath(os.path.expanduser(p)) for p in args.pdb_files]
    if args.cache_root:
        args.cache_root = os.path.realpath(os.path.expanduser(args.cache_root))

    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.diagnose:
        run_diagnostics()
        return 0

    config_payload = {}
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config_payload = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config_payload = {}

    mutation_cfg = config_payload.get("mutation", {}) if isinstance(config_payload, dict) else {}
    if isinstance(mutation_cfg, dict):
        if args.pro_remodel_window == "tripeptide" and "pro_remodel_window" in mutation_cfg:
            args.pro_remodel_window = str(mutation_cfg["pro_remodel_window"])
        if args.pro_remodel_max_steps == 24 and "pro_remodel_max_steps" in mutation_cfg:
            args.pro_remodel_max_steps = int(mutation_cfg["pro_remodel_max_steps"])
    if args.pro_remodel_window not in {"residue_only", "tripeptide", "pentapeptide"}:
        raise ValueError(
            "`pro_remodel_window` must be one of residue_only|tripeptide|pentapeptide. "
            f"Got `{args.pro_remodel_window}`."
        )
    if args.pro_remodel_max_steps < 1:
        raise ValueError("`pro_remodel_max_steps` must be >= 1.")

    backend_name = args.backend
    if backend_name is None:
        try:
            backend_name = config_payload.get("backend", "native")
        except Exception:
            backend_name = "native"

    request = InferenceRequest(
        config=args.config,
        seed=args.seed,
        output_dir=args.output_dir,
        pdb_files=args.pdb_files,
        center_residues=args.center_residues,
        repack_radius=args.repack_radius,
        mutations=args.mutations,
        pro_remodel_window=args.pro_remodel_window,
        pro_remodel_max_steps=args.pro_remodel_max_steps,
        hetero_policy=args.hetero_policy,
        device=args.device,
        fast=args.fast,
        profile=args.profile,
        memory_mode=args.memory_mode,
        cache_root=args.cache_root,
        cache_read_only=bool(args.cache_read_only),
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
