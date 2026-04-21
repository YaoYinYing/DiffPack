#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBParser


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Backend x confidence x task matrix benchmark")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--pdb_file", default=str((root / "1ubq.pdb").resolve()))
    parser.add_argument("--cache_root", default="/tmp/diffpack_matrix_cache")
    parser.add_argument("--output_dir", default=str((root / "outputs").resolve()))
    parser.add_argument("--repeats", type=int, default=1, help="number of repeats per matrix cell")
    parser.add_argument("--timeout_sec", type=int, default=900)
    parser.add_argument("--clash_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_clashes", type=int, default=20)
    parser.add_argument("--parity_max_tol", type=float, default=10.0)
    parser.add_argument("--parity_mean_tol", type=float, default=2.0)
    parser.add_argument("--parity_mode", choices=["default", "strict"], default="default")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def _time_prefix() -> list[str]:
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l"]
    return ["/usr/bin/time", "-v"]


def _parse_timed_stderr(stderr: str) -> dict[str, float | int | None]:
    rss = None
    real = user = sys_time = None
    for line in stderr.splitlines():
        match = re.search(r"^\s*([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", line)
        if match:
            real, user, sys_time = float(match.group(1)), float(match.group(2)), float(match.group(3))
        if "maximum resident set size" in line:
            token = line.strip().split()[0]
            if token.isdigit():
                rss = int(token)
        if "Maximum resident set size (kbytes):" in line:
            token = line.rsplit(":", 1)[-1].strip()
            if token.isdigit():
                rss = int(token)
    return {
        "time_real_sec": real,
        "time_user_sec": user,
        "time_sys_sec": sys_time,
        "peak_rss_kib": rss,
    }


def _pdb_delta(a: str, b: str) -> dict[str, float]:
    parser = PDBParser(QUIET=True)
    pa = parser.get_structure("a", a)
    pb = parser.get_structure("b", b)
    deltas = [float(np.linalg.norm(x.coord - y.coord)) for x, y in zip(pa.get_atoms(), pb.get_atoms())]
    if not deltas:
        return {"num_atoms_compared": 0, "max_abs_delta": 0.0, "mean_abs_delta": 0.0, "p95_abs_delta": 0.0}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "num_atoms_compared": int(arr.size),
        "max_abs_delta": float(arr.max()),
        "mean_abs_delta": float(arr.mean()),
        "p95_abs_delta": float(np.percentile(arr, 95)),
    }


def _run_checker(
    repo_root: Path,
    input_pdb: str,
    output_pdb: str,
    metadata_path: Path,
    report_path: Path,
    *,
    centers: list[str],
    radius: float | None,
    clash_threshold: float,
    top_n_clashes: int,
) -> tuple[int, dict[str, Any] | None]:
    cmd = [
        sys.executable,
        "-m",
        "diffpack.cli.check_structure",
        "--input",
        input_pdb,
        "--output",
        output_pdb,
        "--metadata",
        str(metadata_path),
        "--strict_geometry",
        "--clash_threshold",
        str(clash_threshold),
        "--top_n_clashes",
        str(top_n_clashes),
        "--report",
        str(report_path),
    ]
    if centers:
        cmd += ["--center_residues", *centers]
    if radius is not None:
        cmd += ["--repack_radius", str(radius)]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    payload = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return proc.returncode, payload


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).resolve()
    runs = out / "_backend_conf_task_matrix_runs"
    out.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    pdb = str(Path(args.pdb_file).resolve())
    cache_root = str(Path(args.cache_root).resolve()) if args.cache_root else None

    backends = [("td", "torchdrug"), ("native", "native"), ("pyg", "pyg")]
    modes = [("full", [], None), ("local", ["A:72"], 10.0)]
    configs = [
        ("no_conf", str((repo_root / "src/diffpack/config/inference.yaml").resolve())),
        ("conf", str((repo_root / "src/diffpack/config/inference_confidence.yaml").resolve())),
    ]

    env = os.environ.copy()
    if args.parity_mode == "strict":
        env["DIFFPACK_ENABLE_CLASH_GUARD"] = "0"
        args.parity_max_tol = min(args.parity_max_tol, 1e-3)
        args.parity_mean_tol = min(args.parity_mean_tol, 1e-4)
    else:
        env["DIFFPACK_ENABLE_CLASH_GUARD"] = "0"
    rows: list[dict[str, Any]] = []

    for repeat in range(1, args.repeats + 1):
        for conf_name, cfg in configs:
            for mode_name, centers, radius in modes:
                ref_pdb = None
                for short_backend, backend in backends:
                    cell = f"1ubq_{conf_name}_{mode_name}_{short_backend}_{args.device}_seed{args.seed}_r{repeat}"
                    run_dir = runs / cell
                    run_dir.mkdir(parents=True, exist_ok=True)
                    infer = [
                        sys.executable,
                        "-m",
                        "diffpack.cli.infer",
                        "-c",
                        cfg,
                        "--backend",
                        backend,
                        "--device",
                        args.device,
                        "--seed",
                        str(args.seed),
                        "--output_dir",
                        str(run_dir),
                        "--pdb_files",
                        pdb,
                        "--memory_mode",
                        "quality",
                    ]
                    if cache_root:
                        infer += ["--cache_root", cache_root]
                    if centers:
                        infer += ["--center_residues", *centers]
                    if radius is not None:
                        infer += ["--repack_radius", str(radius)]

                    start = time.perf_counter()
                    proc = subprocess.run(
                        _time_prefix() + infer,
                        cwd=str(repo_root),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout_sec,
                        check=False,
                    )
                    elapsed = time.perf_counter() - start
                    row: dict[str, Any] = {
                        "repeat": repeat,
                        "cell_id": cell,
                        "confidence": conf_name,
                        "task_mode": mode_name,
                        "backend": backend,
                        "device": args.device,
                        "seed": args.seed,
                        "elapsed_sec_wall": elapsed,
                        "infer_returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-1500:],
                        "stderr_tail": proc.stderr[-1500:],
                    }
                    row.update(_parse_timed_stderr(proc.stderr))

                    metadata_path = run_dir / "run_metadata.json"
                    if proc.returncode == 0 and metadata_path.exists():
                        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                        row["elapsed_sec_backend"] = payload.get("elapsed_sec")
                        outputs = payload.get("output_files", [])
                        if outputs:
                            src_pdb = Path(outputs[0])
                            out_prefix = f"1ubq_{conf_name}_{mode_name}_{short_backend}_{args.device}_seed{args.seed}_r{repeat}"
                            out_pdb = out / f"{out_prefix}.pdb"
                            out_md = out / f"{out_prefix}.metadata.json"
                            out_ck = out / f"{out_prefix}.checker.json"
                            shutil.copy2(src_pdb, out_pdb)
                            out_md.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                            ck_code, ck_payload = _run_checker(
                                repo_root,
                                pdb,
                                str(out_pdb),
                                out_md,
                                out_ck,
                                centers=centers,
                                radius=radius,
                                clash_threshold=args.clash_threshold,
                                top_n_clashes=args.top_n_clashes,
                            )
                            row["output_pdb"] = str(out_pdb)
                            row["checker_path"] = str(out_ck)
                            row["checker_returncode"] = ck_code
                            row["checker_status"] = (ck_payload or {}).get("status", "fail")
                            row["checker_failed_checks"] = [
                                c.get("name")
                                for c in ((ck_payload or {}).get("checks", []))
                                if not c.get("ok", False)
                            ]
                            if backend == "torchdrug":
                                ref_pdb = str(out_pdb)
                            elif ref_pdb:
                                delta = _pdb_delta(str(out_pdb), ref_pdb)
                                row["delta_vs_torchdrug"] = delta
                                row["parity_passes_tolerance"] = (
                                    delta["max_abs_delta"] <= args.parity_max_tol
                                    and delta["mean_abs_delta"] <= args.parity_mean_tol
                                )
                    rows.append(row)
                    print(f"DONE {cell} rc={row.get('infer_returncode')} checker={row.get('checker_status')}")

    summary = {
        "policy": {
            "repeats": args.repeats,
            "seed": args.seed,
            "pdb_file": pdb,
            "device": args.device,
            "cache_root": cache_root,
            "clash_guard": "disabled",
            "backends": [b for _, b in backends],
            "configs": [c for c, _ in configs],
            "task_modes": [m for m, _, _ in modes],
            "parity_tolerance": {"max_abs_delta": args.parity_max_tol, "mean_abs_delta": args.parity_mean_tol},
            "parity_mode": args.parity_mode,
        },
        "counts": {
            "total": len(rows),
            "infer_ok": sum(1 for r in rows if r.get("infer_returncode") == 0),
            "checker_pass": sum(1 for r in rows if r.get("checker_status") == "pass"),
        },
        "rows": rows,
    }

    stem = f"backend_conf_task_matrix_{args.repeats}r_1ubq_seed{args.seed}"
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# Backend/Confidence/Task Matrix ({args.repeats}r, {Path(pdb).stem}, {args.device})",
        "",
        "| repeat | confidence | task | backend | infer_rc | checker | wall_s | backend_s | peak_rss_kib | max_delta_vs_td | mean_delta_vs_td | pdb |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda x: (x["repeat"], x["confidence"], x["task_mode"], x["backend"])):
        delta = row.get("delta_vs_torchdrug", {})
        lines.append(
            f"| {row['repeat']} | {row['confidence']} | {row['task_mode']} | {row['backend']} | "
            f"{row.get('infer_returncode')} | {row.get('checker_status', '-')} | "
            f"{float(row.get('elapsed_sec_wall') or 0):.3f} | {float(row.get('elapsed_sec_backend') or 0):.3f} | "
            f"{int(row.get('peak_rss_kib') or 0)} | {float(delta.get('max_abs_delta') or 0):.3f} | "
            f"{float(delta.get('mean_abs_delta') or 0):.3f} | "
            f"{Path(row.get('output_pdb', '-')).name if row.get('output_pdb') else '-'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
