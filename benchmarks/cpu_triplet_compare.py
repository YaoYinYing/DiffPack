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
from statistics import mean, median
from typing import Any

import numpy as np
from Bio.PDB import PDBParser


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CPU triplet backend comparison for DiffPack")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--pdb_file", default=str((root / "1ubq.pdb").resolve()))
    parser.add_argument("--cache_root", default=None)
    parser.add_argument("--output_dir", default=str((root / "outputs").resolve()))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--timeout_sec", type=int, default=600)
    parser.add_argument("--parity_max_tol", type=float, default=10.0)
    parser.add_argument("--parity_mean_tol", type=float, default=2.0)
    parser.add_argument("--parity_mode", choices=["default", "strict"], default="default")
    parser.add_argument("--clash_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_clashes", type=int, default=20)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["torchdrug", "native", "pyg"],
        default=["torchdrug", "native", "pyg"],
    )
    parser.add_argument(
        "--reference_dir",
        default=None,
        help="optional directory to copy summary artifacts into (e.g., benchmarks/reference)",
    )
    return parser.parse_args()


def _benchmark_items(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "item": "inference_full",
            "config": str((root / "src/diffpack/config/inference.yaml").resolve()),
            "centers": [],
            "radius": None,
            "mode": "full",
        },
        {
            "item": "inference_confidence_local",
            "config": str((root / "src/diffpack/config/inference_confidence.yaml").resolve()),
            "centers": ["A:72"],
            "radius": 10.0,
            "mode": "local",
        },
    ]


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
        "time_cmd_real_sec": real,
        "time_cmd_user_sec": user,
        "time_cmd_sys_sec": sys_time,
        "peak_rss_kib": rss,
    }


def _timed_prefix() -> list[str]:
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l"]
    return ["/usr/bin/time", "-v"]


def _run_timed_command(cmd: list[str], cwd: Path, env: dict[str, str], timeout_sec: int) -> tuple[subprocess.CompletedProcess[str], float]:
    full_cmd = _timed_prefix() + cmd
    start = time.perf_counter()
    proc = subprocess.run(
        full_cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    return proc, time.perf_counter() - start


def _pdb_delta(pred_pdb: str, ref_pdb: str) -> dict[str, float]:
    parser = PDBParser(QUIET=True)
    pred = parser.get_structure("pred", pred_pdb)
    ref = parser.get_structure("ref", ref_pdb)
    deltas = [float(np.linalg.norm(a.coord - b.coord)) for a, b in zip(pred.get_atoms(), ref.get_atoms())]
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
    metadata_path: str,
    report_path: Path,
    *,
    centers: list[str],
    radius: float | None,
    clash_threshold: float,
    top_n_clashes: int,
) -> tuple[int, dict[str, Any] | None, str, str]:
    cmd = [
        sys.executable,
        "-m",
        "diffpack.cli.check_structure",
        "--input",
        input_pdb,
        "--output",
        output_pdb,
        "--metadata",
        metadata_path,
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
    return proc.returncode, payload, proc.stdout, proc.stderr


def _aggregate(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    return float(median(values) if method == "median" else mean(values))


def _rank(rows: list[dict[str, Any]], field: str) -> None:
    sortable = [(idx, row[field]) for idx, row in enumerate(rows) if row.get(field) is not None]
    sortable.sort(key=lambda x: x[1])
    for pos, (idx, _) in enumerate(sortable, start=1):
        rows[idx][f"{field}_rank"] = pos


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    runs_root = output_dir / "_cpu_triplet_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    pdb_file = str(Path(args.pdb_file).resolve())
    cache_root = str(Path(args.cache_root).resolve()) if args.cache_root else None
    items = _benchmark_items(repo_root)

    env = os.environ.copy()
    if args.parity_mode == "strict":
        env["DIFFPACK_ENABLE_CLASH_GUARD"] = "0"
        args.parity_max_tol = min(args.parity_max_tol, 1e-3)
        args.parity_mean_tol = min(args.parity_mean_tol, 1e-4)
    else:
        env["DIFFPACK_ENABLE_CLASH_GUARD"] = "0"

    raw_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for item in items:
            for backend in args.backends:
                cell_prefix = f"1ubq_{item['item']}_{backend}_cpu_seed{args.seed}_r{repeat}"
                run_dir = runs_root / cell_prefix
                run_dir.mkdir(parents=True, exist_ok=True)
                infer_cmd = [
                    sys.executable,
                    "-m",
                    "diffpack.cli.infer",
                    "-c",
                    item["config"],
                    "--backend",
                    backend,
                    "--device",
                    "cpu",
                    "--seed",
                    str(args.seed),
                    "--output_dir",
                    str(run_dir),
                    "--pdb_files",
                    pdb_file,
                    "--memory_mode",
                    "quality",
                ]
                if cache_root:
                    infer_cmd += ["--cache_root", cache_root]
                if item["centers"]:
                    infer_cmd += ["--center_residues", *item["centers"]]
                if item["radius"] is not None:
                    infer_cmd += ["--repack_radius", str(item["radius"])]

                proc, elapsed_wall = _run_timed_command(infer_cmd, repo_root, env, args.timeout_sec)
                row: dict[str, Any] = {
                    "repeat": repeat,
                    "cell_id": cell_prefix,
                    "item": item["item"],
                    "backend": backend,
                    "device": "cpu",
                    "config": Path(item["config"]).name,
                    "mode": item["mode"],
                    "seed": args.seed,
                    "elapsed_sec_wall": elapsed_wall,
                    "infer_returncode": proc.returncode,
                    "infer_stdout_tail": proc.stdout[-2000:],
                    "infer_stderr_tail": proc.stderr[-2000:],
                }
                row.update(_parse_timed_stderr(proc.stderr))

                metadata_path = run_dir / "run_metadata.json"
                if proc.returncode == 0 and metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    row["elapsed_sec_backend"] = metadata.get("elapsed_sec")
                    row["backend_metadata_path"] = str(metadata_path)
                    output_files = metadata.get("output_files", [])
                    if output_files:
                        src_pdb = Path(output_files[0])
                        out_prefix = f"1ubq_{item['item']}_{backend}_cpu_seed{args.seed}_r{repeat}"
                        out_pdb = output_dir / f"{out_prefix}.pdb"
                        out_md = output_dir / f"{out_prefix}.metadata.json"
                        out_checker = output_dir / f"{out_prefix}.checker.json"
                        shutil.copy2(src_pdb, out_pdb)
                        out_md.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
                        row["output_pdb"] = str(out_pdb)
                        row["output_metadata"] = str(out_md)
                        c_code, c_payload, c_stdout, c_stderr = _run_checker(
                            repo_root,
                            pdb_file,
                            str(out_pdb),
                            str(out_md),
                            out_checker,
                            centers=item["centers"],
                            radius=item["radius"],
                            clash_threshold=args.clash_threshold,
                            top_n_clashes=args.top_n_clashes,
                        )
                        row["checker_returncode"] = c_code
                        row["checker_status"] = (c_payload or {}).get("status", "fail")
                        row["checker_report_path"] = str(out_checker)
                        row["checker_failed_checks"] = [
                            chk.get("name")
                            for chk in ((c_payload or {}).get("checks", []))
                            if not chk.get("ok", False)
                        ]
                        row["checker_stdout_tail"] = c_stdout[-2000:]
                        row["checker_stderr_tail"] = c_stderr[-2000:]
                raw_rows.append(row)

    rows_by_item_repeat: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        rows_by_item_repeat.setdefault((row["item"], int(row["repeat"])), []).append(row)

    for (item_name, repeat), rows in rows_by_item_repeat.items():
        ref = next((r for r in rows if r["backend"] == "torchdrug"), None)
        for row in rows:
            if row.get("infer_returncode") != 0 or row.get("checker_status") != "pass":
                row["status"] = "FAIL"
                continue
            if row["backend"] == "torchdrug":
                row["status"] = "PASS"
                continue
            if ref and ref.get("output_pdb") and row.get("output_pdb"):
                delta = _pdb_delta(row["output_pdb"], ref["output_pdb"])
                row["delta_vs_torchdrug"] = delta
                parity_ok = delta["max_abs_delta"] <= args.parity_max_tol and delta["mean_abs_delta"] <= args.parity_mean_tol
                row["parity_passes_tolerance"] = bool(parity_ok)
                row["status"] = "PASS" if parity_ok else "PASS_WITH_DELTA"
            else:
                row["status"] = "FAIL"

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["item"], row["backend"]), []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for (item_name, backend), rows in grouped.items():
        wall = [float(r["elapsed_sec_wall"]) for r in rows if r.get("elapsed_sec_wall") is not None]
        backend_elapsed = [float(r["elapsed_sec_backend"]) for r in rows if r.get("elapsed_sec_backend") is not None]
        peak_rss = [float(r["peak_rss_kib"]) for r in rows if r.get("peak_rss_kib") is not None]
        status = "PASS"
        if any(r.get("status") == "FAIL" for r in rows):
            status = "FAIL"
        elif any(r.get("status") == "PASS_WITH_DELTA" for r in rows):
            status = "PASS_WITH_DELTA"
        agg_row: dict[str, Any] = {
            "item": item_name,
            "backend": backend,
            "device": "cpu",
            "seed": args.seed,
            "repeats": args.repeats,
            "aggregate_method": args.aggregate,
            "elapsed_sec_wall": _aggregate(wall, args.aggregate),
            "elapsed_sec_backend": _aggregate(backend_elapsed, args.aggregate) if backend_elapsed else None,
            "peak_rss_kib": int(round(_aggregate(peak_rss, args.aggregate))) if peak_rss else None,
            "status": status,
        }
        deltas = [r.get("delta_vs_torchdrug") for r in rows if r.get("delta_vs_torchdrug")]
        if deltas:
            agg_row["delta_vs_torchdrug"] = {
                "num_atoms_compared": int(_aggregate([float(d["num_atoms_compared"]) for d in deltas], args.aggregate)),
                "max_abs_delta": _aggregate([float(d["max_abs_delta"]) for d in deltas], args.aggregate),
                "mean_abs_delta": _aggregate([float(d["mean_abs_delta"]) for d in deltas], args.aggregate),
                "p95_abs_delta": _aggregate([float(d["p95_abs_delta"]) for d in deltas], args.aggregate),
            }
        aggregated_rows.append(agg_row)

    for item in {row["item"] for row in aggregated_rows}:
        subset = [row for row in aggregated_rows if row["item"] == item]
        _rank(subset, "elapsed_sec_wall")
        _rank(subset, "peak_rss_kib")

    decision = {"torchdrug_dominates_all_items": True, "item_checks": {}}
    for item in sorted({row["item"] for row in aggregated_rows}):
        subset = [row for row in aggregated_rows if row["item"] == item]
        td = next((row for row in subset if row["backend"] == "torchdrug"), None)
        if td is None:
            decision["torchdrug_dominates_all_items"] = False
            decision["item_checks"][item] = {"reason": "missing torchdrug row", "dominates": False}
            continue
        runtime_best = min((row.get("elapsed_sec_wall_rank", 999) for row in subset), default=999)
        memory_best = min((row.get("peak_rss_kib_rank", 999) for row in subset), default=999)
        dominates = (
            td.get("status") == "PASS"
            and td.get("elapsed_sec_wall_rank") == runtime_best
            and td.get("peak_rss_kib_rank") == memory_best
        )
        decision["item_checks"][item] = {
            "torchdrug_status": td.get("status"),
            "torchdrug_runtime_rank": td.get("elapsed_sec_wall_rank"),
            "torchdrug_memory_rank": td.get("peak_rss_kib_rank"),
            "best_runtime_rank": runtime_best,
            "best_memory_rank": memory_best,
            "dominates": dominates,
        }
        decision["torchdrug_dominates_all_items"] &= bool(dominates)

    status_counts = {"PASS": 0, "PASS_WITH_DELTA": 0, "FAIL": 0}
    for row in aggregated_rows:
        status_counts[row["status"]] += 1

    summary_payload = {
        "policy": {
            "reference_backend": "torchdrug",
            "device": "cpu",
            "seed": args.seed,
            "pdb_file": pdb_file,
            "cache_root": cache_root,
            "clash_guard": "disabled",
            "repeats": args.repeats,
            "aggregate_method": args.aggregate,
            "parity_mode": args.parity_mode,
            "parity_tolerance": {"max_abs_delta": args.parity_max_tol, "mean_abs_delta": args.parity_mean_tol},
        },
        "counts": {
            "total_rows": len(aggregated_rows),
            "pass": status_counts["PASS"],
            "pass_with_delta": status_counts["PASS_WITH_DELTA"],
            "fail": status_counts["FAIL"],
        },
        "decision": decision,
        "aggregated_rows": aggregated_rows,
        "raw_rows": raw_rows,
    }
    runtime_payload = {
        "rows": [
            {
                "item": row["item"],
                "backend": row["backend"],
                "device": row["device"],
                "elapsed_sec_wall": row["elapsed_sec_wall"],
                "elapsed_sec_backend": row.get("elapsed_sec_backend"),
                "peak_rss_kib": row.get("peak_rss_kib"),
                "elapsed_sec_wall_rank": row.get("elapsed_sec_wall_rank"),
                "peak_rss_kib_rank": row.get("peak_rss_kib_rank"),
                "status": row["status"],
            }
            for row in aggregated_rows
        ]
    }

    stem = f"1ubq_seed{args.seed}"
    summary_path = output_dir / f"cpu_triplet_compare_{stem}.json"
    runtime_path = output_dir / f"cpu_triplet_runtime_{stem}.json"
    markdown_path = output_dir / f"cpu_triplet_compare_{stem}.md"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        f"# CPU Triplet Compare ({Path(pdb_file).stem}, seed={args.seed}, repeats={args.repeats}, aggregate={args.aggregate})",
        "",
        "| item | backend | status | wall_s | backend_s | peak_rss_kib | rt_rank | mem_rank | max_delta_vs_td | mean_delta_vs_td |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregated_rows, key=lambda r: (r["item"], r["backend"])):
        delta = row.get("delta_vs_torchdrug", {})
        md_lines.append(
            f"| {row['item']} | {row['backend']} | {row['status']} | "
            f"{float(row['elapsed_sec_wall']):.3f} | "
            f"{float(row.get('elapsed_sec_backend') or 0.0):.3f} | "
            f"{int(row.get('peak_rss_kib') or 0)} | "
            f"{row.get('elapsed_sec_wall_rank', '-')} | "
            f"{row.get('peak_rss_kib_rank', '-')} | "
            f"{float(delta.get('max_abs_delta') or 0.0):.3f} | "
            f"{float(delta.get('mean_abs_delta') or 0.0):.3f} |"
        )
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")

    if args.reference_dir:
        ref_dir = Path(args.reference_dir).resolve()
        ref_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, ref_dir / summary_path.name)
        shutil.copy2(runtime_path, ref_dir / runtime_path.name)
        shutil.copy2(markdown_path, ref_dir / markdown_path.name)

    print(f"Wrote {summary_path}")
    print(f"Wrote {runtime_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
