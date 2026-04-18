from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffpack.checker import run_structure_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DiffPack structure output")
    parser.add_argument("--input", required=True, help="input pdb path")
    parser.add_argument("--output", required=True, help="output pdb path")
    parser.add_argument("--center_residues", nargs="*", default=[], help="center residues CHAIN:RESID")
    parser.add_argument("--repack_radius", type=float, default=None, help="radius for local repack check")
    parser.add_argument("--metadata", default=None, help="run metadata json path")
    parser.add_argument("--report", required=True, help="output report json path")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    report = run_structure_checks(
        input_pdb=args.input,
        output_pdb=args.output,
        center_residues=args.center_residues,
        repack_radius=args.repack_radius,
        metadata_path=args.metadata,
    )
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote checker report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
