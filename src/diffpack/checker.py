from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from diffpack import repack
from diffpack.structure_checker import check_structure, compare_reports


@dataclass(frozen=True)
class AtomRecord:
    key: tuple[str, int, str, str, str]
    coord: tuple[float, float, float]


def _parse_atom_records(pdb_path: str) -> list[AtomRecord]:
    records: list[AtomRecord] = []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip() or "_"
            res_num = int(line[22:26].strip())
            ins_code = line[26:27].strip() or "_"
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                # Fallback for loose test fixtures that are not strictly fixed-width PDB columns.
                fields = line.split()
                x, y, z = float(fields[6]), float(fields[7]), float(fields[8])
            records.append(
                AtomRecord(
                    key=(chain_id, res_num, ins_code, res_name, atom_name),
                    coord=(x, y, z),
                )
            )
    return records


def _residue_index(records: list[AtomRecord]):
    residue_ids: list[tuple[str, int]] = []
    atom2residue = []
    residue_to_idx = {}
    coords = []
    for rec in records:
        residue_id = (rec.key[0], rec.key[1])
        if residue_id not in residue_to_idx:
            residue_to_idx[residue_id] = len(residue_ids)
            residue_ids.append(residue_id)
        atom2residue.append(residue_to_idx[residue_id])
        coords.append(rec.coord)
    return (
        residue_ids,
        torch.as_tensor(atom2residue, dtype=torch.long),
        torch.as_tensor(coords, dtype=torch.float32),
    )


def run_structure_checks(
    *,
    input_pdb: str,
    output_pdb: str,
    center_residues: Iterable[str] | None = None,
    repack_radius: float | None = None,
    metadata_path: str | None = None,
    strict_geometry: bool = False,
    clash_threshold: float = 1.0,
    top_n_clashes: int = 20,
) -> dict:
    checks = []
    errors = []

    input_atoms = _parse_atom_records(input_pdb)
    output_atoms = _parse_atom_records(output_pdb)

    checks.append(
        {
            "name": "output_parseable",
            "ok": len(output_atoms) > 0,
            "details": {"output_atom_count": len(output_atoms)},
        }
    )
    checks.append(
        {
            "name": "atom_count_consistency",
            "ok": len(input_atoms) == len(output_atoms),
            "details": {"input_atom_count": len(input_atoms), "output_atom_count": len(output_atoms)},
        }
    )

    input_residues, atom2residue, input_coords = _residue_index(input_atoms)
    output_residues, _, output_coords = _residue_index(output_atoms)
    checks.append(
        {
            "name": "residue_count_consistency",
            "ok": len(input_residues) == len(output_residues),
            "details": {"input_residue_count": len(input_residues), "output_residue_count": len(output_residues)},
        }
    )

    finite_ok = torch.isfinite(output_coords).all().item() if output_coords.numel() else False
    checks.append(
        {
            "name": "finite_coordinates",
            "ok": bool(finite_ok),
            "details": {},
        }
    )

    # Optional selective-repack freeze check.
    selectors = list(center_residues or [])
    if repack_radius is not None and selectors:
        selector_pairs = repack.parse_center_residue_selectors(selectors)
        mask, _ = repack.select_residues_by_radius(
            atom_positions=input_coords,
            atom2residue=atom2residue,
            num_residue=len(input_residues),
            residue_identifiers=input_residues,
            center_selectors=selector_pairs,
            radius=float(repack_radius),
        )
        out_coord_by_key = {a.key: a.coord for a in output_atoms}
        outside_max = 0.0
        missing_atoms = 0
        for atom_idx, atom in enumerate(input_atoms):
            out = out_coord_by_key.get(atom.key)
            if out is None:
                missing_atoms += 1
                continue
            if not bool(mask[atom2residue[atom_idx]].item()):
                dx = atom.coord[0] - out[0]
                dy = atom.coord[1] - out[1]
                dz = atom.coord[2] - out[2]
                outside_max = max(outside_max, math.sqrt(dx * dx + dy * dy + dz * dz))
        checks.append(
            {
                "name": "outside_mask_frozen",
                "ok": outside_max == 0.0 and missing_atoms == 0,
                "details": {
                    "outside_mask_max_abs_delta": outside_max,
                    "missing_atom_mappings": missing_atoms,
                    "selected_residue_count": int(mask.sum().item()),
                },
            }
        )

    # Optional metrics sanity checks from run metadata.
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
        metric_checks = {
            "atom_rmsd_per_residue": metrics.get("atom_rmsd_per_residue"),
            "chi_0_mae_deg": metrics.get("chi_0_mae_deg"),
            "chi_1_mae_deg": metrics.get("chi_1_mae_deg"),
            "chi_2_mae_deg": metrics.get("chi_2_mae_deg"),
            "chi_3_mae_deg": metrics.get("chi_3_mae_deg"),
        }
        metric_ok = True
        for value in metric_checks.values():
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                metric_ok = False
                break
        checks.append({"name": "metric_sanity", "ok": metric_ok, "details": metric_checks})

    geometry_report = None
    try:
        input_geom = check_structure(
            input_pdb,
            clash_threshold=float(clash_threshold),
            top_n_clashes=int(top_n_clashes),
        )
        output_geom = check_structure(
            output_pdb,
            clash_threshold=float(clash_threshold),
            top_n_clashes=int(top_n_clashes),
        )
        delta_geom = compare_reports(
            before=input_geom,
            after=output_geom,
            clash_threshold=float(clash_threshold),
            top_n=int(top_n_clashes),
        )
        geometry_report = {
            "clash_threshold": float(clash_threshold),
            "input": {
                "heavy_atom_count": int(input_geom.heavy_atom_count),
                "min_inter_residue_distance": float(input_geom.min_inter_residue_distance),
                "p1_inter_residue_distance": float(input_geom.p1_inter_residue_distance),
                "p5_inter_residue_distance": float(input_geom.p5_inter_residue_distance),
                "missing_sidechain_atoms_count": len(input_geom.missing_sidechain_atoms),
                "bond_length_outliers_count": len(input_geom.bond_length_outliers),
                "severe_clashes_count": len(input_geom.severe_clashes),
            },
            "output": {
                "heavy_atom_count": int(output_geom.heavy_atom_count),
                "min_inter_residue_distance": float(output_geom.min_inter_residue_distance),
                "p1_inter_residue_distance": float(output_geom.p1_inter_residue_distance),
                "p5_inter_residue_distance": float(output_geom.p5_inter_residue_distance),
                "missing_sidechain_atoms_count": len(output_geom.missing_sidechain_atoms),
                "bond_length_outliers_count": len(output_geom.bond_length_outliers),
                "severe_clashes_count": len(output_geom.severe_clashes),
            },
            "delta": {
                "before_min_inter_residue_distance": float(delta_geom.before_min_inter_residue_distance),
                "after_min_inter_residue_distance": float(delta_geom.after_min_inter_residue_distance),
                "worsened_clash_count": int(delta_geom.worsened_clash_count),
            },
            "top_severe_clashes_output": [
                {
                    "distance": float(c.distance),
                    "atom_a": c.atom_a,
                    "atom_b": c.atom_b,
                }
                for c in output_geom.severe_clashes
            ],
        }

        if strict_geometry:
            checks.append(
                {
                    "name": "no_severe_clashes",
                    "ok": len(output_geom.severe_clashes) == 0,
                    "details": {"count": len(output_geom.severe_clashes)},
                }
            )
            checks.append(
                {
                    "name": "no_bond_length_outliers",
                    "ok": len(output_geom.bond_length_outliers) == 0,
                    "details": {"count": len(output_geom.bond_length_outliers)},
                }
            )
            checks.append(
                {
                    "name": "no_worsened_severe_clashes",
                    "ok": delta_geom.worsened_clash_count == 0,
                    "details": {"count": int(delta_geom.worsened_clash_count)},
                }
            )
    except Exception as exc:
        geometry_report = {"error": str(exc)}
        if strict_geometry:
            checks.append({"name": "geometry_checker_runtime", "ok": False, "details": {"error": str(exc)}})

    status = "pass" if all(c["ok"] for c in checks) else "fail"
    if status == "fail":
        errors = [c["name"] for c in checks if not c["ok"]]
    return {
        "status": status,
        "input_pdb": str(Path(input_pdb).resolve()),
        "output_pdb": str(Path(output_pdb).resolve()),
        "checks": checks,
        "errors": errors,
        "geometry": geometry_report,
    }
