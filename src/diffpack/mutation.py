from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from diffpack import rotamer
from diffpack.pdb_connectivity import append_conect_records_inplace


_COMPACT_MUTATION_PATTERN = re.compile(r"^([A-Za-z0-9])([A-Za-z])(-?\d+)([A-Za-z])$")
_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
_BOND_MIN = 1.05
_BOND_MAX = 1.95
_PRO_PREV_C_MIN_DIST = 1.45
_PRO_CONTEXT_MIN_DIST = 1.65
_PRO_SCOPE_OFFSETS = {
    "residue_only": (0,),
    "tripeptide": (-1, 0, 1),
    "pentapeptide": (-2, -1, 0, 1, 2),
}
_PRO_BACKBONE_IDEAL_BOND = {
    ("CA", "N"): 1.46,
    ("C", "CA"): 1.53,
    ("C", "O"): 1.24,
    ("C", "N"): 1.33,
}
_PRO_RING_IDEAL_BOND = {
    ("CA", "CB"): 1.53,
    ("CB", "CG"): 1.53,
    ("CD", "CG"): 1.53,
    ("CD", "N"): 1.47,
}
_FROZEN_PRO_TEMPLATES = [
    {
        "N": (26.559, 20.220, 7.288),
        "CA": (25.829, 19.825, 8.494),
        "C": (26.541, 18.732, 9.251),
        "CB": (24.469, 19.332, 7.952),
        "CG": (24.299, 20.134, 6.704),
        "CD": (25.714, 20.108, 6.073),
    },
    {
        "N": (41.189, 32.085, 19.031),
        "CA": (41.461, 30.751, 19.594),
        "C": (40.168, 30.026, 19.918),
        "CB": (42.195, 31.142, 20.913),
        "CG": (42.904, 32.414, 20.553),
        "CD": (41.822, 33.188, 19.813),
    },
]


@dataclass(frozen=True)
class MutationSpec:
    chain: str
    position: int
    old_res_1: str
    new_res_1: str
    old_res_3: str
    new_res_3: str

    @property
    def selector(self) -> str:
        return f"{self.chain}:{self.position}"

    @property
    def compact(self) -> str:
        return f"{self.chain}{self.old_res_1}{self.position}{self.new_res_1}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "position": self.position,
            "old_res": self.old_res_1,
            "new_res": self.new_res_1,
            "old_res_3": self.old_res_3,
            "new_res_3": self.new_res_3,
        }


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _rotate_point_around_axis(
    point: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    axis = axis_end - axis_start
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return point
    axis_unit = axis / axis_norm
    v = point - axis_start
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    rotated = (
        v * cos_t
        + np.cross(axis_unit, v) * sin_t
        + axis_unit * np.dot(axis_unit, v) * (1.0 - cos_t)
    )
    return axis_start + rotated


def _local_clash_score(
    residue_coords: dict[str, np.ndarray],
    context_coords: list[np.ndarray],
) -> float:
    score = 0.0
    for atom_name, coord in residue_coords.items():
        if atom_name in {"N", "CA", "C", "O", "OXT"}:
            continue
        for other in context_coords:
            dist = np.linalg.norm(coord - other)
            if dist < 1.8:
                score += (1.8 - dist) ** 2
    return float(score)


def _atom_element(atom_name: str) -> str:
    token = atom_name.strip().upper()
    if not token:
        return "C"
    if token[0].isdigit():
        token = token[1:]
    return token[0]


def _nonbond_min_distance(atom_a: str, atom_b: str) -> float:
    pair = tuple(sorted((_atom_element(atom_a), _atom_element(atom_b))))
    if pair == ("C", "C"):
        return 1.90
    if pair in {("C", "N"), ("C", "O")}:
        return 1.85
    if pair == ("N", "N"):
        return 1.80
    return _PRO_CONTEXT_MIN_DIST


def _build_context_atoms(
    key: tuple[str, int],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
) -> list[tuple[str, np.ndarray]]:
    context_atoms: list[tuple[str, np.ndarray]] = []
    for other_key, atom_coord_map in residue_coords.items():
        if other_key == key:
            continue
        chain, position = other_key
        for atom_name, coord in atom_coord_map.items():
            context_atoms.append((f"{chain}:{position}:{atom_name}", np.asarray(coord, dtype=np.float64)))
    return context_atoms


def _proline_hard_violations(
    key: tuple[str, int],
    residue_coords: dict[str, np.ndarray],
    residue_coords_by_key: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    context_atoms: list[tuple[str, np.ndarray]],
) -> list[tuple[str, str, float, float]]:
    violations: list[tuple[str, str, float, float]] = []
    if "CA" in residue_coords and "CB" in residue_coords:
        dist = float(np.linalg.norm(residue_coords["CA"] - residue_coords["CB"]))
        if dist < _BOND_MIN or dist > _BOND_MAX:
            violations.append(("CA", "CB", dist, _BOND_MAX))
    if "N" in residue_coords and "CD" in residue_coords:
        dist = float(np.linalg.norm(residue_coords["N"] - residue_coords["CD"]))
        if dist < _BOND_MIN or dist > _BOND_MAX:
            violations.append(("N", "CD", dist, _BOND_MAX))

    chain, position = key
    prev_key = (chain, position - 1)
    if "CD" in residue_coords and prev_key in residue_coords_by_key and "C" in residue_coords_by_key[prev_key]:
        prev_c = np.asarray(residue_coords_by_key[prev_key]["C"], dtype=np.float64)
        dist = float(np.linalg.norm(residue_coords["CD"] - prev_c))
        if dist < _PRO_PREV_C_MIN_DIST:
            violations.append(("CD", f"{chain}:{position-1}:C", dist, _PRO_PREV_C_MIN_DIST))

    for atom_name in ("CB", "CG", "CD"):
        if atom_name not in residue_coords:
            continue
        for label, other in context_atoms:
            dist = float(np.linalg.norm(residue_coords[atom_name] - other))
            context_atom = label.rsplit(":", 1)[-1]
            if atom_name == "CD" and label == f"{chain}:{position-1}:C":
                if dist < _PRO_PREV_C_MIN_DIST:
                    violations.append((atom_name, label, dist, _PRO_PREV_C_MIN_DIST))
                continue
            min_allowed = _nonbond_min_distance(atom_name, context_atom)
            if dist < min_allowed:
                violations.append((atom_name, label, dist, min_allowed))
    return violations


def _rotate_atoms(
    residue_coords: dict[str, np.ndarray],
    atom_names: list[str],
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_deg: float,
) -> dict[str, np.ndarray]:
    if abs(angle_deg) < 1e-12:
        return residue_coords
    out = dict(residue_coords)
    angle_rad = np.deg2rad(float(angle_deg))
    for atom_name in atom_names:
        if atom_name not in out:
            continue
        out[atom_name] = _rotate_point_around_axis(out[atom_name], axis_start, axis_end, angle_rad)
    return out


def _top_closest_pairs(
    residue_coords: dict[str, np.ndarray],
    context_atoms: list[tuple[str, np.ndarray]],
    topk: int = 3,
) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for atom_name in ("CB", "CG", "CD"):
        if atom_name not in residue_coords:
            continue
        for label, other in context_atoms:
            dist = float(np.linalg.norm(residue_coords[atom_name] - other))
            pairs.append((atom_name, label, dist))
    pairs.sort(key=lambda x: x[2])
    return pairs[:topk]


def _ideal_bond_length(atom_a: str, atom_b: str) -> float:
    key = tuple(sorted((atom_a, atom_b)))
    if key in _PRO_RING_IDEAL_BOND:
        return _PRO_RING_IDEAL_BOND[key]
    if key in _PRO_BACKBONE_IDEAL_BOND:
        return _PRO_BACKBONE_IDEAL_BOND[key]
    return 1.52


def _pro_window_keys(
    key: tuple[str, int],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    window: str,
) -> list[tuple[str, int]]:
    offsets = _PRO_SCOPE_OFFSETS[window]
    chain, position = key
    out = []
    for offset in offsets:
        k = (chain, position + offset)
        if k in residue_coords:
            out.append(k)
    return out


def _collect_window_updates(
    key: tuple[str, int],
    nodes: dict[str, np.ndarray],
    window: str,
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    include_static: bool = False,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    updates: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for scope_key in _pro_window_keys(key, residue_coords, window):
        chain, pos = scope_key
        prefix = f"{chain}:{pos}:"
        atom_names = ["N", "CA", "C", "O"]
        if scope_key == key:
            atom_names.extend(["CB", "CG", "CD"])
        atom_updates: dict[str, np.ndarray] = {}
        for atom_name in atom_names:
            label = prefix + atom_name
            if label not in nodes:
                continue
            coord = nodes[label]
            if not include_static and scope_key in residue_coords and atom_name in residue_coords[scope_key]:
                ref = np.asarray(residue_coords[scope_key][atom_name], dtype=np.float64)
                if float(np.linalg.norm(coord - ref)) < 1e-8:
                    continue
            atom_updates[atom_name] = coord
        if atom_updates:
            updates[scope_key] = atom_updates
    return updates


def _validate_pro_solution(
    key: tuple[str, int],
    nodes: dict[str, np.ndarray],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    context_atoms: list[tuple[str, np.ndarray]],
) -> tuple[bool, str, list[tuple[str, str, float]]]:
    chain, pos = key
    required = ("N", "CA", "C", "CB", "CG", "CD")
    missing = [atom for atom in required if f"{chain}:{pos}:{atom}" not in nodes]
    if missing:
        return False, f"missing required PRO atoms: {', '.join(missing)}", []

    residue_map = {
        atom: nodes[f"{chain}:{pos}:{atom}"]
        for atom in required
    }
    violations = _proline_hard_violations(
        key=key,
        residue_coords=residue_map,
        residue_coords_by_key=residue_coords,
        context_atoms=context_atoms,
    )
    if violations:
        top = sorted(violations, key=lambda item: item[2])[:3]
        detail = ", ".join(f"{a}-{b}:{d:.3f}" for a, b, d, _ in top)
        return False, f"hard clash/constraint violation ({detail})", [(a, b, d) for a, b, d, _ in top]

    prev_key = (chain, pos - 1)
    next_key = (chain, pos + 1)
    prev_c = nodes.get(f"{chain}:{pos-1}:C")
    if prev_c is None and prev_key in residue_coords and "C" in residue_coords[prev_key]:
        prev_c = np.asarray(residue_coords[prev_key]["C"], dtype=np.float64)
    next_n = nodes.get(f"{chain}:{pos+1}:N")
    if next_n is None and next_key in residue_coords and "N" in residue_coords[next_key]:
        next_n = np.asarray(residue_coords[next_key]["N"], dtype=np.float64)
    if prev_c is not None:
        d = float(np.linalg.norm(prev_c - nodes[f"{chain}:{pos}:N"]))
        if d < 1.15 or d > 1.55:
            return False, f"continuity violation prev:C-mut:N distance={d:.3f}", [("prev:C", "mut:N", d)]
    if next_n is not None:
        d = float(np.linalg.norm(nodes[f"{chain}:{pos}:C"] - next_n))
        if d < 1.15 or d > 1.55:
            return False, f"continuity violation mut:C-next:N distance={d:.3f}", [("mut:C", "next:N", d)]

    # Keep neighbor backbone adjustments local and small.
    for label, coord in nodes.items():
        l_chain, l_pos_txt, atom = label.split(":")
        l_pos = int(l_pos_txt)
        if (l_chain, l_pos) == key:
            continue
        if (l_chain, l_pos) in residue_coords and atom in residue_coords[(l_chain, l_pos)]:
            ref = np.asarray(residue_coords[(l_chain, l_pos)][atom], dtype=np.float64)
            if float(np.linalg.norm(coord - ref)) > 0.18:
                return False, f"neighbor displacement too large at {label}", [(label, "ref", float(np.linalg.norm(coord - ref)))]
    return True, "", []


def _pro_solver_cost(
    key: tuple[str, int],
    nodes: dict[str, np.ndarray],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    covalent_pairs: list[tuple[str, str]],
    context_atoms: list[tuple[str, np.ndarray]],
    initial_nodes: dict[str, np.ndarray],
) -> float:
    cost = 0.0
    bonded = {frozenset((a, b)) for a, b in covalent_pairs}
    for label_a, label_b in covalent_pairs:
        if label_a not in nodes or label_b not in nodes:
            continue
        atom_a = label_a.rsplit(":", 1)[-1]
        atom_b = label_b.rsplit(":", 1)[-1]
        d = float(np.linalg.norm(nodes[label_a] - nodes[label_b]))
        ideal = _ideal_bond_length(atom_a, atom_b)
        cost += 18.0 * (d - ideal) ** 2

    node_items = sorted(nodes.items())
    for i, (label_a, coord_a) in enumerate(node_items):
        atom_a = label_a.rsplit(":", 1)[-1]
        for label_b, coord_b in node_items[i + 1 :]:
            atom_b = label_b.rsplit(":", 1)[-1]
            if frozenset((label_a, label_b)) in bonded:
                continue
            min_allowed = _nonbond_min_distance(atom_a, atom_b)
            d = float(np.linalg.norm(coord_a - coord_b))
            if d < min_allowed:
                cost += 25.0 * (min_allowed - d) ** 2
    for label_a, coord_a in node_items:
        atom_a = label_a.rsplit(":", 1)[-1]
        for context_label, coord_b in context_atoms:
            atom_b = context_label.rsplit(":", 1)[-1]
            d = float(np.linalg.norm(coord_a - coord_b))
            min_allowed = _nonbond_min_distance(atom_a, atom_b)
            if d < min_allowed:
                cost += 20.0 * (min_allowed - d) ** 2

    for label, coord in nodes.items():
        ref = initial_nodes.get(label)
        if ref is None:
            continue
        _, pos_txt, atom = label.split(":")
        pos = int(pos_txt)
        chain = key[0]
        if (chain, pos) == key and atom in {"CB", "CG", "CD"}:
            weight = 0.08
        elif (chain, pos) == key:
            weight = 0.25
        else:
            weight = 0.9
        cost += weight * float(np.linalg.norm(coord - ref) ** 2)
    return float(cost)


def _classify_pro_failure(reason: str) -> str:
    message = reason.lower()
    if "n-cd" in message or "ring" in message:
        return "ring_closure"
    if "closest rejected pair" in message or "clash" in message:
        return "clash"
    if "continuity" in message:
        return "continuity"
    if "sanitize" in message:
        return "sanitize"
    return "geometry"


def _build_proline_seeds(
    key: tuple[str, int],
    atom_lines: dict[str, str],
) -> list[dict[str, np.ndarray]]:
    required_atoms = {"N", "CA", "C", "CB", "CG", "CD"}
    target_backbone = np.asarray(
        [
            [float(atom_lines["N"][30:38]), float(atom_lines["N"][38:46]), float(atom_lines["N"][46:54])],
            [float(atom_lines["CA"][30:38]), float(atom_lines["CA"][38:46]), float(atom_lines["CA"][46:54])],
            [float(atom_lines["C"][30:38]), float(atom_lines["C"][38:46]), float(atom_lines["C"][46:54])],
        ],
        dtype=np.float64,
    )
    seeds: list[dict[str, np.ndarray]] = []
    side_atoms = ["CB", "CG", "CD"]
    downstream_atoms = ["CG", "CD"]
    angle_scan_main = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    angle_scan_minor = (0.0, 120.0, 240.0)
    for template in _FROZEN_PRO_TEMPLATES:
        template_map = {k: np.asarray(v, dtype=np.float64) for k, v in template.items()}
        donor_backbone = np.asarray([template_map["N"], template_map["CA"], template_map["C"]], dtype=np.float64)
        rot, trans = _kabsch_rotation_translation(donor_backbone, target_backbone)
        trial_map: dict[str, np.ndarray] = {}
        for atom_name in required_atoms:
            if atom_name in {"N", "CA", "C"}:
                trial_map[atom_name] = np.asarray(
                    [float(atom_lines[atom_name][30:38]), float(atom_lines[atom_name][38:46]), float(atom_lines[atom_name][46:54])],
                    dtype=np.float64,
                )
            else:
                trial_map[atom_name] = rot @ template_map[atom_name] + trans
        for angle_n_ca in angle_scan_main:
            step_1 = _rotate_atoms(
                residue_coords=trial_map,
                atom_names=side_atoms,
                axis_start=trial_map["N"],
                axis_end=trial_map["CA"],
                angle_deg=angle_n_ca,
            )
            for angle_ca_c in angle_scan_main:
                step_2 = _rotate_atoms(
                    residue_coords=step_1,
                    atom_names=side_atoms,
                    axis_start=step_1["CA"],
                    axis_end=step_1["C"],
                    angle_deg=angle_ca_c,
                )
                for angle_ca_cb in angle_scan_minor:
                    seeds.append(
                        _rotate_atoms(
                            residue_coords=step_2,
                            atom_names=downstream_atoms,
                            axis_start=step_2["CA"],
                            axis_end=step_2["CB"],
                            angle_deg=angle_ca_cb,
                        )
                    )
    return seeds


def _build_pro_optimizer_state(
    key: tuple[str, int],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    seed: dict[str, np.ndarray],
    window: str,
) -> tuple[dict[str, np.ndarray], set[str], list[tuple[str, str]]]:
    chain, pos = key
    scope_keys = _pro_window_keys(key, residue_coords, window)
    nodes: dict[str, np.ndarray] = {}
    movable: set[str] = set()
    covalent_pairs: list[tuple[str, str]] = []

    def add_node(label: str, coord: np.ndarray, is_movable: bool):
        nodes[label] = np.asarray(coord, dtype=np.float64)
        if is_movable:
            movable.add(label)

    def connect(label_a: str, label_b: str):
        if label_a in nodes and label_b in nodes:
            covalent_pairs.append((label_a, label_b))

    for scope_key in scope_keys:
        c, p = scope_key
        atom_map = residue_coords[scope_key]
        base = f"{c}:{p}:"
        for atom in ("N", "CA", "C", "O"):
            if atom not in atom_map:
                continue
            if scope_key == key and atom in seed:
                coord = seed[atom]
            else:
                coord = np.asarray(atom_map[atom], dtype=np.float64)
            add_node(base + atom, coord, is_movable=True)
        if scope_key == key:
            for atom in ("CB", "CG", "CD"):
                if atom in seed:
                    add_node(base + atom, seed[atom], is_movable=True)

    for scope_key in scope_keys:
        c, p = scope_key
        base = f"{c}:{p}:"
        connect(base + "N", base + "CA")
        connect(base + "CA", base + "C")
        connect(base + "C", base + "O")
    for scope_key in scope_keys:
        c, p = scope_key
        connect(f"{c}:{p}:C", f"{c}:{p+1}:N")
    mut_base = f"{chain}:{pos}:"
    connect(mut_base + "CA", mut_base + "CB")
    connect(mut_base + "CB", mut_base + "CG")
    connect(mut_base + "CG", mut_base + "CD")
    connect(mut_base + "N", mut_base + "CD")
    return nodes, movable, covalent_pairs


def _optimize_pro_state(
    key: tuple[str, int],
    nodes: dict[str, np.ndarray],
    movable: set[str],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    covalent_pairs: list[tuple[str, str]],
    context_atoms: list[tuple[str, np.ndarray]],
    max_steps: int,
) -> tuple[dict[str, np.ndarray], float]:
    optimized = {k: v.copy() for k, v in nodes.items()}
    initial_nodes = {k: v.copy() for k, v in nodes.items()}
    pair_lookup = {}
    for a, b in covalent_pairs:
        pair_lookup.setdefault(a, []).append(b)
        pair_lookup.setdefault(b, []).append(a)
    step_size = 0.035
    max_step = 0.12
    for _ in range(max_steps):
        updated = {k: v.copy() for k, v in optimized.items()}
        bonded_pairs = {frozenset((a, b)) for a, b in covalent_pairs}
        for label in sorted(movable):
            atom = label.rsplit(":", 1)[-1]
            force = np.zeros(3, dtype=np.float64)
            pos = optimized[label]
            for nei in pair_lookup.get(label, []):
                nei_pos = optimized[nei]
                vec = nei_pos - pos
                dist = float(np.linalg.norm(vec))
                if dist < 1e-8:
                    continue
                nei_atom = nei.rsplit(":", 1)[-1]
                ideal = _ideal_bond_length(atom, nei_atom)
                force += 2.2 * (dist - ideal) * (vec / dist)

            for context_label, other in context_atoms:
                vec = pos - other
                dist = float(np.linalg.norm(vec))
                if dist < 1e-8:
                    continue
                other_atom = context_label.rsplit(":", 1)[-1]
                min_allowed = _nonbond_min_distance(atom, other_atom)
                if dist < min_allowed:
                    force += 1.6 * (min_allowed - dist) * (vec / dist)

            for other_label, other_pos in optimized.items():
                if other_label == label:
                    continue
                if frozenset((label, other_label)) in bonded_pairs:
                    continue
                vec = pos - other_pos
                dist = float(np.linalg.norm(vec))
                if dist < 1e-8:
                    continue
                other_atom = other_label.rsplit(":", 1)[-1]
                min_allowed = _nonbond_min_distance(atom, other_atom)
                if dist < min_allowed:
                    force += 1.2 * (min_allowed - dist) * (vec / dist)

            _, pos_txt, atom_name = label.split(":")
            if (key[0], int(pos_txt)) == key and atom_name in {"CB", "CG", "CD"}:
                reg = 0.07
            elif (key[0], int(pos_txt)) == key:
                reg = 0.22
            else:
                reg = 0.8
            force += reg * (initial_nodes[label] - pos)
            delta = step_size * force
            norm = float(np.linalg.norm(delta))
            if norm > max_step:
                delta *= max_step / norm
            trial = pos + delta
            ref = initial_nodes[label]
            _, pos_txt, atom_name = label.split(":")
            if (key[0], int(pos_txt)) == key and atom_name in {"CB", "CG", "CD"}:
                max_disp = 1.4
            elif (key[0], int(pos_txt)) == key:
                max_disp = 0.35
            else:
                max_disp = 0.12
            vec_from_ref = trial - ref
            disp = float(np.linalg.norm(vec_from_ref))
            if disp > max_disp:
                trial = ref + vec_from_ref * (max_disp / disp)
            updated[label] = trial
        optimized = updated

    final_cost = _pro_solver_cost(
        key=key,
        nodes=optimized,
        residue_coords=residue_coords,
        covalent_pairs=covalent_pairs,
        context_atoms=context_atoms,
        initial_nodes=initial_nodes,
    )
    return optimized, final_cost


def _solve_proline_internal(
    key: tuple[str, int],
    atom_lines: dict[str, str],
    residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    window: str,
    max_steps: int,
) -> tuple[dict[tuple[str, int], dict[str, np.ndarray]] | None, str | None, dict[str, Any]]:
    del window  # template transplant currently updates only the mutated residue
    del max_steps
    chain, pos = key
    target_atom_map = residue_coords[key]
    required = ("N", "CA", "C", "CB", "CG", "CD")
    target_has_o = "O" in target_atom_map
    context_atoms = _build_context_atoms(key=key, residue_coords=residue_coords)

    candidates: list[dict[str, np.ndarray]] = [
        {atom: np.asarray(coords, dtype=np.float64) for atom, coords in template.items()}
        for template in _FROZEN_PRO_TEMPLATES
    ]
    try:
        rdkit_pro = _get_rdkit_residue_template_coords("PRO")
        candidates.append({atom: np.asarray(coords, dtype=np.float64) for atom, coords in rdkit_pro.items()})
    except ValueError:
        pass

    if not candidates:
        return None, "no PRO template candidates available", {}

    best_updates: dict[tuple[str, int], dict[str, np.ndarray]] | None = None
    best_diag: dict[str, Any] = {}
    best_reason = "no valid PRO template transplant"
    best_score = float("inf")
    best_offender: tuple[str, str, float] | None = None

    for candidate_idx, template in enumerate(candidates):
        missing = [atom for atom in required if atom not in template]
        if missing:
            continue
        align_atoms = ["N", "CA", "C"]
        if target_has_o and "O" in template:
            align_atoms.append("O")
        src = np.asarray([template[a] for a in align_atoms], dtype=np.float64)
        dst = np.asarray([target_atom_map[a] for a in align_atoms], dtype=np.float64)
        rot, trans = _kabsch_rotation_translation(src, dst)

        backbone_map: dict[str, np.ndarray] = {}
        for atom in ("N", "CA", "C", "O"):
            if atom in target_atom_map:
                backbone_map[atom] = np.asarray(target_atom_map[atom], dtype=np.float64)
        side_seed = {
            "CB": rot @ template["CB"] + trans,
            "CG": rot @ template["CG"] + trans,
            "CD": rot @ template["CD"] + trans,
        }
        angle_scan_main = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
        angle_scan_minor = (0.0, 120.0, 240.0)
        for angle_n_ca in angle_scan_main:
            step_1 = _rotate_atoms(
                residue_coords=side_seed,
                atom_names=["CB", "CG", "CD"],
                axis_start=backbone_map["N"],
                axis_end=backbone_map["CA"],
                angle_deg=angle_n_ca,
            )
            for angle_ca_c in angle_scan_main:
                step_2 = _rotate_atoms(
                    residue_coords=step_1,
                    atom_names=["CB", "CG", "CD"],
                    axis_start=backbone_map["CA"],
                    axis_end=backbone_map["C"],
                    angle_deg=angle_ca_c,
                )
                for angle_ca_cb in angle_scan_minor:
                    side_map = _rotate_atoms(
                        residue_coords=step_2,
                        atom_names=["CG", "CD"],
                        axis_start=backbone_map["CA"],
                        axis_end=step_2["CB"],
                        angle_deg=angle_ca_cb,
                    )
                    residue_map = dict(backbone_map)
                    residue_map.update(side_map)

                    violations = _proline_hard_violations(
                        key=key,
                        residue_coords=residue_map,
                        residue_coords_by_key=residue_coords,
                        context_atoms=context_atoms,
                    )
                    if violations:
                        top = sorted(violations, key=lambda v: v[2])[0]
                        if best_offender is None or top[2] < best_offender[2]:
                            best_offender = (str(top[0]), str(top[1]), float(top[2]))
                        continue

                    sidechain_map = {atom: residue_map[atom] for atom in ("CB", "CG", "CD")}
                    score = _local_clash_score(sidechain_map, [coord for _, coord in context_atoms])
                    if score < best_score:
                        best_score = score
                        best_updates = {key: residue_map}
                        best_diag = {
                            "pro_solver_mode": "template_transplant",
                            "pro_solver_window": "residue_only",
                            "pro_solver_iterations": 0,
                            "pro_solver_final_cost": float(score),
                            "pro_template_candidate_index": int(candidate_idx),
                            "pro_template_alignment_atoms": align_atoms,
                            "pro_template_rotation_deg": {
                                "n_ca": float(angle_n_ca),
                                "ca_c": float(angle_ca_c),
                                "ca_cb": float(angle_ca_cb),
                            },
                            "pro_applied_remodel_residues": [f"{chain}:{pos}"],
                        }

    if best_updates is not None:
        return best_updates, None, best_diag
    if best_offender is not None:
        atom_a, atom_b, dist = best_offender
        best_reason = f"closest rejected pair {atom_a}-{atom_b} at {dist:.3f}A"
    return None, best_reason, {}


def _kabsch_rotation_translation(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    cov = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    trans = target_center - (rot @ source_center)
    return rot, trans


_RESIDUE_TEMPLATE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _get_rdkit_residue_template_coords(residue_name_3: str) -> dict[str, np.ndarray]:
    cached = _RESIDUE_TEMPLATE_CACHE.get(residue_name_3)
    if cached is not None:
        return cached

    residue_one = rotamer.three_to_one[residue_name_3]
    mol = Chem.MolFromSequence(residue_one)
    if mol is None:
        raise ValueError(f"Failed to build RDKit residue template for `{residue_name_3}`.")
    mol = Chem.AddHs(mol, addCoords=True)
    status = AllChem.EmbedMolecule(mol, randomSeed=20260421)
    if status != 0:
        raise ValueError(f"RDKit embedding failed for residue template `{residue_name_3}`.")

    conf = mol.GetConformer()
    coords: dict[str, np.ndarray] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        name = info.GetName().strip().upper()
        if not name or name == "OXT":
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords[name] = np.asarray([pos.x, pos.y, pos.z], dtype=np.float64)

    for atom_name in ("N", "CA", "C"):
        if atom_name not in coords:
            raise ValueError(
                f"RDKit residue template `{residue_name_3}` is missing required backbone atom `{atom_name}`."
            )
    _RESIDUE_TEMPLATE_CACHE[residue_name_3] = coords
    return coords


def _check_bond(
    spec: MutationSpec,
    coords: dict[str, tuple[float, float, float]],
    atom_a: str,
    atom_b: str,
    errors: list[str],
):
    if atom_a not in coords or atom_b not in coords:
        return
    dist = _distance(coords[atom_a], coords[atom_b])
    if dist < _BOND_MIN or dist > _BOND_MAX:
        errors.append(
            f"{spec.compact}: invalid local geometry `{atom_a}-{atom_b}` distance {dist:.3f}A "
            f"(expected within {_BOND_MIN:.2f}-{_BOND_MAX:.2f}A)"
        )


def _rdkit_validate_residue_block(spec: MutationSpec, residue_lines: list[str]):
    block = "".join(residue_lines) + "TER\nEND\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(
            f"{spec.compact}: RDKit failed to parse rebuilt residue block."
        )


def _validate_rebuilt_residue_geometry(
    spec: MutationSpec,
    residue_lines: list[str],
    residue_name: str,
):
    coords: dict[str, tuple[float, float, float]] = {}
    for line in residue_lines:
        atom_name = line[12:16].strip().upper()
        coords[atom_name] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )

    errors: list[str] = []
    for atom in ("N", "CA", "C"):
        if atom not in coords:
            errors.append(f"{spec.compact}: rebuilt residue missing required backbone atom `{atom}`")

    residue_id = rotamer.residue_vocab[residue_name]
    expected_atoms = {
        atom_name
        for atom_name, present in zip(
            rotamer.restype_name_to_atom14_names[residue_name],
            rotamer.restype_atom14_mask[residue_id].tolist(),
        )
        if atom_name and bool(present)
    }
    missing = sorted(expected_atoms.difference(coords))
    if missing:
        errors.append(f"{spec.compact}: rebuilt residue missing expected atoms: {', '.join(missing)}")

    if residue_name != "GLY":
        _check_bond(spec, coords, "CA", "CB", errors)
    if residue_name == "PRO":
        _check_bond(spec, coords, "N", "CD", errors)

    if errors:
        raise ValueError("; ".join(errors))

    _rdkit_validate_residue_block(spec, residue_lines)


def _normalize_one_letter(residue: str, *, field: str) -> str:
    if not isinstance(residue, str):
        raise ValueError(f"`{field}` must be a residue string.")
    token = residue.strip().upper()
    if len(token) == 3:
        one = rotamer.three_to_one.get(token)
        if one is None:
            raise ValueError(f"`{field}` has unsupported residue `{residue}`.")
        return one
    if len(token) != 1 or token not in rotamer.one_to_three:
        raise ValueError(f"`{field}` has unsupported residue `{residue}`.")
    return token


def parse_compact_mutation(token: str) -> MutationSpec:
    match = _COMPACT_MUTATION_PATTERN.match(token.strip())
    if not match:
        raise ValueError(
            f"Invalid mutation token `{token}`. Expected format `[chain][old][position][new]`, e.g. `AG76A`."
        )
    chain, old_res, position, new_res = match.groups()
    old_res_1 = _normalize_one_letter(old_res, field="old_res")
    new_res_1 = _normalize_one_letter(new_res, field="new_res")
    return MutationSpec(
        chain=chain,
        position=int(position),
        old_res_1=old_res_1,
        new_res_1=new_res_1,
        old_res_3=rotamer.one_to_three[old_res_1],
        new_res_3=rotamer.one_to_three[new_res_1],
    )


def parse_mutation_dict(payload: dict[str, Any]) -> MutationSpec:
    missing = [key for key in ("chain", "old_res", "position", "new_res") if key not in payload]
    if missing:
        raise ValueError(f"Mutation dict is missing required keys: {', '.join(missing)}")
    chain = str(payload["chain"]).strip()
    if not chain:
        raise ValueError("Mutation `chain` must be non-empty.")
    try:
        position = int(payload["position"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Mutation `position` must be an integer. Got `{payload['position']}`.") from error
    old_res_1 = _normalize_one_letter(str(payload["old_res"]), field="old_res")
    new_res_1 = _normalize_one_letter(str(payload["new_res"]), field="new_res")
    return MutationSpec(
        chain=chain,
        position=position,
        old_res_1=old_res_1,
        new_res_1=new_res_1,
        old_res_3=rotamer.one_to_three[old_res_1],
        new_res_3=rotamer.one_to_three[new_res_1],
    )


def normalize_mutations(mutations: Any) -> list[MutationSpec]:
    if mutations is None:
        return []
    parsed: list[MutationSpec] = []
    if isinstance(mutations, str):
        chunks = [chunk.strip() for chunk in mutations.split(",") if chunk.strip()]
        parsed.extend(parse_compact_mutation(chunk) for chunk in chunks)
        return parsed
    if isinstance(mutations, dict):
        return [parse_mutation_dict(mutations)]
    if isinstance(mutations, (list, tuple)):
        for item in mutations:
            if isinstance(item, str):
                parsed.append(parse_compact_mutation(item))
            elif isinstance(item, dict):
                parsed.append(parse_mutation_dict(item))
            else:
                raise ValueError(
                    f"Unsupported mutation entry type `{type(item).__name__}`. Use compact strings or dict entries."
                )
        return parsed
    raise ValueError("Unsupported `mutations` payload. Use a compact string, list of compact strings, or list[dict].")


def preprocess_pdb_files_with_mutations(
    pdb_files: list[str],
    mutations: list[MutationSpec],
    output_dir: str,
    pro_remodel_window: str = "tripeptide",
    pro_remodel_max_steps: int = 24,
) -> tuple[list[str], list[str]]:
    if not mutations:
        return pdb_files, []
    mutation_dir = Path(output_dir).resolve() / "_mutation_inputs"
    mutation_dir.mkdir(parents=True, exist_ok=True)
    mutation_sites = sorted({m.selector for m in mutations})
    mutation_targets = {(m.chain, m.position): m for m in mutations}
    if pro_remodel_window not in _PRO_SCOPE_OFFSETS:
        raise ValueError(
            f"`pro_remodel_window` must be one of {sorted(_PRO_SCOPE_OFFSETS)}. Got `{pro_remodel_window}`."
        )
    if pro_remodel_max_steps < 1:
        raise ValueError("`pro_remodel_max_steps` must be >= 1.")
    out_files: list[str] = []
    pro_diagnostics: list[dict[str, Any]] = []

    for pdb_file in pdb_files:
        with open(pdb_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        original_text = "".join(lines)
        enforce_full_sanitize = Chem.MolFromPDBBlock(original_text, sanitize=True, removeHs=False) is not None

        present_residues: dict[tuple[str, int], str] = {}
        residue_atoms: dict[tuple[str, int], dict[str, str]] = {}
        residue_coords: dict[tuple[str, int], dict[str, tuple[float, float, float]]] = {}
        for line in lines:
            if not line.startswith("ATOM"):
                continue
            insertion_code = line[26].strip()
            if insertion_code:
                continue
            chain = line[21].strip()
            if not chain:
                continue
            try:
                position = int(line[22:26])
            except ValueError:
                continue
            resname = line[17:20].strip().upper()
            if resname not in rotamer.three_to_one:
                continue
            present_residues.setdefault((chain, position), resname)
            atom_name = line[12:16].strip().upper()
            residue_atoms.setdefault((chain, position), {}).setdefault(atom_name, line)
            residue_coords.setdefault((chain, position), {}).setdefault(
                atom_name,
                (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ),
            )

        errors: list[str] = []
        for key, spec in mutation_targets.items():
            current = present_residues.get(key)
            if current is None:
                errors.append(f"{spec.compact}: target residue `{spec.chain}:{spec.position}` not found")
                continue
            if current != spec.old_res_3:
                errors.append(
                    f"{spec.compact}: source residue mismatch at `{spec.chain}:{spec.position}` "
                    f"(expected {spec.old_res_3}, found {current})"
                )
        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"Mutation validation failed for `{os.path.basename(pdb_file)}`: {joined}")

        rebuilt_residue_lines: dict[tuple[str, int], list[str]] = {}

        def _build_rebuilt_lines(
            *,
            target_key: tuple[str, int],
            coords_update: dict[str, np.ndarray],
            target_resname: str,
            allowed_atoms: list[str],
        ) -> list[str]:
            atom_lines_local = residue_atoms.get(target_key, {})
            rebuilt: list[str] = []
            local_serial = 1
            for atom_name in allowed_atoms:
                if atom_name not in atom_lines_local and atom_name not in coords_update:
                    continue
                if atom_name in coords_update:
                    coord = coords_update[atom_name]
                    x = float(coord[0])
                    y = float(coord[1])
                    z = float(coord[2])
                else:
                    src = atom_lines_local[atom_name]
                    x = float(src[30:38])
                    y = float(src[38:46])
                    z = float(src[46:54])
                src_line = atom_lines_local.get(atom_name)
                if src_line is not None:
                    chain_local = src_line[21]
                    resseq_local = int(src_line[22:26])
                    insertion_local = src_line[26]
                    try:
                        occupancy_local = float(src_line[54:60])
                    except ValueError:
                        occupancy_local = 1.00
                    try:
                        bfactor_local = float(src_line[60:66])
                    except ValueError:
                        bfactor_local = 20.00
                    element_local = src_line[76:78].strip() or atom_name[0]
                else:
                    chain_local = target_key[0]
                    resseq_local = target_key[1]
                    insertion_local = " "
                    occupancy_local = 1.00
                    bfactor_local = 20.00
                    element_local = atom_name[0]
                rebuilt.append(
                    f"ATOM  {local_serial:5d} {atom_name:>4s} {target_resname:>3s} {chain_local:1s}{resseq_local:4d}{insertion_local:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy_local:6.2f}{bfactor_local:6.2f}          {element_local:>2s}\n"
                )
                local_serial += 1
            return rebuilt
        for key, spec in mutation_targets.items():
            atom_lines = residue_atoms.get(key, {})
            if not {"N", "CA", "C"}.issubset(atom_lines):
                raise ValueError(
                    f"Mutation preprocessing failed for `{spec.compact}` in `{os.path.basename(pdb_file)}`: "
                    "backbone atoms N/CA/C are required."
                )

            n_line = atom_lines["N"]
            ca_line = atom_lines["CA"]
            c_line = atom_lines["C"]
            residue_id = rotamer.residue_vocab[spec.new_res_3]
            atom14_names = rotamer.restype_name_to_atom14_names[spec.new_res_3]
            atom14_mask = rotamer.restype_atom14_mask[residue_id]
            expected_atoms = [
                atom_name
                for atom_name, present in zip(atom14_names, atom14_mask.tolist())
                if atom_name and bool(present)
            ]

            template_coords = _get_rdkit_residue_template_coords(spec.new_res_3)
            missing_template_atoms = sorted(
                atom_name for atom_name in expected_atoms if atom_name not in template_coords and atom_name not in _BACKBONE_ATOMS
            )
            if missing_template_atoms:
                raise ValueError(
                    f"{spec.compact}: RDKit residue template `{spec.new_res_3}` is missing atoms: "
                    f"{', '.join(missing_template_atoms)}"
                )
            source_points = np.asarray(
                [template_coords["N"], template_coords["CA"], template_coords["C"]],
                dtype=np.float64,
            )
            target_points = np.asarray(
                [residue_coords[key]["N"], residue_coords[key]["CA"], residue_coords[key]["C"]],
                dtype=np.float64,
            )
            rot, trans = _kabsch_rotation_translation(source_points, target_points)

            chain = key[0]
            resseq = key[1]
            insertion_code = n_line[26] if len(n_line) > 26 else " "
            occupancy = 1.00
            bfactor = 20.00
            try:
                occupancy = float(ca_line[54:60])
            except ValueError:
                pass
            try:
                bfactor = float(ca_line[60:66])
            except ValueError:
                pass

            residue_coord_map: dict[str, np.ndarray] = {}
            proline_updates = None
            proline_donor_error = None
            proline_diag: dict[str, Any] = {}
            if spec.new_res_3 == "PRO":
                proline_updates, proline_donor_error, proline_diag = _solve_proline_internal(
                    key=key,
                    atom_lines=atom_lines,
                    residue_coords=residue_coords,
                    window=pro_remodel_window,
                    max_steps=pro_remodel_max_steps,
                )
                if proline_updates is None:
                    failure_category = _classify_pro_failure(proline_donor_error or "")
                    raise ValueError(
                        f"{spec.compact}: no valid PRO pose for local environment. "
                        f"failing constraint category: {failure_category}. {proline_donor_error or ''}".strip()
                    )
                # Use solver updates as source-of-truth for all movable atoms in the remodel window.
                for update_key, update_coords in proline_updates.items():
                    update_resname = spec.new_res_3 if update_key == key else present_residues[update_key]
                    if update_key == key:
                        allowed = expected_atoms
                    else:
                        allowed = [name for name in residue_atoms.get(update_key, {}).keys()]
                    rebuilt_lines = _build_rebuilt_lines(
                        target_key=update_key,
                        coords_update=update_coords,
                        target_resname=update_resname,
                        allowed_atoms=allowed,
                    )
                    if update_key == key:
                        try:
                            _validate_rebuilt_residue_geometry(spec, rebuilt_lines, spec.new_res_3)
                        except ValueError as error:
                            raise ValueError(
                                f"{spec.compact}: no valid PRO pose for local environment. "
                                f"failing constraint category: {_classify_pro_failure(str(error))}. {error}"
                            ) from error
                    rebuilt_residue_lines[update_key] = rebuilt_lines
                proline_diag = dict(proline_diag)
                proline_diag.update(
                    {
                        "mutation": spec.compact,
                        "selector": spec.selector,
                        "pro_applied_remodel_residues": sorted(
                            f"{chain}:{position}" for chain, position in proline_updates.keys()
                        ),
                    }
                )
                pro_diagnostics.append(proline_diag)
                continue

            for atom_name in expected_atoms:
                if atom_name in _BACKBONE_ATOMS and atom_name in atom_lines:
                    src = atom_lines[atom_name]
                    residue_coord_map[atom_name] = np.asarray(
                        [float(src[30:38]), float(src[38:46]), float(src[46:54])],
                        dtype=np.float64,
                    )
                else:
                    residue_coord_map[atom_name] = rot @ template_coords[atom_name] + trans

            # Reduce severe local clashes by scanning a deterministic chi1-like rotation
            # around CA-CB for downstream sidechain atoms.
            if spec.new_res_3 != "PRO" and "CA" in residue_coord_map and "CB" in residue_coord_map:
                rotate_candidates = [a for a in expected_atoms if a not in _BACKBONE_ATOMS and a not in {"CB"}]
                if rotate_candidates:
                    context_coords = [
                        np.asarray(coord, dtype=np.float64)
                        for other_key, atom_coord_map in residue_coords.items()
                        if other_key != key
                        for coord in atom_coord_map.values()
                    ]
                    ca = residue_coord_map["CA"]
                    cb = residue_coord_map["CB"]
                    best_coords = residue_coord_map
                    best_score = _local_clash_score(best_coords, context_coords)
                    for angle_deg in (60, 120, 180, 240, 300):
                        trial = dict(residue_coord_map)
                        angle_rad = np.deg2rad(float(angle_deg))
                        for atom_name in rotate_candidates:
                            trial[atom_name] = _rotate_point_around_axis(trial[atom_name], ca, cb, angle_rad)
                        score = _local_clash_score(trial, context_coords)
                        if score < best_score:
                            best_score = score
                            best_coords = trial
                    residue_coord_map = best_coords

            # Secondary orientation scan around N-CA for full sidechain placement.
            if spec.new_res_3 != "PRO" and "N" in residue_coord_map and "CA" in residue_coord_map:
                rotate_candidates = [a for a in expected_atoms if a not in _BACKBONE_ATOMS]
                if rotate_candidates:
                    context_coords = [
                        np.asarray(coord, dtype=np.float64)
                        for other_key, atom_coord_map in residue_coords.items()
                        if other_key != key
                        for coord in atom_coord_map.values()
                    ]
                    n_atom = residue_coord_map["N"]
                    ca_atom = residue_coord_map["CA"]
                    best_coords = residue_coord_map
                    best_score = _local_clash_score(best_coords, context_coords)
                    for angle_deg in (60, 120, 180, 240, 300):
                        trial = dict(residue_coord_map)
                        angle_rad = np.deg2rad(float(angle_deg))
                        for atom_name in rotate_candidates:
                            trial[atom_name] = _rotate_point_around_axis(trial[atom_name], n_atom, ca_atom, angle_rad)
                        score = _local_clash_score(trial, context_coords)
                        if score < best_score:
                            best_score = score
                            best_coords = trial
                    residue_coord_map = best_coords

            # Tertiary orientation scan around CA-C to further reduce clashes with
            # previous-residue carbonyl atoms (important for bulky/proline cases).
            if spec.new_res_3 != "PRO" and "CA" in residue_coord_map and "C" in residue_coord_map:
                rotate_candidates = [a for a in expected_atoms if a not in _BACKBONE_ATOMS]
                if rotate_candidates:
                    context_coords = [
                        np.asarray(coord, dtype=np.float64)
                        for other_key, atom_coord_map in residue_coords.items()
                        if other_key != key
                        for coord in atom_coord_map.values()
                    ]
                    ca_atom = residue_coord_map["CA"]
                    c_atom = residue_coord_map["C"]
                    best_coords = residue_coord_map
                    best_score = _local_clash_score(best_coords, context_coords)
                    for angle_deg in (60, 120, 180, 240, 300):
                        trial = dict(residue_coord_map)
                        angle_rad = np.deg2rad(float(angle_deg))
                        for atom_name in rotate_candidates:
                            trial[atom_name] = _rotate_point_around_axis(trial[atom_name], ca_atom, c_atom, angle_rad)
                        score = _local_clash_score(trial, context_coords)
                        if score < best_score:
                            best_score = score
                            best_coords = trial
                    residue_coord_map = best_coords

            rebuilt_lines: list[str] = []
            local_serial = 1
            for atom_name in expected_atoms:
                coord = residue_coord_map[atom_name]
                x = float(coord[0])
                y = float(coord[1])
                z = float(coord[2])
                element = atom_name[0]
                rebuilt_lines.append(
                    f"ATOM  {local_serial:5d} {atom_name:>4s} {spec.new_res_3:>3s} {chain:1s}{resseq:4d}{insertion_code:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}\n"
                )
                local_serial += 1
            try:
                _validate_rebuilt_residue_geometry(spec, rebuilt_lines, spec.new_res_3)
            except ValueError as error:
                raise
            rebuilt_residue_lines[key] = rebuilt_lines

        mutated_lines: list[str] = []
        serial = 1
        emitted_rebuilt_keys: set[tuple[str, int]] = set()
        for line in lines:
            if not line.startswith("ATOM"):
                if line.startswith("TER"):
                    mutated_lines.append(f"TER   {serial:5d}\n")
                    serial += 1
                elif line.startswith("CONECT") or line.startswith("MASTER"):
                    # Atom serials are rewritten during mutation output; stale CONECT / MASTER
                    # records would reference wrong atoms and create invalid bonds in viewers.
                    continue
                else:
                    mutated_lines.append(line)
                continue
            insertion_code = line[26].strip()
            chain = line[21].strip()
            if insertion_code or not chain:
                mutated_lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
                serial += 1
                continue
            try:
                position = int(line[22:26])
            except ValueError:
                mutated_lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
                serial += 1
                continue
            key = (chain, position)
            if key not in rebuilt_residue_lines:
                mutated_lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
                serial += 1
                continue
            if key in emitted_rebuilt_keys:
                continue
            for rebuilt_line in rebuilt_residue_lines[key]:
                mutated_lines.append(f"{rebuilt_line[:6]}{serial:5d}{rebuilt_line[11:]}")
                serial += 1
            emitted_rebuilt_keys.add(key)

        mutated_text = "".join(mutated_lines)
        sanitized_mol = Chem.MolFromPDBBlock(mutated_text, sanitize=True, removeHs=False)
        if enforce_full_sanitize and sanitized_mol is None:
            mutation_tokens = ", ".join(spec.compact for spec in mutation_targets.values())
            pro_tokens = [spec.compact for spec in mutation_targets.values() if spec.new_res_3 == "PRO"]
            if pro_tokens:
                raise ValueError(
                    f"{','.join(pro_tokens)}: no valid PRO pose for local environment. "
                    "failing constraint category: sanitize. "
                    f"Mutation preprocessing produced an invalid full-structure geometry for "
                    f"`{os.path.basename(pdb_file)}` ({mutation_tokens})."
                )
            raise ValueError(
                f"Mutation preprocessing produced an invalid full-structure geometry for "
                f"`{os.path.basename(pdb_file)}` ({mutation_tokens})."
            )
        out_path = mutation_dir / f"{Path(pdb_file).stem}.mutated.pdb"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(mutated_text)
        # Keep explicit connectivity to avoid viewer distance-based bond hallucinations.
        append_conect_records_inplace(str(out_path))
        out_files.append(str(out_path))

    diagnostics_payload: dict[str, Any] = {
        "pro_remodel_window": pro_remodel_window,
        "pro_remodel_max_steps": pro_remodel_max_steps,
        "pro_mutation_count": len(pro_diagnostics),
        "per_mutation": pro_diagnostics,
    }
    if pro_diagnostics:
        applied_residue_set = sorted(
            {
                residue
                for diag in pro_diagnostics
                for residue in diag.get("pro_applied_remodel_residues", [])
            }
        )
        diagnostics_payload.update(
            {
                "pro_solver_mode": pro_diagnostics[0].get("pro_solver_mode", "template_transplant"),
                "pro_solver_window": pro_diagnostics[0].get("pro_solver_window", "residue_only"),
                "pro_solver_iterations": max(d.get("pro_solver_iterations", 0) for d in pro_diagnostics),
                "pro_solver_final_cost": min(d.get("pro_solver_final_cost", float("inf")) for d in pro_diagnostics),
                "pro_applied_remodel_residues": applied_residue_set,
            }
        )
    (mutation_dir / "mutation_diagnostics.json").write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return out_files, mutation_sites
