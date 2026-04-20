from __future__ import annotations

import os
import tempfile

import torch

from diffpack.structure_checker import check_structure


def _residue_key_to_index(protein):
    mapping = {}
    for idx, (chain, num) in enumerate(zip(protein.residue_chain, protein.residue_number)):
        mapping[(str(chain).strip() or "A", int(num))] = idx
    return mapping


def _severe_clash_residue_ids(protein, threshold: float) -> set[int]:
    fd, tmp_path = tempfile.mkstemp(suffix=".pdb", prefix="diffpack_clash_guard_")
    os.close(fd)
    try:
        protein.cpu().to_pdb(tmp_path)
        report = check_structure(tmp_path, clash_threshold=float(threshold), top_n_clashes=200)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    residue_map = _residue_key_to_index(protein)
    bad_residues: set[int] = set()
    for clash in report.severe_clashes:
        a_chain, a_resnum, _, _ = clash.atom_a
        b_chain, b_resnum, _, _ = clash.atom_b
        a = residue_map.get((str(a_chain).strip() or "A", int(a_resnum)))
        b = residue_map.get((str(b_chain).strip() or "A", int(b_resnum)))
        if a is not None:
            bad_residues.add(a)
        if b is not None:
            bad_residues.add(b)
    return bad_residues


def mitigate_severe_clashes(pred_protein, reference_protein, threshold: float = 1.0, max_iter: int = 2):
    bad_before = _severe_clash_residue_ids(pred_protein, threshold=threshold)
    if not bad_before:
        return pred_protein, {"clash_guard_applied": False, "clashes_before": 0, "clashes_after": 0, "reverted_residues": 0}

    working = pred_protein
    reverted: set[int] = set()
    for _ in range(max_iter):
        bad = _severe_clash_residue_ids(working, threshold=threshold)
        if not bad:
            break
        reverted.update(bad)
        bad_tensor = torch.as_tensor(sorted(bad), dtype=torch.long, device=working.device)
        atom_mask = torch.isin(working.atom2residue, bad_tensor)
        working.node_position[atom_mask] = reference_protein.node_position[atom_mask]

    bad_after = _severe_clash_residue_ids(working, threshold=threshold)
    return working, {
        "clash_guard_applied": True,
        "clashes_before": len(bad_before),
        "clashes_after": len(bad_after),
        "reverted_residues": len(reverted),
    }
