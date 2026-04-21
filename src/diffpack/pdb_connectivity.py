from __future__ import annotations

from collections import defaultdict
from pathlib import Path


EXPECTED_BONDS = {
    "ALA": [("CA", "CB")],
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("CA", "CB"), ("CB", "SG")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2")],
    "ILE": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("N", "CD")],
    "SER": [("CA", "CB"), ("CB", "OG")],
    "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("CD2", "CE2"), ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2")],
    "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
}


def _residue_key(line: str, segment: int) -> tuple[int, str, int, str]:
    chain = line[21].strip()
    resseq = int(line[22:26])
    icode = line[26].strip()
    return segment, chain, resseq, icode


def append_conect_records_inplace(pdb_path: str) -> bool:
    path = Path(pdb_path)
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    stripped_lines = [line for line in lines if not line.startswith("CONECT") and not line.startswith("MASTER")]
    residues: dict[tuple[int, str, int, str], dict[str, int]] = defaultdict(dict)
    residue_names: dict[tuple[int, str, int, str], str] = {}
    residue_order: list[tuple[int, str, int, str]] = []
    segment = 0

    for line in stripped_lines:
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        key = _residue_key(line, segment)
        if key not in residue_names:
            residue_order.append(key)
            residue_names[key] = line[17:20].strip()
        if atom_name not in residues[key]:
            residues[key][atom_name] = int(line[6:11])

    adjacency: dict[int, set[int]] = defaultdict(set)

    def connect(a_serial: int | None, b_serial: int | None):
        if a_serial is None or b_serial is None or a_serial == b_serial:
            return
        adjacency[a_serial].add(b_serial)
        adjacency[b_serial].add(a_serial)

    # Intra-residue backbone and sidechain topology.
    for key in residue_order:
        atom_map = residues[key]
        resname = residue_names[key]
        connect(atom_map.get("N"), atom_map.get("CA"))
        connect(atom_map.get("CA"), atom_map.get("C"))
        connect(atom_map.get("C"), atom_map.get("O"))
        connect(atom_map.get("C"), atom_map.get("OXT"))
        for atom_a, atom_b in EXPECTED_BONDS.get(resname, []):
            connect(atom_map.get(atom_a), atom_map.get(atom_b))

    # Peptide C-N links for adjacent residues in the same TER segment.
    for prev_key, next_key in zip(residue_order[:-1], residue_order[1:]):
        if prev_key[0] != next_key[0]:
            continue
        prev_atoms = residues[prev_key]
        next_atoms = residues[next_key]
        connect(prev_atoms.get("C"), next_atoms.get("N"))

    if not adjacency:
        return False

    conect_lines: list[str] = []
    for serial in sorted(adjacency):
        neighbors = sorted(adjacency[serial])
        if not neighbors:
            continue
        for offset in range(0, len(neighbors), 4):
            chunk = neighbors[offset : offset + 4]
            conect_lines.append(f"CONECT{serial:5d}" + "".join(f"{n:5d}" for n in chunk) + "\n")

    end_idx = None
    for idx in range(len(stripped_lines) - 1, -1, -1):
        if stripped_lines[idx].startswith("END"):
            end_idx = idx
            break
    if end_idx is None:
        stripped_lines.extend(conect_lines)
    else:
        stripped_lines[end_idx:end_idx] = conect_lines

    with path.open("w", encoding="utf-8") as handle:
        handle.write("".join(stripped_lines))
    return True
