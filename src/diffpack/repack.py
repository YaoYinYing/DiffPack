import re
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import torch


CenterSelector = Tuple[str, int]

_CENTER_SELECTOR_PATTERN = re.compile(r"^([^:\s]+):(-?\d+)$")


def parse_center_residue_selector(selector: str) -> CenterSelector:
    """
    Parse a residue selector in `CHAIN:RESID` format, such as `A:72`.
    """
    match = _CENTER_SELECTOR_PATTERN.match(selector.strip())
    if not match:
        raise ValueError(
            f"Invalid residue selector `{selector}`. Expected format `CHAIN:RESID`, e.g. `A:72`."
        )
    chain_id, residue_number = match.group(1), int(match.group(2))
    return chain_id, residue_number


def parse_center_residue_selectors(selectors: Iterable[str]) -> List[CenterSelector]:
    parsed = []
    for selector in selectors:
        parsed.append(parse_center_residue_selector(selector))
    return parsed


@lru_cache(maxsize=256)
def _load_residue_identifiers_from_pdb_cached(pdb_file: str, allowed_residue_names_key: Tuple[str, ...]) -> List[CenterSelector]:
    """
    Load residue identifiers (chain_id, residue_number) from a PDB file.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError as error:
        raise ImportError(
            "Biopython is required for residue selector mapping. Install with `pip install biopython`."
        ) from error

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = next(structure.get_models())
    allowed_residue_names = set(allowed_residue_names_key)

    identifiers = []
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            hetero_flag, residue_number, insertion_code = residue.id
            if hetero_flag.strip():
                continue
            if insertion_code.strip():
                # `CHAIN:RESID` selectors do not encode insertion code, so skip insertion variants.
                continue
            if residue.resname not in allowed_residue_names:
                continue
            identifiers.append((chain_id, int(residue_number)))
    return identifiers


def load_residue_identifiers_from_pdb(pdb_file: str, allowed_residue_names: Sequence[str]) -> List[CenterSelector]:
    return _load_residue_identifiers_from_pdb_cached(pdb_file, tuple(sorted(allowed_residue_names)))


def load_residue_identifiers_from_protein(protein) -> List[CenterSelector]:
    """
    Prefer native torchdrug residue metadata when available.
    """
    if not hasattr(protein, "chain_id") or not hasattr(protein, "residue_number"):
        raise ValueError("Protein is missing `chain_id` / `residue_number` metadata.")
    if not hasattr(protein, "id2alphabet"):
        raise ValueError("Protein is missing `id2alphabet` mapping for chain decoding.")

    chain_ids = protein.chain_id.detach().cpu().tolist()
    residue_numbers = protein.residue_number.detach().cpu().tolist()
    identifiers: List[CenterSelector] = []
    for chain_idx, residue_number in zip(chain_ids, residue_numbers):
        chain_symbol = protein.id2alphabet.get(int(chain_idx), "")
        identifiers.append((str(chain_symbol), int(residue_number)))
    return identifiers


def select_residues_by_radius(
    atom_positions: torch.Tensor,
    atom2residue: torch.Tensor,
    num_residue: int,
    residue_identifiers: Sequence[CenterSelector],
    center_selectors: Sequence[CenterSelector],
    radius: float,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Select residues whose any atom lies within `radius` from any center residue centroid.
    """
    if radius <= 0:
        raise ValueError(f"`repack_radius` must be > 0, got {radius}.")
    if not center_selectors:
        raise ValueError("`center_residues` must be provided when `repack_radius` is set.")
    if len(residue_identifiers) != int(num_residue):
        raise ValueError(
            "Residue identifier count mismatch: "
            f"{len(residue_identifiers)} from PDB vs {int(num_residue)} in protein graph."
        )

    selector_set = set(center_selectors)
    center_indices = [i for i, residue_id in enumerate(residue_identifiers) if residue_id in selector_set]

    missing_selectors = sorted(selector_set.difference(set(residue_identifiers)))
    if missing_selectors:
        missing = ", ".join([f"{chain}:{resid}" for chain, resid in missing_selectors])
        raise ValueError(f"Center residues not found in PDB: {missing}")

    repack_residue_mask = torch.zeros(num_residue, dtype=torch.bool, device=atom_positions.device)
    radius_tensor = torch.tensor(radius, dtype=atom_positions.dtype, device=atom_positions.device)

    for center_idx in center_indices:
        center_atom_mask = atom2residue == center_idx
        if not center_atom_mask.any():
            continue
        center_centroid = atom_positions[center_atom_mask].mean(dim=0)
        dist = torch.linalg.norm(atom_positions - center_centroid, dim=-1)
        close_atom_mask = dist < radius_tensor
        touched_residue = atom2residue[close_atom_mask]
        repack_residue_mask[touched_residue] = True

    return repack_residue_mask, center_indices


def select_residues_exact(
    num_residue: int,
    residue_identifiers: Sequence[CenterSelector],
    selectors: Sequence[CenterSelector],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not selectors:
        raise ValueError("`selectors` must be non-empty for exact residue selection.")
    if len(residue_identifiers) != int(num_residue):
        raise ValueError(
            "Residue identifier count mismatch: "
            f"{len(residue_identifiers)} from PDB vs {int(num_residue)} in protein graph."
        )
    selector_set = set(selectors)
    repack_residue_mask = torch.zeros(num_residue, dtype=torch.bool, device=device)
    for residue_idx, residue_identifier in enumerate(residue_identifiers):
        if residue_identifier in selector_set:
            repack_residue_mask[residue_idx] = True
    if not repack_residue_mask.any():
        missing = ", ".join([f"{chain}:{resid}" for chain, resid in sorted(selector_set)])
        raise ValueError(f"Mutation residues not found in PDB: {missing}")
    return repack_residue_mask
