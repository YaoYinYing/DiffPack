import numpy as np
import pytest
torch = pytest.importorskip("torch")
from diffpack.backend_preflight import probe_backend_dependencies

_torchdrug_probe = probe_backend_dependencies("torchdrug")
if _torchdrug_probe["status"] != "pass":
    pytest.skip(
        f"Skipping torchdrug rotamer tests due to ABI preflight failure: {_torchdrug_probe['required_errors']}",
        allow_module_level=True,
    )

from diffpack.torchdrug import data

from diffpack import rotamer


def _atom_only_pdb(tmp_path):
    src = "1ubq.pdb"
    dst = tmp_path / "1ubq_atom_only.pdb"
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                fout.write(line)
    return str(dst)


def test_rotate_side_chain_identity(tmp_path):
    pdb_file = _atom_only_pdb(tmp_path)
    protein = data.Protein.from_pdb(
        pdb_file,
        atom_feature=None,
        bond_feature=None,
        residue_feature=None,
        mol_feature=None,
    )
    protein = protein.subgraph(protein.atom_name != 37)
    with protein.atom():
        protein.atom14index = rotamer.restype_atom14_index_map[
            protein.residue_type[protein.atom2residue], protein.atom_name
        ]
    with protein.residue():
        protein.chi_mask = rotamer.get_chi_mask(protein)
    chis = rotamer.get_chis(protein)
    rotate_angles = torch.zeros_like(chis)
    new_protein = protein.clone()
    _ = rotamer.rotate_side_chain(new_protein, rotate_angles)
    assert torch.allclose(new_protein.node_position, protein.node_position, atol=1e-6, rtol=0.0)


def test_rotate_side_chain_periodicity(tmp_path):
    pdb_file = _atom_only_pdb(tmp_path)
    protein = data.Protein.from_pdb(
        pdb_file,
        atom_feature=None,
        bond_feature=None,
        residue_feature=None,
        mol_feature=None,
    )
    protein = protein.subgraph(protein.atom_name != 37)
    with protein.atom():
        protein.atom14index = rotamer.restype_atom14_index_map[
            protein.residue_type[protein.atom2residue], protein.atom_name
        ]
    with protein.residue():
        protein.chi_mask = rotamer.get_chi_mask(protein)
    chis = rotamer.get_chis(protein)
    rotate_angles = torch.zeros_like(chis)

    for i in range(8):
        new_protein = protein.clone()
        rotamer.rotate_side_chain(new_protein, rotate_angles)
        new_chis = rotamer.get_chis(new_protein)
        diff = (new_chis - chis).fmod(np.pi * 2)
        ok = diff.isnan() | ((diff - np.pi * i / 4).abs() < 1e-4) | ((diff + np.pi * (8 - i) / 4).abs() < 1e-4)
        assert ok.all()
        rotate_angles = rotate_angles + np.pi / 4
