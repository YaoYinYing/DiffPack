from pathlib import Path

import pytest

from diffpack.dataset import SideChainDataset


def _write_hetatm_pdb(path: Path):
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.000  13.000   9.000  1.00 20.00           N",
                "HETATM    2  O   HOH A 101      12.000  13.000   9.000  1.00 20.00           O",
                "TER",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def test_hetero_policy_error_rejects_hetatm(tmp_path: Path):
    pdb = tmp_path / "h.pdb"
    _write_hetatm_pdb(pdb)
    ds = SideChainDataset(pdb_files=[], hetero_policy="error", atom_feature="residue_symbol", bond_feature=None,
                          residue_feature=None, mol_feature=None, sanitize=True, removeHs=True)
    with pytest.raises(ValueError, match="HETATM"):
        ds._prepare_pdb_file(str(pdb))


def test_hetero_policy_exclude_filters_hetatm(tmp_path: Path):
    pdb = tmp_path / "h.pdb"
    _write_hetatm_pdb(pdb)
    ds = SideChainDataset(pdb_files=[], hetero_policy="exclude", atom_feature="residue_symbol", bond_feature=None,
                          residue_feature=None, mol_feature=None, sanitize=True, removeHs=True)
    filtered = Path(ds._prepare_pdb_file(str(pdb)))
    assert filtered.exists()
    text = filtered.read_text(encoding="utf-8")
    assert "HETATM" not in text
