import pytest
import torch

from diffpack import repack


def test_parse_center_residue_selector():
    assert repack.parse_center_residue_selector("A:72") == ("A", 72)
    assert repack.parse_center_residue_selector("B:-1") == ("B", -1)


@pytest.mark.parametrize("selector", ["A72", "A:", ":72", "A:7.2", "A:72:1", ""])
def test_parse_center_residue_selector_invalid(selector):
    with pytest.raises(ValueError):
        repack.parse_center_residue_selector(selector)


def test_select_residues_by_radius():
    atom_positions = torch.tensor([
        [0.0, 0.0, 0.0],   # residue 0
        [0.5, 0.0, 0.0],   # residue 0
        [5.0, 0.0, 0.0],   # residue 1
        [5.5, 0.0, 0.0],   # residue 1
        [12.0, 0.0, 0.0],  # residue 2
        [12.5, 0.0, 0.0],  # residue 2
    ])
    atom2residue = torch.tensor([0, 0, 1, 1, 2, 2])
    residue_identifiers = [("A", 1), ("A", 2), ("A", 3)]

    mask, center_indices = repack.select_residues_by_radius(
        atom_positions=atom_positions,
        atom2residue=atom2residue,
        num_residue=3,
        residue_identifiers=residue_identifiers,
        center_selectors=[("A", 2)],
        radius=6.0,
    )

    assert center_indices == [1]
    assert mask.tolist() == [True, True, False]


def test_select_residues_by_radius_missing_center():
    atom_positions = torch.tensor([[0.0, 0.0, 0.0]])
    atom2residue = torch.tensor([0])

    with pytest.raises(ValueError, match="Center residues not found"):
        repack.select_residues_by_radius(
            atom_positions=atom_positions,
            atom2residue=atom2residue,
            num_residue=1,
            residue_identifiers=[("A", 1)],
            center_selectors=[("A", 2)],
            radius=4.0,
        )
