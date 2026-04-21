from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import json

from diffpack import rotamer
from diffpack.mutation import (
    normalize_mutations,
    preprocess_pdb_files_with_mutations,
)


def _atom_line(serial: int, atom: str, resname: str, chain: str, resid: int, x: float, y: float, z: float) -> str:
    element = atom.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}"
    )


def _write_backbone_only_pdb(path: Path):
    lines: list[str] = []
    serial = 1
    for i, resname in enumerate(rotamer.residue_list, start=1):
        base = i * 4.0
        lines.append(_atom_line(serial, "N", resname, "A", i, base, 1.0, 0.5))
        serial += 1
        lines.append(_atom_line(serial, "CA", resname, "A", i, base + 1.2, 1.3, 0.8))
        serial += 1
        lines.append(_atom_line(serial, "C", resname, "A", i, base + 2.4, 0.7, 1.1))
        serial += 1
        lines.append(_atom_line(serial, "O", resname, "A", i, base + 2.9, 0.2, 2.2))
        serial += 1
    lines.extend(["TER", "END", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _rdkit_parseable(pdb_text: str) -> bool:
    mol = Chem.MolFromPDBBlock(pdb_text, sanitize=True, removeHs=False)
    return mol is not None


def test_normalize_mutations_compact():
    specs = normalize_mutations("AG76A,AG65A")
    assert [spec.compact for spec in specs] == ["AG76A", "AG65A"]
    assert [spec.selector for spec in specs] == ["A:76", "A:65"]


def test_normalize_mutations_dict():
    specs = normalize_mutations([{"chain": "A", "old_res": "G", "position": 76, "new_res": "A"}])
    assert len(specs) == 1
    assert specs[0].old_res_3 == "GLY"
    assert specs[0].new_res_3 == "ALA"


@pytest.mark.parametrize("payload", ["A76A", "AG76", "AG76X", [{"chain": "A", "old_res": "UNK", "position": 1, "new_res": "A"}]])
def test_normalize_mutations_invalid(payload):
    with pytest.raises(ValueError):
        normalize_mutations(payload)


def test_preprocess_pdb_mutation(tmp_path: Path):
    pdb_in = tmp_path / "in.pdb"
    pdb_in.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A  76      11.104  13.207   9.502  1.00 20.00           N",
                "ATOM      2  CA  GLY A  76      12.560  13.132   9.242  1.00 20.00           C",
                "ATOM      3  C   GLY A  76      13.141  11.741   9.523  1.00 20.00           C",
                "ATOM      4  O   GLY A  76      12.545  10.706   9.165  1.00 20.00           O",
                "ATOM      5  HA2 GLY A  76      12.700  13.450   8.000  1.00 20.00           H",
                "ATOM      6  N   ALA A  77      14.309  11.712  10.146  1.00 20.00           N",
                "ATOM      7  CA  ALA A  77      15.000  10.430  10.500  1.00 20.00           C",
                "ATOM      8  CB  ALA A  77      16.000  10.500  11.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    specs = normalize_mutations("AG76A")
    out_files, selectors = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))
    assert selectors == ["A:76"]
    out_text = Path(out_files[0]).read_text(encoding="utf-8")
    assert " GLY A  76" not in out_text
    assert " ALA A  76" in out_text
    assert "CB ALA A  76" in out_text
    assert " HA2 GLY A  76" not in out_text
    assert " ALA A  77" in out_text
    assert "CONECT" in out_text
    assert pdb_in.read_text(encoding="utf-8").count(" GLY A  76") > 0


def test_preprocess_pdb_mutation_old_residue_mismatch(tmp_path: Path):
    pdb_in = tmp_path / "in.pdb"
    pdb_in.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A  10      11.104  13.207   9.502  1.00 20.00           N",
                "ATOM      2  CA  ALA A  10      12.560  13.132   9.242  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    specs = normalize_mutations("AG10A")
    with pytest.raises(ValueError, match="source residue mismatch"):
        preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))


def test_preprocess_pdb_all_aa_to_all_aa_including_self(tmp_path: Path):
    pdb_in = tmp_path / "all_aa_input.pdb"
    _write_backbone_only_pdb(pdb_in)
    one_letters = [rotamer.three_to_one[name] for name in rotamer.residue_list]
    pos_to_old = {i: one_letters[i - 1] for i in range(1, len(one_letters) + 1)}

    for position, old_one in pos_to_old.items():
        for new_one in one_letters:
            spec = normalize_mutations(f"A{old_one}{position}{new_one}")
            if new_one == "P":
                try:
                    preprocess_pdb_files_with_mutations([str(pdb_in)], spec, str(tmp_path / "out"))
                except ValueError as error:
                    assert "no valid PRO pose for local environment" in str(error)
                continue
            out_files, selectors = preprocess_pdb_files_with_mutations([str(pdb_in)], spec, str(tmp_path / "out"))
            assert selectors == [f"A:{position}"]
            out_text = Path(out_files[0]).read_text(encoding="utf-8")
            new_three = rotamer.one_to_three[new_one]
            target_lines = [
                line
                for line in out_text.splitlines()
                if line.startswith("ATOM") and line[21] == "A" and int(line[22:26]) == position
            ]
            assert target_lines, f"mutation produced no atoms for A:{position}"
            assert all(line[17:20].strip() == new_three for line in target_lines)
            atom_names = {line[12:16].strip() for line in target_lines}
            assert {"N", "CA", "C", "O"}.issubset(atom_names)
            if new_three == "GLY":
                assert "CB" not in atom_names
            else:
                assert "CB" in atom_names


def test_preprocess_pdb_multiple_mutations(tmp_path: Path):
    pdb_in = tmp_path / "multi_input.pdb"
    _write_backbone_only_pdb(pdb_in)
    specs = normalize_mutations("AG1A,AA2G,AW20Y")
    out_files, selectors = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))
    assert selectors == ["A:1", "A:2", "A:20"]
    out_text = Path(out_files[0]).read_text(encoding="utf-8")

    def residue_atoms(pos: int) -> set[str]:
        return {
            line[12:16].strip()
            for line in out_text.splitlines()
            if line.startswith("ATOM") and line[21] == "A" and int(line[22:26]) == pos
        }

    def residue_name(pos: int) -> str:
        for line in out_text.splitlines():
            if line.startswith("ATOM") and line[21] == "A" and int(line[22:26]) == pos:
                return line[17:20].strip()
        raise AssertionError(f"residue {pos} not found")

    assert residue_name(1) == "ALA"
    assert "CB" in residue_atoms(1)
    assert residue_name(2) == "GLY"
    assert "CB" not in residue_atoms(2)
    assert residue_name(20) == "TYR"
    assert "CB" in residue_atoms(20)


@pytest.mark.parametrize(
    "mutation",
        [
            "AQ2P",  # non-PRO -> PRO (1ubq A2 is GLN)
            "AP19A",  # PRO -> non-PRO (1ubq A19 is PRO)
            "AP19P",  # PRO -> PRO
            "AP37K",  # PRO -> long sidechain (1ubq A37 is PRO)
        ],
)
def test_preprocess_pdb_proline_cases(tmp_path: Path, mutation: str):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")
    specs = normalize_mutations(mutation)
    out_files, _ = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))
    out_text = Path(out_files[0]).read_text(encoding="utf-8")
    assert _rdkit_parseable(out_text)


@pytest.mark.parametrize("mutation", ["AQ2T", "AT9S", "AV17K", "AI23W"])
def test_preprocess_real_1ubq_short_to_long_parseable(tmp_path: Path, mutation: str):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")
    specs = normalize_mutations(mutation)
    out_files, _ = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))
    assert _rdkit_parseable(Path(out_files[0]).read_text(encoding="utf-8"))


def _atom_coord_by_residue(pdb_text: str) -> dict[tuple[str, int, str], np.ndarray]:
    out: dict[tuple[str, int, str], np.ndarray] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21].strip()
        resseq = int(line[22:26])
        atom = line[12:16].strip()
        out[(chain, resseq, atom)] = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    return out


def test_preprocess_aa46p_proline_collision_guard(tmp_path: Path):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")
    specs = normalize_mutations("AA46P")
    try:
        out_files, _ = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))
        out_text = Path(out_files[0]).read_text(encoding="utf-8")
        assert _rdkit_parseable(out_text)
        coords = _atom_coord_by_residue(out_text)
        pro_cd = coords[("A", 46, "CD")]
        prev_c = coords[("A", 45, "C")]
        assert float(np.linalg.norm(pro_cd - prev_c)) >= 1.45
    except ValueError as error:
        message = str(error)
        assert "no valid PRO pose for local environment" in message
        assert ("closest rejected pair" in message) or ("failing constraint category" in message)


def test_preprocess_aa46p_deterministic_selection(tmp_path: Path):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")
    specs = normalize_mutations("AA46P")

    result_1: str
    result_2: str
    try:
        out_files_1, _ = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out1"))
        result_1 = Path(out_files_1[0]).read_text(encoding="utf-8")
    except ValueError as error:
        result_1 = str(error)

    try:
        out_files_2, _ = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out2"))
        result_2 = Path(out_files_2[0]).read_text(encoding="utf-8")
    except ValueError as error:
        result_2 = str(error)

    assert result_1 == result_2


def test_preprocess_aa46p_writes_dijkstra_diagnostics(tmp_path: Path):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")
    specs = normalize_mutations("AA46P")
    try:
        preprocess_pdb_files_with_mutations(
            [str(pdb_in)],
            specs,
            str(tmp_path / "out"),
            pro_remodel_window="tripeptide",
            pro_remodel_max_steps=12,
        )
    except ValueError:
        pytest.skip("AA46P has no valid deterministic PRO pose in this environment")
    diag_path = tmp_path / "out" / "_mutation_inputs" / "mutation_diagnostics.json"
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["pro_mutation_count"] >= 1
    assert payload["pro_solver_mode"] == "template_transplant"
    assert payload["pro_solver_window"] == "residue_only"
    assert payload["pro_solver_iterations"] == 0
    assert payload["pro_solver_final_cost"] >= 0


def test_preprocess_aa46p_scope_limits_path_atoms(tmp_path: Path):
    pdb_in = tmp_path / "1ubq.pdb"
    pdb_in.write_text(Path("1ubq.pdb").read_text(encoding="utf-8"), encoding="utf-8")
    specs = normalize_mutations("AA46P")
    try:
        preprocess_pdb_files_with_mutations(
            [str(pdb_in)],
            specs,
            str(tmp_path / "out"),
            pro_remodel_window="residue_only",
            pro_remodel_max_steps=8,
        )
    except ValueError:
        pytest.skip("AA46P has no valid deterministic PRO pose in this environment")
    payload = json.loads((tmp_path / "out" / "_mutation_inputs" / "mutation_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["pro_solver_mode"] == "template_transplant"
    assert payload["pro_solver_window"] == "residue_only"
    assert payload["pro_solver_iterations"] == 0


def test_pro_solver_updates_not_clobbered_and_neighbor_written(tmp_path: Path, monkeypatch):
    pdb_in = tmp_path / "in.pdb"
    pdb_in.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   ALA A   1       2.800   0.000   0.000  1.00 20.00           C",
                "ATOM      4  O   ALA A   1       3.600   0.000   0.000  1.00 20.00           O",
                "ATOM      5  CB  ALA A   1       1.400   1.400   0.000  1.00 20.00           C",
                "ATOM      6  N   ALA A   2       4.100   0.100   0.000  1.00 20.00           N",
                "ATOM      7  CA  ALA A   2       5.500   0.100   0.000  1.00 20.00           C",
                "ATOM      8  C   ALA A   2       6.900   0.100   0.000  1.00 20.00           C",
                "ATOM      9  O   ALA A   2       7.700   0.100   0.000  1.00 20.00           O",
                "ATOM     10  CB  ALA A   2       5.500   1.500   0.000  1.00 20.00           C",
                "ATOM     11  N   ALA A   3       8.200   0.200   0.000  1.00 20.00           N",
                "ATOM     12  CA  ALA A   3       9.600   0.200   0.000  1.00 20.00           C",
                "ATOM     13  C   ALA A   3      11.000   0.200   0.000  1.00 20.00           C",
                "ATOM     14  O   ALA A   3      11.800   0.200   0.000  1.00 20.00           O",
                "ATOM     15  CB  ALA A   3       9.600   1.600   0.000  1.00 20.00           C",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def _mock_build_pro(*args, **kwargs):
        updates = {
            ("A", 1): {
                "N": np.asarray([0.050, 0.020, 0.000], dtype=np.float64),
            },
            ("A", 2): {
                "N": np.asarray([4.200, 0.500, 0.300], dtype=np.float64),
                "CA": np.asarray([5.300, 0.700, 0.300], dtype=np.float64),
                "C": np.asarray([6.550, 0.450, 0.200], dtype=np.float64),
                "O": np.asarray([7.200, 0.350, 0.150], dtype=np.float64),
                "CB": np.asarray([5.000, 1.800, 0.300], dtype=np.float64),
                "CG": np.asarray([4.100, 2.000, 0.700], dtype=np.float64),
                "CD": np.asarray([3.200, 1.000, 0.300], dtype=np.float64),
            },
            ("A", 3): {
                "N": np.asarray([8.050, 0.350, 0.120], dtype=np.float64),
            },
        }
        diag = {
            "pro_solver_mode": "template_transplant",
            "pro_solver_window": "tripeptide",
            "pro_solver_iterations": 0,
            "pro_solver_final_cost": 1.23,
            "pro_applied_remodel_residues": ["A:1", "A:2", "A:3"],
        }
        return updates, None, diag

    monkeypatch.setattr("diffpack.mutation._solve_proline_internal", _mock_build_pro)
    specs = normalize_mutations("AA2P")
    out_files, _ = preprocess_pdb_files_with_mutations([str(pdb_in)], specs, str(tmp_path / "out"))
    out_text = Path(out_files[0]).read_text(encoding="utf-8")
    coords = _atom_coord_by_residue(out_text)
    # mutated residue backbone should come from solver updates (not original atom_lines)
    assert np.allclose(coords[("A", 2, "N")], np.asarray([4.200, 0.500, 0.300]), atol=1e-3)
    assert np.allclose(coords[("A", 2, "CA")], np.asarray([5.300, 0.700, 0.300]), atol=1e-3)
    # neighbor residue writeback should also be applied
    assert np.allclose(coords[("A", 3, "N")], np.asarray([8.050, 0.350, 0.120]), atol=1e-3)
    diag = json.loads((tmp_path / "out" / "_mutation_inputs" / "mutation_diagnostics.json").read_text(encoding="utf-8"))
    assert "A:1" in diag.get("pro_applied_remodel_residues", [])
    assert "A:3" in diag.get("pro_applied_remodel_residues", [])
