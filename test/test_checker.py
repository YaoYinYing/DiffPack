import json
from pathlib import Path

from diffpack.checker import run_structure_checks


def _write_minimal_pdb(path: Path, shift: float = 0.0):
    path.write_text(
        "\n".join(
            [
                f"ATOM      1  N   ALA A   1      {11.000 + shift:8.3f}{13.000:8.3f}{9.000:8.3f}  1.00 20.00           N",
                f"ATOM      2  CA  ALA A   1      {12.000 + shift:8.3f}{13.000:8.3f}{9.000:8.3f}  1.00 20.00           C",
                f"ATOM      3  N   GLY A   2      {20.000 + shift:8.3f}{13.000:8.3f}{9.000:8.3f}  1.00 20.00           N",
                f"ATOM      4  CA  GLY A   2      {21.000 + shift:8.3f}{13.000:8.3f}{9.000:8.3f}  1.00 20.00           C",
                "TER",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def test_structure_checker_basic_pass(tmp_path: Path):
    input_pdb = tmp_path / "in.pdb"
    output_pdb = tmp_path / "out.pdb"
    _write_minimal_pdb(input_pdb, shift=0.0)
    _write_minimal_pdb(output_pdb, shift=0.0)
    report = run_structure_checks(input_pdb=str(input_pdb), output_pdb=str(output_pdb))
    assert report["status"] == "pass"


def test_structure_checker_selective_freeze_fail(tmp_path: Path):
    input_pdb = tmp_path / "in.pdb"
    output_pdb = tmp_path / "out.pdb"
    _write_minimal_pdb(input_pdb, shift=0.0)
    # Move all atoms; outside-mask check should fail.
    _write_minimal_pdb(output_pdb, shift=1.0)
    report = run_structure_checks(
        input_pdb=str(input_pdb),
        output_pdb=str(output_pdb),
        center_residues=["A:1"],
        repack_radius=0.1,
    )
    assert report["status"] == "fail"
    assert "outside_mask_frozen" in report["errors"]


def test_structure_checker_metric_sanity(tmp_path: Path):
    input_pdb = tmp_path / "in.pdb"
    output_pdb = tmp_path / "out.pdb"
    metadata = tmp_path / "meta.json"
    _write_minimal_pdb(input_pdb, shift=0.0)
    _write_minimal_pdb(output_pdb, shift=0.0)
    metadata.write_text(
        json.dumps(
            {
                "metrics": {
                    "atom_rmsd_per_residue": 0.1,
                    "chi_0_mae_deg": 1.0,
                    "chi_1_mae_deg": 2.0,
                    "chi_2_mae_deg": 3.0,
                    "chi_3_mae_deg": 4.0,
                }
            }
        ),
        encoding="utf-8",
    )
    report = run_structure_checks(
        input_pdb=str(input_pdb),
        output_pdb=str(output_pdb),
        metadata_path=str(metadata),
    )
    assert report["status"] == "pass"
