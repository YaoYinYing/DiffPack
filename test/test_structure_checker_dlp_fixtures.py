from __future__ import annotations

from pathlib import Path

from diffpack.structure_checker import check_structure, compare_reports


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dlpacker_1ubq"


def test_dlp_1ubq_fixtures_are_geometry_clean():
    fixture_files = [
        FIXTURE_DIR / "1ubq.pdb",
        FIXTURE_DIR / "1ubq_live_test.pdb",
        FIXTURE_DIR / "1ubq_live_test_afterfix_cpu.pdb",
        FIXTURE_DIR / "1ubq_live_test_afterfix_mps.pdb",
    ]
    for pdb_file in fixture_files:
        report = check_structure(str(pdb_file), clash_threshold=1.0, top_n_clashes=20)
        assert report.heavy_atom_count > 0, pdb_file.name
        assert report.min_inter_residue_distance >= 1.0, pdb_file.name
        assert len(report.severe_clashes) == 0, pdb_file.name
        assert len(report.bond_length_outliers) == 0, pdb_file.name
        assert len(report.missing_sidechain_atoms) == 0, pdb_file.name


def test_dlp_1ubq_outputs_do_not_worsen_clashes_vs_input():
    before = check_structure(str(FIXTURE_DIR / "1ubq.pdb"), clash_threshold=1.0, top_n_clashes=20)
    outputs = [
        FIXTURE_DIR / "1ubq_live_test.pdb",
        FIXTURE_DIR / "1ubq_live_test_afterfix_cpu.pdb",
        FIXTURE_DIR / "1ubq_live_test_afterfix_mps.pdb",
    ]
    for out_file in outputs:
        after = check_structure(str(out_file), clash_threshold=1.0, top_n_clashes=20)
        delta = compare_reports(before=before, after=after, clash_threshold=1.0, top_n=20)
        assert delta.worsened_clash_count == 0, out_file.name
        assert delta.after_min_inter_residue_distance >= 1.0, out_file.name

