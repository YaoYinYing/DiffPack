from __future__ import annotations

from pathlib import Path

from diffpack.backends.base import InferenceRequest
from diffpack.backends.mutation_request import prepare_mutation_request


PDB_TEXT = """\
ATOM      1  N   GLY A  76      11.104   8.447   2.073  1.00 20.00           N
ATOM      2  CA  GLY A  76      12.560   8.679   2.180  1.00 20.00           C
ATOM      3  C   GLY A  76      13.122   7.897   3.394  1.00 20.00           C
ATOM      4  O   GLY A  76      12.450   7.061   3.987  1.00 20.00           O
TER
END
"""


def _make_request(tmp_path: Path, *, repack_radius: float, center_residues=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdb = tmp_path / "input.pdb"
    pdb.write_text(PDB_TEXT, encoding="utf-8")
    return InferenceRequest(
        config="dummy.yaml",
        seed=0,
        output_dir=str(tmp_path / "out"),
        pdb_files=[str(pdb)],
        center_residues=center_residues or [],
        repack_radius=repack_radius,
        mutations="AG76A",
        pro_remodel_window="tripeptide",
        pro_remodel_max_steps=24,
        hetero_policy="exclude",
        device="cpu",
        fast=False,
        profile=False,
        memory_mode="quality",
        cache_root=None,
        cache_read_only=True,
    )


def test_prepare_mutation_request_radius_modes(tmp_path):
    req_local = _make_request(tmp_path / "local", repack_radius=10.0, center_residues=["A:10"])
    updated_local, meta_local = prepare_mutation_request(req_local)
    assert sorted(updated_local.center_residues) == ["A:10", "A:76"]
    assert meta_local["radius_mode"] == "local_repack"
    assert meta_local["mutation_site_count"] == 1
    assert meta_local["effective_center_count"] == 2
    assert meta_local["pro_remodel_window"] == "tripeptide"
    assert meta_local["pro_remodel_max_steps"] == 24

    req_mut_only = _make_request(tmp_path / "mut_only", repack_radius=-1.0)
    updated_mut_only, meta_mut_only = prepare_mutation_request(req_mut_only)
    assert updated_mut_only.center_residues == []
    assert meta_mut_only["radius_mode"] == "mutated_only"
    assert meta_mut_only["effective_center_count"] == 0

    req_full = _make_request(tmp_path / "full", repack_radius=0.0)
    _, meta_full = prepare_mutation_request(req_full)
    assert meta_full["radius_mode"] == "full_repack"


def test_prepare_mutation_request_freezes_mutated_proline(tmp_path):
    req = _make_request(tmp_path / "pro", repack_radius=10.0, center_residues=["A:10"])
    req.mutations = "AG76P"
    updated, meta = prepare_mutation_request(req)
    assert updated.frozen_residues == ["A:76"]
    assert meta["frozen_residues"] == ["A:76"]
