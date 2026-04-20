import json

import pytest

pytest.importorskip("torch")

from diffpack.cli import infer


def test_parse_args_requires_center_residue_when_radius_set():
    with pytest.raises(SystemExit):
        infer.parse_args(["--repack_radius", "10"])


def test_parse_args_requires_radius_when_center_residue_set():
    with pytest.raises(SystemExit):
        infer.parse_args(["--center_residues", "A:72"])


def test_parse_args_success():
    args = infer.parse_args(["--center_residues", "A:72", "--repack_radius", "10", "--hetero_policy", "exclude"])
    assert args.repack_radius == 10
    assert args.center_residues == ["A:72"]
    assert args.hetero_policy == "exclude"


def test_run_diagnostics_outputs_json(capsys):
    infer.run_diagnostics()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "torch_version" in payload
    assert "platform" in payload
    native_preview = payload["backend_resolution_preview"]["native"]
    assert native_preview["backend_effective"] == "native"
    assert native_preview["backend_mode"] == "native"
    torchdrug_preview = payload["backend_resolution_preview"]["torchdrug"]
    assert torchdrug_preview["backend_effective"] == "torchdrug"
    assert torchdrug_preview["backend_mode"] == "native"
    pyg_preview = payload["backend_resolution_preview"]["pyg"]
    assert pyg_preview["backend_effective"] == "pyg"
    assert pyg_preview["backend_mode"] == "native"
