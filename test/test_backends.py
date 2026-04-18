import pytest

from diffpack.backends import get_backend_adapter
from diffpack.backends.base import InferenceRequest


def test_get_torchdrug_backend():
    adapter = get_backend_adapter("torchdrug_fork")
    assert adapter.name == "torchdrug_fork"


def test_get_pyg_backend():
    adapter = get_backend_adapter("pyg")
    assert adapter.name == "pyg"


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend_adapter("unknown_backend")


def test_pyg_transitional_metadata(monkeypatch, tmp_path):
    adapter = get_backend_adapter("pyg")

    captured = {}

    def fake_run(request, **kwargs):
        captured.update(kwargs)
        return {"backend_requested": kwargs["backend_requested"], "backend_effective": kwargs["backend_effective"]}

    monkeypatch.setattr(adapter._runner, "run", fake_run)
    request = InferenceRequest(
        config=str(tmp_path / "x.yaml"),
        seed=0,
        output_dir=str(tmp_path / "out"),
        pdb_files=[],
        center_residues=[],
        repack_radius=None,
        hetero_policy="exclude",
        device="cpu",
        fast=False,
        profile=False,
    )
    result = adapter.run_inference(request)
    assert result["backend_requested"] == "pyg"
    assert result["backend_effective"] == "torchdrug_fork"
    assert captured["backend_mode"] == "fallback"
