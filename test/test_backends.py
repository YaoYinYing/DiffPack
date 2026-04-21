import pytest
torch = pytest.importorskip("torch")
import numpy as np

from diffpack.backends import get_backend_adapter
from diffpack.backends.base import InferenceRequest
from diffpack.backends import native_runtime
from diffpack.backends.native_runtime import NativeConfigTranslator, PygTorsionalDiffusion
from diffpack.util import get_default_config_path, load_config


@pytest.fixture(autouse=True)
def _skip_abi_preflight(monkeypatch):
    monkeypatch.setenv("DIFFPACK_SKIP_ABI_PREFLIGHT", "1")


def test_get_torchdrug_backend():
    pytest.importorskip("torch_scatter")
    adapter = get_backend_adapter("torchdrug")
    assert adapter.name == "torchdrug"


def test_get_native_backend():
    adapter = get_backend_adapter("native")
    assert adapter.name == "native"


def test_get_pyg_backend():
    pytest.importorskip("torch_geometric")
    adapter = get_backend_adapter("pyg")
    assert adapter.name == "pyg"


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend_adapter("unknown_backend")


def test_native_metadata(monkeypatch, tmp_path):
    adapter = get_backend_adapter("native")

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
        memory_mode="quality",
    )
    result = adapter.run_inference(request)
    assert result["backend_requested"] == "native"
    assert result["backend_effective"] == "native"
    assert captured["backend_mode"] == "native"
    assert captured["fallback_reason"] is None


def test_native_translator_builds_torsional_task():
    cfg = load_config(get_default_config_path("inference.yaml"))
    fake_p = np.ones((5001, 5001), dtype=np.float64)
    fake_score = np.ones((5001, 5001), dtype=np.float64)
    fake_norm = np.ones((5001,), dtype=np.float64)
    original = native_runtime.load_schedule_tables_readonly
    native_runtime.load_schedule_tables_readonly = lambda *_args, **_kwargs: (fake_p, fake_score, fake_norm)
    translator = NativeConfigTranslator(cfg)
    try:
        task = translator.build_task()
        assert isinstance(task, PygTorsionalDiffusion)
        assert task.graph_construction_model is not None
    finally:
        native_runtime.load_schedule_tables_readonly = original


def test_native_translator_rejects_unknown_model_class():
    cfg = load_config(get_default_config_path("inference.yaml"))
    cfg.task.model["class"] = "UnknownModel"
    fake_p = np.ones((5001, 5001), dtype=np.float64)
    fake_score = np.ones((5001, 5001), dtype=np.float64)
    fake_norm = np.ones((5001,), dtype=np.float64)
    original = native_runtime.load_schedule_tables_readonly
    native_runtime.load_schedule_tables_readonly = lambda *_args, **_kwargs: (fake_p, fake_score, fake_norm)
    translator = NativeConfigTranslator(cfg)
    try:
        with pytest.raises(ValueError, match="supports.*GearNet"):
            translator.build_task()
    finally:
        native_runtime.load_schedule_tables_readonly = original
