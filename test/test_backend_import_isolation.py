from __future__ import annotations

import sys

import pytest

from diffpack.backends import get_backend_adapter


def _clear_backend_modules():
    for name in list(sys.modules.keys()):
        if name.startswith("diffpack.backends.native_runtime"):
            sys.modules.pop(name, None)
        if name.startswith("diffpack.backends.pyg_runtime"):
            sys.modules.pop(name, None)
        if name.startswith("diffpack.backends.torchdrug_compatible_runner"):
            sys.modules.pop(name, None)


def test_native_backend_does_not_import_other_runtimes(monkeypatch):
    monkeypatch.setenv("DIFFPACK_SKIP_ABI_PREFLIGHT", "1")
    _clear_backend_modules()
    get_backend_adapter("native")
    assert "diffpack.backends.native_runtime" in sys.modules
    assert "diffpack.backends.pyg_runtime" not in sys.modules
    assert "diffpack.backends.torchdrug_compatible_runner" not in sys.modules


def test_pyg_backend_does_not_import_other_runtimes(monkeypatch):
    pytest.importorskip("torch_geometric")
    monkeypatch.setenv("DIFFPACK_SKIP_ABI_PREFLIGHT", "1")
    _clear_backend_modules()
    get_backend_adapter("pyg")
    assert "diffpack.backends.pyg_runtime" in sys.modules
    assert "diffpack.backends.native_runtime" not in sys.modules
    assert "diffpack.backends.torchdrug_compatible_runner" not in sys.modules


def test_torchdrug_backend_does_not_import_other_runtimes(monkeypatch):
    pytest.importorskip("torch_scatter")
    monkeypatch.setenv("DIFFPACK_SKIP_ABI_PREFLIGHT", "1")
    _clear_backend_modules()
    get_backend_adapter("torchdrug")
    assert "diffpack.backends.torchdrug_compatible_runner" in sys.modules
    assert "diffpack.backends.native_runtime" not in sys.modules
    assert "diffpack.backends.pyg_runtime" not in sys.modules
