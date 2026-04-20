from __future__ import annotations

import pytest

from diffpack import backend_preflight


def test_probe_backend_dependencies_has_expected_shape():
    probe = backend_preflight.probe_backend_dependencies("native")
    assert probe["backend"] == "native"
    assert "numpy_version" in probe
    assert "dependency_readiness" in probe
    assert "torch" in probe["dependency_readiness"]
    assert "numpy" in probe["dependency_readiness"]


def test_ensure_backend_ready_raises_on_required_failure(monkeypatch):
    def fake_probe(_backend):
        return {
            "backend": "torchdrug",
            "numpy_version": "2.4.4",
            "status": "fail",
            "required_errors": ["rdkit.Chem: AttributeError: _ARRAY_API not found"],
            "optional_errors": [],
            "dependency_readiness": {},
            "remediation": "pip install -U -c requirements/constraints-numpy2.txt ...",
        }

    monkeypatch.setenv("DIFFPACK_SKIP_ABI_PREFLIGHT", "0")
    monkeypatch.setattr(backend_preflight, "probe_backend_dependencies", fake_probe)
    with pytest.raises(RuntimeError, match="preflight failed"):
        backend_preflight.ensure_backend_ready("torchdrug")


def test_ensure_backend_ready_sets_skip_rdkit_for_optional_failure(monkeypatch):
    def fake_probe(_backend):
        return {
            "backend": "native",
            "numpy_version": "2.4.4",
            "status": "pass",
            "required_errors": [],
            "optional_errors": ["rdkit.Chem: AttributeError: _ARRAY_API not found"],
            "dependency_readiness": {"rdkit.Chem": {"ok": False}},
            "remediation": "pip install -U -c requirements/constraints-numpy2.txt ...",
        }

    monkeypatch.delenv("DIFFPACK_SKIP_RDKIT", raising=False)
    monkeypatch.setenv("DIFFPACK_SKIP_ABI_PREFLIGHT", "0")
    monkeypatch.setattr(backend_preflight, "probe_backend_dependencies", fake_probe)
    backend_preflight.ensure_backend_ready("native")
    assert backend_preflight.os.environ.get("DIFFPACK_SKIP_RDKIT") == "1"
