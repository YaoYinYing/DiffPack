import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.mark.smoke
def test_legacy_script_shim_importable():
    script_path = Path(__file__).resolve().parents[1] / "script" / "inference.py"
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location("legacy_inference", script_path)
    assert spec is not None and spec.loader is not None
