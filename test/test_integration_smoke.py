import importlib

import pytest

pytest.importorskip("torch")


@pytest.mark.smoke
def test_legacy_script_shim_importable():
    module = importlib.import_module("script.inference")
    assert hasattr(module, "main")
