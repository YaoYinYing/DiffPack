from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "cpu_triplet_compare.py"
_SPEC = spec_from_file_location("diffpack_cpu_triplet_compare", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_aggregate_median():
    assert _MODULE._aggregate([1.0, 5.0, 3.0], "median") == 3.0


def test_aggregate_mean():
    assert _MODULE._aggregate([1.0, 5.0, 3.0], "mean") == 3.0
