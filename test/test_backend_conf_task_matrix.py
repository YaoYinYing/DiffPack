from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "backend_conf_task_matrix.py"
_SPEC = spec_from_file_location("diffpack_backend_conf_task_matrix", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parse_timed_stderr_schema():
    payload = _MODULE._parse_timed_stderr("1.23 real 4.56 user 0.78 sys\n12345  maximum resident set size")
    assert "time_real_sec" in payload
    assert "time_user_sec" in payload
    assert "time_sys_sec" in payload
    assert "peak_rss_kib" in payload
