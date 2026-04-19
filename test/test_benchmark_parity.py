from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"
_SPEC = spec_from_file_location("diffpack_run_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_compute_pdb_deltas = _MODULE._compute_pdb_deltas


def _write_minimal_pdb(path: Path, shift: float = 0.0):
    base_lines = [
        "ATOM      1  N   ALA A   1      11.000  13.000   9.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A   1      12.000  13.000   9.000  1.00 20.00           C",
        "TER",
        "END",
    ]
    if shift:
        base_lines[0] = base_lines[0].replace("11.000", f"{11.0 + shift:.3f}")
        base_lines[1] = base_lines[1].replace("12.000", f"{12.0 + shift:.3f}")
    path.write_text("\n".join(base_lines), encoding="utf-8")


def test_compute_pdb_deltas(tmp_path: Path):
    pred = tmp_path / "pred.pdb"
    ref = tmp_path / "ref.pdb"
    _write_minimal_pdb(pred, shift=0.5)
    _write_minimal_pdb(ref, shift=0.0)

    report = _compute_pdb_deltas(str(pred), str(ref))
    assert report["num_atoms_compared"] == 2
    assert report["max_abs_delta"] > 0
    assert report["mean_abs_delta"] > 0
