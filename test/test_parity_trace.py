from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_parity_trace.py"
_SPEC = importlib.util.spec_from_file_location("run_parity_trace", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["run_parity_trace"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
StageDiff = _MODULE.StageDiff
_first_divergence = _MODULE._first_divergence


def test_first_divergence_none_when_all_within_tolerance():
    diffs = [
        StageDiff(stage="dataset", max_abs_delta=0.0, mean_abs_delta=0.0, same_shape=True, compared_values=10),
        StageDiff(stage="graph", max_abs_delta=1e-6, mean_abs_delta=1e-7, same_shape=True, compared_values=10),
    ]
    assert _first_divergence(diffs, atol=1e-5, mtol=1e-6) is None


def test_first_divergence_reports_first_failing_stage():
    diffs = [
        StageDiff(stage="dataset", max_abs_delta=0.0, mean_abs_delta=0.0, same_shape=True, compared_values=10),
        StageDiff(stage="graph", max_abs_delta=1e-3, mean_abs_delta=1e-4, same_shape=True, compared_values=10),
        StageDiff(stage="schedule", max_abs_delta=1e-2, mean_abs_delta=1e-3, same_shape=True, compared_values=10),
    ]
    assert _first_divergence(diffs, atol=1e-5, mtol=1e-6) == "graph"
