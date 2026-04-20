from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_parity_trace.py"
_SPEC = importlib.util.spec_from_file_location("run_parity_trace", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["run_parity_trace"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
StageDiff = _MODULE.StageDiff
_first_divergence = _MODULE._first_divergence
_summarize_delta = _MODULE._summarize_delta


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


def test_summarize_delta_equal_nan_treats_matching_nan_as_equal():
    lhs = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    rhs = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    diff = _summarize_delta(lhs, rhs, "nan_stage", equal_nan=True)
    assert diff.same_shape is True
    assert diff.max_abs_delta == 0.0
    assert diff.mean_abs_delta == 0.0


def test_summarize_delta_without_equal_nan_flags_non_finite():
    lhs = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    rhs = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    diff = _summarize_delta(lhs, rhs, "nan_stage", equal_nan=False)
    assert diff.same_shape is True
    assert np.isinf(diff.max_abs_delta)
    assert np.isinf(diff.mean_abs_delta)
