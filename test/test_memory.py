import pytest
torch = pytest.importorskip("torch")

from diffpack.memory import MemoryTracker, mps_memory_probe


def test_mps_memory_probe_schema():
    payload = mps_memory_probe()
    assert "mps_backend_available" in payload
    assert "has_torch_mps_module" in payload
    assert "has_current_allocated_memory" in payload
    assert "has_driver_allocated_memory" in payload
    assert "has_empty_cache" in payload


def test_memory_tracker_cpu_metadata():
    tracker = MemoryTracker(device=torch.device("cpu"), mode="quality")
    tracker.sample("dataset_load")
    tracker.sample("generation_loop")
    payload = tracker.metadata()
    assert payload["memory_mode"] == "quality"
    assert payload["mps_allocated_peak_bytes"] is None
    assert payload["mps_reserved_peak_bytes"] is None
    assert "dataset_load" in payload["memory_phase_peaks"]
    assert "generation_loop" in payload["memory_phase_peaks"]
