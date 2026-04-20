import pytest
torch = pytest.importorskip("torch")

from diffpack.memory import MemoryTracker, cuda_memory_probe


def test_cuda_memory_probe_schema():
    payload = cuda_memory_probe()
    assert "cuda_available" in payload
    assert "has_memory_allocated" in payload
    assert "has_memory_reserved" in payload
    assert "has_empty_cache" in payload


def test_memory_tracker_cpu_metadata():
    tracker = MemoryTracker(device=torch.device("cpu"), mode="quality")
    tracker.sample("dataset_load")
    tracker.sample("generation_loop")
    payload = tracker.metadata()
    assert payload["memory_mode"] == "quality"
    assert payload["device_allocated_peak_bytes"] is None
    assert payload["device_reserved_peak_bytes"] is None
    assert "dataset_load" in payload["memory_phase_peaks"]
    assert "generation_loop" in payload["memory_phase_peaks"]
