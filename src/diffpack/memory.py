from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def mps_memory_probe() -> dict[str, Any]:
    has_backend = hasattr(torch.backends, "mps")
    backend_available = bool(has_backend and torch.backends.mps.is_available())
    module = getattr(torch, "mps", None)
    return {
        "mps_backend_available": backend_available,
        "has_torch_mps_module": module is not None,
        "has_current_allocated_memory": bool(module and hasattr(module, "current_allocated_memory")),
        "has_driver_allocated_memory": bool(module and hasattr(module, "driver_allocated_memory")),
        "has_empty_cache": bool(module and hasattr(module, "empty_cache")),
    }


def read_mps_memory_bytes() -> tuple[int | None, int | None]:
    probe = mps_memory_probe()
    if not probe["mps_backend_available"] or not probe["has_torch_mps_module"]:
        return None, None
    module = torch.mps  # type: ignore[attr-defined]
    allocated = None
    reserved = None
    if probe["has_current_allocated_memory"]:
        allocated = int(module.current_allocated_memory())  # type: ignore[call-arg]
    if probe["has_driver_allocated_memory"]:
        reserved = int(module.driver_allocated_memory())  # type: ignore[call-arg]
    return allocated, reserved


def release_device_cache(device: torch.device, *, aggressive: bool = False) -> None:
    if device.type != "mps":
        return
    if aggressive or hasattr(torch.mps, "empty_cache"):  # type: ignore[attr-defined]
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            return


@dataclass
class MemoryTracker:
    device: torch.device
    mode: str = "quality"
    phase_peaks: dict[str, dict[str, int | None]] = field(default_factory=dict)
    allocated_peak_bytes: int | None = None
    reserved_peak_bytes: int | None = None

    def sample(self, phase: str) -> None:
        allocated, reserved = read_mps_memory_bytes() if self.device.type == "mps" else (None, None)
        if phase not in self.phase_peaks:
            self.phase_peaks[phase] = {"allocated_peak_bytes": allocated, "reserved_peak_bytes": reserved}
        else:
            current = self.phase_peaks[phase]
            current["allocated_peak_bytes"] = _max_nullable(current["allocated_peak_bytes"], allocated)
            current["reserved_peak_bytes"] = _max_nullable(current["reserved_peak_bytes"], reserved)
        self.allocated_peak_bytes = _max_nullable(self.allocated_peak_bytes, allocated)
        self.reserved_peak_bytes = _max_nullable(self.reserved_peak_bytes, reserved)

    def metadata(self) -> dict[str, Any]:
        return {
            "memory_mode": self.mode,
            "mps_allocated_peak_bytes": self.allocated_peak_bytes,
            "mps_reserved_peak_bytes": self.reserved_peak_bytes,
            "memory_phase_peaks": self.phase_peaks,
        }


def _max_nullable(lhs: int | None, rhs: int | None) -> int | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return max(lhs, rhs)
