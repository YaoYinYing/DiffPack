from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def cuda_memory_probe() -> dict[str, Any]:
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "has_memory_allocated": bool(hasattr(torch.cuda, "memory_allocated")),
        "has_memory_reserved": bool(hasattr(torch.cuda, "memory_reserved")),
        "has_empty_cache": bool(hasattr(torch.cuda, "empty_cache")),
    }


def read_device_memory_bytes(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    allocated = int(torch.cuda.memory_allocated(device=device))
    reserved = int(torch.cuda.memory_reserved(device=device))
    return allocated, reserved


def release_device_cache(device: torch.device, *, aggressive: bool = False) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if aggressive or hasattr(torch.cuda, "empty_cache"):
        try:
            torch.cuda.empty_cache()
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
        allocated, reserved = read_device_memory_bytes(self.device)
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
            "device_allocated_peak_bytes": self.allocated_peak_bytes,
            "device_reserved_peak_bytes": self.reserved_peak_bytes,
            "memory_phase_peaks": self.phase_peaks,
        }


def _max_nullable(lhs: int | None, rhs: int | None) -> int | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return max(lhs, rhs)
