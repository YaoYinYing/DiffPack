from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceRequest:
    config: str
    seed: int
    output_dir: str
    pdb_files: list[str]
    center_residues: list[str]
    repack_radius: float | None
    hetero_policy: str
    device: str
    fast: bool
    profile: bool
    memory_mode: str = "quality"
    cache_root: str | None = None
    cache_read_only: bool = True


class BackendAdapter(ABC):
    name: str

    @abstractmethod
    def run_inference(self, request: InferenceRequest) -> dict[str, Any]:
        """Run inference and return telemetry metadata."""
