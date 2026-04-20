from __future__ import annotations

import platform
from typing import Any

import torch


def choose_torch_device(device: str | None) -> torch.device:
    requested = (device or "cpu").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this runtime.")
        return torch.device("cuda")
    if requested == "mps":
        raise RuntimeError(
            "MPS is not supported by DiffPack. Use `--device cpu` on Apple Silicon, or `--device cuda` on NVIDIA GPUs."
        )
    raise ValueError(f"Unsupported device `{device}`. Expected one of: cpu, cuda.")


def move_to_device(batch: Any, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if hasattr(batch, "to") and callable(batch.to):
        try:
            return batch.to(device)
        except TypeError:
            pass
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def device_diagnostics() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "omp_env_omp_num_threads": __import__("os").environ.get("OMP_NUM_THREADS"),
        "openmp_default_on_macos": platform.system() == "Darwin",
    }
