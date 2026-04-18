from __future__ import annotations

from diffpack.backends.base import BackendAdapter, InferenceRequest


def get_backend_adapter(name: str) -> BackendAdapter:
    normalized = (name or "torchdrug_fork").strip().lower()
    if normalized in {"torchdrug", "torchdrug_fork"}:
        from diffpack.backends.torchdrug_fork import TorchDrugForkAdapter
        return TorchDrugForkAdapter()
    if normalized in {"pyg", "pyg_native"}:
        from diffpack.backends.pyg_native import PygNativeAdapter
        return PygNativeAdapter()
    raise ValueError(f"Unsupported backend `{name}`. Expected one of: torchdrug_fork, pyg.")


__all__ = ["InferenceRequest", "BackendAdapter", "get_backend_adapter"]
