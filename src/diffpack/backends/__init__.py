from __future__ import annotations

from diffpack.backends.base import BackendAdapter, InferenceRequest


def get_backend_adapter(name: str) -> BackendAdapter:
    normalized = (name or "native").strip().lower()
    if normalized == "native":
        from diffpack.backends.native import NativeAdapter
        return NativeAdapter()
    if normalized == "torchdrug":
        from diffpack.backends.torchdrug import TorchDrugAdapter
        return TorchDrugAdapter()
    if normalized == "pyg":
        from diffpack.backends.pyg import PygAdapter
        return PygAdapter()
    raise ValueError(f"Unsupported backend `{name}`. Expected one of: native, torchdrug, pyg.")


__all__ = ["InferenceRequest", "BackendAdapter", "get_backend_adapter"]
