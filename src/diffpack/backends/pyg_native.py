from __future__ import annotations

import warnings

from diffpack.backends.base import BackendAdapter, InferenceRequest
from diffpack.backends.torchdrug_compatible_runner import TorchDrugCompatibleRunner


class PygNativeAdapter(BackendAdapter):
    """
    Transitional adapter.

    The long-term goal is a fully native PyG execution path. During migration
    this adapter preserves the CLI / API contract while delegating execution to
    the maintained torchdrug fork backend.
    """

    name = "pyg"

    def __init__(self):
        self._runner = TorchDrugCompatibleRunner()

    def run_inference(self, request: InferenceRequest) -> dict[str, object]:
        warnings.warn(
            "PyG backend is in transitional mode and currently falls back to torchdrug_fork runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._runner.run(
            request,
            backend_requested=self.name,
            backend_effective="torchdrug_fork",
            backend_mode="fallback",
            fallback_reason="pyg_transitional_adapter",
        )
