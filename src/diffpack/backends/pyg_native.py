from __future__ import annotations

from diffpack.backends.base import BackendAdapter, InferenceRequest
from diffpack.backends.pyg_runtime import PygNativeRunner


class PygNativeAdapter(BackendAdapter):
    """Native PyG inference adapter."""

    name = "pyg"

    def __init__(self):
        self._runner = PygNativeRunner()

    def run_inference(self, request: InferenceRequest) -> dict[str, object]:
        return self._runner.run(
            request,
            backend_requested=self.name,
            backend_effective=self.name,
            backend_mode="native",
            fallback_reason=None,
        )
