from __future__ import annotations

from diffpack.backends.base import BackendAdapter, InferenceRequest
from diffpack.backends.native_runtime import NativeRunner


class NativeAdapter(BackendAdapter):
    """Custom native inference adapter (no torch_geometric dependency)."""

    name = "native"

    def __init__(self):
        self._runner = NativeRunner()

    def run_inference(self, request: InferenceRequest) -> dict[str, object]:
        return self._runner.run(
            request,
            backend_requested=self.name,
            backend_effective=self.name,
            backend_mode="native",
            fallback_reason=None,
        )
