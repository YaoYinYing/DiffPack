from __future__ import annotations

from diffpack.backend_preflight import ensure_backend_ready
from diffpack.backends.base import BackendAdapter, InferenceRequest
from diffpack.backends.mutation_request import prepare_mutation_request


class NativeAdapter(BackendAdapter):
    """Custom native inference adapter (no torch_geometric dependency)."""

    name = "native"

    def __init__(self):
        self._abi_probe = ensure_backend_ready(self.name)
        from diffpack.backends.native_runtime import NativeRunner

        self._runner = NativeRunner()

    def run_inference(self, request: InferenceRequest) -> dict[str, object]:
        request, mutation_meta = prepare_mutation_request(request)
        metadata = self._runner.run(
            request,
            backend_requested=self.name,
            backend_effective=self.name,
            backend_mode="native",
            fallback_reason=None,
        )
        metadata.update(mutation_meta)
        metadata.setdefault("numpy_version", self._abi_probe.get("numpy_version"))
        metadata.setdefault("abi_probe_status", self._abi_probe.get("status"))
        metadata.setdefault("abi_probe_errors", self._abi_probe.get("required_errors", []))
        metadata.setdefault("dependency_readiness", self._abi_probe.get("dependency_readiness", {}))
        return metadata
