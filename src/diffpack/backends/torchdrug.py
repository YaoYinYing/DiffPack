from __future__ import annotations

from diffpack.backends.base import BackendAdapter, InferenceRequest
from diffpack.backends.torchdrug_compatible_runner import TorchDrugCompatibleRunner


class TorchDrugAdapter(BackendAdapter):
    name = "torchdrug"

    def __init__(self):
        self._runner = TorchDrugCompatibleRunner()

    def run_inference(self, request: InferenceRequest) -> dict[str, object]:
        return self._runner.run(
            request,
            backend_requested=self.name,
            backend_effective=self.name,
            backend_mode="native",
            fallback_reason=None,
        )
