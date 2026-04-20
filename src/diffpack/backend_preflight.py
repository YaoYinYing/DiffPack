from __future__ import annotations

import contextlib
import importlib
import io
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProbeSpec:
    module: str
    required: bool


def _probe_module(module: str) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            importlib.import_module(module)
    except Exception as error:  # pragma: no cover - exercised by integration/runtime checks
        stderr_text = stderr.getvalue().strip()
        return {
            "module": module,
            "ok": False,
            "error": f"{error.__class__.__name__}: {error}",
            "stderr": stderr_text[-1200:] if stderr_text else "",
        }
    return {"module": module, "ok": True, "error": None, "stderr": ""}


def _backend_specs(backend: str) -> list[ProbeSpec]:
    if backend == "native":
        return [
            ProbeSpec("torch", True),
            ProbeSpec("numpy", True),
            ProbeSpec("rdkit.Chem", False),
            ProbeSpec("rdkit.Chem.AllChem", False),
            ProbeSpec("rdkit.Chem.Descriptors", False),
            ProbeSpec("torch_cluster", False),
        ]
    if backend == "pyg":
        return [
            ProbeSpec("torch", True),
            ProbeSpec("numpy", True),
            ProbeSpec("torch_geometric", True),
            ProbeSpec("rdkit.Chem", False),
            ProbeSpec("rdkit.Chem.AllChem", False),
            ProbeSpec("rdkit.Chem.Descriptors", False),
            ProbeSpec("torch_cluster", False),
        ]
    if backend == "torchdrug":
        return [
            ProbeSpec("torch", True),
            ProbeSpec("numpy", True),
            ProbeSpec("rdkit.Chem", True),
            ProbeSpec("rdkit.Chem.AllChem", True),
            ProbeSpec("rdkit.Chem.Descriptors", True),
            ProbeSpec("torch_geometric", True),
            ProbeSpec("torch_scatter", True),
            ProbeSpec("torch_cluster", True),
        ]
    raise ValueError(f"Unsupported backend `{backend}` for dependency probe")


def _build_remediation(backend: str) -> str:
    if backend == "native":
        return (
            "pip install -U -c requirements/constraints-numpy2.txt "
            "\"numpy>=2,<3\" \"torch>=2.3\" \"rdkit>=2024.3.5\""
        )
    if backend == "pyg":
        return (
            "pip install -U -c requirements/constraints-numpy2.txt "
            "\"numpy>=2,<3\" \"torch>=2.3\" \"torch-geometric>=2.6\" "
            "\"torch-scatter>=2.1.2\" \"torch-cluster>=1.6.3\" \"rdkit>=2024.3.5\""
        )
    return (
        "pip install -U -c requirements/constraints-numpy2.txt "
        "\"numpy>=2,<3\" \"rdkit>=2024.3.5\" \"torch-geometric>=2.6\" "
        "\"torch-scatter>=2.1.2\" \"torch-cluster>=1.6.3\""
    )


def _probe_torchdrug_runtime_subprocess() -> dict[str, Any]:
    code = """
import rdkit.Chem.AllChem  # noqa: F401
import rdkit.Chem.Draw  # noqa: F401
import rdkit.Chem.Descriptors  # noqa: F401
import diffpack.torchdrug.utils.plot  # noqa: F401
import diffpack.torchdrug.metrics.metric  # noqa: F401
"""
    cmd = [
        sys.executable,
        "-c",
        code,
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/diffpack_torch_extensions")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    ok = proc.returncode == 0
    stderr = (proc.stderr or "").strip()[-1200:]
    stdout = (proc.stdout or "").strip()[-400:]
    error = None if ok else f"subprocess import probe failed with code={proc.returncode}"
    if error and stderr:
        error = f"{error} stderr={stderr}"
    elif error and stdout:
        error = f"{error} stdout={stdout}"
    return {
        "module": "torchdrug_runtime_subprocess",
        "ok": ok,
        "error": error,
        "stderr": stderr,
    }


def probe_backend_dependencies(backend: str) -> dict[str, Any]:
    specs = _backend_specs(backend)
    readiness: dict[str, Any] = {}
    required_errors: list[str] = []
    optional_errors: list[str] = []

    for spec in specs:
        result = _probe_module(spec.module)
        readiness[spec.module] = result
        if result["ok"]:
            continue
        message = f"{spec.module}: {result['error']}"
        if result.get("stderr"):
            message += f" | stderr={result['stderr']}"
        if spec.required:
            required_errors.append(message)
        else:
            optional_errors.append(message)

    if backend == "torchdrug":
        runtime_probe = _probe_torchdrug_runtime_subprocess()
        readiness[runtime_probe["module"]] = runtime_probe
        if not runtime_probe["ok"]:
            required_errors.append(f"{runtime_probe['module']}: {runtime_probe['error']}")

    np_version = np.__version__
    status = "pass" if not required_errors else "fail"
    return {
        "backend": backend,
        "numpy_version": np_version,
        "status": status,
        "required_errors": required_errors,
        "optional_errors": optional_errors,
        "dependency_readiness": readiness,
        "remediation": _build_remediation(backend),
    }


def ensure_backend_ready(backend: str) -> dict[str, Any]:
    if os.environ.get("DIFFPACK_SKIP_ABI_PREFLIGHT", "").strip() == "1":
        return {
            "backend": backend,
            "numpy_version": np.__version__,
            "status": "skipped",
            "required_errors": [],
            "optional_errors": [],
            "dependency_readiness": {},
            "remediation": "",
        }

    probe = probe_backend_dependencies(backend)
    if probe["status"] != "pass":
        raise RuntimeError(
            "Backend dependency preflight failed for "
            f"`{backend}` under numpy {probe['numpy_version']}. "
            f"Errors: {probe['required_errors']}. "
            f"Remediation: {probe['remediation']}"
        )

    # Native / pyg can run without rdkit via geometric bond fallback.
    rdkit_probe = probe["dependency_readiness"].get("rdkit.Chem")
    if rdkit_probe and not rdkit_probe.get("ok"):
        os.environ["DIFFPACK_SKIP_RDKIT"] = "1"
    return probe
