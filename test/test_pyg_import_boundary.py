from __future__ import annotations

import ast
from pathlib import Path


def _imports_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def test_pyg_runtime_has_no_torchdrug_imports():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "src" / "diffpack" / "backends" / "pyg.py",
        repo_root / "src" / "diffpack" / "backends" / "pyg_runtime.py",
    ]
    bad = []
    for path in targets:
        for name in _imports_in_file(path):
            if name == "diffpack.torchdrug" or name.startswith("diffpack.torchdrug."):
                bad.append((str(path), name))
    assert not bad, f"PyG backend import boundary violated: {bad}"


def test_pyg_runtime_has_torch_geometric_references():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "src" / "diffpack" / "backends" / "pyg.py",
        repo_root / "src" / "diffpack" / "backends" / "pyg_runtime.py",
    ]
    found = False
    for path in targets:
        for name in _imports_in_file(path):
            if name == "torch_geometric" or name.startswith("torch_geometric."):
                found = True
                break
        if found:
            break
    assert found, "PyG backend must import torch_geometric directly"


def test_native_runtime_has_no_torch_geometric_or_pyg_imports():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "src" / "diffpack" / "backends" / "native.py",
        repo_root / "src" / "diffpack" / "backends" / "native_runtime.py",
    ]
    bad = []
    for path in targets:
        for name in _imports_in_file(path):
            if name == "torch_geometric" or name.startswith("torch_geometric."):
                bad.append((str(path), name))
            if name == "diffpack.backends.pyg_runtime" or name.startswith("diffpack.backends.pyg"):
                bad.append((str(path), name))
    assert not bad, f"Native backend import boundary violated: {bad}"
