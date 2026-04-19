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
        repo_root / "src" / "diffpack" / "backends" / "pyg_native.py",
        repo_root / "src" / "diffpack" / "backends" / "pyg_runtime.py",
    ]
    bad = []
    for path in targets:
        for name in _imports_in_file(path):
            if name == "diffpack.torchdrug" or name.startswith("diffpack.torchdrug."):
                bad.append((str(path), name))
    assert not bad, f"PyG backend import boundary violated: {bad}"
