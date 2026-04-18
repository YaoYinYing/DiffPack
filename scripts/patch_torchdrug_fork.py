#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_in_file(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Apply DiffPack compatibility patches to a torchdrug fork")
    parser.add_argument("--torchdrug-root", default="/Users/yyy/Documents/protein_design/torchdrug")
    args = parser.parse_args()

    root = Path(args.torchdrug_root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"torchdrug root does not exist: {root}")

    setup_py = root / "setup.py"
    utils_torch = root / "torchdrug" / "utils" / "torch.py"
    engine_py = root / "torchdrug" / "core" / "engine.py"

    patched = []

    if replace_in_file(setup_py, 'python_requires=">=3.7,<3.11"', 'python_requires=">=3.10,<3.15"'):
        patched.append(str(setup_py))

    # Disable unconditional OpenMP flags (macOS clang compatibility).
    if replace_in_file(
        utils_torch,
        'if torch.backends.openmp.is_available():\n            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]',
        'if torch.backends.openmp.is_available() and __import__("os").environ.get("TORCHDRUG_ENABLE_OPENMP", "0") == "1":\n'
        '            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]',
    ):
        patched.append(str(utils_torch))

    # Basic MPS acceptance in engine GPU gate.
    if replace_in_file(
        engine_py,
        "if gpus is None or (not torch.cuda.is_available()):",
        "if gpus is None or ((not torch.cuda.is_available()) and (not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))):",
    ):
        patched.append(str(engine_py))

    if not patched:
        print("No patch hunks applied. Check torchdrug fork version and patch script patterns.")
    else:
        print("Patched files:")
        for p in patched:
            print(f"- {p}")
        print("Done.")


if __name__ == "__main__":
    main()
