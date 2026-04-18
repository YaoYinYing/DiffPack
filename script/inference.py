"""Compatibility shim for legacy entrypoint.

Deprecated: use `diffpack-infer` (installed console script) instead.
"""

from diffpack.cli.infer import main


if __name__ == "__main__":
    raise SystemExit(main())

