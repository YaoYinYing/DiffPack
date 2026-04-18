# Troubleshooting

## `clang++: error: unsupported option '-fopenmp'` on macOS

This error typically comes from backend extension compilation (torchdrug/related deps), not DiffPack core.

Recommended:

1. Use the vendored runtime shipped in `src/diffpack/torchdrug` (default).
2. If OpenMP is required, install `libomp` and use a matching clang toolchain.
3. Run diagnostics:

```bash
diffpack-infer --diagnose
```

## MPS unavailable

Check:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If `False`, run with `--device cpu` or install a torch build with MPS enabled.

## Runtime extension build failures

DiffPack uses a safe fallback path (`torch.sparse_coo_tensor`) when the optional C++ extension cannot be built.
This keeps CPU/MPS paths functional with deterministic behavior, at the cost of some performance.
