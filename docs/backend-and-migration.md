# Backend and Migration Guide

## Backends

- `torchdrug_fork`: default production backend.
- `pyg`: transitional backend contract; currently routed through torchdrug runtime while native migration lands.

## Why a torchdrug fork

DiffPack relies on APIs and runtime behavior not fully maintained upstream for modern Python / macOS / MPS needs.
The maintained runtime is vendored in this repository under `src/diffpack/torchdrug`, so common users do not need
an editable external fork checkout.

## Sunset criteria for `torchdrug_fork`

The fork can be sunset after the `pyg` backend reaches:

1. Inference feature parity (including selective radius repacking),
2. Deterministic regression parity on fixed-seed test set,
3. Performance target parity or improvement,
4. CI stability across Python 3.10-3.14.
