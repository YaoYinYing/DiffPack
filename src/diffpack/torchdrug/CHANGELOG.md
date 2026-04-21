# TorchDrug (Vendored) Changelog

This changelog tracks DiffPack-specific changes applied to the vendored TorchDrug copy in this directory.

## 2026-04-21

### Changed
- Migrated RDKit drawing helper to modern RDKit APIs in `data/rdkit/draw.py`.
  - Removed legacy imports:
    - `rdkit.Chem.Draw.MolDrawing`
    - `rdkit.Chem.Draw.mplCanvas`
  - Reimplemented drawing via `rdkit.Chem.Draw.MolToImage`.
  - Kept `MolToMPL(...)` function entrypoint for internal compatibility.

### Why
- Current RDKit builds used by DiffPack do not expose the legacy `mplCanvas` path.
- This avoids import/runtime failures from deprecated drawing internals.

### Scope
- Visualization helper path only.
- No inference model math or sidechain packing logic changes.
