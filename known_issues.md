# Known Issues (Live Test Snapshot)

## Environment

- Date: 2026-04-20
- Test env: `conda env DiffPack`
- Input: `1ubq.pdb`
- Cache root: `/Users/yyy/Library/Caches/DiffPackCache`

## Setup

```bash
pip install '.[pyg,torchdrug]'
diffpack-prepare-cache
```

## Runtime Command Template

```bash
diffpack-infer \
  -c config/inference_confidence.yaml \
  --seed 2023 \
  --output_dir output_<backend>_<device> \
  --pdb_files 1ubq.pdb \
  --center_residues A:72 \
  --repack_radius 10 \
  --hetero_policy exclude \
  --backend <backend> \
  --device <device> \
  --cache_root /Users/yyy/Library/Caches/DiffPackCache
```

## Live Matrix Results (Current)

| Backend | Device | Status | Notes |
|---|---|---|---|
| `native` | `cpu` | PASS | Full repack strict checker PASS |
| `native` | `cpu` + local repack | PASS | `outside_mask_frozen` PASS (`max_abs_delta=0.0`) |
| `pyg` | `cpu` | PASS | Full repack strict checker PASS |
| `pyg` | `cpu` + local repack | PASS | `outside_mask_frozen` PASS (`max_abs_delta=0.0`) |
| `torchdrug` | `cpu` | PASS | Full repack completes, metrics reported |
| `native` | `mps` | BLOCKED (runtime stall) | Command starts, no output PDB produced in this run environment |
| `pyg` | `mps` | BLOCKED (runtime stall) | Command starts, no output PDB produced in this run environment |
| `torchdrug` | `mps` | FAIL | `RuntimeError: Could not infer dtype of NoneType` in torchdrug `to()` path |

## CPU Artifacts (2026-04-20)

- Full repack outputs:
  - `outputs/live_native_after_fix/1ubq.pdb`
  - `outputs/live_pyg_after_fix/1ubq.pdb`
  - `outputs/live_torchdrug_after_fix/1ubq.pdb`
- Full repack strict checker reports:
  - `outputs/live_native_after_fix/structure_report_strict.json`
  - `outputs/live_pyg_after_fix/structure_report_strict.json`
  - `outputs/live_torchdrug_after_fix/structure_report_strict.json`
- Local repack outputs (`A:72`, radius `10`):
  - `outputs/live_native_local_after_fix/1ubq.pdb`
  - `outputs/live_pyg_local_after_fix/1ubq.pdb`
- Local repack strict checker reports:
  - `outputs/live_native_local_after_fix/structure_report_strict.json`
  - `outputs/live_pyg_local_after_fix/structure_report_strict.json`

## Parity Status (TorchDrug Reference)

- Stage parity now matches at:
  - dataset mapping/features/masks,
  - graph edge list + edge features,
  - schedule scalars (`t`, `dt`, `sigma`),
  - generation masks (`mask_1pi`, `mask_2pi`),
  - `score_norms`.
- First remaining divergence:
  - `generation.pred_scores` (then `chi_states` / final coordinates).
- Latest parity artifacts:
  - `outputs/parity_native_after_noise_fix/parity_trace_report.json`
  - `outputs/parity_pyg_after_noise_fix/parity_trace_report.json`

## Remaining Issues

1. **MPS runtime unresolved**
   - `native` / `pyg` MPS commands did not complete in this environment.
   - `torchdrug` MPS crashes with:
     ```text
     RuntimeError: Could not infer dtype of NoneType
     at diffpack/torchdrug/data/protein.py::__init__
     (called from diffpack/torchdrug/data/graph.py::to)
     ```

2. **Non-zero generation parity drift**
   - Even after mask / sigma-embedding / geometric-conv alignment, `pred_scores` remains the first divergent stage.
   - Current drift is much smaller than earlier runs but still above strict parity tolerance.

## Notes

- Previous `native` / `pyg` CPU structural crash symptom is not reproduced in current `1ubq` runs.
- `clash_guard` remains non-default; these PASS results are from raw generation outputs.
