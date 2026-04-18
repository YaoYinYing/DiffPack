# NumPy Compatibility

## Policy

- Required baseline: `numpy>=1.26.4`.
- Experimental lane: `numpy>=2,<3` (quarantined in CI with `allow_failure=true`).

## Current Status (2026-04-18)

- `numpy==1.26.4`: pass on current test and smoke suite.
- `numpy==2.0.2`: **blocked** in current runtime stack due a hard crash in the
  RDKit / TorchDrug molecule construction path (segmentation fault during protein parsing).

## Practical Guidance

- Use `numpy==1.26.4` for production and release builds.
- Keep NumPy 2 lane running in CI as an early-warning signal while upstream stack compatibility matures.
