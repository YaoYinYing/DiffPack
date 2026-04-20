# NumPy Compatibility

## Policy

- Required baseline: `numpy>=1.26.4`.
- Required compatibility lane: `numpy>=2,<3`.
- Dependency strategy: pip-only wheels with explicit constraints files.

## Current Status (2026-04-20)

- `numpy==1.26.4`: supported via `requirements/constraints-numpy126.txt`.
- `numpy>=2,<3`: supported target via `requirements/constraints-numpy2.txt`.
- Runtime now performs backend ABI preflight checks and fails fast with remediation commands if
  binary dependencies are incompatible (instead of late segfaults).

## Practical Guidance

- Install with one of:
  - `pip install -e ".[dev,torchdrug,pyg]" -c requirements/constraints-numpy126.txt`
  - `pip install -e ".[dev,torchdrug,pyg]" -c requirements/constraints-numpy2.txt`
- Run `diffpack-infer --diagnose` before inference to validate backend dependency readiness.
