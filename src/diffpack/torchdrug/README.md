# Vendored TorchDrug in DiffPack

This directory contains a vendored copy of TorchDrug code used by DiffPack.

## Origin

- Upstream project: https://github.com/DeepGraphLearning/torchdrug
- This copy is maintained inside DiffPack to ensure reproducible installs and runtime behavior across supported Python versions.

## Why it exists here

DiffPack relies on TorchDrug internals for parts of the inference stack. Using an external pip install of upstream TorchDrug is not always sufficient for this repository because:

- DiffPack needs repository-specific compatibility fixes.
- Upstream release constraints may not match DiffPack's supported environment matrix.
- Users should be able to install and run DiffPack without patching TorchDrug manually.

## Scope and policy

- This is a runtime dependency implementation detail for DiffPack.
- Changes here should be minimal and targeted to DiffPack needs.
- When possible, keep structure close to upstream TorchDrug to reduce maintenance risk.
- Document notable local deviations in commit messages and PR descriptions.

## Optional visualization dependencies

Some TorchDrug visualization paths use matplotlib / RDKit drawing utilities.  
DiffPack inference workflows do not require those visualization APIs by default.
