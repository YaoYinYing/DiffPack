# DiffPack

DiffPack is a torsional diffusion model for protein side-chain packing.

This repository now ships as a pip-installable package built with `flit_core`, with a modern CLI and backend abstraction.

## Quick Start

### 1) Install package

```bash
pip install -e ".[torch]"
```

### 2) Install backend runtime dependencies

Vendored runtime code is included under `src/diffpack/torchdrug`.

```bash
pip install -e ".[torchdrug_fork]"
```

### 3) Run inference

```bash
diffpack-infer \
  -c src/diffpack/config/inference_confidence.yaml \
  --seed 2023 \
  --output_dir output \
  --pdb_files 1ubq.pdb \
  --center_residues A:72 \
  --repack_radius 10 \
  --hetero_policy exclude \
  --backend torchdrug_fork \
  --device cpu
```

Legacy entrypoint is still available temporarily:

```bash
python script/inference.py --help
```

## CLI

```text
diffpack-infer [options]

--backend {torchdrug_fork,pyg}
--device {cpu,cuda,mps}
--diagnose
--profile
--fast
--center_residues CHAIN:RESID ...
--repack_radius FLOAT
--hetero_policy {exclude,context_only,error}
```

`--diagnose` prints runtime/compiler/backend capabilities and exits.

## Backend Notes

- `torchdrug_fork`: default and production path.
- `pyg`: native PyG inference path (no implicit fallback to `torchdrug_fork` when explicitly selected).
- Current native PyG support target is the shipped inference configs (`inference.yaml`, `inference_confidence.yaml`).

## MPS and macOS

- MPS is supported at the DiffPack runtime layer (`--device mps`) when backend runtime supports it.
- OpenMP flags are not required in DiffPack itself; TorchDrug fork/toolchain setup controls extension compile flags.

## Testing

```bash
pytest -q
```

## Benchmarks

```bash
python benchmarks/run_benchmark.py --device cpu --backend torchdrug_fork
python benchmarks/run_benchmark.py --device mps --backend torchdrug_fork
python benchmarks/run_benchmark.py --device cpu --backend pyg --reference_backend torchdrug_fork
```

NumPy compatibility status is tracked in [docs/numpy-compatibility.md](docs/numpy-compatibility.md).

## Structure Checker

```bash
diffpack-check-structure \
  --input 1ubq.pdb \
  --output output/1ubq.pdb \
  --metadata output/run_metadata.json \
  --report output/checker.json
```

## Citation

```bibtex
@article{zhang2023diffpack,
  title={DiffPack: A Torsional Diffusion Model for Autoregressive Protein Side-Chain Packing},
  author={Zhang, Yangtian and Zhang, Zuobai and Zhong, Bozitao and Misra, Sanchit and Tang, Jian},
  journal={arXiv preprint arXiv:2306.01794},
  year={2023}
}
```
