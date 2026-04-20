# DiffPack

DiffPack is a torsional diffusion model for protein side-chain packing.

This repository now ships as a pip-installable package built with `flit_core`, with a modern CLI and backend abstraction.

## Quick Start

### 1) Install package

```bash
pip install -e .
```

### 2) Optional backend runtime dependencies

Vendored TorchDrug runtime code is included under `src/diffpack/torchdrug`.

```bash
pip install -e ".[torchdrug]"
pip install -e ".[pyg]"
```

### 2.1) Reproducible NumPy compatibility lanes (pip-only)

```bash
# NumPy 2.x lane
pip install -e ".[dev,torchdrug,pyg]" -c requirements/constraints-numpy2.txt

# NumPy 1.26 legacy lane
pip install -e ".[dev,torchdrug,pyg]" -c requirements/constraints-numpy126.txt
```

### 3) Run inference

```bash
diffpack-prepare-cache

diffpack-infer \
  -c src/diffpack/config/inference_confidence.yaml \
  --seed 2023 \
  --output_dir output \
  --pdb_files 1ubq.pdb \
  --center_residues A:72 \
  --repack_radius 10 \
  --hetero_policy exclude \
  --backend native \
  --device cpu
```

Inference enforces read-only cache usage. If cache validation fails, run:

```bash
diffpack-prepare-cache [--cache_root /path/to/cache]
```

Legacy entrypoint is still available temporarily:

```bash
python script/inference.py --help
```

## CLI

```text
diffpack-infer [options]

--backend {native,torchdrug,pyg}
--device {cpu,cuda}
--diagnose
--profile
--fast
--center_residues CHAIN:RESID ...
--repack_radius FLOAT
--hetero_policy {exclude,context_only,error}
```

`--diagnose` prints runtime/compiler/backend capabilities and exits.
It also reports NumPy/ABI dependency probe status for each backend.

## Backend Notes

- `native`: default backend, no `torch_geometric` dependency path.
- `torchdrug`: internal TorchDrug backend (vendored code under `src/diffpack/torchdrug`).
- `pyg`: torch_geometric-native backend.

## Apple Silicon / MPS

- MPS is intentionally **not supported** in DiffPack.
- Reason: we observed unstable runtime and memory behavior on MPS that is not release-grade compared with CPU/CUDA execution.
- On Apple Silicon, use `--device cpu`.
- OpenMP flags are not required in DiffPack itself; vendored TorchDrug runtime/toolchain setup controls extension compile flags.

## Testing

```bash
pytest -q
```

## Benchmarks

```bash
python benchmarks/run_benchmark.py --device cpu --backend native
python benchmarks/run_benchmark.py --device cpu --backend torchdrug
python benchmarks/run_benchmark.py --device cpu --backend pyg --reference_backend torchdrug

# Repeated CPU backend comparison (torchdrug/native/pyg)
python benchmarks/cpu_triplet_compare.py --repeats 3 --aggregate median
```

Benchmarks run strict geometry checks (ported from DLPacker checker logic) and fail checker status when severe clashes or bond outliers are detected.

Reference CPU comparison artifacts are tracked in `benchmarks/reference/` and mirrored into `outputs/` when you run the comparison script.

NumPy compatibility status is tracked in [docs/numpy-compatibility.md](docs/numpy-compatibility.md).

## Structure Checker

```bash
diffpack-check-structure \
  --input 1ubq.pdb \
  --output output/1ubq.pdb \
  --metadata output/run_metadata.json \
  --strict_geometry \
  --report output/checker.json
```

## Live Test

```bash
python scripts/live_test_inference.py --backend native --device cpu --pdb_file 1ubq.pdb
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
