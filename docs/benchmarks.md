# Benchmark Methodology

## Goal

Track runtime and regression across CPU and MPS as backend migration progresses.

## Command

```bash
python benchmarks/run_benchmark.py --backend torchdrug --device cpu --output_dir benchmark_output
```

## Output

- `benchmark_output/benchmark_result.json`
- `output/run_metadata.json` from inference path
- `checker_reports` and aggregate `checker_status` in `benchmark_result.json`

## Fields to compare

- `elapsed_sec`
- `wall_elapsed_sec`
- backend effective/requested labels
- checker status (`pass` / `fail`)
