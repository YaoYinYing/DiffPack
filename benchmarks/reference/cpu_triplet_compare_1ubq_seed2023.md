# CPU Triplet Compare (1ubq, seed=2023, repeats=3, aggregate=median)

| item | backend | status | wall_s | backend_s | peak_rss_kib | rt_rank | mem_rank | max_delta_vs_td | mean_delta_vs_td |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| inference_confidence_local | native | PASS | 40.326 | 38.417 | 2494480384 | 2 | 2 | 4.914 | 0.109 |
| inference_confidence_local | pyg | PASS | 46.285 | 42.571 | 2581889024 | 3 | 3 | 4.914 | 0.109 |
| inference_confidence_local | torchdrug | PASS | 27.434 | 20.758 | 2126528512 | 1 | 1 | 0.000 | 0.000 |
| inference_full | native | PASS | 11.519 | 9.453 | 2166538240 | 1 | 2 | 6.841 | 0.459 |
| inference_full | pyg | PASS | 14.225 | 10.505 | 2537717760 | 3 | 3 | 6.841 | 0.459 |
| inference_full | torchdrug | PASS | 12.261 | 5.436 | 1986379776 | 2 | 1 | 0.000 | 0.000 |