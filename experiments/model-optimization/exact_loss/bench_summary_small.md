# MMD benchmark summary

torch 2.12.0, macOS-26.0.1-arm64-arm-64bit-Mach-O, arm, threads=4

Speedup = reference median forward+backward time / variant median time (same device, dtype, n_cond, n_target, dim). Geometric mean over the grid, then by regime.

## device = cpu

| variant | dtype | gmean speedup (all) | n<=8,m=250 | n=100,m=2000 | dim=768 | dim<=10 |
|---|---|---|---|---|---|---|
| stacked_mm | float32 | 0.98 | 0.98 | nan | 1.03 | 0.93 |
| stacked_mm | float64 | 1.02 | 1.02 | nan | 1.08 | 0.96 |
| stacked_powchain | float32 | 0.93 | 0.89 | nan | 0.95 | 0.91 |
| stacked_powchain | float64 | 1.17 | 1.14 | nan | 1.12 | 1.21 |
| fixed_cdist | float32 | 4.67 | 5.74 | nan | 4.06 | 5.37 |
| fixed_cdist | float64 | 7.25 | 9.29 | nan | 6.63 | 7.92 |
| fixed_mm | float32 | 6.29 | 7.71 | nan | 7.40 | 5.35 |
| fixed_mm | float64 | 10.16 | 12.98 | nan | 12.91 | 8.00 |
| fixed_mm_loop | float32 | 4.80 | 5.39 | nan | 5.72 | 4.03 |
| fixed_mm_loop | float64 | 7.39 | 8.82 | nan | 9.68 | 5.65 |
| fixed_mm_powchain | float32 | 6.00 | 6.42 | nan | 7.07 | 5.09 |
| fixed_mm_powchain | float64 | 9.55 | 11.04 | nan | 11.90 | 7.66 |
| fixed_mm_chunked256 | float32 | 6.07 | 7.25 | nan | 7.06 | 5.22 |
| fixed_mm_chunked256 | float64 | 9.47 | 11.92 | nan | 11.80 | 7.60 |
| fixed_mm_adaptive | float32 | 1.91 | 1.98 | nan | 2.33 | 1.56 |
| fixed_mm_adaptive | float64 | 1.93 | 2.00 | nan | 2.53 | 1.47 |
| batched_fixed_mm_B3 | float32 | 12.62 | 15.82 | nan | 14.36 | 11.09 |
| batched_fixed_mm_B3 | float64 | 17.89 | 23.93 | nan | 20.08 | 15.93 |

### Absolute median forward+backward times (ms)

| n_cond | n_target | dim | dtype | reference | stacked_mm | stacked_powchain | fixed_cdist | fixed_mm | fixed_mm_loop | fixed_mm_powchain | fixed_mm_chunked256 | fixed_mm_adaptive | reference_adaptive | batched_fixed_mm_B3 | batched_reference_B3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 250 | 2 | float32 | 0.740 | 0.793 | 0.901 | 0.095 | 0.101 | 0.164 | 0.136 | 0.117 | 0.577 | 0.928 | 0.150 | 2.131 |
| 1 | 250 | 2 | float64 | 1.249 | 1.291 | 1.102 | 0.098 | 0.101 | 0.169 | 0.136 | 0.119 | 1.128 | 1.655 | 0.145 | 3.731 |
| 8 | 250 | 2 | float32 | 0.776 | 0.835 | 0.846 | 0.132 | 0.132 | 0.173 | 0.147 | 0.128 | 0.642 | 1.004 | 0.184 | 2.347 |
| 8 | 250 | 2 | float64 | 1.266 | 1.325 | 1.049 | 0.142 | 0.146 | 0.206 | 0.157 | 0.150 | 1.161 | 1.795 | 0.233 | 3.968 |
| 32 | 250 | 2 | float32 | 0.922 | 1.004 | 0.910 | 0.273 | 0.259 | 0.286 | 0.200 | 0.246 | 0.836 | 1.276 | 0.392 | 2.953 |
| 32 | 250 | 2 | float64 | 1.446 | 1.535 | 1.123 | 0.331 | 0.304 | 0.365 | 0.239 | 0.291 | 1.513 | 2.122 | 0.495 | 4.583 |
| 32 | 250 | 768 | float32 | 1.607 | 1.529 | 1.571 | 0.568 | 0.326 | 0.357 | 0.269 | 0.331 | 0.929 | 1.909 | 0.619 | 5.302 |
| 32 | 250 | 768 | float64 | 3.550 | 3.275 | 3.077 | 0.797 | 0.435 | 0.520 | 0.422 | 0.491 | 1.787 | 4.097 | 1.022 | 11.025 |

### Max RSS growth over the grid (MB, per worker process; the grid is run in ascending size order so this is dominated by the largest cell n=100, m=2000, d=768 float64)

| variant | rss_delta_mb (end of grid) | first-call s (largest cell) |
|---|---|---|
| reference | 56.6 | 0.002 |
| stacked_mm | 62.5 | 0.002 |
| stacked_powchain | 56.4 | 0.002 |
| fixed_cdist | 35.2 | 0.001 |
| fixed_mm | 35.1 | 0.000 |
| fixed_mm_loop | 33.1 | 0.000 |
| fixed_mm_powchain | 33.9 | 0.000 |
| fixed_mm_chunked256 | 34.7 | 0.000 |
| fixed_mm_adaptive | 46.5 | 0.001 |
| reference_adaptive | 65.8 | 0.002 |
| batched_fixed_mm_B3 | 36.9 | 0.001 |
| batched_reference_B3 | 92.1 | 0.006 |
