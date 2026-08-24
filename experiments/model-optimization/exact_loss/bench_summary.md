# MMD benchmark summary

torch 2.10.0+cu128, Linux-6.12.90-aufs-1-x86_64-with-glibc2.36, cpu, threads=4

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

## device = cuda

| variant | dtype | gmean speedup (all) | n<=8,m=250 | n=100,m=2000 | dim=768 | dim<=10 |
|---|---|---|---|---|---|---|
| stacked_mm | float32 | 0.96 | 0.92 | 1.03 | 0.95 | 0.96 |
| stacked_mm | float64 | 0.95 | 0.94 | 1.02 | 0.91 | 0.97 |
| stacked_powchain | float32 | 0.97 | 0.78 | 1.45 | 0.94 | 0.98 |
| stacked_powchain | float64 | 0.86 | 0.80 | 1.03 | 0.84 | 0.87 |
| fixed_cdist | float32 | 2.03 | 1.01 | 9.31 | 2.15 | 2.00 |
| fixed_cdist | float64 | 3.18 | 1.34 | 20.87 | 5.30 | 2.69 |
| fixed_mm | float32 | 2.19 | 1.04 | 10.97 | 2.31 | 2.15 |
| fixed_mm | float64 | 3.45 | 1.37 | 27.12 | 5.86 | 2.89 |
| fixed_mm_loop | float32 | 1.42 | 0.67 | 7.13 | 1.50 | 1.39 |
| fixed_mm_loop | float64 | 2.33 | 0.89 | 19.79 | 4.22 | 1.91 |
| fixed_mm_powchain | float32 | 1.48 | 0.69 | 7.43 | 1.56 | 1.45 |
| fixed_mm_powchain | float64 | 2.44 | 0.93 | 20.76 | 4.42 | 2.00 |
| fixed_mm_chunked256 | float32 | 1.13 | 0.79 | 2.56 | 1.19 | 1.11 |
| fixed_mm_chunked256 | float64 | 1.94 | 1.06 | 8.03 | 3.46 | 1.60 |
| fixed_mm_adaptive | float32 | 0.82 | 0.65 | 1.40 | 0.85 | 0.81 |
| fixed_mm_adaptive | float64 | 1.03 | 0.83 | 2.00 | 1.74 | 0.86 |
| batched_fixed_mm_B3 | float32 | 5.52 | 2.54 | 26.54 | 5.82 | 5.43 |
| batched_fixed_mm_B3 | float64 | 7.52 | 3.17 | 30.63 | 9.87 | 6.87 |

### Absolute median forward+backward times (ms)

| n_cond | n_target | dim | dtype | reference | stacked_mm | stacked_powchain | fixed_cdist | fixed_mm | fixed_mm_loop | fixed_mm_powchain | fixed_mm_chunked256 | fixed_mm_adaptive | reference_adaptive | batched_fixed_mm_B3 | batched_reference_B3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 250 | 2 | float32 | 0.648 | 0.706 | 0.832 | 0.629 | 0.615 | 0.957 | 0.918 | 0.804 | 1.100 | 0.727 | 0.684 | 1.766 |
| 1 | 250 | 2 | float64 | 0.749 | 0.782 | 0.932 | 0.633 | 0.615 | 0.959 | 0.917 | 0.802 | 1.145 | 0.839 | 0.682 | 1.848 |
| 8 | 250 | 2 | float32 | 0.650 | 0.706 | 0.835 | 0.651 | 0.631 | 0.977 | 0.937 | 0.820 | 1.117 | 0.727 | 0.698 | 1.772 |
| 8 | 250 | 2 | float64 | 0.747 | 0.778 | 0.930 | 0.651 | 0.642 | 0.985 | 0.941 | 0.836 | 1.170 | 0.837 | 0.701 | 1.852 |
| 32 | 250 | 2 | float32 | 0.653 | 0.705 | 0.832 | 0.761 | 0.645 | 0.991 | 0.950 | 0.838 | 1.133 | 0.727 | 0.698 | 1.773 |
| 32 | 250 | 2 | float64 | 0.652 | 0.708 | 0.836 | 0.753 | 0.639 | 0.987 | 0.943 | 0.835 | 1.174 | 0.743 | 0.708 | 1.776 |
| 8 | 250 | 10 | float32 | 0.652 | 0.707 | 0.833 | 0.652 | 0.634 | 0.976 | 0.937 | 0.824 | 1.117 | 0.729 | 0.695 | 1.770 |
| 8 | 250 | 10 | float64 | 0.745 | 0.781 | 0.932 | 0.649 | 0.643 | 0.983 | 0.939 | 0.833 | 1.172 | 0.838 | 0.705 | 1.846 |
| 8 | 100 | 768 | float32 | 0.646 | 0.699 | 0.826 | 0.654 | 0.629 | 0.975 | 0.944 | 0.821 | 1.120 | 0.722 | 0.699 | 1.749 |
| 8 | 100 | 768 | float64 | 0.784 | 0.990 | 1.140 | 0.652 | 0.705 | 0.977 | 0.931 | 0.825 | 1.158 | 0.876 | 0.743 | 1.977 |
| 32 | 250 | 768 | float32 | 0.649 | 0.705 | 0.831 | 0.767 | 0.649 | 0.990 | 0.952 | 0.837 | 1.139 | 0.727 | 0.701 | 1.774 |
| 32 | 250 | 768 | float64 | 2.004 | 2.150 | 2.316 | 0.787 | 0.645 | 0.991 | 0.951 | 0.842 | 1.178 | 2.130 | 0.939 | 5.815 |
| 100 | 2000 | 10 | float32 | 6.678 | 6.525 | 4.482 | 0.763 | 0.649 | 0.998 | 0.960 | 2.833 | 7.850 | 10.567 | 0.804 | 19.581 |
| 100 | 2000 | 10 | float64 | 19.112 | 18.766 | 18.533 | 0.891 | 0.704 | 1.034 | 0.991 | 2.790 | 16.749 | 27.058 | 1.905 | 57.178 |
| 100 | 2000 | 768 | float32 | 8.376 | 8.243 | 6.199 | 0.769 | 0.659 | 1.009 | 0.968 | 2.796 | 7.904 | 12.164 | 0.839 | 24.925 |
| 100 | 2000 | 768 | float64 | 67.996 | 67.262 | 67.338 | 3.633 | 2.406 | 2.562 | 2.381 | 4.832 | 18.281 | 75.803 | 8.603 | 203.589 |

### CUDA peak allocated memory (MB) per cell, largest cells

| variant | n=32,m=250,d=768 f32 | n=100,m=2000,d=768 f32 | n=100,m=2000,d=768 f64 |
|---|---|---|---|
| reference | 23.8 | 314.5 | 612.8 |
| stacked_mm | 24.6 | 314.5 | 612.8 |
| stacked_powchain | 23.8 | 180.0 | 346.5 |
| fixed_cdist | 22.8 | 308.8 | 538.0 |
| fixed_mm | 22.8 | 309.5 | 538.7 |
| fixed_mm_loop | 22.8 | 309.5 | 538.7 |
| fixed_mm_powchain | 22.8 | 309.5 | 538.7 |
| fixed_mm_chunked256 | 22.8 | 309.5 | 538.7 |
| fixed_mm_adaptive | 23.7 | 425.8 | 834.7 |
| reference_adaptive | 26.8 | 482.8 | 949.3 |
| batched_fixed_mm_B3 | 23.0 | 310.1 | 540.6 |
| batched_reference_B3 | 29.3 | 529.3 | 1043.1 |

### Max RSS growth over the grid (MB, per worker process; the grid is run in ascending size order so this is dominated by the largest cell n=100, m=2000, d=768 float64)

| variant | rss_delta_mb (end of grid) | first-call s (largest cell) |
|---|---|---|
| reference | 603.3 | 0.008 |
| stacked_mm | 554.9 | 0.008 |
| stacked_powchain | 653.4 | 0.006 |
| fixed_cdist | 627.8 | 0.003 |
| fixed_mm | 563.8 | 0.003 |
| fixed_mm_loop | 560.9 | 0.003 |
| fixed_mm_powchain | 617.2 | 0.003 |
| fixed_mm_chunked256 | 564.8 | 0.004 |
| fixed_mm_adaptive | 577.0 | 0.008 |
| reference_adaptive | 602.0 | 0.012 |
| batched_fixed_mm_B3 | 553.8 | 0.003 |
| batched_reference_B3 | 605.8 | 0.025 |
