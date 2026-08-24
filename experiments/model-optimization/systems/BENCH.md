# Agent 5 — systems benchmarks on the synthetic loop (`_guided.run`, 2D, T=100)

Files: `runners.py` (alternative runners), `bench.py` (driver -> `bench_rows.csv`,
`bench_full.log`), `bench_scaling_quiet.log` (second restart-batching scan),
`microbatch_mmd.py` (part C, `microbatch_mmd.log`). Python
`/Users/stolk/miniconda3/bin/python`, torch 2.12, CPU float32 (the repo loop is float32
end to end, see AUDIT.md §1.0), 4 threads, Apple-silicon Mac. Commit `6af2081`.

**Caveat on absolute times.** The machine was shared with the other campaign agents'
benchmarks during both runs (load average 3-16 on 4 cores). Within one run the variants
were measured back-to-back against a freshly timed reference, so the *ratios* are
meaningful; absolute seconds move by up to 3x between runs (reference no-LGD n=8: 0.19 s
in the Agent 1 baseline, 0.20 s in `bench_full.log`, 0.60 s in
`bench_scaling_quiet.log`). Medians of 5 repeats after 1 warm-up, `time.perf_counter()`.
Peak RSS is `ru_maxrss` of the whole process and therefore a running maximum across
rows (it never decreases down the table); the synthetic graphs are tiny (<< 100 MB),
no variant changes memory measurably.

## How equivalence was judged

* **end-to-end**: `max |x_final(variant) - x_final(_guided.run)|` for the same
  restart(s) (restarts 0..3 for B >= 4).
* **per-step (teacher-forced)**: the variant is fed the *reference* `x_t` at every step
  (`force_traj`) and `max_t |g_variant(x_t) - g_ref(x_t)|` over the 99 steps is reported.
  This isolates numerical agreement of one guidance evaluation from the chaotic
  amplification documented in AUDIT.md §1.0 (float64 vs float32 *of the same
  algorithm* moves the final `x` by 0.04-1.1 and flips modes). Gradient magnitudes
  are 1e-2..1, so per-step diffs of 1e-7..1e-4 are float32 reduction-order round-off.
* Variants with end-to-end **0.0** are bit-identical (EXACT). Variants with per-step
  ~1e-6 but end-to-end O(1e-2..1) are REORDER: same arithmetic, different float32
  summation order (e.g. `cdist` kernel entries are bit-identical, only `.mean()` order
  differs; BLAS picks a different GEMM kernel above ~32 rows — checked: CM forward on
  the same rows differs by 7.6e-6 once the batch has >= 32 rows, 0.0 below).

## Results (`bench_rows.csv`, run 1)

| variant | cell | B | wall s (median of 5) | per-restart s | restarts/s | peak RSS MB (running max) | end-to-end max abs dx | per-step max abs dg (teacher-forced) | status |
|---|---|---|---|---|---|---|---|---|---|
| reference(_guided.run) | no_lgd/none/n8 | 1 | 0.2034 | 0.2034 | 4.917 | 322.0 | 0.000e+00 | 0.000e+00 | ok |
| generator_seed | no_lgd/none/n8 | 1 | 0.2422 | 0.2422 | 4.129 | 333.3 | 0.000e+00 | 0.000e+00 | ok |
| frozen_params | no_lgd/none/n8 | 1 | 0.2608 | 0.2608 | 3.834 | 333.4 | 0.000e+00 | 0.000e+00 | ok |
| batched_mmd(+batched_lgd) | no_lgd/none/n8 | 1 | 0.1349 | 0.1349 | 7.412 | 333.6 | 1.335e-05 | 1.311e-06 | ok |
| batched_restarts | no_lgd/none/n8 | 1 | 0.1618 | 0.1618 | 6.179 | 334.5 | 1.335e-05 | 1.311e-06 | ok |
| batched_restarts | no_lgd/none/n8 | 2 | 0.1784 | 0.0892 | 11.212 | 335.3 | 3.809e-02 | 1.587e-04 | ok |
| batched_restarts | no_lgd/none/n8 | 4 | 0.1916 | 0.0479 | 20.881 | 337.3 | 3.043e+00 | 2.151e-04 | ok |
| batched_restarts | no_lgd/none/n8 | 8 | 0.2170 | 0.0271 | 36.858 | 338.8 | 3.043e+00 | 2.151e-04 | ok |
| batched_restarts+lean_ddim | no_lgd/none/n8 | 8 | 0.2135 | 0.0267 | 37.477 | 338.8 | 3.043e+00 | - | ok |
| batched_restarts | no_lgd/none/n8 | 16 | 0.2827 | 0.0177 | 56.605 | 338.8 | 3.043e+00 | 2.151e-04 | ok |
| batched_restarts | no_lgd/none/n8 | 32 | 0.4528 | 0.0141 | 70.674 | 338.8 | 5.183e-01 | 2.217e-04 | ok |
| float64_throughout | no_lgd/none/n8 | 1 | 0.4249 | 0.4249 | 2.354 | 355.8 | 3.807e-02 | - | changes_numerics(float64) |
| reference(_guided.run) | no_lgd/none/n32 | 1 | 0.2958 | 0.2958 | 3.380 | 355.9 | 0.000e+00 | 0.000e+00 | ok |
| generator_seed | no_lgd/none/n32 | 1 | 0.2966 | 0.2966 | 3.371 | 361.9 | 0.000e+00 | 0.000e+00 | ok |
| frozen_params | no_lgd/none/n32 | 1 | 0.2867 | 0.2867 | 3.488 | 361.9 | 0.000e+00 | 0.000e+00 | ok |
| batched_mmd(+batched_lgd) | no_lgd/none/n32 | 1 | 0.1549 | 0.1549 | 6.454 | 361.9 | 3.648e-03 | 5.960e-07 | ok |
| batched_restarts | no_lgd/none/n32 | 1 | 0.1816 | 0.1816 | 5.508 | 362.0 | 3.648e-03 | 5.960e-07 | ok |
| batched_restarts | no_lgd/none/n32 | 2 | 0.2926 | 0.1463 | 6.835 | 363.8 | 3.767e-03 | 1.347e-05 | ok |
| batched_restarts | no_lgd/none/n32 | 4 | 0.2323 | 0.0581 | 17.221 | 363.8 | 1.447e-02 | 4.798e-05 | ok |
| batched_restarts | no_lgd/none/n32 | 8 | 0.3125 | 0.0391 | 25.603 | 363.8 | 1.447e-02 | 4.798e-05 | ok |
| batched_restarts+lean_ddim | no_lgd/none/n32 | 8 | 0.3089 | 0.0386 | 25.900 | 363.8 | 1.447e-02 | - | ok |
| batched_restarts | no_lgd/none/n32 | 16 | 0.6066 | 0.0379 | 26.375 | 369.8 | 1.447e-02 | 4.798e-05 | ok |
| batched_restarts | no_lgd/none/n32 | 32 | 1.0982 | 0.0343 | 29.138 | 444.0 | 1.439e-02 | 1.828e-04 | ok |
| float64_throughout | no_lgd/none/n32 | 1 | 0.3827 | 0.3827 | 2.613 | 444.2 | 2.183e-01 | - | changes_numerics(float64) |
| reference(_guided.run) | lgd/none/n8 | 1 | 0.7406 | 0.7406 | 1.350 | 445.4 | 0.000e+00 | 0.000e+00 | ok |
| generator_seed | lgd/none/n8 | 1 | 0.6860 | 0.6860 | 1.458 | 445.8 | 0.000e+00 | 0.000e+00 | ok |
| frozen_params | lgd/none/n8 | 1 | 0.9145 | 0.9145 | 1.094 | 447.1 | 0.000e+00 | 0.000e+00 | ok |
| batched_lgd | lgd/none/n8 | 1 | 0.5364 | 0.5364 | 1.864 | 447.2 | 0.000e+00 | 0.000e+00 | ok |
| batched_mmd(+batched_lgd) | lgd/none/n8 | 1 | 0.1655 | 0.1655 | 6.041 | 447.2 | 1.193e-01 | 1.013e-06 | ok |
| batched_restarts | lgd/none/n8 | 1 | 0.1606 | 0.1606 | 6.225 | 447.6 | 4.864e-02 | 1.013e-06 | ok |
| batched_restarts | lgd/none/n8 | 2 | 0.1737 | 0.0869 | 11.514 | 447.6 | 1.200e-01 | 9.429e-05 | ok |
| batched_restarts | lgd/none/n8 | 4 | 0.2174 | 0.0544 | 18.395 | 447.6 | 1.276e-01 | 9.429e-05 | ok |
| batched_restarts | lgd/none/n8 | 8 | 0.3079 | 0.0385 | 25.982 | 447.6 | 1.276e-01 | 9.429e-05 | ok |
| batched_restarts+lean_ddim | lgd/none/n8 | 8 | 0.2962 | 0.0370 | 27.010 | 447.6 | 1.276e-01 | - | ok |
| batched_restarts | lgd/none/n8 | 16 | 0.6668 | 0.0417 | 23.996 | 447.6 | 1.276e-01 | 9.429e-05 | ok |
| batched_restarts | lgd/none/n8 | 32 | 0.9529 | 0.0298 | 33.582 | 462.4 | 2.824e-01 | 4.244e-05 | ok |
| float64_throughout | lgd/none/n8 | 1 | 0.8317 | 0.8317 | 1.202 | 462.4 | 1.199e-01 | - | changes_numerics(float64) |
| reference(_guided.run) | no_lgd/adam/n8 | 1 | 0.1895 | 0.1895 | 5.278 | 462.4 | 0.000e+00 | 0.000e+00 | ok |
| generator_seed | no_lgd/adam/n8 | 1 | 0.1826 | 0.1826 | 5.477 | 462.4 | 0.000e+00 | 0.000e+00 | ok |
| frozen_params | no_lgd/adam/n8 | 1 | 0.2014 | 0.2014 | 4.964 | 462.4 | 0.000e+00 | 0.000e+00 | ok |
| batched_mmd(+batched_lgd) | no_lgd/adam/n8 | 1 | 0.1078 | 0.1078 | 9.278 | 462.4 | 7.153e-06 | 1.192e-06 | ok |
| batched_restarts | no_lgd/adam/n8 | 1 | 0.1119 | 0.1119 | 8.936 | 462.6 | 7.153e-06 | 1.192e-06 | ok |
| batched_restarts | no_lgd/adam/n8 | 2 | 0.1329 | 0.0665 | 15.046 | 462.6 | 1.120e+00 | 6.348e-06 | ok |
| batched_restarts | no_lgd/adam/n8 | 4 | 0.1528 | 0.0382 | 26.175 | 462.6 | 1.675e+00 | 3.111e-05 | ok |
| batched_restarts | no_lgd/adam/n8 | 8 | 0.1857 | 0.0232 | 43.084 | 462.6 | 1.675e+00 | 3.111e-05 | ok |
| batched_restarts+lean_ddim | no_lgd/adam/n8 | 8 | 0.1767 | 0.0221 | 45.282 | 462.6 | 1.675e+00 | - | ok |
| batched_restarts | no_lgd/adam/n8 | 16 | 0.2267 | 0.0142 | 70.592 | 462.6 | 1.675e+00 | 3.111e-05 | ok |
| batched_restarts | no_lgd/adam/n8 | 32 | 0.3522 | 0.0110 | 90.860 | 462.6 | 2.175e+00 | 3.111e-05 | ok |
| float64_throughout | no_lgd/adam/n8 | 1 | 0.2969 | 0.2969 | 3.368 | 462.6 | 1.124e+00 | - | changes_numerics(float64) |
| torch.compile[default] | no_lgd/none/n8 | 1 | 0.2432 | 0.2432 | 4.111 | 626.7 | 1.097e-05 | - | ok |
| batched_restarts[mps] | no_lgd/none/n8 | 1 | 1.0065 | 1.0065 | 0.994 | 702.2 | 4.768e-07 | - | ok(mps,float32) |
| batched_restarts[mps] | no_lgd/none/n8 | 8 | 1.2810 | 0.1601 | 6.245 | 733.1 | 1.090e+01 | - | ok(mps,float32) |
| batched_restarts[mps] | no_lgd/none/n8 | 32 | 1.3098 | 0.0409 | 24.431 | 760.3 | 5.801e+00 | - | ok(mps,float32) |

`torch.compile[default]`: first call including compilation **16.0 s**; steady state
0.243 s vs 0.203 s reference (no gain: the per-step work is ~1-2 ms of tiny kernels +
autograd, inductor CPU kernels do not help; `dynamic=True` was needed because the batch
size changes between the 1-row denoiser and the n-row CM). End-to-end diff 1.1e-5
(REORDER). RSS +270 MB for the compiler.

MPS (float32 only, no float64 on MPS): 1.0 s per run at B=1 — **5x slower than CPU** —
launch/sync-bound (each step is ~40 tiny kernels and an `autograd.grad`), reaching 24
restarts/s at B=32 vs 70-90 on CPU. End-to-end diffs are O(1-10) because MPS float32
kernels round differently and the trajectory is chaotic; per-step agreement was not
measured on MPS. Not worth pursuing for this problem size.

## Second restart-batching scan (`bench_scaling_quiet.log`; machine still loaded, load avg 4-15)

```
load avg (4.44580078125, 10.806640625, 14.9423828125) threads 4
cell               ref_1restart B=1   B=2   B=4   B=8   B=16  B=32  B=64    (restarts/s; lean_ddim=True, batched_lgd+mmd)
no_lgd/none/n8             1.65   5.0   5.9   9.8  14.5  24.2  33.7  32.3
no_lgd/none/n32            1.65   4.7   5.9   6.2   9.5  10.0  14.2  19.9
lgd/none/n8                1.34   6.1   7.4   9.7  12.3   9.5  13.2  14.9
no_lgd/adam/n8             1.49   4.0   4.9   8.5  13.0  19.3  23.8  19.8
no_lgd/none/n256           0.39   0.6   0.7   1.2   1.9   1.3   1.6   1.5
```

## Reading the numbers (speedup = variant throughput / reference throughput *in the same run*)

| change | exactness | no_lgd/none/n8 | no_lgd/none/n32 | lgd/none/n8 | no_lgd/adam/n8 |
|---|---|---|---|---|---|
| (0) `torch.Generator` instead of global `manual_seed` (99 us/call) | EXACT (0.0) | ~1.0x (noise) | 1.0x | 1.08x | 1.04x |
| (i) `requires_grad_(False)` on both models | EXACT (0.0) | 1.0x (noise) | 1.03x | 0.8x (noise) | 0.94x (noise) |
| (ii) batched LGD (3n rows, same noise) | EXACT (0.0) | n/a | n/a | **1.4x** (0.74->0.54 s; 0.53->0.40 in the interleaved A/B) | n/a |
| (iii) batched MMD + cached YY (+ii) | REORDER, per-step <= 1.3e-6 | **1.5x** | **1.9x** | **4.5x** (0.74->0.17 s) | **1.8x** |
| (iv) batched restarts, B=8 | REORDER, per-step <= 2.2e-4 | 7.5x | 7.6x | 19x | 8.2x |
| (iv) batched restarts, B=32 | REORDER | **14x** (70 r/s) | 8.6x (29 r/s; saturates, 1024 CM rows x 6 steps + 32x(32x250) kernels on 4 threads) | 25x (34 r/s) | **17x** (91 r/s) |
| (iv)+lean DDIM step (no per-call `.to(device)`) | EXACT relative to (iv) | +2% | +1% | +4% | +5% |
| (v) `torch.compile` | REORDER | 0.84x (+16 s compile) | - | - | - |
| (vi) float64 throughout | NUMERICS | 0.48x | 0.77x | 0.89x | 0.64x |
| (vii) MPS batched restarts | NUMERICS (device kernels) | 0.2x at B=1, 5x at B=32 | - | - | - |

Where the time goes in one no-LGD n=8 step (~2 ms; micro-timings in AUDIT.md §1.3):
DDIM step 0.27 ms (0.075 of it the per-call `self.to(device)`), `manual_seed` 0.10,
CM 6-step ladder 0.50, MMD forward 0.46 (94% of the kernel entries are the constant
target-target block), backward ~0.6. This is why the MMD restructuring (iii) is the
single largest 1-restart win and why batching restarts (iv) — which amortises the
fixed per-step Python/autograd overhead over B rows — is the big lever: at B=32 the
per-restart cost falls from 200 ms to 11-14 ms.

## Recommendation for the experiment scripts (all call-site changes, no engine edits)

1. Run the restarts of one `(setting, arm, n)` cell as one batch through
   `run_batched_restarts` (B = all restarts of the cell, e.g. 20-50): 10-25x on CPU.
   Equivalence to report: per-step teacher-forced gradient agreement (<= 1e-4 abs) and
   the `L2`/`success_rate` distribution over restarts (paired by restart index) — NOT
   a per-restart `x_final` match, which the chaos makes impossible for any REORDER change.
2. Replace `MMDLoss` on the stacked matrix by XX/XY blocks with cached YY (1.5-4.5x alone;
   Agent 2's exact-loss work overlaps here — coordinate).
3. Batch the three LGD perturbations (EXACT, 1.4x on LGD cells) and pre-draw the
   conditional noise from a `torch.Generator` (EXACT).
4. Keep `requires_grad_(False)` and the lean DDIM step as hygiene (EXACT, ~0-5%).
5. Do not bother with `torch.compile` or MPS at this problem size.

## Cluster re-run (heavy compute off the Mac)

`submit_bench.sh` (glacier, CPU, 8 threads; 2D/5D/10D full factor scan incl. `torch.compile`,
no MPS) and `submit_bench_gpu.sh` (catfish L4; CPU factors + `batched_restarts[cuda]` at
B=1,8,32,128 — the only runner that can gain from a GPU). Both expect the repo synced to
`/sci/labs/orzuk/shaulytolk/cdm-perf/` and write CSVs + logs to
`/sci/labs/orzuk/shaulytolk/cdm-perf/logs/`. The local numbers above are from a shared
4-core Mac and should be superseded by those runs. Submit (not done by Agent 5):

    ssh -p 2222 shaulytolk@localhost "bash -lc 'mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs && sbatch /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/model-optimization/systems/submit_bench.sh'"
    ssh -p 2222 shaulytolk@localhost "bash -lc 'sbatch /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/model-optimization/systems/submit_bench_gpu.sh'"
