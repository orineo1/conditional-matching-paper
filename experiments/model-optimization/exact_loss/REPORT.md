# Agent 2 -- exact MMD accelerations (REPORT)

Files (all under `experiments/model-optimization/exact_loss/`):

| file | what |
|---|---|
| `fast_mmd.py` | `MMDFixedTarget` + helpers (`repo_scales`, `sd_scale`, `sd_median_bandwidth`, `sq_dists`, `kernel_from_d2`, `chunked_kernel_sum`, `mmd_reference_like`) |
| `test_fast_mmd.py` | 236 pytest cases (value + dL/dX vs `LossFunctions.MMDLoss`, gradcheck, SD `compute_mmd`, edge cases). `cd simulations && python -m pytest ../experiments/model-optimization/exact_loss -q` -> **236 passed** |
| `bench_mmd.py` | micro-benchmark driver (resumable, `bench_raw/*.json`), `--grid small|quick|full`, `--device cpu|cuda` |
| `bench_results_small.csv`, `bench_summary_small.md` | local Mac validation grid (n in {1,8,32}, m=250, d in {2,768}, f32/f64) |
| `submit_bench.sh`, `submit_bench_gpu.sh` | cluster sbatch (glacier CPU full grid; catfish L4 CUDA full grid). **Not submitted.** |
| `bench_compile.py`, `bench_compile_cpu.md` | torch.compile: compile time vs steady state |
| `end_to_end_check.py`, `end_to_end_results.csv` | `_guided.run` (Exp 2-7 loop) with the reference MMD vs fast variant, by monkeypatching `_guided.MMDLoss` (no edit to `simulations/`) |
| `../hypotheses/agent2.yaml` | hypotheses |

## 1. What the reference does, and what is exact

Reference `simulations/src/LossFunctions.py`: `K = sum_{k=0..4} exp(-D / (bw * 2^(k-2)))` on
`D = cdist(Z,Z)^2`, `Z = vstack(X,Y)`; biased V-statistic `XX.mean - 2 XY.mean + YY.mean`.
`bw` is fixed (synthetic: `_common.fixed_bandwidth(S_G)`) or, with `bandwidth=None`,
`D.sum()/(N^2-N)` on the stacked matrix, **not detached** (gradient flows through bw).

Facts established by reading + tests:

* The reference **does** materialise the `(5, n+m, n+m)` tensor (`L2[None]/(bw*mults[:,None,None])`
  then `exp(...).sum(0)`), and keeps the exp output for backward. For n=8, m=250 that is
  5 x 258^2 = 333k entries, of which 312.5k (94%) are the constant Y-Y block -- recomputed
  and back-propagated (to nothing) on every call (Agent 1's accounting agrees).
* `torch.cdist` switches to the `||x||^2+||y||^2-2xy` matmul formula for > 25 rows, so the
  reference at m=250 is already matmul-based; its diagonal is therefore not exactly 0
  (rounding ~1e-15), which is irrelevant unless bw < ~1e-3 (see edge cases).
* Dtype quirk: `RBF.bandwidth_multipliers` are built in the **default dtype**; a Python-float
  or 0-dim bandwidth is promoted to that dtype, so under default float32 the reference
  rounds `bw*mult` to float32 even for float64 inputs. `fast_mmd.repo_scales` reproduces
  this on purpose; tests run under default float64 (like `simulations/tests/conftest.py`).
  The synthetic experiments actually run the models in float32 (`S_G = ...float()`).
* SD pipeline (`SD_cond_SD_controlnet/src/metrics.py::compute_mmd`, used by
  `scripts/run_mlgd_f.py` via `run_dps_step_clip`): **unbiased U-statistic**, single
  generalised kernel `exp(-(D/(2 bw^2))^alpha)`, bandwidth = detached median heuristic on
  `X[:1000]` vs `Y[:1000]` each call times `bandwidth_scale`, output `sqrt(|MMD^2|+1e-8)`,
  float32; `zeta_i = base_zeta / loss.detach()` (adaptive zeta), grad of `loss*loss_scale`.
  Targets (CLIP-768, ~100-120) are fixed. Covered by `MMDFixedTarget(kernel="sd",
  alpha, unbiased=True, sd_output=True)` + `sd_median_bandwidth`; tested against
  `compute_mmd` at float32 tolerance (1e-5 value, 1e-4 grad).
* MNIST (`MNIST/run_mlgdf.py`) uses **sliced Wasserstein** (50 random projections), not MMD.

Variants in `fast_mmd.MMDFixedTarget` (all opt-in kwargs), and equivalence status:

| variant | mechanism | exact? | tested |
|---|---|---|---|
| fixed-target cache (default) | cache Y, `||y||^2`, `D_yy`, scalar YY term; compute XX, XY only | yes | value+grad 1e-12 vs reference, fixed + adaptive bw |
| `dist="mm"` | norms + matmul, clamp >= 0 | yes | same |
| `kernel_eval="powchain"` | `E=exp(-D/(4bw))`, `K=E+E^2+E^4+E^8+E^16` (alpha=1, integer mul_factor) | yes (few ulp) | same + elementwise 1e-14 |
| `kernel_eval="loop"` | accumulate kernel-by-kernel, no (5,n,m) tensor | yes | same |
| `chunk=c` | blockwise XY with fused `autograd.Function` (stores only (n,d)+(K,)) | yes, 1st order | same + gradcheck (X and scales) |
| `batched(Xb)` | B sample sets vs one Y in one call, (B,) losses | yes | == B reference calls, 1e-12 |
| `bandwidth=None` | stacked adaptive rule incl. grad through bw; YY(bw) re-attached by first-order-exact `f(bw0)+f'(bw0)(bw-bw0)` (or `reattach_yy="autograd"`) | yes, 1st order | 1e-12 vs reference adaptive |
| `kernel="sd"`, `alpha`, `unbiased`, `sd_output` | SD conventions | yes | vs naive double loop 1e-12 and vs `compute_mmd` at f32 tol |

Grid tested: n in {1,2,8,32,100} x m in {5,250} x d in {1,2,8,768}, fixed/adaptive, 7
variants, plus finite-difference gradchecks and edge cases:

* **identical X rows / X row equal to a Y row**: finite, matches reference to 1e-12.
* **n = 1**: matches (V-stat); U-stat defines xx term = 0 as SD does.
* **bw = 1e-12**: no NaN/inf in reference or variants; off-diagonal kernels underflow to 0,
  gradient is exactly 0 everywhere. BUT the value is rounding-noise dependent in the
  reference itself (its cdist diagonal ~1e-15 divided by 2.5e-13 is O(1e-2) in the exp):
  reference gives 0.95831 vs the ideal 5/6+5/40 = 0.95833; variants agree with it only to
  ~1e-3 there. Documented deviation; irrelevant for any bw >~ 1e-3*||x||^2 (the repo
  bandwidth is the mean squared distance, O(1)); at bw=1e-2 the 1e-12 equivalence holds.
  bw = 0 exactly gives 0/0 = NaN in the reference; not reproduced.
* Under default float32 the reference and fast_mmd still agree to 1e-12 (both round bw).

## 2. Speedups (local Mac, M4, 4 threads, forward+backward, median of 20-50)

From `bench_summary_small.md` (speedup = reference / variant, geometric mean over the
small grid):

| variant | float32 | float64 | n<=8,m=250 f64 | d=768 f64 |
|---|---|---|---|---|
| stacked_mm (control: reference algorithm, mm distances) | 0.98 | 1.02 | 1.02 | 1.08 |
| stacked_powchain (control: reference algorithm, powchain) | 0.93 | 1.17 | 1.14 | 1.12 |
| fixed_cdist | 4.7 | 7.3 | 9.3 | 6.6 |
| **fixed_mm** | 6.3 | 10.2 | 13.0 | 12.9 |
| fixed_mm_loop | 4.8 | 7.4 | 8.8 | 9.7 |
| fixed_mm_powchain | 6.0 | 9.6 | 11.0 | 11.9 |
| fixed_mm_chunked256 | 6.1 | 9.5 | 11.9 | 11.8 |
| fixed_mm_adaptive (vs reference_adaptive) | 1.9 | 1.9 | 2.0 | 2.5 |
| batched_fixed_mm_B3 (vs 3 reference calls) | 12.6 | 17.9 | 23.9 | 20.1 |

Absolute: n=8, m=250, d=2, float32: reference 0.78 ms -> fixed_mm 0.13 ms; n=32, m=250,
d=768, float64: 3.55 ms -> 0.44 ms. RSS growth over the grid: reference 57 MB, fixed_mm
35 MB (the (5,N,N) stacked tensor + its saved exp vs (5,n,m)).

Reading: essentially all of the gain is **not recomputing the Y-Y block** (94% of kernel
entries at n=8). The distance formula and the powchain trick are second-order (+/-20%,
within noise at these sizes; powchain helps more in the float64 end-to-end run below). The
adaptive-bandwidth case gains only ~2x because YY(bw) must still be re-evaluated each call
(m x m exps, though without autograd). Chunking is speed-neutral; it only buys memory.
Batched B=3 is ~2x better than 3 separate fixed_mm calls (dispatch overhead).

Full grid (n up to 100, m up to 2000, d=1..768, CUDA; produces the unsuffixed
`bench_results.csv` / `bench_summary.md`) is left to the cluster scripts
(`submit_bench.sh` glacier, `submit_bench_gpu.sh` catfish L4); by construction the
speedup is ~ (n+m)^2 / (n^2 + n m), i.e. ~4x at n=100, m=250 and ~30x at n=8, m=250,
shrinking toward 1 only when n ~ m.

## 3. Does MMD time matter? (honest accounting)

**Synthetic loop (Exp 2-7, `_guided.run`)**: yes, it matters -- the conditional model is a
tiny MLP, so the stacked (258 x 258 x 5) kernel is a large share. Agent 1's profile:
mmd forward 17-22% of step time, backward 42-46% (mostly the kernel block). End-to-end
(`end_to_end_check.py`, 2D, 99 steps, restart 0; times taken while another benchmark was
running so absolute ms are inflated, ratios are paired):

| arm | n | reference | control stacked_mm | fixed_mm | fixed_mm_powchain |
|---|---|---|---|---|---|
| no_lgd | 8 | 340 ms | 0.97x | **1.66x** | **1.77x** |
| no_lgd | 32 | 351 ms | 1.12x | 1.84x | 2.02x |
| lgd (M=3) | 8 | 745 ms | 0.97x | 1.93x | 1.92x |
| lgd (M=3) | 32 | 877 ms | 0.98x | 1.80x | 1.97x |

So the cached-target MMD roughly **halves the wall time of the whole synthetic loop**
with identical conditional-model call counts (the hardware-independent cost is unchanged:
this is pure overhead removal).
Trajectory agreement: the loop runs in float32 and is chaotic w.r.t. rounding -- the
*control* (reference algorithm with only the distance formula changed) already moves the
final x_hat by up to 0.14; the fast variants move it by the same order (1e-5 .. 0.19).
This is float32 sensitivity of a 99-step guided sampler, not a bias; in float64 the losses
and gradients agree to 1e-12 per step (tests). Exp 2-7 score over restarts, so this is
noise of the same kind as a BLAS/threading change, but bit-for-bit reproduction of old
runs should not be expected after switching.

**SD pipeline**: the MMD is on n ~ 8-32 x 768 vs m ~ 100-120 x 768 in float32 on GPU.
Measured here on CPU: reference ~1.6 ms (n=32, m=250, d=768 f32) vs 0.3 ms; on an L4 both
are << 1 ms. One DPS step runs n sprinter SDXL-Turbo+ControlNet passes (2 steps each) +
CLIP + a backward through all of it -- O(seconds). The MMD is < 0.1% of a step; the
equivalent SD version (`kernel="sd"`, unbiased, sqrt) is exact but **not worth
integrating for speed**; its only practical value is the shared, tested implementation.
(Also: `compute_mmd` re-estimates the median bandwidth every call from the current X, so
caching would only cover `||y||^2`/`D_yy`; the YY kernel depends on the per-call bw.)

**MNIST**: SWD, not affected.

## 4. torch.compile

`torch.compile(lambda X: f(X), dynamic=False)` on `MMDFixedTarget(dist="mm",
kernel_eval="powchain")`, CPU inductor, one compile per fresh process (`bench_compile.py`;
inductor's compile pool hung intermittently on this Mac -- 9 hangs in 12 attempts -- so
the run was stopped after 5 cells to keep compute off the laptop; `bench_compile_cpu.md`):

| dtype | n | m | d | compile (first call) | compiled | eager powchain | reference | x vs eager | x vs ref | break-even calls | max err vs eager |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float32 | 1 | 250 | 2 | 4.5 s | 0.067 ms | 0.168 ms | 0.973 ms | 2.5 | 14 | 44,555 | 4.8e-07 |
| float32 | 8 | 250 | 2 | 5.1 s | 0.129 ms | 0.271 ms | 1.707 ms | 2.1 | 13 | 36,158 | 9.5e-07 |
| float32 | 8 | 250 | 10 | 5.8 s | 0.123 ms | 0.261 ms | 1.756 ms | 2.1 | 14 | 41,846 | 4.8e-07 |
| float64 | 1 | 250 | 2 | 6.6 s | 0.149 ms | 0.263 ms | 3.658 ms | 1.8 | 25 | 57,783 | 0 |
| float64 | 8 | 250 | 2 | 2.2 s | 0.218 ms | 0.303 ms | 3.540 ms | 1.4 | 16 | 26,134 | 6.9e-17 |

Steady state is a further 1.4-2.5x over eager (fusing the elementwise kernel stage), exact
(float64 err <= 1e-16), but compile costs 2-7 s **per shape** (`dynamic=False`; `n_t`
varies per step under the `n_for_step` schedules, and float32/float64 are separate
shapes) and break-even is 26k-58k calls -- an Exp 2-7 restart makes 99-297 MMD calls,
an experiment a few thousand. Verdict: not worth it for the synthetic loop unless n is
constant and many restarts share one process; the hang-prone macOS inductor makes it a
poor default anyway. Report separately: compile overhead 2-7 s vs 0.1-0.3 ms steady.

## 5. Recommendation

* **Integrate `MMDFixedTarget(Y=S_G, bandwidth=fixed, dist="mm")` (optionally
  `kernel_eval="powchain"`) into the synthetic loop** -- exact, ~10x on the loss
  micro-benchmark, ~1.7-2.0x on the whole Exp 2-7 loop, fewer allocations, no change in
  conditional-model calls. Minimal patch: in `_guided.run`, replace
  `mmd = MMDLoss(kernel=RBF(bandwidth=bandwidth))` by a fixed-target object built once from
  `S_G` (the `FastMMDLossShim` in `end_to_end_check.py` shows the drop-in shape:
  `mmd(y, S_G)` signature preserved). Expect float32 trajectory differences of the
  rounding kind (see Section 3); re-validate Exp 2-7 scores once after switching.
* `batched()` is useful only if the M=3 LGD sampler outputs are concatenated before the
  loss (Agent 4/5 territory); the loss-side gain is ~2x of an already small cost.
* Do not bother for the SD pipeline (negligible share); do not use `chunk` unless
  m >> 2000 on a memory-limited GPU; treat torch.compile as optional (see Section 4).
* Cluster runs to finalise the numbers (not submitted, per instruction):
  ```
  ssh -p 2222 shaulytolk@localhost "bash -lc 'cd /sci/labs/orzuk/shaulytolk/cdm-perf && sbatch experiments/model-optimization/exact_loss/submit_bench.sh'"
  ssh -p 2222 shaulytolk@localhost "bash -lc 'cd /sci/labs/orzuk/shaulytolk/cdm-perf && sbatch experiments/model-optimization/exact_loss/submit_bench_gpu.sh'"
  # then: python experiments/model-optimization/exact_loss/bench_mmd.py --aggregate --device cpu,cuda
  ```
  (both expect the repo checked out at `/sci/labs/orzuk/shaulytolk/cdm-perf`, logs in
  `/sci/labs/orzuk/shaulytolk/cdm-perf/logs/`; the GPU script also runs the merged
  aggregation; `bench_compile.py cuda` can be appended to the GPU script if wanted).
