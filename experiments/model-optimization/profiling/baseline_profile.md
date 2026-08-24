# Baseline profile of the CDM / distributional-TFG synthetic loop (Agent 1)

Commit `6af2081`, branch `tfg-generalization-v2`. Python 3.13.11, torch 2.12.0 (CPU,
4 intra-op threads), macOS-26.0.1-arm64 (Apple M4, 10 cores, 16 GB).
Loop profiled: `simulations/experiments/_guided.py::run` (the code Exp 2-7 use).
Models and target set are **float32** (checkpoints + `_common.target_set` use `.float()`);
only `evaluate` and the `tfg` schedule are float64.

Files: `run_baseline.py` (baseline reproduction, writes `baseline_rows.csv`,
`baseline_runs.json`), `profile_guided.py` (hierarchical perf_counter profile by
monkeypatching call sites in-process, writes `profile_buckets.{md,json}`),
`torch_profile.py` (operator-level `torch.profiler`, writes `torch_profile_*.txt`),
`profile.json` (machine-readable summary of everything below).

## 1. Baseline reproduction

Arms `no_lgd/none`, `no_lgd/adam`, `lgd/none`; `n in {8, 32}`; restarts 0..4; each
(setting, arm, n) cell in a fresh subprocess (so peak RSS is per cell); per restart one
warm-up run then 5 timed repeats. The 2D numbers reproduce
`results/tfg/exp2_lgd_vs_adam_n8_canonical.json` restart-by-restart (e.g. restart 0
`no_lgd/none`: L2 = 0.2182, x_hat = -5.447157 in both). All repeats are bit-identical
(`deterministic_repeats = True`).

| setting | arm | n | wall s median [min,max] (25 runs) | peak RSS MB | cond samples/run | exact L2 restarts 0..4 | mean L2 | final MMD^2 (r0) |
|---|---|---|---|---|---|---|---|---|
| 2D | no_lgd/none | 8 | 0.188 [0.184,0.194] | 328 | 792 | 0.218, 0.314, 0.044, 0.181, 1.096 | 0.371 | 0.2074 |
| 2D | no_lgd/none | 32 | 0.215 [0.209,0.233] | 340 | 3168 | 0.231, 0.255, 0.262, 0.346, 0.138 | 0.247 | 0.0680 |
| 2D | no_lgd/adam | 8 | 0.189 [0.186,0.196] | 327 | 792 | 0.077, 0.327, 0.786, 0.783, 0.207 | 0.436 | 0.1790 |
| 2D | no_lgd/adam | 32 | 0.216 [0.211,0.238] | 331 | 3168 | 0.314, 0.252, 0.433, 0.409, 0.155 | 0.313 | 0.0679 |
| 2D | lgd/none | 8 | 0.512 [0.505,0.541] | 352 | 2376 | 0.077, 0.218, 0.819, 0.053, 0.030 | 0.240 | 0.1539 |
| 2D | lgd/none | 32 | 0.589 [0.582,0.619] | 352 | 9504 | 0.187, 0.236, 0.277, 0.236, 0.250 | 0.237 | 0.0676 |
| 5D | no_lgd/none | 8 | 0.183 [0.181,0.194] | 331 | 792 | 0.411, 1.087, 1.078, 0.399, 1.077 | 0.810 | 0.3589 |
| 5D | no_lgd/none | 32 | 0.212 [0.209,0.215] | 327 | 3168 | 0.412, 0.418, 0.471, 0.417, 0.442 | 0.432 | 0.2914 |
| 5D | no_lgd/adam | 8 | 0.185 [0.184,0.195] | 322 | 792 | 0.396, 0.568, 0.384, 0.399, 0.412 | 0.432 | 0.3566 |
| 5D | no_lgd/adam | 32 | 0.216 [0.213,0.222] | 330 | 3168 | 0.405, 0.281, 0.622, 0.420, 0.454 | 0.437 | 0.2431 |
| 5D | lgd/none | 8 | 0.502 [0.495,0.540] | 344 | 2376 | 0.399, 0.412, 0.444, 0.419, 0.413 | 0.417 | 0.2996 |
| 5D | lgd/none | 32 | 0.606 [0.587,0.718] | 367 | 9504 | 0.429, 0.425, 0.422, 0.423, 0.427 | 0.425 | 0.2570 |
| 10D | no_lgd/none | 8 | 0.183 [0.180,0.198] | 333 | 792 | 0.527, 0.581, 0.450, 0.536, 0.730 | 0.565 | 0.3381 |
| 10D | no_lgd/none | 32 | 0.216 [0.210,0.239] | 348 | 3168 | 0.597, 0.571, 0.403, 0.443, 0.534 | 0.510 | 0.0533 |
| 10D | no_lgd/adam | 8 | 0.186 [0.184,0.191] | 327 | 792 | 0.558, 0.910, 0.921, 0.505, 0.682 | 0.715 | 0.2443 |
| 10D | no_lgd/adam | 32 | 0.216 [0.210,0.279] | 330 | 3168 | 0.765, 0.399, 0.729, 0.438, 0.359 | 0.538 | 0.0310 |
| 10D | lgd/none | 8 | 0.502 [0.488,0.584] | 356 | 2376 | 0.416, 0.323, 0.441, 0.438, 0.399 | 0.404 | 0.2962 |
| 10D | lgd/none | 32 | 0.597 [0.583,0.807] | 359 | 9504 | 0.451, 0.293, 0.461, 0.412, 0.420 | 0.407 | 0.0509 |

"final MMD^2" = one fresh n-sample MMD^2 of the CM conditional at the returned x_hat
against S_G (diagnostic; the loop itself does not log a loss). RSS after model load
(before any run) is 293 MB, so the per-run increment is 30-75 MB.

Observations

* Wall time is almost flat in `n` (n=8 -> n=32 is +14%) and in dimension
  (2D = 5D = 10D within noise): the loop is **per-step overhead bound**
  (~1.9-2.2 ms per diffusion step, ~100 small aten ops + autograd), not FLOP bound.
* LGD costs 2.7x (not 3x) because the unconditional denoiser and per-step bookkeeping are
  shared; its three perturbations are three *separate* sampler + MMD + graph
  constructions, never batched.
* These timings were taken in a quiet window (load average < 3). Later profiling runs
  (section 2) ran while other agents loaded the machine (load average 20-30 on 10
  cores) and are 1-4x slower in absolute terms; their *fractions* are stable.

## 2. Hierarchical profile (`profile_guided.py`)

Per-bucket wall (medians over 5 restarts x 5 repeats) as a fraction of `run` wall.
Buckets are top-level exclusive except the indented sub-buckets, which are contained in
the one above. `other` = total - (ddim + cond_sample + mmd + backward + adam).
Absolute numbers below are from the contended run; the first row gives the quiet-window
wall for scale.

| cell | quiet wall ms/step | ddim (uncond fwd) | cond_sample (cond fwd) | mmd (kernel: cdist / exp) | backward | adam | other |
|---|---|---|---|---|---|---|---|
| 2D no_lgd/none n=8 | 1.90 | 10.2% (5.4%) | 18.9% (17.4%) | 19.3% (16.8%: 2.7% / 5.9%) | **41.8%** | - | 9.3% |
| 2D no_lgd/none n=32 | 2.17 | 9.6% (5.1%) | 20.3% (18.9%) | 18.7% (16.6%: 2.6% / 5.2%) | **43.1%** | - | 8.9% |
| 2D lgd/none n=8 | 5.17 | 4.4% (2.3%) | 20.9% (19.3%) | 20.6% (17.8%: 3.0% / 5.9%) | **43.7%** | - | 10.2% |
| 2D lgd/none n=32 | 5.95 | 3.6% (1.8%) | 19.7% (18.3%) | 22.3% (19.8%: 3.2% / 6.3%) | **45.7%** | - | 8.7% |
| 2D no_lgd/adam n=32 | 2.18 | 9.1% (4.8%) | 16.5% (15.2%) | 19.3% (16.3%: 2.4% / 4.3%) | **44.3%** | 0.8% | 8.9% |
| 10D no_lgd/none n=32 | 2.18 | 9.5% (4.9%) | 18.9% (17.6%) | 19.5% (16.8%: 2.6% / 5.3%) | **43.0%** | - | 9.2% |
| 10D lgd/none n=32 | 6.03 | 3.9% (2.0%) | 21.5% (20.1%) | 21.1% (18.7%: 2.9% / 5.9%) | **43.8%** | - | 9.3% |

An earlier (quieter, but double-wrapped for the later cells) run of the same script
gave for 2D no_lgd/none n=8 (total 189.7 ms): ddim 11.1%, cond_sample 20.9%, mmd 16.7%,
backward 41.4%, other 10.0% -- the same picture. Full per-run data: `profile_buckets.json`.

### Call accounting (per run, T=100 -> 99 guided steps, m = 250 targets, 5 bandwidths)

| quantity | no_lgd n=8 | no_lgd n=32 | lgd n=8 | lgd n=32 |
|---|---|---|---|---|
| denoiser (uncond) forward calls | 99 | 99 | 99 | 99 |
| conditional sampler calls (`model_cond.sample`) | 99 | 99 | 297 | 297 |
| conditional network forwards (5 per sampler call: ladder [150,50,20,10,5,1] has 6 levels, 5 transitions) | 495 | 495 | 1485 | 1485 |
| conditional samples (`conditional_calls`) | 792 | 3168 | 2376 | 9504 |
| target samples | 250 (fixed, drawn once) | 250 | 250 | 250 |
| MMD evaluations | 99 | 99 | 297 | 297 |
| kernel entries per MMD, (n+m)^2 x 5 | 332,820 | 397,620 | 332,820 | 397,620 |
| kernel entries per run | 32.9 M | 39.4 M | 98.8 M | 118.1 M |
| target-target entries per MMD, m^2 x 5 | 312,500 | 312,500 | 312,500 | 312,500 |
| target-target share of kernel | **93.9%** | **78.6%** | 93.9% | 78.6% |
| times the target-target block is recomputed per run | 99 | 99 | 297 | 297 |
| backward passes (`torch.autograd.grad`) | 99 | 99 | 99 | 99 |
| Adam steps (adam arms) | 99 | 99 | 99 | 99 |

The target-target block is constant (S_G fixed, bandwidth frozen, detached), so every
recomputation is wasted in **both** the forward (cdist on the stacked 258x258 or
282x282 matrix, exp over 5 bandwidths) and the backward (EuclideanDistBackward /
ExpBackward over the full stacked matrix even though only the n X-rows need gradients).

Micro-benchmark (2D, m=250, bandwidth 50.198, single MMD forward+backward, median of
200): stacked repo form 1.22 ms (n=8) / 1.51 ms (n=32); computing K_XX, K_XY only with
YY cached: 0.18 ms / 0.40 ms. Max |difference| 7e-7 in float32 (summation order only).
That is ~1.0-1.1 ms of the ~1.9-2.2 ms per step => ~45-50% of no-LGD wall, and about
3.1-3.3 ms of the ~5.2-6.0 ms per LGD step.

### Operator-level view (`torch_profile_*.txt`)

2D no_lgd n=8 (one run, CPU): top self-time ops are `aten::mul` (8.5%), `aten::div`
(7.9%), `aten::exp` (7.1%), `aten::addmm` (6.3%), `aten::sum` (6.0%), `aten::neg`
(5.1%), `aten::fill_`, `aten::copy_`, `aten::mm`, then `AddmmBackward0`. The
`div`/`neg`/`exp`/`mul`/`sum` entries are dominated by the kernel (the 5 x 258 x 258
broadcasts `L2 / (bw * mult)`, `-scaled`, `exp`, `.sum(0)`): they allocate
300 MB / 251 MB / 126 MB / 178 MB cumulative over a run (CPU Mem column), i.e. the
kernel intermediates are the main per-step memory. `addmm`/`mm` (the MLPs, 4554 calls)
are only ~10% combined. Memory ranking by self CPU mem: `aten::div` > `aten::neg` >
`aten::mul` > `aten::exp` > `aten::empty` > `aten::mm`.

## 3. Bottleneck ranking (synthetic task)

1. **Backward pass (42-46% of wall).** One `torch.autograd.grad` per step through:
   5 CM network evals x n samples, the RBF kernel on the stacked (n+m) matrix (5
   bandwidths), and the unconditional denoiser. The kernel part of the backward scales
   with (n+m)^2 although the gradient only needs the n x (n+m) rows.
2. **MMD forward (17-22%)**, 79-94% of it the constant target-target block recomputed
   99-297 times per run; `exp` is the single most expensive op inside it, `cdist` is
   small.
3. **Conditional CM forward (15-21%)**: 495 (no-LGD) / 1485 (LGD) MLP evaluations per
   run, cost flat in n (framework overhead per call, tiny matmuls). LGD triples the
   number of calls instead of batching the 3 perturbations into one (3n) call.
4. Unconditional denoiser 4-10% (includes `self.to(device)` and `t_batch` plumbing every
   call); Python/other 8-11%; Adam < 1%.

Memory: not a bottleneck (peak RSS 320-370 MB, ~293 MB is import + models). The per-run
increment is the 5 x (n+m)^2 float32 kernel intermediates kept for backward
(1.3-1.6 MB per MMD eval) plus autograd bookkeeping.

Static redundancies visible in `_guided.py` / `LossFunctions.py` / `Diffusion.py` /
`ConsistencyModels.py`:

* Target-target kernel block recomputed every MMD (above).
* Kernel computed on the stacked `(X;Y)` matrix: `cdist` and backward touch m x n +
  m x m entries that are never needed for the gradient.
* Model parameters have `requires_grad=True` (both models): autograd records
  parameter-gradient edges and saves inputs for weight grads on every Linear; the
  gradient is never taken w.r.t. them. (`torch.autograd.grad(loss, x)` prunes the
  weight branches at run time, but the graph is still built; cost is the saved
  activations and graph nodes, not the matmuls.)
* `sample_ddim_step` calls `self.to(device)` and re-reads `next(self.parameters()).dtype`
  every step; `DiffusionModel.forward` / CM `forward` rebuild the time embedding
  (`torch.exp(arange*...)`) on every call although `t` takes only 100 (resp. 5) values.
* Under LGD the three perturbation sets are sampled, kernelised and graphed separately
  (`for j, n_j in enumerate(alloc)`): 3 sampler calls of n instead of one call of 3n,
  and 3 separate kernel matrices of size (n+m)^2 instead of one (3n+m)^2 or three
  n x m blocks.
* `torch.manual_seed(key_seed(...))` (global RNG reseed) once per perturbation per step.
* Bandwidth multipliers `mul_factor ** (arange - 2)` are a tensor built once (fine), but
  `bandwidth * multipliers` is re-broadcast per call.

## 4. Static cost analysis: MNIST and Stable Diffusion

Neither can be run locally (no GPU / no checkpoints). The per-step call graphs and the
cost expression `T x N_recur x N_iter x n_t x C(f_phi)` with the *implemented* meaning of
each symbol:

### 4a. Synthetic (`simulations/experiments/_guided.py::run`) -- for comparison

| symbol | implemented meaning |
|---|---|
| T | `model_uncond.diffusion_steps = 100`; loop `t = 99..1` => **99** guided steps (no final t=0 step) |
| N_recur | **1** (no re-noising; the engine's `N_recur` exists but `_guided` does not use it) |
| N_iter | **1** gradient evaluation per step (`guidance_target="x_t"`); MPGD arm uses 1 leaf gradient |
| M_t (spatial) | 1 (`no_lgd`) or 3 (`lgd`), each an *independent* sampler call + MMD; combined by `-logsumexp` |
| n_t | `n_max` constant (schedule="constant"); `time`/`noise` schedules via `tfg.n_schedule.n_at` |
| C(f_phi) | one `ConsistencyModeliCT.sample` of n: 5 MLP forwards (128 units, depth 3) on n rows; plus the MMD on (n+250)^2 x 5 and one backward through all of it |
| cost per run | denoiser 99; cond network forwards 99 x M_t x 5; cond samples 99 x M_t x n; MMD 99 x M_t; backward 99 |

### 4b. MNIST (`MNIST/run_mlgdf.py::optimize_LGD`)

Per step `i, t in enumerate(timesteps[:-1])` (DDIM, `num_inference_steps=130` =>
**129** guided steps, then one final no-grad DDIM step):

1. `model_uncond(x_t, t)` -- HF `UNet2DModel` (28x28, blocks 32/64/128 with attention),
   batch 1, with grad. **1 call/step.**
2. for `_ in range(num_x_t)` (**num_x_t = 3** spatial LGD perturbations, `r_t^2`-scaled noise
   added to `pred_x0`):
   * `model_cond_cm.sample(nsamples=1500, condition_x=x0_sample, ts=[150,50,20,10,5,1])`:
     `CircularAngleConsistencyModel` -- 5 forwards; each forward runs the conv image
     encoder `cond_embed` on the **batch-1** image (broadcast to 1500 rows) and the
     128-unit MLP on 1500 x 2. The image encoder is recomputed 5x per sampler call
     (15x per step) for the same `x0_sample`.
   * target angles: fresh `generate_mog_samples(1500)` every perturbation every step
     (target resampled, not fixed).
   * loss = `sliced_wasserstein_distance(..., n_projections=50)` -- **SWD, not MMD**
     (fresh random projections each call; sort of 1500 x 50).
3. `log_me = -logsumexp(losses) + log(num_x_t)`; `grad = autograd.grad(log_me, x_t,
   retain_graph=True)` -- **retain_graph=True keeps the whole graph alive** until the
   next iteration rebinds; memory, not time.
4. `x_t = x_{t-1} - step_size * grad` (no momentum; `compute_step_size` schedule).
5. After the loop: one sampler call with `nsamples = 4 x 1500` for the final loss.

| symbol | implemented meaning |
|---|---|
| T | 129 guided DDIM steps (130 timesteps, last one unguided) |
| N_recur | 1 |
| N_iter | 1 |
| M_t | `num_x_t = 3` |
| n_t | `nsamples = 1500` (constant), each perturbation; final eval 6000 |
| C(f_phi) | 5 CM forwards: 5 x (conv encoder on 1 image + MLP on 1500 x 2) + SWD sort (1500 x 50) |
| per seed | UNet 129; CM sampler calls 387; CM forwards 1935; conditional samples **580,500** (+6000 final); SWD evals 387; backward 129 (through UNet + 3 x 5 MLP forwards + sorts) |
| per experiment | x `n_seeds = 15` |

Redundant/static: image encoder recomputed per ladder level; `retain_graph=True`;
target set resampled every call (could be fixed once, like the synthetic loop); SWD
projections resampled per call (adds gradient noise independent of the conditional
model); UNet is evaluated **once** per step (no double evaluation here).

### 4c. Stable Diffusion (`SD_cond_SD_controlnet/scripts/run_mlgd_f.py` + `src/generation.py`)

(The brief names `run_dps_synthetic_targets.py`; that file is not in this checkout --
`scripts/run_mlgd_f.py` is the MLGD-F entry point and is what was analysed.)

Per guided step `i, t in enumerate(timesteps[start_step:])` (`n_steps=30`,
`start_step=15` in the submit scripts => **15** guided steps; earlier memory notes
250/500-step runs use the same loop):

1. Architect SDXL-base UNet, CFG batch of 2, **with grad** (`predict_noise_cfg`) -- and
   a **second** UNet call on the parallel *unguided* trajectory (`noise_pred_regular`,
   no grad): the denoiser is evaluated twice per step (the second is a control arm, not
   the guided path, but it is paid every step).
2. `compute_pred_x0_direct`, architect VAE decode of `pred_x0` (512x512, fp32,
   checkpointed) -> `pixel_x0_norm`.
3. `run_dps_step_clip`: for each of `num_variations = 6` (**variation_batch_size = 1
   hard-coded**): one full `StableDiffusionXLControlNetPipeline.__call__` (prompt
   re-encoded through both SDXL text encoders every call, ControlNet + UNet x
   `num_inference_steps = 2`, output latent), sprinter VAE decode (fp32), CLIP ViT-L/14
   encode -> 768-d; all wrapped in `torch.utils.checkpoint` (forward recomputed during
   backward). Then `compute_mmd` (unbiased, single bandwidth from the median heuristic
   recomputed from `x[:ss]`, `y[:ss]` each call; `sqrt`), `zeta_i = base_zeta / loss`,
   `autograd.grad(loss, latents_step)`.
4. `evaluate_distribution_mmd` every `eval_interval` steps: architect VAE decode,
   `n_eval = 10` extra sprinter calls (batch 2), CLIP encode, MMD; **moves CLIP to
   CPU afterwards** and back to GPU at the next `run_dps_step_clip`.
5. `visualize_step` **every step**: 4 VAE decodes + **`num_cond = 5` extra sprinter
   pipeline calls** (no grad, PIL output) + PCA plot; also casts `sprinter.vae` to
   fp16 and back to fp32 **every step** (full weight copy twice per step, also done
   inside `generate_and_store_cs` and `evaluate_distribution_mmd`).
6. `denoise_step` for guided and regular latents; `gc.collect(); torch.cuda.empty_cache()`
   every step.

| symbol | implemented meaning |
|---|---|
| T | `len(timesteps[start_step:])` = n_steps - start_step (15 in the submit scripts) |
| N_recur | 1 |
| N_iter | 1 |
| M_t | 1 (no spatial LGD perturbation in the SD path) |
| n_t | `num_variations = 6` conditional samples, each a separate batch-1 pipeline call |
| C(f_phi) | sprinter: 2 x (ControlNet + SDXL-turbo UNet) + text encoders + VAE decode + CLIP ViT-L; x2 because of checkpoint recompute in backward; plus backward through all of it |
| per step | architect UNet 2 (1 with grad); VAE decodes: 1 (guided, checkpointed) + 4 (vis) + 6 (sprinter) + eval; sprinter pipeline calls **6 guided + 5 visualisation (+10 eval on eval steps)**; CLIP encodes 6 (+10); MMD 1 (+1); backward 1 |

Where MMD/feature/backward sit relative to the conditional model: the MMD itself is
negligible (6 x ~40-200 embeddings of 768-d). The cost is the conditional model
(`f_phi` = sprinter + VAE + CLIP, run 6x forward, 6x recompute, 6x backward) and the
architect VAE/UNet backward. The 5 visualisation sprinter calls per step are ~45% of the
per-step sprinter forward count (5 vs 6+6 recompute) and are pure diagnostics.

Redundant work visible statically: double denoiser evaluation (guided + unguided
control); per-step visualisation sprinter calls; per-call prompt re-encoding in the
sprinter pipeline (the prompt is constant); VAE dtype casts every step; CLIP CPU<->GPU
moves around eval; median-heuristic bandwidth recomputed every step from detached
copies (cheap but makes the loss non-stationary across steps); `variation_batch_size=1`
(6 serial pipeline calls instead of one batch of 6, where memory allows); target
embeddings are cached once (`all_clip_embeddings`, detached) -- that part is fine; the
`K_yy` block (N_target^2, 768-d) is recomputed every step although targets are fixed.
