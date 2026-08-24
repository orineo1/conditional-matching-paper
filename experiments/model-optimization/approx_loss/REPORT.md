# Agent 3 report — approximate / linear-time distributional objectives

Files: `approx_mmd.py` (candidates), `diagnostics.py` -> `diagnostics.csv`, `DIAGNOSTICS.md`,
`diagnostics_grad_cos_and_time.png`, `diagnostics.log`; `THEORY.md`; `test_approx_mmd.py` (15 tests, all pass);
`../hypotheses/agent3.yaml`. Run from `simulations/` with `/Users/stolk/miniconda3/bin/python`, float64 CPU.

## Bottom line

1. **At m=250 (synthetic) the exact kernel costs `n(n+m)` kernel evaluations once the constant target block is
   cached; an RFF with D=256 features costs 1.2-2.2x MORE than that and is less accurate. Random-feature and
   low-rank approximations of the MMD do not pay off in any of this paper's regimes.** The only thing that is
   expensive in the repository call is the recomputed YY block (9-11x wall at m=250, 2.6x at m=120/d=768), and
   removing it is exact (Agent 2).
2. **The one approximation-class idea that is worth a Stage-1 screen is not an approximation but an exact
   simplification: the population-GMM target** (`PopulationGMMMMD`): closed-form cross term via
   `tfg.gmm_mmd.kernel_mean_embedding_multibandwidth`, cost 0.02-0.12x (d=1) / 0.2-0.55x (d=8,16) of the cached-exact
   loss, gradient cosine 0.989-0.998 to the empirical-target gradient, deterministic, and it removes the
   target-sampling noise from the objective. Synthetic / MNIST-GMM only; not SD.
3. Everything else — RFF, ORF, Nystrom with target landmarks, sliced distances, FFT — is rejected for the
   approximation question (reasons below). Target subsampling with a fixed subset is kept only as a documented
   cost-accuracy knob (B=64: 0.3x cost, cos 0.96-0.98 in d<=16) since it is literally "use fewer targets".

## Key numbers (n=8 and 32; gradient cosine vs exact; cost ratio vs exact-cached; wall ratio vs repository call)

| setting | n | candidate | D | grad cos | loss rel err | grad-norm rel err | cost/cachedYY | wall/ref |
|---|---|---|---|---|---|---|---|---|
| synth_d1 | 8 | exact_cachedYY |  | 1.0000 | 8.71e-15 | 5.00e-16 | 1.00 | 0.09 |
| synth_d1 | 8 | rff | 16 | 0.9316 | 3.76e-01 | 4.02e-01 | 0.08 | 0.03 |
| synth_d1 | 8 | rff | 64 | 0.9778 | 1.50e-01 | 1.72e-01 | 0.34 | 0.04 |
| synth_d1 | 8 | rff | 256 | 0.9891 | 1.05e-01 | 1.10e-01 | 1.34 | 0.09 |
| synth_d1 | 8 | rff | 1024 | 0.9977 | 3.53e-02 | 3.98e-02 | 5.37 | 0.19 |
| synth_d1 | 8 | nystrom | 16 | 1.0000 | 2.21e-03 | 6.90e-03 | 0.89 | 0.09 |
| synth_d1 | 8 | nystrom | 64 | 1.0000 | 6.20e-04 | 2.04e-03 | 13.48 | 0.11 |
| synth_d1 | 8 | subsample | 16 | 0.8713 | 6.54e-01 | 3.66e-01 | 0.09 | 0.05 |
| synth_d1 | 8 | subsample | 64 | 0.9685 | 2.60e-01 | 1.27e-01 | 0.28 | 0.08 |
| synth_d1 | 8 | sliced_w2 | 1 | 0.5240 | 3.08e+01 | 4.40e+01 | 0.00 | 0.03 |
| synth_d1 | 8 | population_gmm |  | 0.9972 | 2.40e-02 | 1.95e-02 | 0.04 | 0.31 |
| synth_d1 | 8 | tab_kme_1d | 2048 | 0.9994 | 7.37e-04 | 1.05e-02 | 0.04 | 0.05 |
| synth_d1 | 32 | exact_cachedYY |  | 1.0000 | 1.77e-14 | 4.13e-16 | 1.00 | 0.22 |
| synth_d1 | 32 | rff | 16 | 0.9888 | 3.16e-01 | 3.87e-01 | 0.07 | 0.03 |
| synth_d1 | 32 | rff | 64 | 0.9968 | 1.14e-01 | 1.56e-01 | 0.29 | 0.08 |
| synth_d1 | 32 | rff | 256 | 0.9991 | 7.39e-02 | 1.02e-01 | 1.16 | 0.17 |
| synth_d1 | 32 | rff | 1024 | 0.9998 | 2.94e-02 | 3.61e-02 | 4.63 | 0.50 |
| synth_d1 | 32 | nystrom | 16 | 1.0000 | 1.30e-03 | 4.43e-03 | 0.81 | 0.09 |
| synth_d1 | 32 | nystrom | 64 | 1.0000 | 3.17e-04 | 1.19e-03 | 12.33 | 0.10 |
| synth_d1 | 32 | subsample | 16 | 0.8306 | 1.78e+00 | 2.60e-01 | 0.17 | 0.08 |
| synth_d1 | 32 | subsample | 64 | 0.9781 | 5.85e-01 | 7.86e-02 | 0.34 | 0.10 |
| synth_d1 | 32 | sliced_w2 | 1 | 0.3421 | 9.83e+01 | 3.03e+01 | 0.00 | 0.02 |
| synth_d1 | 32 | population_gmm |  | 0.9988 | 4.82e-02 | 4.82e-02 | 0.12 | 0.30 |
| synth_d1 | 32 | tab_kme_1d | 2048 | 0.9992 | 1.59e-03 | 1.48e-03 | 0.12 | 0.07 |
| synth_d8 | 8 | exact_cachedYY |  | 1.0000 | 1.51e-14 | 3.74e-16 | 1.00 | 0.09 |
| synth_d8 | 8 | rff | 16 | 0.5230 | 2.23e-01 | 1.39e+00 | 0.12 | 0.03 |
| synth_d8 | 8 | rff | 64 | 0.7539 | 1.22e-01 | 3.82e-01 | 0.49 | 0.04 |
| synth_d8 | 8 | rff | 256 | 0.8940 | 5.68e-02 | 1.35e-01 | 1.96 | 0.10 |
| synth_d8 | 8 | rff | 1024 | 0.9678 | 2.85e-02 | 5.81e-02 | 7.82 | 0.19 |
| synth_d8 | 8 | nystrom | 16 | 0.8925 | 3.20e-01 | 3.19e-01 | 0.44 | 0.08 |
| synth_d8 | 8 | nystrom | 64 | 0.9417 | 2.05e-01 | 2.47e-01 | 6.35 | 0.10 |
| synth_d8 | 8 | subsample | 16 | 0.8807 | 4.59e-01 | 1.41e-01 | 0.09 | 0.05 |
| synth_d8 | 8 | subsample | 64 | 0.9806 | 1.72e-01 | 4.26e-02 | 0.28 | 0.07 |
| synth_d8 | 8 | sliced_w2 | 32 | 0.7199 | 6.36e+00 | 7.04e+00 | 0.10 | 0.03 |
| synth_d8 | 8 | population_gmm |  | 0.9941 | 2.83e-02 | 1.38e-02 | 0.22 | 0.28 |
| synth_d8 | 32 | exact_cachedYY |  | 1.0000 | 5.29e-14 | 6.38e-16 | 1.00 | 0.19 |
| synth_d8 | 32 | rff | 16 | 0.5239 | 2.46e-01 | 1.41e+00 | 0.11 | 0.04 |
| synth_d8 | 32 | rff | 64 | 0.7408 | 1.29e-01 | 3.75e-01 | 0.44 | 0.09 |
| synth_d8 | 32 | rff | 256 | 0.8970 | 5.20e-02 | 1.25e-01 | 1.76 | 0.16 |
| synth_d8 | 32 | rff | 1024 | 0.9667 | 2.76e-02 | 5.31e-02 | 7.03 | 0.50 |
| synth_d8 | 32 | nystrom | 16 | 0.9011 | 2.89e-01 | 2.94e-01 | 0.41 | 0.09 |
| synth_d8 | 32 | nystrom | 64 | 0.9416 | 1.96e-01 | 2.34e-01 | 5.81 | 0.11 |
| synth_d8 | 32 | subsample | 16 | 0.8165 | 1.07e+00 | 1.84e-01 | 0.17 | 0.08 |
| synth_d8 | 32 | subsample | 64 | 0.9646 | 3.52e-01 | 4.39e-02 | 0.34 | 0.09 |
| synth_d8 | 32 | sliced_w2 | 32 | 0.5533 | 1.53e+01 | 7.30e+00 | 0.11 | 0.03 |
| synth_d8 | 32 | population_gmm |  | 0.9895 | 1.93e-02 | 1.26e-02 | 0.29 | 0.30 |
| synth_d16 | 8 | exact_cachedYY |  | 1.0000 | 8.07e-15 | 8.84e-17 | 1.00 | 0.10 |
| synth_d16 | 8 | rff | 16 | 0.4001 | 1.99e-01 | 1.91e+00 | 0.13 | 0.03 |
| synth_d16 | 8 | rff | 64 | 0.6332 | 1.35e-01 | 6.84e-01 | 0.54 | 0.04 |
| synth_d16 | 8 | rff | 256 | 0.8230 | 6.84e-02 | 2.06e-01 | 2.16 | 0.08 |
| synth_d16 | 8 | rff | 1024 | 0.9389 | 2.90e-02 | 7.38e-02 | 8.62 | 0.18 |
| synth_d16 | 8 | nystrom | 16 | 0.7591 | 4.21e-01 | 3.89e-01 | 0.30 | 0.07 |
| synth_d16 | 8 | nystrom | 64 | 0.9415 | 2.37e-01 | 2.73e-01 | 4.03 | 0.09 |
| synth_d16 | 8 | subsample | 16 | 0.8830 | 4.15e-01 | 1.32e-01 | 0.09 | 0.04 |
| synth_d16 | 8 | subsample | 64 | 0.9782 | 1.67e-01 | 2.31e-02 | 0.28 | 0.07 |
| synth_d16 | 8 | sliced_w2 | 32 | 0.6253 | 6.69e+00 | 8.83e+00 | 0.11 | 0.02 |
| synth_d16 | 8 | population_gmm |  | 0.9958 | 8.18e-03 | 5.57e-03 | 0.50 | 0.26 |
| synth_d16 | 32 | exact_cachedYY |  | 1.0000 | 2.89e-14 | 2.21e-16 | 1.00 | 0.18 |
| synth_d16 | 32 | rff | 16 | 0.4143 | 2.17e-01 | 2.11e+00 | 0.12 | 0.04 |
| synth_d16 | 32 | rff | 64 | 0.6392 | 1.32e-01 | 7.54e-01 | 0.49 | 0.08 |
| synth_d16 | 32 | rff | 256 | 0.8226 | 5.70e-02 | 2.52e-01 | 1.95 | 0.18 |
| synth_d16 | 32 | rff | 1024 | 0.9353 | 2.64e-02 | 7.99e-02 | 7.81 | 0.56 |
| synth_d16 | 32 | nystrom | 16 | 0.7708 | 4.12e-01 | 3.75e-01 | 0.27 | 0.09 |
| synth_d16 | 32 | nystrom | 64 | 0.9297 | 2.23e-01 | 2.46e-01 | 3.69 | 0.11 |
| synth_d16 | 32 | subsample | 16 | 0.8194 | 1.10e+00 | 2.03e-01 | 0.17 | 0.08 |
| synth_d16 | 32 | subsample | 64 | 0.9626 | 3.61e-01 | 3.63e-02 | 0.34 | 0.10 |
| synth_d16 | 32 | sliced_w2 | 32 | 0.5291 | 1.70e+01 | 1.03e+01 | 0.11 | 0.03 |
| synth_d16 | 32 | population_gmm |  | 0.9923 | 1.54e-02 | 1.14e-02 | 0.55 | 0.37 |
| clip_d768 | 8 | exact_cachedYY |  | 1.0000 | 2.66e-16 | 2.00e-16 | 1.00 | 0.36 |
| clip_d768 | 8 | rff | 16 | 0.0629 | 2.72e-01 | 1.57e+01 | 0.31 | 0.06 |
| clip_d768 | 8 | rff | 64 | 0.1147 | 1.09e-01 | 7.35e+00 | 1.25 | 0.08 |
| clip_d768 | 8 | rff | 256 | 0.2301 | 6.40e-02 | 3.38e+00 | 4.98 | 0.38 |
| clip_d768 | 8 | rff | 1024 | 0.4287 | 3.21e-02 | 1.35e+00 | 19.93 | 1.17 |
| clip_d768 | 8 | orf | 16 | 0.0479 | 2.48e-01 | 1.36e+01 | 0.31 | 0.06 |
| clip_d768 | 8 | orf | 64 | 0.0946 | 8.85e-02 | 5.91e+00 | 1.25 | 0.07 |
| clip_d768 | 8 | orf | 256 | 0.1757 | 5.56e-02 | 2.37e+00 | 4.98 | 0.45 |
| clip_d768 | 8 | orf | 1024 | 0.3521 | 2.53e-02 | 9.64e-01 | 19.93 | 1.19 |
| clip_d768 | 8 | nystrom | 16 | 0.2268 | 8.16e-01 | 7.17e-01 | 0.14 | 0.18 |
| clip_d768 | 8 | nystrom | 64 | 0.2509 | 7.74e-01 | 6.82e-01 | 0.71 | 0.29 |
| clip_d768 | 8 | subsample | 16 | 0.7211 | 1.02e-01 | 2.13e-01 | 0.19 | 0.18 |
| clip_d768 | 8 | subsample | 64 | 0.7425 | 3.07e-02 | 1.95e-01 | 0.56 | 0.31 |
| clip_d768 | 8 | sliced_w2 | 32 | 0.1140 | 9.99e-01 | 9.96e-01 | 0.25 | 0.04 |
| clip_d768 | 32 | exact_cachedYY |  | 1.0000 | 1.02e-15 | 3.45e-17 | 1.00 | 0.39 |
| clip_d768 | 32 | rff | 16 | 0.0657 | 1.80e-01 | 1.49e+01 | 0.26 | 0.05 |
| clip_d768 | 32 | rff | 64 | 0.1231 | 1.24e-01 | 6.81e+00 | 1.05 | 0.14 |
| clip_d768 | 32 | rff | 256 | 0.2463 | 7.37e-02 | 3.10e+00 | 4.19 | 0.33 |
| clip_d768 | 32 | rff | 1024 | 0.4561 | 3.92e-02 | 1.21e+00 | 16.78 | 1.28 |
| clip_d768 | 32 | orf | 16 | 0.0712 | 2.67e-01 | 1.67e+01 | 0.26 | 0.05 |
| clip_d768 | 32 | orf | 64 | 0.1368 | 7.60e-02 | 7.08e+00 | 1.05 | 0.12 |
| clip_d768 | 32 | orf | 256 | 0.1948 | 6.64e-02 | 2.30e+00 | 4.19 | 0.31 |
| clip_d768 | 32 | orf | 1024 | 0.3871 | 2.51e-02 | 9.26e-01 | 16.78 | 1.20 |
| clip_d768 | 32 | nystrom | 16 | 0.1949 | 7.65e-01 | 7.19e-01 | 0.12 | 0.14 |
| clip_d768 | 32 | nystrom | 64 | 0.2196 | 7.13e-01 | 6.86e-01 | 0.60 | 0.26 |
| clip_d768 | 32 | subsample | 16 | 0.7347 | 1.31e-01 | 2.49e-01 | 0.32 | 0.31 |
| clip_d768 | 32 | subsample | 64 | 0.7581 | 3.93e-02 | 2.34e-01 | 0.63 | 0.40 |
| clip_d768 | 32 | sliced_w2 | 32 | 0.1176 | 9.99e-01 | 9.97e-01 | 0.21 | 0.04 |
Full tables incl. n=4, across-seed dispersion (loss CV, pairwise gradient cosine of feature seeds) and peak-memory
model: `DIAGNOSTICS.md` / `diagnostics.csv`.  Separate sweep at d=768, n=8 (`THEORY.md` sec. 2a): RFF cos
0.20/0.41/0.67/0.88/0.97 and ORF 0.25/0.47/0.75/0.92/0.98 at D = 256/1024/4096/16384/65536 — the exact-cached loss
costs the equivalent of D ~ 70.

## Verdicts by regime

| regime | exact-cached cost/call | viable approximation? |
|---|---|---|
| synthetic d=1, m=250, n<=32 | `n (n+250) * 6` ~ 1.5e4 at n=8 | none needed; **population-GMM closed form** (0.02-0.12x, exact) or tabulated KME (d=1 only, 0.04-0.12x, cos 0.999) if the target were only available as samples |
| synthetic d=8,16, m=250 | `n(n+250)(d+5)` | population-GMM (0.2-0.55x, cos 0.99); RFF/Nystrom need D,L >~ m for cos>0.95 -> no gain; subsample B=64 is 0.3x at cos 0.96-0.98 |
| SD CLIP-768, m~120, n 8-32 | `n(n+120) 773` | none: RFF/ORF need D ~ 2^16 >> m; Nystrom on target landmarks collapses (cos 0.25) when X is off-target; alpha=2 kernel is not PD so RFF/FFT do not exist for it; the exact cached loss is already ~0.5-0.8 ms |
| MNIST (SWD on 2-D angles) | `n log n * 50` | already linear-time; its target is a GMM -> population-GMM MMD would be an exact alternative objective; final SWD metric must use independent projections |

## Recommended Stage-1 configuration

* Candidate: `A3-pop-gmm-target` — `PopulationGMMMMD(params["target_means"], stack(params["target_variances"]),
  params["target_weights"], bw)` with the SAME frozen `bw = fixed_bandwidth(S_G)` and 5 bandwidths as the baseline
  (so only the target changes, not the kernel). Plug in as a drop-in for `mmd(y, S_G)` in `_guided.run` (it
  exposes `loss(X)`); evaluation unchanged (`|x - x*|`).
* Screen: Exp-2 style, 2D params and `dimy` d in {1,8,16}, n in {4,8,16,32}, 50 restarts, paired with the
  exact-empirical baseline (`paired_stats`), both `temporal=none` and `adam`. Rejection: paired L2 worse at any n
  (perm p<0.05), or loss wall share already <5% of the step in Agent 1's profile (then it is an estimator-cleanliness
  change for Agent 4, not a performance one).
* Optional second knob: `SubsampledTargetMMD(B in {32,64,125})`, fixed subset, to measure how much of m the
  gradient uses; reject if L2 degrades at B<=125.

## Rejected, with reasons

* **RFF / ORF**: gradient variance scales like d/D; D must exceed m for cos>0.95 at d>=8, so it is never cheaper
  than the exact cross term; at d=768 hopeless (D ~ 2^16). ORF helps loss/grad-norm error marginally, not the
  cosine at usable D. Does not exist for the SD alpha=2 kernel (not PD).
* **Nystrom (target landmarks)**: accurate and cheap-ish only in d=1 (where nothing needs accelerating); in d>=8
  cos 0.93-0.94 at L=64 for 4-6x the exact cost (n L^2 term); in d=768 the gradient collapses far from the target
  (cos 0.25) — structurally wrong for guidance unless X is added to the landmarks (L^3 refactorisation per step).
* **Linear-time / block estimators**: the only O(m) term is the cross term; subsampling the target is all they
  amount to here (n<=32 makes XX negligible, YY is constant). Fresh subsets add per-step variance (no CRN).
* **Sliced W2**: a different objective (cos 0.3-0.9 vs MMD in d<=16, 0.1 in d=768 with P=32); if used, final
  metric must be independent (MNIST note). Hand to the estimator track as an alternative objective, not here.
* **FFT**: no grid in CLIP space; for d=1 the grid KME table can be filled by a one-off direct sum (5e5 kernel
  evals, ~0.2 ms) so the FFT saves nothing; the per-call gain is the table, and the population-GMM form beats the
  table whenever the target is parametric. No FFT candidate proceeds.

## Caveats

* Gradient cosine is measured w.r.t. the n conditional samples `X`, not w.r.t. the diffusion state `x_t`; the
  chain rule through the conditional model is common to both so cosines are indicative, but the threshold at which
  an approximation becomes harmful should be compared with the Monte-Carlo gradient noise of the n draws (Agent 4).
* Wall numbers are CPU float64 on this machine, median of 7; all candidates are sub-millisecond, so the loss is not
  the bottleneck of a step once YY is cached.
* Repository `MMDLoss` and the cached variant differ at the 1e-8 relative level because `torch.cdist` switches to
  the matmul formula for >25 rows (numerical only, see `test_cached_yy_equals_reference`).
