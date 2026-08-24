# TFG experiments

## Motivation

Conditional distribution matching needs **many conditional generations per
diffusion step**, because the guidance loss is an MMD between a sample of the
model's conditional and a target set. LGD then multiplies that already
expensive cost: it evaluates the objective at `M_t` spatially perturbed copies
of `x_{0|t}`, so the conditional cost per step is

```
C_t = M_t * n_t          (legacy default: 3 * 250 = 750 per step)
```

The spatial perturbations exist to stabilise a noisy estimate of `x_{0|t}`.
**Temporal momentum is a candidate substitute**: it stabilises the guidance
gradient by accumulating information *across* diffusion steps, which costs no
extra conditional generations at all. If it works, the 3x spatial spend can be
removed. This is the question the experiments below are built around.

Adam is the momentum rule we test, following *Adaptive Moments are Surprisingly
Effective for Plug-and-Play Diffusion Sampling* (ICLR 2026, arXiv:2603.16797).
Its moment update is verified identical to the authors' released code. Because
our guidance loss is an MMD rather than the paper's pointwise likelihood, we
call the combination **Adam-CDM**, not AdamDPS.

## Experiments

| # | Experiment | Question | Comparison | Primary metrics | Status | Command | Output |
|---|---|---|---|---|---|---|---|
| 1 | `exp1_delta_target_equivalence.py` | Is the engine a strict generalization of ordinary TFG? | engine (point objective, delta target) vs frozen Algorithm 1 reference, all traced intermediates | max abs error over every intermediate | **supported** | `python experiments/exp1_delta_target_equivalence.py` | `results/tfg/exp1_delta_target_equivalence.json` |
| 2 | `exp2_lgd_vs_adam.py` | Can momentum replace LGD's 3x spatial cost? | 2x2: {no LGD, LGD} x {none, Adam} | exact GMM L2, success, paired CI + permutation p, conditional calls, runtime | **run, negative** | `python experiments/exp2_lgd_vs_adam.py --n 8 --restarts 100` | `results/tfg/exp2_lgd_vs_adam_n{n}.json` |
| 3 | `exp3_sample_scaling.py` | Where does Adam stop helping? | no-LGD x {none, Adam} across n | `Delta(n)`, its 95% CI, `n*` | **run, `n* = 8`** | `python experiments/exp3_sample_scaling.py --restarts 100` | `results/tfg/exp3_sample_scaling_{setting}.json` |
| 4 | `exp4_nt_schedules.py` | Does an uneven `n_t` schedule beat a uniform one at equal budget? | constant vs time- vs noise-increasing, x {none, Adam}, plus a budget-matched constant | exact GMM L2, `sum_t n_t`, paired CI | **planned** | `python experiments/exp4_nt_schedules.py --n-max 16 --restarts 100` | `results/tfg/exp4_nt_schedules_nmax{n}.json` |
| 5 | `exp5_dimy_scaling.py` | Does momentum help more as **dim(Y)** grows? | no-LGD x {none, Adam} across `d` and `n` | `Delta(d,n)`, `n*(d)` | **run, INVALID for its stated question** | `python experiments/exp5_dimy_scaling.py --restarts 50` | `results/tfg/exp5_dimy_scaling_*.json` |
| 5A | `exp5a_plateau_mechanism.py` | Which part of Adam crosses the plateau? | none vs full Adam vs normalisation-only (`beta1 = 0`) | escape rate, near-optimum rate, exact GMM L2 | **run, supported** | `python experiments/exp5a_plateau_mechanism.py --d 8 --restarts 40` | `results/tfg/exp5a_plateau_mechanism_d{d}*.json` |
| 5B | `exp5b_zeta_calibration.py` | What guidance strength makes the *baseline* a working optimiser at each dim(Y)? | zeta sweep per `d`, no-momentum arm only | reached-`x*` rate at large `n`, `zeta_d*` | **run; gate FAILED, and it corrects Exp 5** | `python experiments/exp5b_zeta_calibration.py --restarts 16` | `results/tfg/exp5b_zeta_calibration.json` |
| 6 | `exp6_mpgd.py` | Does MPGD-style guidance help, and does momentum help on top of it? | 2x2: {`x_t`, `x0` (MPGD)} x {none, Adam} | exact GMM L2, success, paired CI | **run, negative** | `python experiments/exp6_mpgd.py --n 8 --restarts 100` | `results/tfg/exp6_mpgd_*.json` |
| 7 | `exp7_adam_hyperparams.py` | Are the inherited Adam constants right for an MMD loss? | `beta1` x `beta2` x `n`, vs no-momentum | exact GMM L2, paired CI vs none | **run, null** | `python experiments/exp7_adam_hyperparams.py --restarts 40` | `results/tfg/exp7_adam_hyperparams_2D.json` |
| 8 | `exp8_trust_region.py` | Does a noise-level trust region on the guidance step (`\|\|Delta_t\|\| <= sqrt(1-alphabar_t)`, `trust_noise1`) improve the no-LGD estimator at equal cost, and transfer across 2D/5D/10D? | **through the engine** (`GeneralizedTFG`, `step_clip="noise"`): baseline vs `trust_noise1` at `n in {4,8,16,32}`, paired; LGD and larger-n baselines for the Pareto frontier | exact GMM L2, paired CI + permutation p, conditional draws, wall time | **run (held-out, cluster), supported in 2D and 10D, null in 5D** | `python experiments/exp8_trust_region.py --setting 2D --restarts 100 --offset 1000 --lgd` | `results/tfg/exp8_trust_region_{setting}.json` |

Experiment 1 is cheap (seconds). Experiments 2-7 train or load models on first
run (~1 min each, cached in `artifacts/checkpoints/`, git-ignored).

## Configuration

Frozen across every experiment unless stated:

| item | value | source |
|---|---|---|
| joint distribution | `params/2D_cond_1D_gmm_params.pt` | the paper's canonical file |
| target `S_G` | 250 samples, drawn once, seed 987654, reused everywhere | — |
| MMD kernel | repository `RBF`, 5 bandwidths, `mul_factor=2` | `src/LossFunctions.py` |
| bandwidth | selected once from `S_G`, then frozen across all n, methods, restarts | `_common.fixed_bandwidth` |
| conditional model | `ConsistencyModeliCT`, blocks 3, units 128, 20k epochs, batch 1024 | paper Table 3 |
| unconditional model | `DiffusionModel`, blocks 3, units 128, 20k epochs, batch 1024 | paper Table 3 |
| sampling ladder | `[150, 50, 20, 10, 5, 1]` | paper Appendix A.5 |
| Adam | `rho=0.4`, `beta1=0.9`, `beta2=0.995`, `delta=1e-8` | betas/delta official; rho tuned on tuning data |
| model seeds | 20240401 (main), 20240402/3 (robustness) | — |
| restarts | 100 per cell; use `--offset` for a disjoint held-out block | — |
| selection | failure-penalised mean L2 (`PENALTY=2.0`) with a success-rate floor; median and win rate secondary | `_common.penalised_score` |

**Compute accounting.** Every arm reports `conditional_calls` (averaged over
completed runs, not read off restart 0) and `sum_t n_t`. Cost per step is
`C_t = M_t * n_t`: no-LGD costs `n_t`, LGD costs `3 n_t`, and each individual
MMD uses at most `n_t` samples in both.

## Results

All numbers below are 100 paired restarts on the paper's canonical 2D
parameters, MAIN model seed 20240401, Adam `rho = 0.4`. Score is the
failure-penalised mean exact GMM L2 (lower is better).

### Supported

**Engine correctness.** With the point objective and a delta target the engine
reproduces the frozen Algorithm 1 transcription exactly: 30 configuration cells
(`N_recur` x `N_iter` x `gamma_bar` x guidance mode, plus a deep-recurrence and a
`T=2` boundary cell), **max absolute error `0.000e+00`** over every traced
intermediate. Five injected mutations are all detected, so the comparison is
not vacuous.

**Adam parity.** The engine's Adam path is identical to the authors' released
`adaptive_moment_estimate` to `atol 1e-12`, and to our standalone implementation
to `atol 1e-14`, across beta pairs, gradient scales `1e-8 ... 1e8`, and
tensor shapes up to 3-D.

**Benchmark.** Under the canonical parameters the target is realisable:
`x_opt = -5.0000328`, `L2^2 = 2.29e-08`.

**Momentum helps when conditional samples are scarce** (Experiment 3, no-LGD,
`Delta = S_none - S_Adam`, positive means Adam better):

| n | none | Adam | Delta | 95% CI | p |
|---|---|---|---|---|---|
| **4** | 0.5873 | **0.3598** | **+0.2275** | [+0.137, +0.319] | **<0.0001** |
| **8** | 0.3925 | **0.3228** | +0.0697 | [+0.0005, +0.141] | 0.054 |
| 16 | 0.2277 | 0.2262 | +0.0014 | [-0.034, +0.037] | 0.938 |
| 32 | **0.2521** | 0.2764 | -0.0243 | [-0.049, -0.001] | 0.056 |

`n* = 8`: the largest n whose 95% lower confidence bound on `Delta` is still
positive. The benefit decays monotonically and reverses by `n = 32`. Success
rate at `n = 4` rises with Adam; at `n = 32` it falls.

### Not supported

**Momentum does not replace LGD's 3x spatial cost** (Experiment 2, `n = 8`):

| arm | score | success | conditional calls |
|---|---|---|---|
| no-LGD / none | 0.3925 | 33% | 792 |
| no-LGD / Adam | 0.3228 | 45% | 792 |
| **LGD / none** | **0.2542** | **60%** | 2376 |
| LGD / Adam | 0.2585 | 52% | 2376 |

- LGD without momentum beats no-LGD without momentum: `+0.1383`,
  CI [+0.076, +0.202], **p < 0.0001**. The 3x spatial spend buys real accuracy here.
- LGD without momentum also beats no-LGD **with** Adam at 1/3 the calls:
  `-0.0686`, CI [-0.117, -0.021], **p = 0.0073**.
- Adam adds nothing on top of LGD: `-0.0043`, p = 0.84.

So at `n = 8` the honest statement is: momentum improves the *no-LGD* estimator,
but does **not** let it match LGD at one third of the conditional cost.

### Experiment 5A -- within Adam, the operative part is normalisation

`dim(X) = 1`, `dim(Y) = 8`, `n = 32` (3,168 conditional calls/run), 250-sample
target, 99 steps, 40 paired restarts, `zeta = 1.715`.

| arm | score | escaped the plateau | landed near `x*` |
|---|---|---|---|
| none | 0.3769 | **0%** | 0% |
| adam (`beta1 = 0.9, beta2 = 0.995`) | 0.1997 | 52.5% | 52.5% |
| **normalise-only (`beta1 = 0`)** | **0.1131** | **100%** | **97.5%** |

- `none` vs normalise-only: `+0.2638`, CI [+0.239, +0.288], **p < 0.0001**, 40/40 wins.
- normalise-only vs full Adam: `-0.0866`, CI [-0.151, -0.022], **p = 0.016**,
  favouring normalise-only.

Normalisation alone is not merely sufficient, it is **better than full Adam**;
the `beta1` accumulation term actively hurts here.

**Scope.** "Here" is load-bearing. Experiment 7 sweeps the same two constants on
the paper's 2D setting, where the baseline is already a working optimiser, and
finds `beta1` makes no difference at all (`beta1 = 0` vs `0.9`: p = 0.81 at
`n = 8`, p = 0.62 at `n = 4`). The `beta1` penalty is therefore specific to the
plateau regime, not a general property of the guidance rule. Read 5A as "what
crosses a plateau is normalisation", not as "momentum is useless". So within Adam the operative
part is **per-coordinate normalisation**: dividing by `sqrt(v_hat)` turns a tiny
gradient into a full-size `~rho` step instead of a `~1e-3` one. Experiment 5B
then explains why that mattered *here* -- the baseline's `zeta` was about 4x too
small, and normalisation supplies the missing factor. Against a properly
calibrated baseline this is a step-size effect, not evidence that accumulating
gradients across diffusion steps buys anything.

An accumulation-only arm was run and then dropped from the script and from this
README: with `beta2 -> 1` the update still divides by a frozen `sqrt(v_hat)`,
which is normalisation by a constant, so it does not isolate accumulation and is
not a usable signal.

### Experiment 6 -- MPGD is worse, and momentum does not rescue it

`n = 8`, 100 paired restarts, no LGD, canonical 2D parameters. The `x_t` arm
differentiates through the denoiser back to `x_t`; the `x0` (MPGD) arm treats
`x_{0|t}` as a leaf and moves the clean estimate directly (`N_iter >= 1`, `rho = 0`).

| arm | score | success | conditional calls |
|---|---|---|---|
| `x_t` / none | 0.3925 | 33% | 792 |
| `x_t` / Adam | 0.3228 | 45% | 792 |
| **`x0` (MPGD) / none** | **0.7114** | **1%** | 792 |
| **`x0` (MPGD) / Adam** | **0.7171** | **0%** | 792 |

- `x_t` beats MPGD without momentum: `-0.3189`, CI [-0.376, -0.257],
  **p < 0.0001**, MPGD wins only 21/100.
- Momentum adds nothing on top of MPGD: `-0.0057`, CI [-0.025, +0.013], p = 0.58.
- With momentum on both sides the gap widens: `-0.3943`, **p < 0.0001**.

Skipping the backpropagation through the denoiser also discards the sensitivity
of `x_{0|t}` to `x_t`, which is what carries the guidance signal in this setup.
At the same conditional cost, MPGD is the wrong trade here.

### Experiment 7 -- the inherited Adam constants are not a live variable

Every other experiment uses `beta1 = 0.9, beta2 = 0.995, delta = 1e-8`, the
official AdamDPS defaults, which were tuned for a **pointwise likelihood** loss.
Ours is an MMD between a sample of the model conditional and a target set, whose
gradient has a different noise structure, so the transfer needed checking.

4 x 4 sweep over `beta1 in {0, 0.3, 0.6, 0.9}` x `beta2 in {0.9, 0.99, 0.995,
0.999}`, at `n in {4, 8}`, 2D canonical, `rho = 0.4`, 40 restarts per cell, run
on a **tuning block at `--offset 1000`** that is disjoint from the 0..99 block
every reported number uses.

- At `n = 4`, **all 16 cells** beat plain gradient guidance (`p <= 0.044`);
  scores span 0.30-0.40.
- At `n = 8`, 5 of 16 beat it; scores span 0.21-0.33.

The apparent ranking among cells is sweep noise. Re-running the nominal winners
against the default as a **paired** comparison:

| n | default (0.9, 0.995) | challenger | diff | p |
|---|---|---|---|---|
| 4 | 0.3279 | 0.3035 (0.9, 0.9) | +0.0244 | 0.64 |
| 8 | 0.2253 | 0.2076 (0.6, 0.995) | +0.0177 | 0.65 |
| 8 | 0.2253 | 0.2160 (`beta1 = 0`) | +0.0094 | 0.81 |
| 4 | 0.3279 | 0.3588 (`beta1 = 0`) | -0.0309 | 0.62 |

Nothing is distinguishable from the default. Two consequences: the inherited
constants cost us nothing on this setting and need no retuning, and the
`beta1` effect found in Experiment 5A does not generalise beyond the plateau
regime (see the scope note there).

### Experiment 5B -- the Exp 5 "plateau" was a step-size artifact

For each `d`, sweep `zeta` and take the smallest value at which the
**no-momentum** baseline reaches `x*` on >= 80% of restarts at `n = 128`.
Calibrating on the baseline, never on Adam, is what makes the downstream
comparison fair. 16 restarts per cell; multipliers are on the Exp 5
magnitude-matched `zeta`.

| d | zeta_d* | multiplier | what the baseline does |
|---|---|---|---|
| 1 | **none found** | -- | converges to `x ~ +5.9` at every zeta; diverges above x8 |
| 2 | 0.4926 | x0.5 | 100% reach `x*` |
| 4 | 0.3884 | x0.5 | 100% reach `x*` |
| **8** | **6.858** | **x4** | **100% reach `x*`** (0% at x0.5, x1, x2) |
| 16 | none in grid | -- | 62% at x4, divergence by x8 |

**This overturns the Experiment 5 conclusion.** At `d = 8`, four times the
guidance strength makes plain gradient guidance reach the optimum on **100%** of
restarts. So "Adam crosses a plateau that plain gradient guidance cannot" is
**false**: plain guidance crosses it once `zeta` is large enough. What Exp 5
actually compared was Adam against a baseline whose step size was about 4x too
small -- and Adam's normalisation supplies roughly that missing factor.

This is consistent with the `n = 2048` check rather than contradicting it: at a
fixed, too-small `zeta`, more samples cannot help, because the deficiency is in
the step size and not in the gradient estimate.

**The gate fails at `d = 1`.** Chasing it turned up two real defects and one
non-defect.

*Defect 1 (fixed).* `dimy_benchmark.BASE_X[8]` read `-7.0` where the canonical
file has `-8.0`, so `d = 1` was NOT the 2-D benchmark, contrary to the module's
central claim. No test asserted that claim.
`tests/test_dimy_benchmark_d1.py` now pins means, covariances, weights, `x*`,
the target, and the population objective on a grid against
`params/2D_cond_1D_gmm_params.pt`.

*Defect 2 (worked around, not fixed).* Checkpoint filenames key on
`seed` and `dim` but **not on the parameters**, so after the `BASE_X` fix the
re-run silently reloaded a prior trained on the old distribution. The stale
files are renamed `*.STALE_basex7.pt` rather than deleted. The cache key should
include a hash of the parameters; changing it invalidates every checkpoint, so
it is left as a decision rather than done here.

*Not a defect.* The local minimum near `x ~ +6` is **genuine, and it is present
in the paper's own canonical 2-D benchmark** (MMD^2 = 1.135 at `x = +6` against
1.637 at `x = +2` and 1.839 at `x = +7`). Probing the gradient the runner
actually sees at `d = 1` shows the objective is multimodal in `x`, with basins
near `x ~ +1` and `x ~ +6` besides the global one at `x = -5`, and with
`dL/dx < 0` at the `x_T = 0` start -- the guidance pushes *right*, away from the
optimum, from the first step.

So the gate failure is a **protocol** problem, not a benchmark defect: a single
start at `x_T = 0` with a plain zeta-scaled gradient lands in a local minimum,
and raising zeta diverges instead of escaping (0% reached, 75-100% diverged
above x8). Both defects above were fixed or bypassed and the failure persisted
unchanged, which is what identifies the cause as the protocol.

`d = 16` is separately unresolved: its useful window lies between the x4 and x8
multipliers and needs a finer grid.

### Experiment 5 -- what it does and does not establish

**It does NOT answer the dim(Y) question it was built for.** At `d >= 8` the
no-momentum baseline never leaves the diffusion prior's basin: `x_hat ~ 0.39` on
every restart at `n = 4` and at `n = 32`, so its score is essentially identical
across an 8x change in sample count (d = 8: 0.3703 / 0.3771 / 0.3771 / 0.3770;
d = 16: 0.1562 / 0.1558 / 0.1556 / 0.1556). `Delta(d, n)` therefore measures
*escapes a flat region* versus *does not*, and the `n*(d)` it reports is
meaningless. The `d = 1` anchor also fails to reproduce Experiment 3, which is
the clearest sign the calibration is not comparable across `d`.

**The sticking is not finite-sample noise.** At `n = 2048` -- 512x the data of
`n = 4` -- the baseline still escapes **0/8** times, and `x_hat` concentrates
*more* tightly (0.35-0.37) rather than less. The plateau is a feature of the
population landscape: the MMD surface at `d = 8` reads 3.295 at `x = 0`, 3.252 at
`x = -2`, and 0.085 at `x = -4`.

**WHAT IT ESTABLISHES -- CORRECTED BY EXPERIMENT 5B.** The original reading was
that Adam crosses a plateau plain gradient guidance *cannot*. That is wrong: the
baseline crosses it on 100% of restarts at four times the guidance strength, so
the barrier was the step size, not the rule. The surviving, weaker statement is
that **Adam is far less sensitive to the choice of `zeta`** -- its normalisation
rescales the update to a fixed magnitude, so it reaches `x*` at a `zeta` where
the un-normalised baseline stalls. That is a robustness result, not evidence that
momentum unlocks something otherwise unreachable.

### Experiment 8 -- a noise-level trust region on the guidance step

Found by the 2026-08 performance campaign (`experiments/model-optimization/`,
Agent 4 screening of estimator/update rules, verified independently on a
held-out block by Agent 6: `experiments/model-optimization/VERIFICATION.md`).
The rule is a single engine switch, `TFGConfig.temporal.step_clip = "noise"`,
`step_tau = 1`: after the temporal operator and `rho_t` scaling the step is
rescaled so that `||Delta_t|| <= sqrt(1 - alphabar_t)` (line 9's `/sqrt(alpha_t)`
unchanged; direction unchanged). It costs no conditional calls and no
measurable wall time. Numbers below: **held-out block, offset 1000, 100 paired
restarts**, no-LGD, no momentum, `rng=tape`, float32 (the experiments' dtype),
engine path (`exp8_trust_region.py` / `engine_runner.py`). Diff = baseline - rule
(+ = rule better), `*` = permutation p <= 0.05.

| n | 2D baseline | 2D rule | diff | 5D baseline | 5D rule | diff | 10D baseline | 10D rule | diff |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.597 | 0.196 | **+0.401*** | 0.534 | 0.508 | +0.026 | 0.667 | 0.615 | **+0.053*** |
| 8 | 0.418 | 0.168 | **+0.250*** | 0.508 | 0.472 | **+0.036*** | 0.658 | 0.535 | **+0.123*** |
| 16 | 0.282 | 0.192 | **+0.090*** | 0.449 | 0.441 | +0.008 | 0.564 | 0.489 | **+0.075*** |
| 32 | 0.247 | 0.223 | **+0.024*** | 0.444 | 0.434 | +0.010 (p=.06) | 0.477 | 0.457 | +0.019 |

Budget-matched (Pareto) statements, held-out:

- **2D**: `trust_noise1` at `n = 8` (792 conditional draws, 0.168) beats the
  plain baseline at `n = 96` (9504 draws, 0.259) and LGD/none at `n = 32`
  (9504 draws, 0.225) and at `n = 8` (2376 draws, 0.201) -- a >= 3x and up to 12x
  reduction in conditional cost at equal or better L2; success rate at `n = 8`
  rises from ~40 % to ~80 %.
- **10D**: `trust_noise1` at `n = 32` (3168 draws, 0.457) matches the baseline
  at `n = 64` (6336 draws, 0.456), and at `n = 16` (1584 draws, 0.489) beats
  LGD/none at `n = 8` (2376 draws, 0.518) -- a ~2x gain at `n <= 16`, null at
  `n = 32`.
- **5D**: null (all four diffs positive but only `n = 8` significant).

**Limitations.** (i) In 5D the effect is within the seed-noise floor. (ii) The
10D setting is **mis-calibrated at `zeta = 1`**: the unguided chain (`rho = 0`)
scores 0.58, better than the guided no-LGD baseline at `n <= 8` (0.62-0.67) and
than Adam at every `n`, and every rule that merely shrinks the 10D step wins
there while the same rules are catastrophic in 2D. The raw gradient-norm median
is 0.29-0.38 in 10D vs 0.04-0.09 in 2D with the same unit step multiplier, so
the 10D gain partly measures "taming an over-sized step"; a fair dim(X) sweep
needs the per-dimension `zeta_d` of Experiment 5B first. `trust_noise1` is
invariant to that rescaling (the bound is in units of the noise level), which
is why it is the one rule that wins in 10D without losing in 2D. (iii) The
`success` metric (`|x - x*| < 0.5`) is 0 % for every arm in 5D/10D; it was
defined for dim(X) = 1 and is not comparable across settings -- the L2 score is
the metric. Related rules that did **not** transfer (held-out): absolute clips
(`clip0.5`, `relclip1`), `relclip2` (best 2D rule, -0.047 at 10D n=4),
`sqrt_floor` / `sqrtfloor_clip0.5` (pass 2D+5D, regress at 10D n=32); full
verdicts in `experiments/model-optimization/VERIFICATION.md`.

### Discrepancy with earlier runs -- read before citing

An earlier 200-restart held-out run, performed in a different workspace against
the **non-canonical** parameter file `mog_2d_full.txt`, reported a stronger
result: momentum significant at `n = 16` (p = 0.0029) and no-LGD/Adam matching
full-LGD at `n = 8`. Under the canonical parameters used here, both weaken:
`n = 16` becomes null (p = 0.94) and LGD wins at `n = 8`.

The two differ in the joint distribution itself. The old file has an
irreducible objective floor (`L2^2 = 9.9e-5` at its optimum, `x_opt = -4.9984`);
the canonical file's target is realisable (`2.3e-08`, `x_opt = -5.0000328`).
**The canonical numbers in this README supersede the earlier ones.**

### Planned

- Experiment 4 has not been run.
- Robustness models (seeds 20240402/3) have not been run under canonical
  parameters.
- **A genuine dim(Y) sweep.** The paper's 5D and 10D settings scale `dim(X)` and
  keep `dim(Y) = 1` (their `mog_means` is `(2,1,1)`), so they do not test how MMD
  sample complexity scales with the dimension of the distribution being matched.
  Experiment 3 reports `dim(X)` and `dim(Y)` explicitly and makes no dimensional
  claim; Experiment 5 attempted the sweep and does not answer it, for the reason
  documented above. **The question is therefore still open.** Answering it needs a
  calibration that guarantees the no-momentum baseline is a *working optimiser*
  at every `d` -- e.g. the smallest `zeta_d` at which it reaches `x*` at large
  `n`, with `d = 1` required to reproduce Experiment 3 -- so that the comparison
  is about sampling noise rather than plateau escape. The current
  `zeta_d = C / median|g|_d` rule equalises update magnitude but does not achieve
  this.
- A clean accumulation-only arm for Experiment 5A (raw momentum, no division at
  all), to close out the normalisation-vs-accumulation decomposition.
- **Give Experiment 5B an escape mechanism before re-running the dim(Y) sweep.**
  The `d = 1` gate fails on a genuine local minimum, not a benchmark defect, and
  larger `zeta` diverges rather than escapes. The two candidates are the
  noise-level trust region of Experiment 8, which is exactly a way to raise the
  step without divergence, and multi-start / annealed `zeta`. The gate should
  then measure basin-of-attraction rate rather than demanding 80% at one start.
- **Key checkpoints on a parameter hash.** See Defect 2 under Experiment 5B: a
  silently stale prior survived a change to the benchmark distribution.
- `d = 16` needs a finer `zeta` grid between the x4 and x8 multipliers.
- **Re-examine every Adam result against a calibrated baseline.** Exp 5B shows a
  4x change in `zeta` moves the baseline from 0% to 100% success at `d = 8`. No
  reported comparison should treat `zeta` as fixed background until the same
  check has been done on the 2D setting used by Experiments 2, 3, 6 and 7.

## Engine options retained but not evaluated

`tfg` also implements an adaptive `lambda_t` temporal rule, a target-resolution
curriculum (`target_hierarchy.py`), a temporal gradient cache, and
improvement-adaptive recurrence. They are configuration options with tests, but
**no experiment here reports them** and none has a supported result. The
2026-08 campaign added (all opt-in, `src/tfg/README.md`): gradient-norm
clipping variants (`temporal.grad_norm`), the step trust region
(`temporal.step_clip`, Experiment 8), adaptive `n_t`, stale-gradient reuse,
early-stopped recurrence, CRN/antithetic sampling, bandwidth policies and
loss transforms (`tfg/distributional.py`), and the exact cached-target MMD
(`tfg/fast_mmd.py`); only the trust region was promoted
(`experiments/model-optimization/VERIFICATION.md`).
