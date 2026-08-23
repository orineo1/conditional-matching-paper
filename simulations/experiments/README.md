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
| 6 | `exp6_mpgd.py` | Does MPGD-style guidance help, and does momentum help on top of it? | 2x2: {`x_t`, `x0` (MPGD)} x {none, Adam} | exact GMM L2, success, paired CI | **run, negative** | `python experiments/exp6_mpgd.py --n 8 --restarts 100` | `results/tfg/exp6_mpgd_*.json` |

Experiment 1 is cheap (seconds). Experiments 2-6 train or load models on first
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

### Experiment 5A -- the mechanism is normalisation, not momentum

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
the `beta1` accumulation term actively hurts here. So what the other experiments
loosely call "momentum helping" is really **per-coordinate normalisation**:
dividing by `sqrt(v_hat)` turns a tiny but persistent gradient into a full-size
`~rho` step instead of a `~1e-3` one, which is what crosses a shelf several units
wide in 99 steps. Accumulation is not what does the work.

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

**What it does establish** (and this is a real result, just a different one):
Adam crosses a plateau that plain gradient guidance never crosses, and crosses it
in the *correct* direction -- landing at -5.12, -5.15, -5.01, -4.98, -4.94 on
roughly 60-75% of restarts. That is an optimisation result about the guidance
rule, not a sample-complexity result about MMD. Experiment 5A isolates which part
of Adam is responsible.

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

## Engine options retained but not evaluated

`tfg` also implements an adaptive `lambda_t` temporal rule, a target-resolution
curriculum (`target_hierarchy.py`), a temporal gradient cache, and
improvement-adaptive recurrence. They are configuration options with tests, but
**no experiment here reports them** and none has a supported result. See
`src/tfg/README.md`.
