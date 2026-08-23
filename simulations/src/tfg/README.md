# `tfg` — Generalized Training-Free Guidance

One engine, `engine.py`, implements TFG Algorithm 1 (Ye et al., NeurIPS 2024,
arXiv:2409.15761) with every mechanism exposed as configuration. There is no
second engine: Adam is a *temporal option inside this one*.

`reference.py` is a frozen, deliberately dumb transcription of Algorithm 1. It
exists only so the tests can prove the engine reproduces it. **Do not use it in
experiments.**

---

## The loop

For `t = T … 1`, and `r = 1 … N_recur` within each `t`:

```
x_{0|t} = (x_t − √(1−ᾱ_t)·ε_θ(x_t,t)) / √ᾱ_t
f̃(x)    = E_δ f(x + γ̄·√(1−ᾱ_t)·δ)                    # n_mc Monte-Carlo draws
Δ_t     = ρ_t · Temporal( ∇_{x_t} log f̃(x_{0|t}) )    # ρ branch
Δ_0     = Δ_0 + μ_t · ∇_{x_{0|t}} log f̃(x_{0|t}+Δ_0)   # μ branch, N_iter times
x_{t−1} = DDIM(x_t, x_{0|t}) + Δ_t/√α_t + √ᾱ_{t−1}·Δ_0
x_t     ~ N(√α_t·x_{t−1}, (1−α_t)I)                    # only if r < N_recur
```

Two details that are easy to get wrong and are pinned by tests:

* **Line 9 divides Δ_t by `√α_t`, the per-step alpha — not `√ᾱ_t`.** Confusing
  them costs up to 64× at T=100. (`α_t = ᾱ_t/ᾱ_{t−1}`.)
* **The μ branch differentiates `x_{0|t}` as a leaf** (detached), so it does not
  backpropagate through `ε_θ`. The ρ branch does.

---

## Configuration (`config.py`)

### Core TFG — `TFGConfig`

| field | meaning |
|---|---|
| `T` | diffusion steps |
| `N_recur` | recurrences per step; re-noises when `r < N_recur` |
| `N_iter` | inner mean-guidance iterations on `x_{0|t}` |
| `gamma_bar` | Gaussian smoothing width of the predictor |
| `rho_scalar`, `rho_structure` | ρ branch strength × structure |
| `mu_scalar`, `mu_structure` | μ branch strength × structure |
| `n_mc` | Monte-Carlo draws for the γ̄ smoothing |

`*_structure ∈ {constant, increase, decrease}`; `increase ∝ α_t` (strongest at
the data end), normalised to mean 1 over `t = 1…T`.

Recovering published methods (Ye et al., Theorem 3.2):

| method | configuration |
|---|---|
| DPS | `N_recur=1, N_iter=0, mu=0, gamma_bar=0` |
| LGD | `N_recur=1, N_iter=0, mu=0` |
| FreeDoM | `N_iter=0, mu=0, gamma_bar=0` |
| MPGD | `N_recur=N_iter=1, rho=0, gamma_bar=0` |
| UGD | `gamma_bar=0` |

### Temporal — `TemporalConfig`

| `mode` | behaviour |
|---|---|
| `none` | raw gradient (ordinary TFG) |
| `adam` | AdamDPS adaptive moments |
| `lambda` | adaptive temporal mixing (**retained, not evaluated**) |

`adam` applies the moment update to the ρ-branch gradient **before** ρ_t scaling;
line 9's `/√α_t` then reproduces upstream's `x_prev += guidance/α_t**0.5`
exactly. `inv_sqrt_alpha` is forced off on the Adam object so the factor is not
applied twice. Defaults are the official ones: `β₁=0.9`, `β₂=0.995`, `δ=1e-8`.

### Sample count — `NScheduleConfig`

`n_t = 1 + ⌊(n_max−1)·p_t^κ⌋`, with `p_t` normalised over the **executed** steps
`t = T…1`, so `n_T = 1` and `n_1 = n_max`:

* `time`: `p_t = (T−t)/(T−1)`
* `noise`: `p_t = (ᾱ_t−ᾱ_T)/(ᾱ_1−ᾱ_T)`
* `constant`: `n_max` at every step (legacy behaviour)

Conditional draws are keyed `("eta", t, i)` — shared across recurrences and
loss evaluations within a step, fresh at the next step.

### Retained but unevaluated

`TemporalCacheConfig`, `AdaptiveRecurrenceConfig`, and `target_hierarchy.py`
(the `K_t` curriculum) are available as options. `validate()` raises if the
first two are enabled. No experiment currently reports them.

---

## Modules

| file | role |
|---|---|
| `engine.py` | the unified engine |
| `reference.py` | frozen Algorithm 1, **tests only** |
| `adam_guidance.py` | Adam moment update, verified against the official repo |
| `config.py` | all configuration dataclasses |
| `schedule.py` | cosine ᾱ schedule (float64) and ρ/μ structures |
| `n_schedule.py` | adaptive `n_t` |
| `noise_tape.py` | semantically-keyed RNG |
| `trace.py` | passive intermediate-state recorder |
| `gmm_l2.py` | exact closed-form GMM L2 — **evaluation only** |
| `gmm_mmd.py` | exact population MMD² (multi-bandwidth) |
| `oracle.py` | analytic conditional + paper parameter loading |
| `target_hierarchy.py`, `lambda_deployable.py`, `selection.py` | retained options / helpers |

## Numerical notes

* Everything is float64. The schedule is built here rather than read from
  `Diffusion.py`, whose `betas`/`baralphas` are float32 plain attributes that
  `model.double()` does not convert.
* `alphabar[T] = 0` exactly for the raw cosine, so β is clipped at 0.999
  (Nichol & Dhariwal). No `alphabar` floor is applied: a floor of 1e-4 changes
  `α_T` from 0.001 to 0.41 at T=100, i.e. a different diffusion process.
* `gmm_l2` clamps mixture weights before `log`. Weights of far components
  underflow to exactly 0; `log(0)` leaves the value correct but makes the
  **gradient NaN** near the optimum.
* `NoiseTape` keys with blake2b, not `hash()`, which Python salts per process.
