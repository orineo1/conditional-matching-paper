# Engine vs `_guided.run` equivalence (Agent 4, Part A)

**Outcome: exact.** With the opt-in legacy switches below, `tfg.engine.GeneralizedTFG`
reproduces `simulations/experiments/_guided.py::run` **bit-for-bit** (max abs diff
`0.000e+00` over the final `x` and every per-step intermediate `x_t`, DDIM
`x_{t-1}`, `x_{0|t}`) for no-LGD/none, no-LGD/adam and LGD/none on the real 2D
checkpoints, restarts 0-2, `n = 8`.  Test:
`simulations/tests/test_engine_matches_guided.py` (asserts `<= 1e-6`, reports the
achieved `0.0`).  The whole test suite: 328 passed (304 before).

Reproduce:
```
cd simulations && /Users/stolk/miniconda3/bin/python -m pytest tests/test_engine_matches_guided.py -q -s
```

## How `_guided.run` differs from Algorithm-1 TFG, and the switch that closes each gap

| # | `_guided.run` (Exp 2-7) | `GeneralizedTFG` default | opt-in switch / object |
|---|---|---|---|
| 1 | `x_T = 0` | `x_T ~ N(0,I)` from the tape (`("x_T",)`) | `TFGConfig.init="zeros"` |
| 2 | schedule = `Diffusion.DiffusionModel` cosine, **float32**, `baralphas[0..99]`, `baralphas[0]=1`, betas `1-ab[t]/ab[t-1]`, no beta clip | `tfg.schedule.DiffusionSchedule`: cosine in float64 with `alphabar[T]=0` -> beta clipped at 0.999, `alphabar` recomputed by `cumprod` | `tfg.distributional.repository_schedule(model, dtype)` -- the model's formula **rebuilt** (not cast) in the requested dtype; in float32 it is bit-identical to `model.baralphas/betas` (`matches_model()`), in float64 it differs from the float32 values by ~1e-7 |
| 3 | loop `t = T-1 .. 1` (99 steps; `T = diffusion_steps = 100`) | loop `t = T .. 1` | the rebuilt schedule has `T = diffusion_steps - 1 = 99` and `alphabar[t] = baralphas[t]`, so the engine's `t=99..1` is the same set of levels; `x_0` is produced from `baralphas[0] = 1` exactly as `sample_ddim_step(t=1)` does |
| 4 | update `x = DDIM(x_t) - g`, no `1/sqrt(alpha_t)` | line 9 adds `Delta_t / sqrt(alpha_t)` (C4a) | `TFGConfig.guidance_scaling="raw"` (+ `rho_scalar = 1`; the engine maximises `log f = -MMD^2`, so `Delta_t = -g` exactly and `x_ddim + Delta_t == x_ddim - g` in IEEE) |
| 5 | LGD perturbation `x0 + r_t * randn`, `r_t = beta_t / sqrt(1+beta_t^2)`, only when `M > 1` | `x0 + gamma_bar * sqrt(1-alphabar_t) * delta`, `delta` keyed `("delta", t, j)` | `TFGConfig.smoothing="lgd_beta"` (+ `n_mc = M = 3`); for `M=1` the default (`gamma_bar = 0`) adds exactly `0 * delta` |
| 6 | `-log((1/M) sum_j exp(-MMD_j^2))` | `logsumexp_j log f_j - log n_mc` (C5) | identical up to sign; the sign flip is exact |
| 7 | Adam on the raw gradient `g`, then `x = x_prev - adam(g)` | Adam on `grad log f = -g` before `rho_t` scaling, then `/sqrt(alpha_t)` | with `guidance_scaling="raw"`, `adam(-g) = -adam(g)` bit-for-bit (sign symmetry of every IEEE op) |
| 8 | conditional noise: `torch.manual_seed(key_seed("cond", restart, t, j))` then (`randn_like(x0)` if `M>1`) then `model_cond.sample` consumes the **global** RNG; independent per perturbation `j` | conditional draws keyed `("eta", t, i)` on the tape, shared across all loss evaluations of the step (C5); the CM sampler itself uses the global RNG | `NScheduleConfig.eta_per_perturbation=True` -> keys `("eta", t, j, i)`; `tfg.distributional.CMSampler` -- a noise-injectable wrapper (`sample_with_noise`) proven identical to `ConsistencyModeliCT.sample` given the same noise (`test_tape_keyed_sampler_agrees_with_manual_seed_path`); `source="legacy"` replays `_guided`'s exact draw order with a private `torch.Generator` (CPU mt19937 == `torch.manual_seed` values), shared with `LegacyTape` which serves the engine's `("delta", t, j)` request from the same stream; `source="tape"` is the order-independent tape keying used for all new experiments |
| 9 | per-step divergence check (`|x| > 50` or non-finite -> stop) | none | the runner applies the same rule post hoc on the traced `x_prev` (`engine_runner.run_engine`) |
| 10 | constant `n`; `time`/`noise` schedules use `DiffusionSchedule(T=100)` progress | `n_at` on the engine's schedule (`T=99`) | not needed for Exp 2-7 (constant `n`); **note** the `time`/`noise` progress variables would differ slightly (normalised over 100 vs 99 steps) -- Exp 4 was never run |
| 11 | dtype: **float32 end to end** (`torch.zeros` default dtype, float32 checkpoints, `S_G.float()`, float32 MMD) | float64 | the runner builds the float32 schedule and tape; `--dtype float64` converts the models and rebuilds the schedule in float64 |

## Notable finding

Experiments 2-7 ran in **float32** end to end (not float64 as the campaign
README assumes for "the repo"): `_guided.run` never sets a dtype, the
checkpoints are float32, and `sample_ddim_step` casts to the model dtype.  The
engine path reproduces this exactly; a float64 engine run (`--dtype float64`)
is a different (more precise) computation and would not reproduce the README's
numbers bit-for-bit.

## Engine switches added (all off by default; default path trace-identical to the frozen reference -- `test_agent4_candidates.py::test_defaults_are_still_the_reference`, and the existing 304 tests)

`TFGConfig.init`, `guidance_scaling`, `smoothing`; `NScheduleConfig.eta_per_perturbation`,
`eta_keying`, `type="adaptive"` (+ policy fields); `TemporalConfig.grad_norm/grad_clip/grad_eps`;
`TemporalCacheConfig.implementation="stale"`, `refresh_every`;
`AdaptiveRecurrenceConfig.implementation="v1"`.  New modules `tfg/adaptive.py`,
`tfg/distributional.py`.  `n_schedule.n_at` now consumes `state` for `kind="adaptive"` only.

## Baseline through the engine

`screen.py` runs every baseline arm twice: `rng=legacy` (reproduces the README
numbers; restarts 0..39 are the first 40 of the README's 100) and `rng=tape`
(the engine's native keyed RNG, used for all paired candidate comparisons).
See REPORT.md.
