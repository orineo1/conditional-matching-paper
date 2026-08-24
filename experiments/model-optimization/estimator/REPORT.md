# Agent 4 -- guidance estimator and update rules: report

Commit `6af2081` (branch `tfg-generalization-v2`), 2026-08-23. Python
`/Users/stolk/miniconda3/bin/python`, torch 2.12, CPU. **All numbers in this file
are float32 end to end**, because that is how Experiments 2-7 ran (see
EQUIVALENCE.md, "Notable finding"). Screening was run on the cluster
(`submit_screen.sh`, glacier CPU, 2 threads per cell); 519 cells in `runs/`
(round 1: 258, round 2: 261 new), full per-cell tables with bootstrap CIs,
wins and permutation p in `screening_tables.md`, one row per cell in
`screening_rows.csv` (results.csv columns), compact matrices in
`round2_matrix.md` (`matrix.py`).

## 1. Equivalence outcome (Part A) -- EXACT

`GeneralizedTFG` + the opt-in legacy switches reproduces
`experiments/_guided.py::run` **bit-for-bit** (max abs diff 0.0 on the final `x`
and on every per-step `x_t`, DDIM `x_{t-1}`, `x_{0|t}`) for no-LGD/none,
no-LGD/adam and LGD/none on the real 2D checkpoints (`tests/test_engine_matches_guided.py`,
asserts <= 1e-6, achieved 0.0). The mapping of the eleven differences and the
switch that closes each is in `EQUIVALENCE.md`. The pre-existing 304 tests
still pass (331 now). Every comparison below is engine-vs-engine with the
engine's NoiseTape keying (`rng=tape`); `rng=legacy` rows reproduce the
README's numbers (first 40 restarts of its 100). **Seed-noise floor:**
baseline(tape) vs baseline(legacy) differ by up to 0.08 at 40 restarts, so a
difference below 0.08 without p <= 0.05 is noise.

Engine additions (all opt-in; defaults trace-identical to the frozen reference):
`TFGConfig.init/guidance_scaling/smoothing`; `TemporalConfig.grad_norm in
{clip, unit, clip_rel, clip_quantile}` with `grad_clip`, `clip_ref (median|ema)`,
`clip_ema`; `TemporalConfig.step_clip in {noise, ddim}` with `step_tau`;
`NScheduleConfig.type="adaptive"` (+ policy fields), `eta_per_perturbation`,
`eta_keying="frozen"`; `TemporalCacheConfig.implementation="stale"`;
`AdaptiveRecurrenceConfig.implementation="v1"`. New modules `tfg/adaptive.py`,
`tfg/distributional.py` (`RepositorySchedule`, noise-injectable `CMSampler`
with tape/legacy sources, antithetic pairs, exact per-row cache;
`DistributionalLoss` with bandwidth policies and transforms). Tests:
`tests/test_agent4_candidates.py` (20), `tests/test_engine_matches_guided.py` (6).
Compute accounting: tables use `cm_samples` = actual conditional generator
draws (the engine's `conditional_calls` counts requested samples, 2x for the
agreement policy whose half batches are served from the sampler cache).

## 2. Candidates (Part B) -- pre-registered in `hypotheses/agent4.yaml`

| id | candidate | engine switch | exact / approx |
|---|---|---|---|
| 1a | normalisation-only (`norm_only`) | `temporal.mode=adam, beta1=0, adam_rho=0.4` | exact (update rule) |
| 1b | absolute clipping (`clip0.5`, `clip0.1`) | `temporal.grad_norm=clip` | exact |
| 1c | unit-norm gradient (`unit0.4`, `unit0.1`) | `temporal.grad_norm=unit, rho_scalar=c` | exact |
| 2 | adaptive `n_t` (`adapt_agree0.5/0.8`, `adapt_improve`) | `n_schedule.type=adaptive`, budget `n*T`, `n_min=max(2,n/4)`, `n_max=4n` | exact, same total calls |
| 3 | adaptive recurrence (`recur2_next_state_tweedie`) | `adaptive_recurrence v1`, max 2 | exact; up to 2x calls |
| 4a | common random numbers (`crn`) | `eta_keying=frozen` | **approximate** |
| 4b | antithetic pairs (`antithetic`) | `CMSampler(antithetic=True)` | exact in distribution |
| 5 | bandwidth / transform (`bw_pooled`, `bw_pooled_floor`, `sqrt_abs_eps`, `sqrt_floor`) | `DistributionalLoss` | different objectives |
| 6 | stale gradient (`stale2`, `stale3`) | `temporal_cache stale, refresh_every=k` | **approximate**; 1/k calls |
| 7a | relative clip (`relclip{0.5,1,2}`, `relclip_ema{0.5,1,2}`) | `grad_norm=clip_rel`, threshold = c x running median / EMA(0.9) of PAST raw norms (causal, scale-free) | exact |
| 7b | quantile clip (`qclip{0.5,0.75}`) | `grad_norm=clip_quantile` | exact (`qclip0.5 == relclip1`) |
| 7c | trust region (`trust_noise{0.1,0.3,1}`, `trust_ddim{0.1,0.3,1}`) | `step_clip`, `||Delta_t|| <= tau*sqrt(1-ab_t)` or `tau*||x_ddim-x_t||` | exact |
| 7d | sqrt_floor + clip (`sqrtfloor_clip{0.5,0.1}`, `sqrtfloor_relclip1`) | transform + `grad_norm` | exact |
| 8 | combinations | clip / relclip on the Adam and LGD arms | exact |

Small-n pathologies (`test_pooled_bandwidth_collapses...`, `test_sqrt_transforms_gradient_bounds`):
the repository's pooled bandwidth is a function of the batch (gradient flows
through it; a tiny tight target + collapsed batch drives it to ~1e-6 and the
gradient to >100x the fixed-bandwidth one; `fixed`/`target`/`pooled_floor` are
immune); the SD code's `sqrt(|MMD^2|+eps)` has gradient `1/(2 sqrt eps)` near the
optimum (>10x the floored transform at MMD^2 ~ 1e-6); `sqrt_floor` bounds it by
`1/(2 sqrt c)`, `c = floor_frac*k(0)*(1/n+1/m)`, same asymptote.

## 3. Screening results (Part C), 2D/5D/10D, 40 paired restarts 0..39

Score = failure-penalised mean exact GMM L2 (lower better); "diff" = paired
`base - cand` (+ = candidate better); p = paired permutation. Baseline arms
through the engine (rng=tape; legacy rng in brackets):

| setting | n | no-LGD/none | no-LGD/adam | LGD/none | unguided (rho=0) |
|---|---|---|---|---|---|
| 2D | 4 / 8 / 16 / 32 | 0.585 / 0.381 / 0.298 / 0.252 (0.612 / 0.381 / - / 0.244) | 0.384 / 0.303 / 0.285 / 0.236 | 0.274 / 0.256 / - / 0.229 | 0.597 |
| 5D | 4 / 8 / 16 / 32 | 0.573 / 0.498 / 0.484 / 0.437 (0.583 / 0.505 / - / 0.451) | 0.587 / 0.610 / 0.551 / 0.526 | 0.455 / 0.454 / - / 0.428 | 0.912 |
| 10D | 4 / 8 / 16 / 32 | 0.621 / 0.615 / 0.524 / 0.494 (0.661 / 0.693 / - / 0.493) | 0.656 / 0.631 / 0.570 / 0.567 | 0.624 / 0.523 / - / 0.434 | 0.581 |

(Pareto baselines no-LGD/none: 2D n=2/24/64/96 0.805/0.265/0.255/0.251; 5D
0.643/0.497/0.434/0.435; 10D 0.637/0.501/0.464/0.439. Unguided = `x_T=0` DDIM,
deterministic, one point.)

### 3.1 Round-2 matrix, no-LGD/none (diff; bold p <= 0.05; score in brackets; full matrix incl. round-1 candidates in `round2_matrix.md`)

| rule | 2D n=4 | 2D n=8 | 2D n=16 | 2D n=32 | 5D n=4 | 5D n=8 | 5D n=16 | 5D n=32 | 10D n=4 | 10D n=8 | 10D n=16 | 10D n=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip0.5 | **+.364** (.221) | **+.193** (.187) | **+.053** (.245) | -.000 | -.011 | -.077 | +.002 | -.001 | -.048 | +.052 | -.008 | **-.067** (.561) |
| clip0.1 | -.028 | **-.131** | **-.231** | **-.356** | +.030 | +.062 | **+.056** | +.006 | **+.092** (.528) | **+.122** (.493) | +.045 | +.001 |
| relclip0.5 | +.125 | +.084 | +.005 | **-.226** | -.031 | -.032 | +.000 | **+.031** | +.048 | +.062 | -.017 | **-.072** |
| relclip1 (=qclip0.5) | **+.370** (.214) | **+.267** (.114) | +.067 | -.001 | -.001 | -.039 | +.021 | +.010 | -.032 | +.037 | +.012 | **-.070** |
| **relclip2** | **+.414** (.171) | **+.237** (.144) | **+.113** (.185) | **+.040** (.212) | +.008 | -.037 | **+.010** | **+.009** | -.049 | +.008 | -.024 | +.008 |
| relclip_ema0.5 | **+.267** | +.016 | -.034 | **-.246** | +.008 | -.021 | +.017 | -.011 | +.039 | **+.074** | -.005 | **-.066** |
| relclip_ema1 | **+.370** | **+.236** | **+.139** (.159) | +.015 | -.005 | -.032 | +.008 | **+.023** | +.012 | **+.091** (.524) | +.028 | -.037 |
| **relclip_ema2** | **+.356** (.229) | **+.199** (.182) | **+.108** (.190) | **+.051** (.201) | +.012 | -.008 | -.014 | **+.018** | -.007 | -.015 | +.012 | +.026 |
| qclip0.75 | **+.394** (.191) | **+.178** | **+.084** | +.016 | +.002 | -.058 | -.039 | -.015 | -.044 | +.024 | -.014 | +.016 |
| trust_noise0.1 | -.012 | **-.167** | **-.180** | **-.382** | +.069 | +.076 | **+.096** | **+.067** | **+.107** | **+.145** (.470) | +.066 | +.027 |
| trust_noise0.3 | **+.367** (.218) | **+.197** (.184) | +.079 | -.037 | **+.094** (.479) | +.059 | **+.066** | **+.032** | **+.078** | **+.105** (.510) | +.008 | -.025 |
| **trust_noise1** | **+.407** (.178) | **+.200** (.181) | **+.094** (.204) | +.021 | +.028 | +.034 | **+.040** | +.006 | -.009 | **+.094** (.522) | +.030 | +.038 |
| trust_ddim0.1 | -.011 | **-.215** | **-.298** | **-.344** | **-.341** | **-.416** | **-.428** | **-.470** | **+.179** (.442) | **+.191** (.424) | **+.111** (.413) | **+.082** (.412) |
| trust_ddim0.3 | -.008 | **-.211** | **-.295** | **-.342** | +.075 | -.026 | **-.125** | **-.315** | **+.145** | **+.165** | **+.085** | **+.059** |
| trust_ddim1 | -.040 | **-.286** | **-.394** | **-.455** | +.082 | **+.093** | **+.113** (.371) | **+.085** (.353) | **+.105** | **+.123** | +.052 | +.015 |
| **sqrt_floor** | **+.268** (.317) | +.007 | +.032 | **+.051** (.201) | +.018 | +.009 | **+.043** (.442) | **+.019** (.418) | -.005 | -.001 | +.034 | -.006 |
| sqrtfloor_clip0.5 | **+.329** | **+.149** | **+.062** | **+.041** | +.021 | -.001 | **+.064** (.421) | **+.030** (.408) | -.004 | +.036 | -.011 | **-.074** |
| sqrtfloor_clip0.1 | -.030 | **-.219** | **-.325** | **-.286** | +.055 | +.059 | +.049 | +.005 | **+.097** | **+.121** | +.042 | +.000 |
| sqrtfloor_relclip1 | **+.201** | +.110 | -.037 | -.032 | -.024 | -.046 | +.029 | +.003 | +.019 | +.023 | -.006 | **-.068** |
| unit0.4 / norm_only (r1) | **+.311** / **+.213** | **+.120** / **+.126** | - | +.040 / +.026 | +.047 / +.054 | -.030 / -.068 | - | +.016 / -.019 | -.023 / **-.121** | **+.072** / **-.137** | - | **-.093** / **-.181** |

### 3.2 Does the rule transfer across 2D/5D/10D with ONE constant?

Criterion: never significantly worse (p <= 0.05) in any (dim, n) cell AND a
significant win in at least two dims.

| rule | transfers? | evidence |
|---|---|---|
| **relclip2** (median x 2) | **yes** | never negative; 2D wins at all n (+.41/+.24/+.11/+.04), 5D n=16/32 small wins (+.010/+.009, p<=.05), 10D null (-.05..+.01) |
| **relclip_ema2** (EMA x 2) | **yes** | never negative; 2D all n (+.36/+.20/+.11/+.05), 5D n=32 +.018, 10D null |
| **trust_noise1** (`||Delta|| <= sqrt(1-ab_t)`) | **yes** (best overall) | never negative; 2D +.41/+.20/+.09/+.02, 5D +.03/+.03/**+.04**/+.01, 10D -.01/**+.09**/+.03/+.04 -- the only rule with a significant 10D win that costs nothing in 2D |
| **sqrt_floor** | yes (small) | never negative; 2D n=4 +.27, n=32 +.05; 5D n=16/32 +.04/+.02; 10D null |
| trust_noise0.3 | almost | 2D n=32 -.037 (n.s.), wins in all three dims at n<=8; the most uniform gain at small n (5D n=4 +.09, 10D n=8 +.11) |
| relclip_ema1, qclip0.75, relclip1, clip0.5, unit0.4 | no | 2D wins but a significant 10D n=32 loss (relclip1 -.07, clip0.5 -.07, unit0.4 -.09) or 2D n=32 loss |
| clip0.1, trust_noise0.1, sqrtfloor_clip0.1 | no | 10D/5D wins, catastrophic in 2D (down to -.38) -- the absolute/too-tight threshold |
| trust_ddim{0.1,0.3,1} | no | wins everywhere in 10D (+.08..+.19) and trust_ddim1 in 5D (+.09..+.11), catastrophic in 2D (-.2..-.46) and trust_ddim0.1/0.3 in 5D; with tau=0.1 the 2D/5D scores (0.595/0.91) equal the UNGUIDED ones (0.597/0.912): the trust radius is so small the guidance is effectively off |
| norm_only, relclip0.5, relclip_ema0.5, sqrtfloor_relclip1 | no | significant 10D losses |

Why the scale-free rules transfer and the absolute ones do not: the raw
gradient-norm medians (baseline, no-LGD/none, n=4/8/16/32) are **2D
0.092/0.076/0.047/0.038, 5D 0.25/0.18/0.13/0.09, 10D 0.30/0.38/0.34/0.29**
(the MMD gradient w.r.t. a higher-dimensional x is larger and falls less with
n). `clip0.5` is ~5x the 2D median (a pure tail cap) but ~1.5x the 10D median
(clips the bulk, i.e. shrinks the effective step): it wins in 2D and loses at
10D n=32; `clip0.1` is the reverse. A threshold of 2x the running median (or
`sqrt(1-ab_t)`, which is 1.0 at t=99 and shrinks toward the data end) is a
tail cap in every dim.

### 3.3 Interactions: Adam and LGD arms

no-LGD/adam (clip applied BEFORE the moments; diff vs Adam baseline):

| | 2D n=4 | 2D n=8 | 2D n=16 | 2D n=32 | 5D n=4 | 5D n=8 | 5D n=16 | 5D n=32 | 10D n=4 | 10D n=8 | 10D n=16 | 10D n=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adam baseline | .384 | .303 | .285 | .236 | .587 | .609 | .551 | .526 | .656 | .631 | .570 | .567 |
| + clip0.5 | +.028 | +.021 | +.012 | +.002 | +.008 | +.024 | +.017 | -.018 | -.039 | -.023 | **-.075** | **-.083** |
| + clip0.1 | +.085 | +.044 | +.040 | +.035 | +.002 | +.000 | -.039 | -.058 | -.081 | **-.071** | **-.106** | **-.122** |
| + relclip1 | +.100 | +.044 | +.060 | +.035 | -.001 | -.009 | -.038 | -.022 | +.012 | -.032 | -.047 | -.051 |

Clipping before Adam never helps significantly (Adam already normalises the
tail) and hurts in 10D. Adam itself is worse than plain guidance in 5D/10D
(0.61 vs 0.50 at 5D n=8; 0.57 vs 0.49 at 10D n=32): the ρ=0.4 fixed step is
too large there. **Plain guidance + relclip2 / trust_noise1 beats Adam in every
dim**: 2D n=8 0.144/0.181 vs 0.303, 5D n=8 0.535/0.464 vs 0.609, 10D n=8
0.607/0.522 vs 0.631.

LGD/none (diff vs LGD baseline): 2D n=8/32: clip0.5 +.017/+.001, relclip1
+.013/**-.103**, clip0.1 **-.252/-.339**; 5D: clip0.5 +.021/**+.017**, relclip1
+.007/**+.018**, clip0.1 +.043/+.031; 10D: clip0.5 -.045/**-.154**, relclip1
-.025/**-.154**, clip0.1 +.040/**-.047**. Clipping adds little to LGD (the 3
perturbations already average the tail) and hurts at 10D n=32.

Does clipping make LGD's 3x spend unnecessary? (no-LGD rule at n=8, 792 calls,
vs LGD/none at n=8, 2376 calls, and at n=32, 9504 calls; diff = LGD - rule):

| setting | rule @ n=8 | score | vs LGD n=8 (score) | vs LGD n=32 (score) |
|---|---|---|---|---|
| 2D | baseline | 0.381 | -0.125 (0.256) p=.04 | -0.152 (0.229) p=.006 |
| 2D | clip0.5 | 0.187 | **+0.068** p=.02 | **+0.041** p=.04 |
| 2D | relclip2 | **0.144** | **+0.112** p=.001 | **+0.085** p<.001 |
| 2D | trust_noise1 | 0.181 | **+0.075** p=.02 | **+0.048** p=.02 |
| 5D | baseline | 0.498 | -0.043 p=.24 | -0.069 p=.03 |
| 5D | trust_noise1 | 0.464 | -0.010 p=.86 | -0.036 p=.21 |
| 5D | clip0.1 | 0.436 | +0.018 p=.55 | -0.008 p=.75 |
| 10D | baseline | 0.615 | -0.092 p=.04 | -0.181 p<.001 |
| 10D | trust_noise1 | 0.522 | +0.001 p=.97 | -0.087 p=.007 |
| 10D | clip0.1 | 0.493 | +0.030 p=.32 | -0.058 p=.009 |

In 2D, yes: a clipped no-LGD at n=8 beats LGD at 3x and 12x the calls. In
5D/10D a clipped no-LGD at n=8 matches LGD at n=8 (3x calls) but not LGD at
n=32 (12x).

### 3.4 Budget-matched Pareto (no-LGD/none, score / calls / s per restart on the cluster CPU)

| setting | rule | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---|---|---|
| 2D | baseline (n=2: .805/198; n=24: .265/2376; n=64: .255/6336; n=96: .251/9504) | .585/396/0.68 | .381/792/0.78 | .298/1584/0.69 | .252/3168/0.91 |
| 2D | relclip2 | **.171**/396/0.66 | **.144**/792/0.68 | .185/1584/0.71 | .212/3168/0.98 |
| 2D | trust_noise1 | .178/396/0.70 | .181/792/0.68 | .204/1584/0.70 | .231/3168/0.80 |
| 2D | sqrt_floor | .317/396/0.74 | .374/792/0.73 | .266/1584/0.73 | .201/3168/0.94 |
| 5D | baseline (n=2: .643; n=24: .497; n=64: .434; n=96: .435) | .573/396 | .498/792 | .484/1584 | .437/3168 |
| 5D | trust_noise1 | .545/396 | .464/792 | .444/1584 | .432/3168 |
| 5D | relclip2 | .565/396 | .535/792 | .475/1584 | .428/3168 |
| 5D | sqrt_floor | .555/396 | .489/792 | .442/1584 | .418/3168 |
| 10D | baseline (n=2: .637; n=24: .501; n=64: .464; n=96: .439) | .621/396 | .615/792 | .524/1584 | .494/3168 |
| 10D | trust_noise1 | .630/396 | .522/792 | .494/1584 | .456/3168 |
| 10D | clip0.1 | .528/396 | .493/792 | .479/1584 | .492/3168 |
| 10D | sqrt_floor | .626/396 | .616/792 | .490/1584 | .500/3168 |

Wall time is unchanged by the rules (0.6-1.0 s per restart; all are O(d)
operations on the gradient); conditional calls are identical to the baseline
at the same n. In 2D relclip2 at n=4 (396 calls, 0.171) is better than the
baseline at any n up to 96 (9504 calls, 0.251): a >24x call reduction at equal
or better score. In 5D the frontier moves by ~0.03-0.05 at n <= 16 (trust_noise1
at n=8 ~ baseline at n=32, 4x fewer calls); in 10D trust_noise1 at n=8 ~
baseline at n=24 (3x), clip0.1 at n=4 ~ baseline at n=24 (6x).

### 3.5 The 10D regime: 0 % success at every n -- a calibration problem, not an estimator problem

`success` is `|x_hat - x*| < 0.5` in **dim(x) = 9** (10D setting), which no
arm reaches (LGD at n=32 included); in 5D it is also 0 % everywhere. That
threshold was set for dim(x) = 1 and is not comparable across settings; the
L2 score is the metric. The tell-tale signs that 10D is mis-calibrated
rather than noise-limited: (i) the **unguided** chain (rho = 0, `x_T=0` DDIM,
one deterministic point) scores 0.581, *better* than the guided no-LGD
baseline at n=4/8 (0.621/0.615) and than Adam at every n (0.57-0.66) -- at
zeta = 1 the guidance step is so large relative to the 10-dim landscape that
it degrades the prior at small n; (ii) every rule that simply *shrinks* the
10D step wins there (clip0.1, trust_noise0.1/0.3, trust_ddim*, unit0.1:
+0.08..+0.19), and the same rules are catastrophic in 2D; (iii) the raw
gradient norm is 3-8x the 2D one (0.29-0.38 vs 0.04-0.09) while the step
multiplier is 1 in both. This is exactly the zeta-calibration issue
Experiment 5B addresses (a per-dimension `zeta_d*`): with a fixed zeta = 1
the 10D (and partly 5D) comparisons measure "how much does the rule tame an
over-sized step", not the estimator's sample efficiency. A fair dim(x) sweep
needs `zeta_d` calibrated first (exp5b), after which the scale-free rules
(relclip2, trust_noise1), which are invariant to that rescaling, are the
ones to re-test. Trust_noise1 already behaves like an annealed calibration
(`||Delta|| <= sqrt(1 - ab_t)`), which is why it is the only rule winning in
10D without losing in 2D.

## 4. Regime explanation -- gradient-noise measurements

`grad_noise.py` (`grad_noise.json`): at fixed points `x_t` of 8 baseline
2D trajectories (no-LGD/none, n=8), 64 independent conditional-noise sets
of size `n` each, gradient `g = d MMD^2 / d x_t`. Medians over restarts:

| n | t | SNR = \|E g\|/sd(g) | P(sign of one draw != sign of E g) | \|E g\| | sd(g) | dist to x* |
|---|---|---|---|---|---|---|
| 4 | 90 | 0.11 | 0.35 | 0.068 | 0.50 | 3.2 |
| 4 | 50 | 0.11 | 0.33 | 0.075 | 0.41 | 1.8 |
| 4 | 15 | 0.15 | 0.33 | 0.057 | 0.42 | 0.85 |
| 4 | 1 | 0.07 | 0.42 | 0.020 | 0.38 | 0.71 |
| 8 | 90 | 0.27 | 0.33 | 0.101 | 0.33 | 3.2 |
| 8 | 50 | 0.19 | 0.40 | 0.052 | 0.25 | 1.8 |
| 8 | 15 | 0.25 | 0.29 | 0.048 | 0.19 | 0.85 |
| 8 | 1 | 0.16 | 0.34 | 0.028 | 0.21 | 0.71 |
| 32 | 90 | 0.61 | 0.20 | 0.122 | 0.16 | 3.2 |
| 32 | 50 | 0.44 | 0.33 | 0.033 | 0.088 | 1.8 |
| 32 | 15 | 0.40 | 0.26 | 0.031 | 0.075 | 0.85 |
| 32 | 1 | 0.37 | 0.33 | 0.023 | 0.074 | 0.71 |

Per-run raw gradient-norm medians (baseline, cluster, n=4/8/16/32): 2D
0.092/0.076/0.047/0.038; 5D 0.25/0.18/0.13/0.09; 10D 0.30/0.38/0.34/0.29;
the within-run p90 is ~6-8x the median and the max 100-500x (heavy tail).

Reading: (1) the per-step gradient is **noise dominated at every n** (SNR < 1
everywhere; a single draw has the wrong sign 20-40 % of the time), its noise
`sd(g)` falls like `1/sqrt(n)` while the signal `|E g|` does not grow. (2) The
raw step is `1 * g`: at n=4 a typical step is 0.4-0.5 of pure noise on a 0.05
drift, and the heavy tail gives occasional steps of several units -- the
divergence/penalty tail of the baseline score. Adam replaces `g` by
`m_hat/sqrt(v_hat)` (magnitude O(1) per coordinate x rho = 0.4): it **caps the
tail** and, with beta1 = 0.9, averages ~10 steps, raising the effective SNR by
~3x -- at n=4 from ~0.1 to ~0.35, comparable to the raw n=32 estimator.
(3) At n=32 the raw estimator already has SNR 0.4-0.6 and `sd(g)` ~0.08-0.16:
a 0.4-sized normalised step is now *larger* than the raw gradient most of the
time (median |g| ~0.04), so Adam over-steps near the optimum; this is the
regime flip (benefit = tail cap + averaging when `sd(g) >> |E g|`; cost = the
rho-sized floor when the raw step is already well scaled). (4) Across
dimensions the same mechanism has the other sign: in 10D the raw norms are
3-8x larger, so a fixed-size normalised step (Adam, norm_only, unit0.4) or a
2D-tuned absolute clip is too big, and rules that only remove the tail
relative to the *local* scale (2x running median, `sqrt(1-ab_t)`) are the
ones that transfer. Pure variance reduction that leaves the step rule alone
(antithetic, crn) does nothing measurable; reallocating samples across steps
(adaptive n) or recurrences (recur2) or reusing stale gradients does not
help either -- the binding constraint is the step rule, not the per-step
sample budget.

Round-1 prediction (pre-registered before the cluster run) vs outcome: clip0.5
kept its edge at n=8 and was neutral at n=32 in 2D (correct); unit0.4 /
norm_only did not turn negative at 2D n=32 but did in 10D; clip0.5 at n=8
DID overturn the LGD ordering in 2D (0.187 vs LGD/none 0.256 at 3x calls) --
wrong in the optimistic direction. Round-2 prediction vs outcome: "relclip1
reproduces clip0.5's 2D gain and clip0.1's 10D gain with one constant" --
half right: relclip1 reproduces 2D but loses at 10D n=32 (-0.07); the
transferable constant is 2x the median (or trust_noise1). "trust_noise anneals
the step and should be the most robust at n=32" -- correct (trust_noise1 never
negative). "clipped no-LGD at n=8 matches LGD/none at n=8 in 2D but not in
10D" -- 2D: it beats it; 10D: matches at n=8, not at n=32.

## 5. Verdicts

Survivors (transfer with one constant, never significantly worse):
**trust_noise1** (`step_clip="noise", step_tau=1`), **relclip2**
(`grad_norm="clip_rel", clip_ref="median", grad_clip=2`), **relclip_ema2**,
**sqrt_floor** (small). Best single choice: trust_noise1 (wins in all three
dims at n=8; 2D 0.181, 5D 0.464, 10D 0.522 vs 0.381/0.498/0.615) or relclip2
for the largest 2D gain (0.144 at n=8). Rejected: absolute clips as a
transferable rule, unit/norm-only, Adam+clip, adaptive n, recurrence,
stale-k, crn, antithetic, bandwidth policies, trust_ddim.

## 6. Interaction hypotheses for a combination run

- `trust_noise1 + relclip2` (tail cap on the gradient AND an annealed step
  bound) -- expected >= each alone, especially 5D/10D at n<=8.
- `trust_noise1` with a calibrated `zeta_d` (exp5b) -- the fair dim sweep.
- `sqrt_floor + trust_noise1` -- the SD loss is already a sqrt; a noise-level
  trust region is the one rule that needs no scale constant there.
- `trust_noise1 + LGD` at n=4 -- whether the annealed bound lets LGD use n=4.

## 7. Reproduction

```
# equivalence + candidate unit tests (local, seconds)
cd simulations && /Users/stolk/miniconda3/bin/python -m pytest tests/test_engine_matches_guided.py tests/test_agent4_candidates.py -q -s
# gradient-noise table (local, ~10 min single process)
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/estimator/grad_noise.py
# one screening cell (local, single process)
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/estimator/engine_runner.py --setting 2D --n 8 --spatial no_lgd --temporal none --candidate trust_noise1 --restarts 40 --offset 0 --out ../experiments/model-optimization/estimator/runs/2D_n8_no_lgd_none_trust_noise1_tape.json
# the grids on the cluster (round 1: 258 cells, round 2: 324 cells; done), then the report (~5 min) and matrices
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
N=$(python ../experiments/model-optimization/estimator/screen.py list 2>/dev/null | wc -l)
sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/estimator/submit_screen.sh
N2=$(python ../experiments/model-optimization/estimator/screen.py list --round 2 2>/dev/null | wc -l)
ROUND=2 sbatch --array=0-$((N2-1))%40 ../experiments/model-optimization/estimator/submit_screen.sh
python ../experiments/model-optimization/estimator/screen.py report        # -> screening_rows.csv, screening_tables.md
python ../experiments/model-optimization/estimator/matrix.py               # -> compact matrices (round2_matrix.md built by the same module)
```
Cluster prerequisites: `simulations/` incl. `artifacts/checkpoints/*.pt`
(git-ignored) and `params/`; `experiments/model-optimization/` alongside.

## 8. Integration (post-verification)

Verifier verdicts (`../VERIFICATION.md`): **promoted** `trust_noise1`
(`TFGConfig.temporal.step_clip="noise"`, `step_tau=1.0`); **conditional (2D/5D)**
`sqrt_floor`, `sqrtfloor_clip0.5`; not promoted: `relclip*`, `clip*`, the rest.
Exact-speed: the cached-target MMD (`exact_loss/fast_mmd.py`) is EXACT.

| item | where |
|---|---|
| trust region switch | `simulations/src/tfg/config.py` (`TemporalConfig.step_clip`, `step_tau`; docstring states the order: grad_norm -> temporal -> `rho_t` -> **step_clip** -> line 9 `/sqrt(alpha_t)` unchanged); `simulations/src/tfg/engine.py::_step_clip`; documented in `simulations/src/tfg/README.md` ("Step-size control") |
| loss transforms | `simulations/src/tfg/distributional.py::DistributionalLoss(transform=mmd2|sqrt_abs_eps|sqrt_floor)`; README paragraph |
| exact cached-target MMD | `simulations/src/tfg/fast_mmd.py::MMDFixedTarget` (minimal port: fixed + adaptive bw, 5 kernels, `dist='mm'`, biased V-stat, first-order-exact YY re-attachment); `DistributionalLoss(backend="fast")`; `engine_runner.py --loss {reference,fast}` (default reference) |
| experiment script | `simulations/experiments/exp8_trust_region.py` (engine path; `--setting --n-grid --pareto-n --lgd --restarts --offset`; writes `results/tfg/exp8_trust_region_{setting}.json`); README row + "Experiment 8" results subsection with the held-out numbers, Pareto statements and limitations |
| tests | `simulations/tests/test_fast_mmd_integration.py` (17: value+gradient == `LossFunctions.MMDLoss` at 1e-12 in float64 for unequal n, fixed & adaptive bw; `DistributionalLoss` backends equal; engine_runner `--loss fast` teacher-forced per-step gradients equal the reference to 2.4e-7 in float32 -- full float32 trajectories then differ by up to 0.18 because the loop is chaotic, which is the expected amplification, not a discrepancy); `test_agent4_candidates.py::test_step_trust_region_noise_and_ddim` |

Commands:
```
cd simulations && /Users/stolk/miniconda3/bin/python -m pytest tests -q                       # 348 pass
cd simulations && /Users/stolk/miniconda3/bin/python experiments/exp8_trust_region.py --setting 2D --restarts 100 --offset 1000 --lgd
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/estimator/engine_runner.py --setting 2D --n 8 --candidate trust_noise1 --loss fast --restarts 40
```
