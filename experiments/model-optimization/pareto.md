# Held-out Pareto: failure-penalised mean exact L2 vs conditional calls (Agent 7)

Source: `verification/heldout_runs/*.json` (Agent 6; offset 1000, 100 restarts per cell, float32, engine path `estimator/engine_runner.py`, same-node pairs). Score = failure-penalised mean exact GMM L2 (cap 2.0); CI = 95% normal interval of the mean over restarts (not the paired CI -- paired diffs and permutation p are in `verification/heldout_tables.md`). Calls = conditional-model samples per restart (99 steps x M_t x n). Frontier = non-dominated points among the arms plotted. Figure: `pareto.png`.

## 2D

| arm | candidate | n | calls | score | 95% CI of mean | success | diverged | mmd2_eval | frontier |
|---|---|---|---|---|---|---|---|---|---|
| no_lgd/none | relclip2 | 4 | 396 | 0.153 | +/-0.027 | 84% | 0 | 0.086 | **yes** |
| no_lgd/none | trust_noise1 | 4 | 396 | 0.196 | +/-0.036 | 76% | 0 | 0.118 |  |
| no_lgd/none | sqrt_floor | 4 | 396 | 0.365 | +/-0.062 | 51% | 0 | 0.360 |  |
| no_lgd/none | baseline | 4 | 396 | 0.597 | +/-0.081 | 28% | 2 | 0.677 |  |
| no_lgd/none | relclip2 | 8 | 792 | 0.139 | +/-0.026 | 85% | 0 | 0.078 | **yes** |
| no_lgd/none | trust_noise1 | 8 | 792 | 0.167 | +/-0.026 | 80% | 0 | 0.086 |  |
| no_lgd/none | sqrt_floor | 8 | 792 | 0.299 | +/-0.051 | 58% | 0 | 0.187 |  |
| no_lgd/none | baseline | 8 | 792 | 0.418 | +/-0.065 | 40% | 0 | 0.390 |  |
| no_lgd/none | relclip2 | 16 | 1584 | 0.168 | +/-0.019 | 85% | 0 | 0.068 |  |
| no_lgd/none | trust_noise1 | 16 | 1584 | 0.192 | +/-0.019 | 74% | 0 | 0.069 |  |
| no_lgd/none | sqrt_floor | 16 | 1584 | 0.233 | +/-0.041 | 67% | 0 | 0.139 |  |
| no_lgd/none | baseline | 16 | 1584 | 0.282 | +/-0.039 | 48% | 0 | 0.184 |  |
| lgd/none | baseline | 8 | 2376 | 0.201 | +/-0.024 | 64% | 0 | 0.081 |  |
| no_lgd/none | sqrt_floor | 32 | 3168 | 0.209 | +/-0.022 | 72% | 0 | 0.072 |  |
| no_lgd/none | relclip2 | 32 | 3168 | 0.221 | +/-0.023 | 75% | 0 | 0.072 |  |
| no_lgd/none | trust_noise1 | 32 | 3168 | 0.223 | +/-0.014 | 70% | 0 | 0.064 |  |
| no_lgd/none | baseline | 32 | 3168 | 0.247 | +/-0.022 | 59% | 0 | 0.069 |  |
| no_lgd/none | baseline | 64 | 6336 | 0.249 | +/-0.015 | 56% | 0 | 0.067 |  |
| lgd/none | baseline | 32 | 9504 | 0.225 | +/-0.009 | 64% | 0 | 0.063 |  |
| no_lgd/none | baseline | 96 | 9504 | 0.259 | +/-0.010 | 43% | 0 | 0.063 |  |

Frontier (increasing calls): relclip2@n=4 (no_lgd, 396 calls, 0.153); relclip2@n=8 (no_lgd, 792 calls, 0.139)

## 5D

| arm | candidate | n | calls | score | 95% CI of mean | success | diverged | mmd2_eval | frontier |
|---|---|---|---|---|---|---|---|---|---|
| no_lgd/none | sqrt_floor | 4 | 396 | 0.504 | +/-0.039 | 0% | 0 | 0.370 | **yes** |
| no_lgd/none | trust_noise1 | 4 | 396 | 0.508 | +/-0.038 | 0% | 0 | 0.287 |  |
| no_lgd/none | relclip2 | 4 | 396 | 0.524 | +/-0.042 | 0% | 0 | 0.336 |  |
| no_lgd/none | baseline | 4 | 396 | 0.534 | +/-0.045 | 0% | 0 | 0.340 |  |
| no_lgd/none | sqrt_floor | 8 | 792 | 0.459 | +/-0.025 | 0% | 0 | 0.281 | **yes** |
| no_lgd/none | trust_noise1 | 8 | 792 | 0.473 | +/-0.030 | 0% | 0 | 0.267 |  |
| no_lgd/none | relclip2 | 8 | 792 | 0.499 | +/-0.038 | 0% | 0 | 0.280 |  |
| no_lgd/none | baseline | 8 | 792 | 0.508 | +/-0.040 | 0% | 0 | 0.294 |  |
| no_lgd/none | trust_noise1 | 16 | 1584 | 0.441 | +/-0.016 | 0% | 0 | 0.250 | **yes** |
| no_lgd/none | relclip2 | 16 | 1584 | 0.441 | +/-0.017 | 0% | 0 | 0.251 |  |
| no_lgd/none | baseline | 16 | 1584 | 0.449 | +/-0.019 | 0% | 0 | 0.269 |  |
| no_lgd/none | sqrt_floor | 16 | 1584 | 0.453 | +/-0.024 | 0% | 0 | 0.307 |  |
| lgd/none | baseline | 8 | 2376 | 0.467 | +/-0.029 | 0% | 0 | 0.276 |  |
| no_lgd/none | sqrt_floor | 32 | 3168 | 0.421 | +/-0.009 | 0% | 0 | 0.214 | **yes** |
| no_lgd/none | relclip2 | 32 | 3168 | 0.429 | +/-0.009 | 0% | 0 | 0.230 |  |
| no_lgd/none | trust_noise1 | 32 | 3168 | 0.434 | +/-0.009 | 0% | 0 | 0.238 |  |
| no_lgd/none | baseline | 32 | 3168 | 0.444 | +/-0.015 | 0% | 0 | 0.274 |  |
| no_lgd/none | baseline | 64 | 6336 | 0.434 | +/-0.005 | 0% | 0 | 0.249 |  |
| no_lgd/none | baseline | 96 | 9504 | 0.432 | +/-0.003 | 0% | 0 | 0.245 |  |
| lgd/none | baseline | 32 | 9504 | 0.433 | +/-0.013 | 0% | 0 | 0.245 |  |

Frontier (increasing calls): sqrt_floor@n=4 (no_lgd, 396 calls, 0.504); sqrt_floor@n=8 (no_lgd, 792 calls, 0.459); trust_noise1@n=16 (no_lgd, 1584 calls, 0.441); sqrt_floor@n=32 (no_lgd, 3168 calls, 0.421)

## 10D

| arm | candidate | n | calls | score | 95% CI of mean | success | diverged | mmd2_eval | frontier |
|---|---|---|---|---|---|---|---|---|---|
| no_lgd/none | trust_noise1 | 4 | 396 | 0.615 | +/-0.034 | 0% | 0 | 0.281 | **yes** |
| no_lgd/none | baseline | 4 | 396 | 0.667 | +/-0.039 | 0% | 0 | 0.896 |  |
| no_lgd/none | sqrt_floor | 4 | 396 | 0.670 | +/-0.037 | 0% | 0 | 0.337 |  |
| no_lgd/none | relclip2 | 4 | 396 | 0.714 | +/-0.038 | 0% | 0 | 0.528 |  |
| no_lgd/none | trust_noise1 | 8 | 792 | 0.535 | +/-0.031 | 0% | 0 | 0.120 | **yes** |
| no_lgd/none | sqrt_floor | 8 | 792 | 0.610 | +/-0.034 | 0% | 0 | 0.200 |  |
| no_lgd/none | relclip2 | 8 | 792 | 0.640 | +/-0.035 | 0% | 0 | 0.225 |  |
| no_lgd/none | baseline | 8 | 792 | 0.658 | +/-0.040 | 0% | 0 | 0.500 |  |
| no_lgd/none | trust_noise1 | 16 | 1584 | 0.489 | +/-0.026 | 0% | 0 | 0.071 | **yes** |
| no_lgd/none | relclip2 | 16 | 1584 | 0.514 | +/-0.024 | 0% | 0 | 0.086 |  |
| no_lgd/none | sqrt_floor | 16 | 1584 | 0.536 | +/-0.030 | 0% | 0 | 0.128 |  |
| no_lgd/none | baseline | 16 | 1584 | 0.563 | +/-0.035 | 0% | 0 | 0.228 |  |
| lgd/none | baseline | 8 | 2376 | 0.518 | +/-0.027 | 0% | 0 | 0.095 |  |
| no_lgd/none | trust_noise1 | 32 | 3168 | 0.457 | +/-0.018 | 0% | 0 | 0.045 | **yes** |
| no_lgd/none | relclip2 | 32 | 3168 | 0.477 | +/-0.022 | 0% | 0 | 0.060 |  |
| no_lgd/none | baseline | 32 | 3168 | 0.477 | +/-0.025 | 0% | 0 | 0.131 |  |
| no_lgd/none | sqrt_floor | 32 | 3168 | 0.515 | +/-0.023 | 0% | 0 | 0.074 |  |
| no_lgd/none | baseline | 64 | 6336 | 0.456 | +/-0.019 | 0% | 0 | 0.085 | **yes** |
| no_lgd/none | baseline | 96 | 9504 | 0.442 | +/-0.017 | 0% | 0 | 0.083 | **yes** |
| lgd/none | baseline | 32 | 9504 | 0.449 | +/-0.016 | 0% | 0 | 0.085 |  |

Frontier (increasing calls): trust_noise1@n=4 (no_lgd, 396 calls, 0.615); trust_noise1@n=8 (no_lgd, 792 calls, 0.535); trust_noise1@n=16 (no_lgd, 1584 calls, 0.489); trust_noise1@n=32 (no_lgd, 3168 calls, 0.457); baseline@n=64 (no_lgd, 6336 calls, 0.456); baseline@n=96 (no_lgd, 9504 calls, 0.442)

## Compact table (score; calls in header)

| setting | arm | n=4 (396) | n=8 (792) | n=16 (1584) | n=32 (3168) | n=64 (6336) | n=96 (9504) | LGD n=8 (2376) | LGD n=32 (9504) |
|---|---|---|---|---|---|---|---|---|---|
| 2D | baseline | 0.597 | 0.418 | 0.282 | 0.247 | 0.249 | 0.259 | 0.201 | 0.225 |
| 2D | trust_noise1 | 0.196 | 0.167 | 0.192 | 0.223 | - | - | - | - |
| 2D | sqrt_floor | 0.365 | 0.299 | 0.233 | 0.209 | - | - | - | - |
| 2D | relclip2 | **0.153** | **0.139** | 0.168 | 0.221 | - | - | - | - |
| 5D | baseline | 0.534 | 0.508 | 0.449 | 0.444 | 0.434 | 0.432 | 0.467 | 0.433 |
| 5D | trust_noise1 | 0.508 | 0.473 | **0.441** | 0.434 | - | - | - | - |
| 5D | sqrt_floor | **0.504** | **0.459** | 0.453 | **0.421** | - | - | - | - |
| 5D | relclip2 | 0.524 | 0.499 | 0.441 | 0.429 | - | - | - | - |
| 10D | baseline | 0.667 | 0.658 | 0.563 | 0.477 | **0.456** | **0.442** | 0.518 | 0.449 |
| 10D | trust_noise1 | **0.615** | **0.535** | **0.489** | **0.457** | - | - | - | - |
| 10D | sqrt_floor | 0.670 | 0.610 | 0.536 | 0.515 | - | - | - | - |
| 10D | relclip2 | 0.714 | 0.640 | 0.514 | 0.477 | - | - | - | - |

Bold = on the frontier for that setting. The LGD/none columns are shown on the baseline row only.

Reading (verifier numbers, VERIFICATION.md 5.3): 2D frontier is relclip2 at n=4/8 (2D-only rule; trust_noise1 at n=8, 0.167, already beats every baseline point incl. n=96 at 12x the calls and LGD/none at n=8/32); 5D frontier moves by 0.01-0.04 only (trust_noise1 at n=16 is on it; sqrtfloor_clip0.5, not plotted, holds n=4/8/32); 10D frontier is trust_noise1 at every n in 4..32, then the baseline at n=64/96 (trust_noise1 n=32 = baseline n=64 at half the calls).
