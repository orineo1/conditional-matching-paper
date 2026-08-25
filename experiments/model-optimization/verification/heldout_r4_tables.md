# Round-4 held-out (offset 5000, Agent 6): FIFO/cohort replay vs trust_noise1 at equal fresh cost

21 cells; paired diff = trust_noise1@f - candidate@f (+ = candidate better); score = failure-penalised exact GMM L2; mmd2_eval = objective at x_hat (256 fresh draws); 'M-10' = the implementer's offset-4000 estimate of the same diff.


## 2D  (reference trust_noise1@8, 792 calls: score 0.1815)

| f | arm | calls | score | success | div | mmd2_eval | diff L2 vs trust@f | 95% CI | wins | p | diff mmd2_eval | p | M-10 (off 4000) | vs trust@8 diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | trust_noise1 | 198 | 0.2676 | 55% | 0 | 0.1797 |  |  |  |  |  |  |  | -0.0861 (p=0.001) |
| 2 | replay_fifo16_trust | 198 | 0.2298 | 74% | 0 | 0.1369 | +0.0379 | [-0.030, +0.104] | 54/100 | 0.277 | +0.0428 | 0.146 | +0.0890 (p=0.007) | -0.0483 (p=0.081) |
| 2 | replay_cohort16_trust | 198 | 0.2473 | 72% | 0 | 0.1463 | +0.0203 | [-0.043, +0.084] | 55/100 | 0.537 | +0.0335 | 0.232 | +0.0660 (p=0.044) | -0.0658 (p=0.030) |
| 4 | trust_noise1 | 396 | 0.1943 | 71% | 0 | 0.0995 |  |  |  |  |  |  |  | -0.0128 (p=0.506) |
| 4 | replay_fifo16_trust | 396 | 0.2031 | 76% | 0 | 0.0989 | -0.0088 | [-0.058, +0.040] | 50/100 | 0.727 | +0.0006 | 0.976 | -0.0319 (p=0.219) | -0.0216 (p=0.388) |
| 4 | replay_cohort16_trust | 396 | 0.1871 | 73% | 0 | 0.0930 | +0.0073 | [-0.046, +0.059] | 59/100 | 0.785 | +0.0064 | 0.757 | -0.0047 (p=0.833) | -0.0055 (p=0.814) |

call-halving check: replay_fifo16_trust@2 (198) vs trust@4 (396): -0.0354 [-0.097, +0.027] p=0.267

call-halving check: replay_cohort16_trust@2 (198) vs trust@4 (396): -0.0529 [-0.117, +0.011] p=0.112

## 5D  (reference trust_noise1@8, 792 calls: score 0.4884)

| f | arm | calls | score | success | div | mmd2_eval | diff L2 vs trust@f | 95% CI | wins | p | diff mmd2_eval | p | M-10 (off 4000) | vs trust@8 diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | trust_noise1 | 198 | 0.5139 | 0% | 0 | 0.3134 |  |  |  |  |  |  |  | -0.0255 (p=0.330) |
| 2 | replay_fifo16_trust | 198 | 0.5310 | 0% | 0 | 0.3153 | -0.0171 | [-0.064, +0.029] | 46/100 | 0.481 | -0.0019 | 0.905 | +0.0201 (p=0.437) | -0.0426 (p=0.103) |
| 2 | replay_cohort16_trust | 198 | 0.4876 | 0% | 0 | 0.3092 | +0.0263 | [-0.022, +0.073] | 50/100 | 0.279 | +0.0043 | 0.778 | +0.0351 (p=0.116) | +0.0008 (p=0.975) |
| 4 | trust_noise1 | 396 | 0.5099 | 0% | 0 | 0.2840 |  |  |  |  |  |  |  | -0.0215 (p=0.418) |
| 4 | replay_fifo16_trust | 396 | 0.4955 | 0% | 0 | 0.2575 | +0.0144 | [-0.031, +0.060] | 49/100 | 0.539 | +0.0265 | 0.000 | -0.0086 (p=0.694) | -0.0071 (p=0.756) |
| 4 | replay_cohort16_trust | 396 | 0.5120 | 0% | 0 | 0.2713 | -0.0021 | [-0.050, +0.045] | 41/100 | 0.933 | +0.0127 | 0.126 | -0.0122 (p=0.553) | -0.0236 (p=0.330) |

call-halving check: replay_fifo16_trust@2 (198) vs trust@4 (396): -0.0211 [-0.076, +0.033] p=0.453

call-halving check: replay_cohort16_trust@2 (198) vs trust@4 (396): +0.0222 [-0.028, +0.073] p=0.391

## 10D  (reference trust_noise1@8, 792 calls: score 0.5776)

| f | arm | calls | score | success | div | mmd2_eval | diff L2 vs trust@f | 95% CI | wins | p | diff mmd2_eval | p | M-10 (off 4000) | vs trust@8 diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | trust_noise1 | 198 | 0.6720 | 0% | 0 | 0.5181 |  |  |  |  |  |  |  | -0.0944 (p=0.000) |
| 2 | replay_fifo16_trust | 198 | 0.5693 | 0% | 0 | 0.3196 | +0.1028 | [+0.055, +0.150] | 72/100 | 0.000 | +0.1985 | 0.000 | +0.0841 (p=0.000) | +0.0083 (p=0.705) |
| 2 | replay_cohort16_trust | 198 | 0.5921 | 0% | 0 | 0.3468 | +0.0800 | [+0.036, +0.123] | 69/100 | 0.000 | +0.1713 | 0.000 | +0.0497 (p=0.044) | -0.0145 (p=0.511) |
| 4 | trust_noise1 | 396 | 0.6075 | 0% | 0 | 0.2701 |  |  |  |  |  |  |  | -0.0300 (p=0.223) |
| 4 | replay_fifo16_trust | 396 | 0.5338 | 0% | 0 | 0.2418 | +0.0738 | [+0.025, +0.122] | 60/100 | 0.004 | +0.0284 | 0.290 | +0.0821 (p=0.001) | +0.0438 (p=0.031) |
| 4 | replay_cohort16_trust | 396 | 0.5245 | 0% | 0 | 0.2058 | +0.0831 | [+0.039, +0.127] | 61/100 | 0.000 | +0.0644 | 0.011 | +0.0796 (p=0.001) | +0.0531 (p=0.007) |

call-halving check: replay_fifo16_trust@2 (198) vs trust@4 (396): +0.0383 [-0.013, +0.089] p=0.150

call-halving check: replay_cohort16_trust@2 (198) vs trust@4 (396): +0.0155 [-0.036, +0.066] p=0.555
