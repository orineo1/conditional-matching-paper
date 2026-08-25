# Round-3 held-out (offset 2000, Agent 6)

24 cells; paired diff = comparator - candidate (+ = candidate better); score = failure-penalised exact GMM L2; calls = mean fresh conditional samples per run.


## 2D

| cell | calls | score | success | div | mmd2_eval |
|---|---|---|---|---|---|
| trust_noise1@n=4 | 396 | 0.1779 | 82% | 0 | 0.0798 |
| replay_geo0.7d5_trust@n=4 | 99 | 0.2998 | 58% | 0 | 0.1830 |
| trust_noise1@n=8 | 792 | 0.1607 | 82% | 0 | 0.0691 |
| replay_geo0.7d5_trust@n=8 | 297 | 0.1699 | 80% | 0 | 0.0867 |
| baseline@n=8 | 792 | 0.3343 | 48% | 0 | 0.1893 |
| trust_noise1@n=32 | 3168 | 0.2163 | 71% | 0 | 0.0577 |
| replay_geo0.7d5_trust@n=32 | 1089 | 0.1945 | 83% | 0 | 0.0694 |
| baseline@n=32 | 3168 | 0.2313 | 61% | 0 | 0.0549 |

**same-n (replay 1/3-1/4 calls)**

| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |
|---|---|---|---|---|---|---|---|---|
| replay_geo0.7d5_trust@n=4 (99) | trust_noise1@n=4 (396) | 0.1779 -> 0.2998 | -0.1219 | [-0.182, -0.062] | 34/100 | 0.000 | -0.1032 | 0.000 |
| replay_geo0.7d5_trust@n=8 (297) | trust_noise1@n=8 (792) | 0.1607 -> 0.1699 | -0.0092 | [-0.050, +0.030] | 49/100 | 0.658 | -0.0176 | 0.330 |
| replay_geo0.7d5_trust@n=32 (1089) | trust_noise1@n=32 (3168) | 0.2163 -> 0.1945 | +0.0218 | [-0.009, +0.050] | 63/100 | 0.154 | -0.0117 | 0.178 |

**call-matched (fewer or ~equal candidate calls)**

| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |
|---|---|---|---|---|---|---|---|---|
| replay_geo0.7d5_trust@n=4 (99) | trust_noise1@n=4 (396) | 0.1779 -> 0.2998 | -0.1219 | [-0.182, -0.062] | 34/100 | 0.000 | -0.1032 | 0.000 |
| replay_geo0.7d5_trust@n=8 (297) | trust_noise1@n=4 (396) | 0.1779 -> 0.1699 | +0.0080 | [-0.038, +0.052] | 49/100 | 0.732 | -0.0069 | 0.688 |
| replay_geo0.7d5_trust@n=32 (1089) | trust_noise1@n=8 (792) | 0.1607 -> 0.1945 | -0.0338 | [-0.069, -0.000] | 37/100 | 0.054 | -0.0003 | 0.988 |
| replay_geo0.7d5_trust@n=32 (1089) | baseline@n=32 (3168) | 0.2313 -> 0.1945 | +0.0368 | [+0.004, +0.067] | 69/100 | 0.023 | -0.0145 | 0.057 |
| replay_geo0.7d5_trust@n=32 (1089) | baseline@n=8 (792) | 0.3343 -> 0.1945 | +0.1398 | [+0.079, +0.204] | 61/100 | 0.000 | +0.1199 | 0.000 |

## 5D

| cell | calls | score | success | div | mmd2_eval |
|---|---|---|---|---|---|
| trust_noise1@n=4 | 396 | 0.5100 | 0% | 0 | 0.2879 |
| replay_geo0.7d5_trust@n=4 | 99 | 0.5983 | 0% | 0 | 0.3704 |
| trust_noise1@n=8 | 792 | 0.5148 | 0% | 0 | 0.2668 |
| replay_geo0.7d5_trust@n=8 | 297 | 0.5175 | 0% | 0 | 0.2951 |
| baseline@n=8 | 792 | 0.5533 | 0% | 0 | 0.3037 |
| trust_noise1@n=32 | 3168 | 0.4272 | 0% | 0 | 0.2337 |
| replay_geo0.7d5_trust@n=32 | 1089 | 0.4472 | 0% | 0 | 0.2470 |
| baseline@n=32 | 3168 | 0.4436 | 0% | 0 | 0.2530 |

**same-n (replay 1/3-1/4 calls)**

| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |
|---|---|---|---|---|---|---|---|---|
| replay_geo0.7d5_trust@n=4 (99) | trust_noise1@n=4 (396) | 0.5100 -> 0.5983 | -0.0883 | [-0.151, -0.026] | 45/100 | 0.006 | -0.0825 | 0.000 |
| replay_geo0.7d5_trust@n=8 (297) | trust_noise1@n=8 (792) | 0.5148 -> 0.5175 | -0.0027 | [-0.053, +0.047] | 48/100 | 0.915 | -0.0283 | 0.004 |
| replay_geo0.7d5_trust@n=32 (1089) | trust_noise1@n=32 (3168) | 0.4272 -> 0.4472 | -0.0200 | [-0.040, -0.004] | 40/100 | 0.027 | -0.0133 | 0.007 |

**call-matched (fewer or ~equal candidate calls)**

| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |
|---|---|---|---|---|---|---|---|---|
| replay_geo0.7d5_trust@n=4 (99) | trust_noise1@n=4 (396) | 0.5100 -> 0.5983 | -0.0883 | [-0.151, -0.026] | 45/100 | 0.006 | -0.0825 | 0.000 |
| replay_geo0.7d5_trust@n=8 (297) | trust_noise1@n=4 (396) | 0.5100 -> 0.5175 | -0.0075 | [-0.059, +0.043] | 37/100 | 0.777 | -0.0072 | 0.583 |
| replay_geo0.7d5_trust@n=32 (1089) | trust_noise1@n=8 (792) | 0.5148 -> 0.4472 | +0.0676 | [+0.033, +0.105] | 49/100 | 0.000 | +0.0198 | 0.000 |
| replay_geo0.7d5_trust@n=32 (1089) | baseline@n=32 (3168) | 0.4436 -> 0.4472 | -0.0036 | [-0.026, +0.018] | 54/100 | 0.763 | +0.0061 | 0.222 |
| replay_geo0.7d5_trust@n=32 (1089) | baseline@n=8 (792) | 0.5533 -> 0.4472 | +0.1061 | [+0.061, +0.155] | 55/100 | 0.000 | +0.0567 | 0.000 |

## 10D

| cell | calls | score | success | div | mmd2_eval |
|---|---|---|---|---|---|
| trust_noise1@n=4 | 396 | 0.6118 | 0% | 0 | 0.2910 |
| replay_geo0.7d5_trust@n=4 | 99 | 0.6563 | 0% | 0 | 0.5197 |
| trust_noise1@n=8 | 792 | 0.5161 | 0% | 0 | 0.1290 |
| replay_geo0.7d5_trust@n=8 | 297 | 0.6215 | 0% | 0 | 0.3263 |
| baseline@n=8 | 792 | 0.6728 | 0% | 0 | 0.3505 |
| trust_noise1@n=32 | 3168 | 0.4434 | 0% | 0 | 0.0548 |
| replay_geo0.7d5_trust@n=32 | 1089 | 0.5237 | 0% | 0 | 0.1156 |
| baseline@n=32 | 3168 | 0.4694 | 0% | 0 | 0.1181 |

**same-n (replay 1/3-1/4 calls)**

| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |
|---|---|---|---|---|---|---|---|---|
| replay_geo0.7d5_trust@n=4 (99) | trust_noise1@n=4 (396) | 0.6118 -> 0.6563 | -0.0445 | [-0.099, +0.011] | 40/100 | 0.120 | -0.2287 | 0.000 |
| replay_geo0.7d5_trust@n=8 (297) | trust_noise1@n=8 (792) | 0.5161 -> 0.6215 | -0.1054 | [-0.146, -0.063] | 27/100 | 0.000 | -0.1973 | 0.000 |
| replay_geo0.7d5_trust@n=32 (1089) | trust_noise1@n=32 (3168) | 0.4434 -> 0.5237 | -0.0803 | [-0.111, -0.051] | 26/100 | 0.000 | -0.0608 | 0.000 |

**call-matched (fewer or ~equal candidate calls)**

| candidate (calls) | comparator (calls) | scores comp -> cand | diff L2 | 95% CI | wins | p | diff mmd2_eval | p |
|---|---|---|---|---|---|---|---|---|
| replay_geo0.7d5_trust@n=4 (99) | trust_noise1@n=4 (396) | 0.6118 -> 0.6563 | -0.0445 | [-0.099, +0.011] | 40/100 | 0.120 | -0.2287 | 0.000 |
| replay_geo0.7d5_trust@n=8 (297) | trust_noise1@n=4 (396) | 0.6118 -> 0.6215 | -0.0096 | [-0.055, +0.036] | 49/100 | 0.681 | -0.0353 | 0.185 |
| replay_geo0.7d5_trust@n=32 (1089) | trust_noise1@n=8 (792) | 0.5161 -> 0.5237 | -0.0076 | [-0.044, +0.029] | 48/100 | 0.686 | +0.0134 | 0.500 |
| replay_geo0.7d5_trust@n=32 (1089) | baseline@n=32 (3168) | 0.4694 -> 0.5237 | -0.0543 | [-0.093, -0.016] | 44/100 | 0.007 | +0.0025 | 0.882 |
| replay_geo0.7d5_trust@n=32 (1089) | baseline@n=8 (792) | 0.6728 -> 0.5237 | +0.1491 | [+0.104, +0.194] | 70/100 | 0.000 | +0.2349 | 0.000 |
