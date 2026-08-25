# Importance-selected backprop screening (Agent B)_smoke

score = failure-penalised mean exact GMM L2 (lower better); diff = paired comparator - candidate (+ = candidate better); p = paired permutation.  fwd = conditional forward samples per run (cm_samples), diff_s = DIFFERENTIATED samples per run (graphs + backward).


## 2D

| n | candidate | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | comparator | diff (p) | vs baseline diff (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | baseline | 0.3008 | 30% | 0 | 792 | 792 | 0.11 | 310 | - | - | - |
| 8 | trust_noise1 | 0.2243 | 70% | 0 | 792 | 792 | 0.11 | 313 | baseline | +0.077 (p=0.311) | +0.077 (p=0.311) |
| 8 | backsel_uni_k2_trust | 0.1364 | 80% | 0 | 990 | 198 | 0.15 | 316 | trust_noise1 | +0.088 (p=0.256) | +0.164 (p=0.033) |
| 8 | backsel_is_k2_trust | 0.2367 | 80% | 0 | 974 | 182 | 0.15 | 318 | trust_noise1 | -0.012 (p=0.911) | +0.064 (p=0.460) |
| 8 | backsel_clust_k2_trust | 0.1211 | 90% | 0 | 990 | 198 | 0.15 | 315 | trust_noise1 | +0.103 (p=0.271) | +0.180 (p=0.027) |
| 8 | replay_cohort16_trust | 0.3273 | 60% | 0 | 792 | 792 | 0.11 | 309 | trust_noise1 | -0.103 (p=0.393) | -0.026 (p=0.820) |
| 8 | backsel_is_k4_cohort16_trust | 0.1955 | 60% | 0 | 1106 | 314 | 0.15 | 316 | replay_cohort16_trust | +0.132 (p=0.211) | +0.105 (p=0.123) |
