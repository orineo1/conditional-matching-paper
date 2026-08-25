# B-R6b stage 2 -- end-to-end quality at n = 128 (corrected protocol)

score = failure-penalised exact L2 (lower better; 10D: GMM L2 of _guided.evaluate, dimY/nuis: exp5 population L2), paired over the same restarts; diff = comparator - arm (+ = arm better), bootstrap 95% CI, permutation p.  fwd = conditional forward samples per run, diff_s = differentiated samples per run.  NOTE: in dimy16/nuis16 the conditional is the ORACLE with a common Jacobian, so kcenter aggregation is exact by construction there -- those cells test the protocol, not the approximation; 10D is the informative setting.


## 10D (dim x=9, dim y=1, zeta=4, x_init=randn, step_clip=noise tau=1, zeta=protocol/zeta_star.json 10D.trust (l2min, n=128, trust on); R=5, offset 9000)

| arm | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | vs full128 diff [CI] p | vs fresh-k diff [CI] p |
|---|---|---|---|---|---|---|---|---|---|
| full128 | 0.5583 | 0% | 0 | 12672 | 12672 | 0.33 | 366 | - | - |
| kcenter16 | 0.5739 | 0% | 0 | 14256 | 1584 | 0.31 | 374 | -0.016 [-0.099,+0.077] p=0.691 | -0.028 [-0.116,+0.060] p=0.627 |
| kcenter32 | 0.4443 | 0% | 0 | 15840 | 3168 | 0.31 | 384 | +0.114 [-0.023,+0.327] p=0.314 | -0.001 [-0.121,+0.138] p=0.936 |
| kcenter_mean2_16 | 0.4433 | 0% | 0 | 14213 | 1541 | 0.29 | 390 | +0.115 [+0.010,+0.303] p=0.065 | +0.102 [-0.013,+0.258] p=0.318 |
| uniform32 | 0.5548 | 0% | 0 | 15840 | 3168 | 0.28 | 397 | +0.003 [-0.218,+0.275] p=0.940 | -0.112 [-0.188,-0.036] p=0.065 |
| fresh16 | 0.5458 | 0% | 0 | 1584 | 1584 | 0.11 | 398 | +0.012 [-0.054,+0.085] p=0.877 | - |
| fresh32 | 0.4431 | 0% | 0 | 3168 | 3168 | 0.14 | 400 | +0.115 [-0.134,+0.354] p=0.445 | - |

kcenter16 diagnostic (495 steps): cluster size min/median/max = 1/6/41, per-step max size median 23, singleton fraction 0.10; selection stability between consecutive steps: index-Jaccard median 0.067 (chance ~0.067, noise is fresh per step), center-set displacement / within-step center spacing median 0.064 [p10 0.047, p90 0.138].

## dimy16 (dim x=1, dim y=16, zeta=1.367, x_init=randn, step_clip=noise tau=1, zeta=exp5b_v2; R=2, offset 9000)

| arm | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | vs full128 diff [CI] p | vs fresh-k diff [CI] p |
|---|---|---|---|---|---|---|---|---|---|
| full128 | 0.1555 | 0% | 0 | 12672 | 12672 | 0.33 | 311 | - | - |
| kcenter16 | 0.1555 | 0% | 0 | 14256 | 1584 | 1.09 | 364 | +0.000 [+0.000,+0.000] p=1.000 | +0.000 [+0.000,+0.000] p=0.498 |
| kcenter32 | 0.1555 | 0% | 0 | 15840 | 3168 | 0.50 | 364 | +0.000 [+0.000,+0.000] p=1.000 | -0.000 [-0.000,-0.000] p=0.498 |
| kcenter_mean2_16 | 0.1555 | 0% | 0 | 14208 | 1536 | 0.42 | 364 | +0.000 [+0.000,+0.000] p=1.000 | +0.000 [+0.000,+0.000] p=0.498 |
| uniform32 | 0.1555 | 0% | 0 | 15840 | 3168 | 0.42 | 364 | -0.000 [-0.000,-0.000] p=0.498 | -0.000 [-0.000,-0.000] p=0.498 |
| fresh16 | 0.1557 | 0% | 0 | 1584 | 1584 | 0.23 | 364 | -0.000 [-0.000,-0.000] p=0.498 | - |
| fresh32 | 0.1553 | 0% | 0 | 3168 | 3168 | 0.25 | 364 | +0.000 [+0.000,+0.000] p=0.498 | - |

kcenter16 diagnostic (198 steps): cluster size min/median/max = 1/6/58, per-step max size median 23, singleton fraction 0.08; selection stability between consecutive steps: index-Jaccard median 0.067 (chance ~0.067, noise is fresh per step), center-set displacement / within-step center spacing median 0.123 [p10 0.112, p90 0.189].

## nuis16 (dim x=1, dim y=16, zeta=1.367, x_init=randn, step_clip=noise tau=1, zeta=exp5b_v2; R=2, offset 9000)

| arm | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | vs full128 diff [CI] p | vs fresh-k diff [CI] p |
|---|---|---|---|---|---|---|---|---|---|
| full128 | 0.0143 | 50% | 0 | 12672 | 12672 | 0.32 | 316 | - | - |
| kcenter16 | 0.0143 | 50% | 0 | 14256 | 1584 | 0.44 | 320 | +0.000 [+0.000,+0.000] p=1.000 | +0.001 [+0.000,+0.002] p=0.498 |
| kcenter32 | 0.0143 | 50% | 0 | 15840 | 3168 | 0.43 | 328 | +0.000 [+0.000,+0.000] p=1.000 | +0.001 [-0.000,+0.002] p=1.000 |
| kcenter_mean2_16 | 0.0143 | 50% | 0 | 14245 | 1573 | 0.44 | 337 | +0.000 [+0.000,+0.000] p=1.000 | +0.001 [+0.000,+0.002] p=0.498 |
| uniform32 | 0.0142 | 50% | 0 | 15840 | 3168 | 0.39 | 338 | +0.000 [+0.000,+0.000] p=0.498 | +0.001 [+0.000,+0.002] p=0.498 |
| fresh16 | 0.0155 | 50% | 0 | 1584 | 1584 | 0.23 | 341 | -0.001 [-0.002,-0.000] p=0.498 | - |
| fresh32 | 0.0153 | 50% | 0 | 3168 | 3168 | 0.24 | 341 | -0.001 [-0.002,+0.000] p=1.000 | - |

kcenter16 diagnostic (198 steps): cluster size min/median/max = 1/6/58, per-step max size median 26, singleton fraction 0.06; selection stability between consecutive steps: index-Jaccard median 0.067 (chance ~0.067, noise is fresh per step), center-set displacement / within-step center spacing median 0.456 [p10 0.331, p90 0.581].
