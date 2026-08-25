# B-R6b stage 2 -- end-to-end quality at n = 128 (corrected protocol)

score = failure-penalised exact L2 (lower better; 10D: GMM L2 of _guided.evaluate, dimY/nuis: exp5 population L2), paired over the same restarts; diff = comparator - arm (+ = arm better), bootstrap 95% CI, permutation p.  fwd = conditional forward samples per run, diff_s = differentiated samples per run.  NOTE: in dimy16/nuis16 the conditional is the ORACLE with a common Jacobian, so kcenter aggregation is exact by construction there -- those cells test the protocol, not the approximation; 10D is the informative setting.


## 10D (dim x=9, dim y=1, zeta=4, x_init=randn, step_clip=noise tau=1, zeta=protocol/zeta_star.json 10D.trust (l2min, n=128, trust on); R=100, offset 9000)

| arm | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | vs full128 diff [CI] p | vs fresh-k diff [CI] p |
|---|---|---|---|---|---|---|---|---|---|
| full128 | 0.5011 | 0% | 0 | 12672 | 12672 | 1.29 | 1697 | - | - |
| kcenter16 | 0.4860 | 0% | 0 | 14256 | 1584 | 1.32 | 2108 | +0.015 [-0.015,+0.045] p=0.323 | +0.035 [-0.005,+0.075] p=0.089 |
| kcenter32 | 0.5025 | 0% | 0 | 15840 | 3168 | 1.43 | 2245 | -0.001 [-0.027,+0.024] p=0.920 | -0.001 [-0.034,+0.031] p=0.943 |
| kcenter_mean2_16 | 0.4784 | 0% | 0 | 14202 | 1530 | 1.40 | 2463 | +0.023 [-0.004,+0.050] p=0.106 | +0.043 [+0.007,+0.078] p=0.021 |
| uniform32 | 0.4868 | 0% | 0 | 15840 | 3168 | 1.37 | 4078 | +0.014 [-0.014,+0.044] p=0.335 | +0.014 [-0.018,+0.047] p=0.389 |
| fresh16 | 0.5210 | 0% | 0 | 1584 | 1584 | 0.67 | 4078 | -0.020 [-0.058,+0.018] p=0.307 | - |
| fresh32 | 0.5013 | 0% | 0 | 3168 | 3168 | 0.76 | 4078 | -0.000 [-0.033,+0.033] p=0.994 | - |

kcenter16 diagnostic (9900 steps): cluster size min/median/max = 1/7/47, per-step max size median 22, singleton fraction 0.11; selection stability between consecutive steps: index-Jaccard median 0.067 (chance ~0.067, noise is fresh per step), center-set displacement / within-step center spacing median 0.063 [p10 0.048, p90 0.105].

## 10D_z003 (dim x=9, dim y=1, zeta=0.03125, x_init=randn, step_clip=noise tau=1, zeta=SENSITIVITY: exp12 none-arm 0.03125 (not the campaign calibration); R=100, offset 9000)

| arm | score | succ | div | fwd/run | diff_s/run | s/run | RSS MB | vs full128 diff [CI] p | vs fresh-k diff [CI] p |
|---|---|---|---|---|---|---|---|---|---|
| full128 | 0.6080 | 0% | 0 | 12672 | 12672 | 1.07 | 1694 | - | - |
| kcenter16 | 0.6082 | 0% | 0 | 14256 | 1584 | 1.07 | 2008 | -0.000 [-0.003,+0.002] p=0.876 | +0.005 [-0.004,+0.014] p=0.322 |
| kcenter32 | 0.6095 | 0% | 0 | 15840 | 3168 | 1.15 | 2145 | -0.002 [-0.004,+0.001] p=0.169 | +0.002 [-0.004,+0.010] p=0.654 |
| kcenter_mean2_16 | 0.6115 | 0% | 0 | 14199 | 1527 | 1.12 | 2314 | -0.004 [-0.012,+0.004] p=0.449 | +0.001 [-0.005,+0.008] p=0.641 |
| uniform32 | 0.6064 | 0% | 0 | 15840 | 3168 | 1.10 | 4244 | +0.002 [-0.001,+0.005] p=0.292 | +0.005 [-0.001,+0.013] p=0.134 |
| fresh16 | 0.6130 | 0% | 0 | 1584 | 1584 | 0.54 | 4244 | -0.005 [-0.014,+0.003] p=0.276 | - |
| fresh32 | 0.6118 | 0% | 0 | 3168 | 3168 | 0.61 | 4244 | -0.004 [-0.012,+0.002] p=0.347 | - |

kcenter16 diagnostic (9900 steps): cluster size min/median/max = 1/7/53, per-step max size median 20, singleton fraction 0.11; selection stability between consecutive steps: index-Jaccard median 0.067 (chance ~0.067, noise is fresh per step), center-set displacement / within-step center spacing median 0.059 [p10 0.046, p90 0.082].
