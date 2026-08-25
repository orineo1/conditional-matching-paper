# Back-selection on SD: why kcenter loses (diagnosis from the saved per-step records)

Data: `output/sd_perf/{novis,backsel_k8_uniform,backsel_k8_kcenter,trust}_seed{1..7}/metrics.json`
(per step: MMD, zeta, `correction_norm_raw`, `trust_cap_tau1`, and for backsel arms the
selected indices, `cluster_sizes` and all 32 `||g_i||`). Paired vs novis (7 seeds):
kcenter +0.104 [+0.04, +0.18], uniform +0.039 [+0.005, +0.086], trust -0.011 n.s.

## 1. Trajectories: steady drift AND late catastrophic steps

Mean guided-MMD trajectory over 7 seeds at steps 1/10/20/30/40/50:

| arm | 1 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| novis | 0.535 | 0.501 | 0.469 | 0.455 | 0.443 | 0.439 |
| trust | 0.535 | 0.522 | 0.478 | 0.451 | 0.440 | 0.420 |
| backsel uniform | 0.528 | 0.502 | 0.484 | 0.467 | 0.461 | 0.466 |
| backsel kcenter | 0.535 | 0.510 | 0.493 | 0.486 | 0.484 | **0.548** |

kcenter drifts above novis from step ~10 (bias, +0.03-0.04 through step 40) and then
**blows up in the last 10 steps** (+0.06 between step 40 and 50, while novis still
improves). The two worst seeds show the mechanism step by step:

* seed 3 (final kc 0.594 vs novis 0.411): corrections of 10.4 / 15.0 at steps 40 / 42
  (novis: 2.5 / 3.1) -> guided MMD jumps 0.454 -> 0.584 at step 43 and never recovers
  (0.599, 0.589 at steps 46, 49). Earlier a 16.7 spike at step 14 (novis 1.5) was
  absorbed because there was still noise to re-mix it.
* seed 4 (final kc 0.670 vs 0.371): 8.3 at step 1 with a 19-member cluster (59% of
  the batch behind one Jacobian), 8.7 at step 15 (cluster 14), 14.1 at step 49
  (cluster 13, novis 2.0) -> 0.508 at step 49 -> 0.670 at the final eval.

Late steps are the killers: `abar_{t_prev}` -> 1 there, so a wrong step can no longer
be undone by the remaining denoising — exactly the regime the noise-level trust region
targets.

## 2. Cluster structure at k = 8 of 32 (CLIP space)

Greedy k-center puts its centers on the OUTLIERS (farthest-point rule) and leaves one
huge cluster around the mode: over all 350 kcenter steps the largest cluster averages
**11 of 32 (34%)**, max **19-20 (60%)**, with ~2 singleton clusters per step. So at
every step one representative's Jacobian carries a third (sometimes 60%) of the
batch's output-gradient mass. The per-variation `||g_i||` are essentially uniform
(p90/median 1.2, max/median 1.3 at every step) — the output-space gradients are not
heavy-tailed on SD; the heavy tail is entirely in the Jacobians. Consequently the IS
rule would behave like uniform, and the failure is the Jacobian substitution
`J_i -> J_{r_c}`, not the weighting.

## 3. Correction norms: kcenter (and uniform) take larger steps

Pooled over seeds 1-7, `||correction||_2 = zeta_i ||grad||`:

| arm | steps | median | p90 | max | frac > 2x median | last-10-steps: median / max / frac > 5 |
|---|---|---|---|---|---|---|
| novis | 350 | 1.71 | 3.27 | 11.4 | 0.09 | — |
| trust (tau .25) | 350 | 1.83 | 3.40 | 13.8 | 0.08 | — |
| backsel uniform | 300 | 2.87 | 6.59 | 17.4 | 0.12 | — |
| backsel kcenter | 350 | **3.11** | **7.21** | **22.6** | 0.13 | 2.99 / 22.6 / **0.24** |

Both selection rules roughly double the step size relative to the full-batch gradient
(zeta is normalised by the LOSS, not by the gradient, so any variance/bias in the
gradient estimate translates 1:1 into a bigger latent step). kcenter is the worst:
`||J_r^T (sum_{i in C} g_i)||` with 11-19 nearly parallel `g_i` (recall they have equal
norms and point at the same target mass) is ~|C| times one Jacobian's response, whereas
the true `sum_i J_i^T g_i` has partially cancelling Jacobian components; uniform's
`(N/k) J_i^T g_i` inflates through the same mechanism but with weight 4 and random
membership, i.e. it is unbiased noise rather than coherent bias. A quarter of
kcenter's last-10 steps exceed 5 (3x novis's median).

## 4. Fixes, smallest first

1. **Trust region on top of back-selection** (already implemented: `trust_backsel*`
   arms). tau = 0.25 would clip 8% of kcenter's / 7% of uniform's steps vs 5% of
   novis's — exactly the spikes in sec 1, and above all the late ones. Does NOT fix
   kcenter's steady drift (bias), so pair it with an unbiased rule.
2. **`strat` rule** (implemented in `src/backsel.py`, tested unbiased): balanced
   k-center strata with capacity `ceil(N/k) = 4`, ONE representative drawn uniformly
   per stratum, weight `|C_c| <= 4`. Unbiased (stratified Horvitz-Thompson), weights no
   larger than uniform's N/k, variance <= uniform's (stratification), and no cluster
   can put more than 4 g's behind one Jacobian. `--backsel_rule strat`.
3. Not recommended: capped-size kcenter with aggregation (still biased), mean2
   (2 members per cluster only halves the amplification), or plain kcenter at k=16
   (largest cluster still ~6-10; the k=16 jobs already running will show whether the
   bias shrinks enough).

## 5. Recommendation

* kcenter is dead on SD as a rule (coherent Jacobian-substitution bias + step inflation).
* Run **`trust_backsel_uniform` k=8, seeds 1-8** as the primary extra arm (cost 17 s/step,
  the mechanism that fails is exactly what trust caps); `backsel_k8_strat` seeds 1-8 as
  the secondary (unbiased, structured, same cost) and, if both look good,
  `trust_backsel_strat`. Compare paired vs novis with the same 8 seeds.
* Keep the cost claim (2.6x, -9 GB at k=8) — it is unaffected by the rule.
