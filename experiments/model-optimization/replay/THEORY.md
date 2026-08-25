# Sample-replay MMD: theory

Agent M, round 3 of the CDM/TFG performance campaign, 2026-08-24. Notation
follows `estimator/REPORT.md`: at outer step `t` the guidance loss is the
biased V-statistic `MMD^2_V(X_t, Y)` between the `n` conditional samples
`X_t = {x_i = G(x_{0|t}, eta_i)}` (differentiable in `x_t` through
`x_{0|t}`) and the fixed target set `Y` (`m = 250` rows, detached), with the
5-bandwidth RBF kernel `k` at a fixed bandwidth.

## 1. The two replay estimators

Let `F` be the `f` fresh rows generated at step `t` (differentiable), and for
`k = 1..D` let `C_k` be the cache of rows generated at step `t+k` (the loop
runs t downward), stored **detached** -- their graphs were freed at their own
step, so they are constants.

**(a) Subsampled replay.** Draw `r_k` rows `R_k` (uniform, without
replacement, NoiseTape-keyed `("replay", t, j, k)`) from `C_k`, stack
`X = [F; R_1; ...; R_D]`, `B = f + sum r_k` rows, and evaluate the ordinary
V-statistic

    L_sub = (1/B^2) sum_{ij} k(x_i, x_j) - (2/(Bm)) sum_{i,a} k(x_i, y_a) + YY.

Counts are geometric: unnormalised step weights `w_k ~ lambda^k`, `k = 0..D`;
in `batch` mode `B` is fixed (the baseline's `n`) and
`r_k = round_LR(B * w_k / sum w)` for `k >= 1` with `f = B - sum r_k`
(largest-remainder rounding, so Ori's `reuse_frac = p` is `D = 1`,
`lambda = p/(1-p)`); in `augment` mode `f = n` is kept and
`r_k = round(n * lambda^k)`, `B = f + sum r_k > n`.

**(b) Weighted replay.** Use every cached row with per-row weights instead of
subsampling. Give group `k` total weight `W_k = w_k / sum_j w_j` (only over
groups actually available), split uniformly inside the group
(`omega_i = W_k / |C_k|` for a row of group `k`; group 0 is `F`). With
`sum_i omega_i = 1` the weighted V-statistic is

    L_w = sum_{ij} omega_i omega_j k(x_i, x_j)
          - 2 sum_i omega_i * (1/m) sum_a k(x_i, y_a) + YY.

`L_sub` with counts `r_k ~ B W_k` is a randomised, equal-weight quadrature of
`L_w`: for rows in different groups
`E[ (1/B^2) 1{i,j drawn} k ] = (r_k r_l / B^2) E[k] ~ W_k W_l E[k]`; the only
systematic difference is inside a group, where sampling without replacement
gives pair weight `r_k(r_k-1)/B^2` on off-diagonal pairs and `r_k/B^2` on the
diagonal, versus `W_k^2` spread over all pairs including the diagonal -- an
`O(1/r_k)` discrepancy in the (constant-in-`x_t`) XX block plus the usual
subsampling variance. `tests/test_replay.py::test_weighted_matches_subsampled_expectation`
checks the agreement statistically.

## 2. Gradient: only the fresh rows are differentiable

Split the batch as `X = [F; R]` with `R` constant. Then

    dL_sub/dx_t = sum_{i in F} (dL/dx_i)^T dx_i/dx_t,

    dL/dx_i = (2/B^2) sum_{j in F u R} d/dx_i k(x_i, x_j)      (XX row, incl. self)
              - (2/(Bm)) sum_a d/dx_i k(x_i, y_a).             (XY row)

Compare the fresh-only estimator at batch `f` (what the baseline computes with
`n = f`): there the same row has XX weight `2/f^2` and XY weight `2/(fm)`.
So replay does exactly three things to the gradient:

1. **Rescales the differentiable mass.** Each fresh row's XY (target
   attraction) coefficient shrinks by `f/B` x `(f/B)`-worth of rows: the total
   XY gradient is `(f/B)` times a fresh-only batch's (per-row `1/(Bm)` vs
   `1/(fm)`, `f` rows either way). Equivalently `L_sub = (f/B)^2 * MMD-like(F)
   + cross terms + const`: replay at fixed `zeta` implicitly multiplies the
   guidance step by ~`f/B` **only through the XY term**. This is a real (and
   benign) interaction: the campaign showed step-size control dominates, and
   `trust_noise1` on top makes the comparison step-size-fair.
2. **Adds fresh-replay XX cross terms.** `(2/B^2) sum_{j in R} dk(x_i, x_j)/dx_i`
   is a repulsion of the fresh rows away from the replayed rows (for RBF,
   `dk/dx_i = -2(x_i - x_j)/bw_l * k`, pointing away). Since `R` is (approximately)
   distributed like the recent conditional law, this is the correct
   self-repulsion term of the V-statistic evaluated against the *smoothed*
   batch: it penalises the fresh rows for collapsing onto where the
   conditional mass already is, exactly as the fresh-fresh XX term does, but
   against a larger, lower-variance sample of that law.
3. **Leaves the target attraction direction unchanged** -- the XY term is
   computed at the current fresh rows, so the gradient still points from the
   *current* `x_{0|t}`'s samples toward the target.

Nothing else: no gradient flows through `R`, and `YY` is a constant.

## 3. Bias: a trajectory-smoothed (EMA) objective

Taking expectation over conditional noise, `L_w` (and `L_sub` up to the
`O(1/r_k)` term above) estimates

    MMD^2( sum_k W_k P_{t+k}, Y ) + sum_k W_k^2 (excess variance terms),

where `P_s` = conditional law given `x_{0|s}`, i.e. the MMD between a
**mixture of the last D+1 conditional laws** and the target -- an EMA of the
objective along the trajectory (geometric weights = first-order low-pass on
the conditional distribution). The gradient we apply is the partial gradient
of this smoothed objective with respect to the *current* argument only.

Bias relative to the fresh objective `MMD^2(P_t, Y)` is controlled by
`||P_{t+k} - P_t||`, i.e. by how far `x_{0|t}` moves per outer step. Two
regimes:

* **Late steps / near convergence:** `x_{0|t}` changes slowly (the DDIM
  update is `O(beta_t)` and guidance is trust-capped), `P_{t+k} ~ P_t`,
  bias -> 0 while the effective sample size stays `~B`: pure variance win.
* **Early steps:** `x_{0|t}` moves fast, the mixture lags. But early
  `x_{0|t}` are dominated by prior noise anyway and the per-step gradient is
  noise-dominated (SNR < 1 at every `n`, `grad_noise.py`); a lagged mean with
  1/3 the variance can beat an unbiased estimate with full variance in MSE.
  The lag acts like momentum **in distribution space** rather than gradient
  space.

**Variance.** With independent rows, the XY term's variance scales like
`sum_i omega_i^2 Var[mean_a k(x_i, y_a)]`, so the effective sample size is
`ESS = 1 / sum_i omega_i^2 = B / sum_k (B W_k)^2 / r_k ... = (sum_k w_k)^2 /
sum_k w_k^2` in group units. For `lambda = 0.5`, depth 3: ESS multiplier
`(1.9375)^2 / 1.328 ~ 2.8x` the fresh count at equal fresh calls (augment
mode), or the same batch at `~0.52x` the fresh calls (batch mode). Rows at
different steps are not exactly independent (they share the trajectory), but
their conditional noises are independent by construction (per-step tape keys),
so within the noise-dominated regime the independence approximation is good.

## 4. Why stale GRADIENTS failed and this may not

The graveyard has two neighbours; the distinction is what is frozen:

* **Stale gradient (`stale2/3`, failed):** re-applies `g(x_t_old)` at
  `x_t_new`. The reused object is a *derivative at the wrong point*: it has a
  direction error that grows with `||x_new - x_old||` and no mechanism ever
  corrects it within the reuse window; worse, on skipped steps there is *no*
  fresh signal at all, so the estimator is pure extrapolation. Replay never
  freezes a derivative: **every step differentiates the loss at the current
  `x_t`**, through fresh samples of the current conditional; the stale part
  only perturbs the objective's landscape (the mixture above), not the
  point of evaluation.
* **CRN / frozen conditional noise (`crn`, failed):** freezing `eta` makes
  the loss a deterministic function of one noise draw -- the chain then
  optimises the idiosyncrasies of that draw (a biased objective whose
  minimiser is not the population one), with the bias *persistent* across the
  whole trajectory. Replay keeps fresh, independent draws every step; a
  replayed row is used for at most `D` steps and then evicted, so any single
  draw's idiosyncrasy is down-weighted by `W_k` and forgotten geometrically.

The failure mode replay *can* have is the lag bias of Section 3 (early steps,
large `lambda`, deep buffers). That predicts: small `lambda`/depth safe,
`lambda -> 1` or large depth degrading first in 2D at large n (where the
baseline is least noise-limited and the trajectory moves most per unit of
distributional distance), helped by `trust_noise1` (which caps how far `x_t`
moves per step and hence the lag).

## 5. Chosen grids (justification)

* **decay `lambda`**: {0.3, 0.5, 0.7} for the geometric arms -- brackets
  Ori's effective one-step ratio (`reuse_frac=0.3` = `lambda=3/7~0.43` at
  depth 1) and spans replay mass 23%-58% of the batch. `lambda >= 0.9`
  excluded: replay mass > 2/3 with depth >= 3 and lag bias dominates
  (Section 4's predicted failure).
* **depth**: 1 (Ori's), 3 (`lambda=0.3/0.5`; `w_3 < 3%` beyond that), 5
  (`lambda=0.7` only, where `w_4 ~ 6%` still matters). Deeper buffers add
  memory and lag for < 3% weight.
* **fresh fraction**: two arms per mechanism. `batch` mode = fixed total
  `B = n`, fresh `f = B w_0 / sum w_k` -- the **calls-saving Pareto arm**
  (e.g. `replay50` at n=8: 4 fresh calls/step, MMD batch 8, vs baseline n=8:
  8 calls -- calls halve at equal batch). `augment` (`_aug`) mode = fresh
  `f = n`, batch grows -- the **equal-calls quality arm**.
* **`+ trust_noise1`** (`_trust`) on the most promising arms, because (i) it
  is the promoted champion, so any promotion claim must survive on top of it,
  and (ii) Section 2.1's implicit `f/B` step shrink and Section 4's lag both
  interact with step size.
