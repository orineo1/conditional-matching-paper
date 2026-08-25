# Importance-selected backpropagation: theory

Agent B, round 4 of the CDM/TFG performance campaign, 2026-08-24. Notation
follows `estimator/REPORT.md` and `replay/THEORY.md`: at outer step `t` the
guidance loss is the biased V-statistic `L = MMD^2_V(Y, S)` between the `n`
conditional samples `Y = {y_i = G(x, eta_i)}` (each differentiable in the
condition `x = x_{0|t}` through the consistency sampler) and the fixed target
set `S` (`m = 250` rows, detached), 5-bandwidth RBF kernel at a fixed
bandwidth. The engine needs `dL/dx`.

## 1. The decomposition

Because `x` enters `L` only through the samples,

    dL/dx = sum_{i=1}^{n} J_i^T g_i,        J_i = dy_i/dx,   g_i = dL/dy_i.

The two factors have wildly different costs:

* `g_i` is **kernel-only**: for the fixed-bandwidth multi-scale RBF,

      g_i = (2/n^2) sum_j dk(y_i,y_j)/dy_i - (2/(nm)) sum_a dk(y_i,s_a)/dy_i,
      dk(u,v)/du = - sum_l (2/(bw m_l)) exp(-||u-v||^2/(bw m_l)) (u - v),

  i.e. `O(n(n+m)d)` arithmetic on already-computed kernel blocks -- no
  conditional model involved. (Implementation: one autograd call on a leaf
  copy of `Y` through `DistributionalLoss`, which is this closed form for any
  backend/bandwidth/transform, at the cost of one extra kernel evaluation.)
* `J_i^T g_i` requires a **backward pass through the conditional sampler**
  for sample `i` -- the expensive part: retaining its autograd graph
  (memory ~ n x sampler depth) and running the VJP.

So: generate all `n` samples under `no_grad` (full forward, zero graphs),
compute the full-batch `L` and all `g_i`, then rebuild graphs for only
`k << n` selected samples by replaying their tape noise keys (`CMSampler`
keys are per-sample: `("eta", t, j, i)` -- the same key returns bit-identical
noise; the regenerated rows equal the no-grad rows to round-off: 1e-14 in
float64, max 1.1e-6 relative in float32 at n=32, because CPU BLAS rounding
depends on the batch dimension -- measured and asserted in
`tests/test_backsel.py`; the full key set regenerates bit-identically), and
backprop

    G_hat = sum_{i in S} w_i J_i^T g_i

through a surrogate scalar `sum_{i in S} (w_i g_i)^T y_i` (with `g_i`
detached), whose gradient w.r.t. `x` is exactly `G_hat`. The returned loss
VALUE stays the full-batch one (`value + surrogate - surrogate.detach()`), so
every diagnostic/adaptive consumer sees the exact MMD.

**Cost currencies (reported separately, never conflated).** Conditional
forward samples per evaluation: `n + k` (the no-grad batch plus the replayed
subset -- forwards go UP by `k/n`). Differentiated samples: `k` instead of
`n` -- graph memory and backward cost drop by `n/k`. On the synthetic CPU
benchmark (backward ~ forward) the wall-time balance is
`(n + 2k) / 2n`-ish, a win for `k < n/2`; on real pipelines (SD sprinter,
where graph memory for n variations is the binding constraint) the
differentiated-samples currency is the relevant one.

## 2. Selection rules and unbiasedness

Write `h_i = J_i^T g_i` (unknown vectors, `sum_i h_i = G` the exact gradient).

**(a) Uniform without replacement** (`backsel_uni_k*`): `S` = k of n uniform
(tape argsort), Horvitz-Thompson weights `w_i = 1/pi_i = n/k` with inclusion
probability `pi_i = k/n`:

    E[G_hat] = sum_i pi_i (n/k) h_i = sum_i h_i = G.        (unbiased)

**(b) Gradient-magnitude importance sampling** (`backsel_is_k*`): `k` iid
draws `d_1..d_k ~ p`, with

    p_i = (1 - eps) ||g_i|| / sum_j ||g_j||  +  eps / n,    eps = floor = 0.25,

and `G_hat = (1/k) sum_m h_{d_m} / p_{d_m}`. Unbiasedness is immediate:

    E[G_hat] = (1/k) * k * sum_i p_i (h_i / p_i) = sum_i h_i = G,

and it is unaffected by de-duplicating repeated draws into weights
`w_i = c_i / (k p_i)` (`c_i` = multiplicity), which is what the
implementation does so that at most `min(k, n)` samples are differentiated.
The floor bounds every weight by `n / (k eps)` and hence the variance; it also
keeps the estimator well-defined when some `||g_i|| = 0`. Proof is restated
in the `tfg/backsel.py` docstring; `tests/test_backsel.py` checks it
statistically (mean over selection draws vs the full gradient, SE-calibrated
tolerance) for both (a) and (b).

**(c) k-center cluster aggregation** (`backsel_clust_k*`): greedy k-center on
the `y_i` (tape-keyed start row, deterministic thereafter), assign every
sample to its nearest center, differentiate only the `k` centers and apply
the cluster-AGGREGATED output gradient through each center's Jacobian:

    G_hat = sum_c J_{r_c}^T ( sum_{i in C_c} g_i ).

This is NOT an inverse-probability estimator: it keeps 100% of the
output-gradient mass and replaces `J_i` by `J_{r_c}` within a cluster, so its
error is `sum_c sum_{i in C_c} (J_i - J_{r_c})^T g_i` -- a deterministic bias
controlled by how fast the sampler Jacobian varies with the conditional
noise `eta` at fixed `x`, times the cluster radius (k-center minimizes the
max radius). Zero selection variance given the tape. Exact at `k = n`
(singleton clusters). Clustering on `y` is chosen over clustering on `g`
because nearby outputs have nearby Jacobians for a smooth sampler, which is
the quantity being substituted; `g`-clustering groups samples with similar
*output* gradients whose Jacobians may differ.

All three rules reduce to the identity (all samples, unit weights) when
`k >= n`, restoring the exact full gradient to 1e-12 (float64 test).

## 3. Variance: when does importance selection beat uniform?

With-replacement analogues (clean formulas; the without-replacement uniform
is strictly better by the finite-population factor `(n-k)/(n-1)`):

    Var_p[G_hat] = (1/k) ( sum_i ||h_i||^2 / p_i  -  ||G||^2 ).

* Uniform `p_i = 1/n`:      `Var_uni = (1/k)( n sum_i ||h_i||^2 - ||G||^2 )`.
* Optimal `p_i ~ ||h_i||`:  `Var_opt = (1/k)( (sum_i ||h_i||)^2 - ||G||^2 )`.

The ratio of the leading terms is

    Var_uni / Var_opt ~ n * sum ||h||^2 / (sum ||h||)^2  =  n / ESS(||h||),

which is 1 for equal norms and grows with the dispersion of `||h_i||`:
importance selection pays off exactly when the per-sample norms are
**heavy-tailed**. That regime is documented, not assumed:
`estimator/REPORT.md` section 4 measured the per-step gradient-norm
distribution of this very pipeline -- **within-run p90 ~ 6-8x the median and
the max 100-500x the median** (2D/5D/10D medians 0.038-0.38). Those numbers
are for the assembled step gradient across steps; the per-sample `||g_i||`
within a batch inherit the same kernel geometry (a sample far from both the
batch and the target carries the large XY pull), so a within-batch spread of
the same order is the expectation the IS arm tests. We use `||g_i||` as the
proxy for the unobservable `||h_i|| <= ||J_i||_op ||g_i||`; the proxy is
exact up to the spread of `||J_i||_op` across `eta` draws at fixed `x`.

The floor caps the harm when the proxy is wrong: `p_i >= eps/n` gives
`Var_is <= (n/eps) sum ||h_i||^2 / k`, i.e. at worst `1/eps = 4x` uniform's
leading term, while the upside in the heavy-tail regime is `~ n/ESS`.

## 4. Expected regime and interaction with the step rule

The campaign's central negative result (REPORT sec 4, verified): the per-step
estimator is noise-dominated (SNR < 1 at every n) and pure variance
manipulation with the step rule left alone moved nothing (antithetic, CRN,
adaptive-n all null); the binding constraint is the step size, and
`trust_noise1` is the promoted fix. Selection ADDS variance (uni/is) --
`G_hat` has the full-backprop gradient as its mean plus selection noise of
relative size `~ sqrt(n/k - 1)` (uniform). The honest predictions are
therefore modest and pre-registered in `hypotheses/agentB.yaml`:

* plain arms at `k = 2, n = 8` lose measurably in 2D (selection noise on top
  of SNR ~ 0.2 estimator);
* the `_trust` arms shrink the gap (the trust region caps exactly the
  inflated-tail steps selection noise produces);
* the promotable point, if any, is `k = 4, n = 32` under trust: 8x fewer
  differentiated samples for a |diff| below the 0.08 seed-noise floor;
* ordering `clust >= is >= uni` at fixed k (variance ordering of sec 2-3),
  unless the Jacobian-substitution bias of clust flips it -- that flip is the
  diagnostic for "J_i varies strongly with eta".

## 5. Rejection criteria (pre-registered)

* A rule is dead if significantly worse (p <= 0.05) than its equal-`n`
  comparator in a majority of its 12 (setting, n) cells at `k = 4`.
* The mechanism is dead if no arm achieves |diff| < 0.08 with p > 0.05 vs
  `trust_noise1` at `n = 32, k = 4` in >= 2 of 3 settings.
* Claims are made ONLY in the differentiated-samples/memory currency;
  cm_samples (forwards) rise by `k/n` and the tables must show it.
* Exactness gates: bit-identical noise replay + round-off row regeneration test, k=n equality at 1e-12,
  statistical unbiasedness of (a)/(b), off-by-default byte-identity --
  any failure blocks screening.

## 6. Where the saving lives: the SD pipeline estimate

The synthetic benchmark cannot show a cost win (screening: wall +25-40%,
`REPORT.md` sec 2) because its conditional model is a 128-wide MLP whose
backward is negligible. The mechanism targets the SD pipeline audited in
`systems/AUDIT.md` (`run_mlgd_f.py`): `N = num_variations` sprinter calls
per outer step, hard-coded `variation_batch_size = 1`, each wrapped in an
outer non-reentrant `checkpoint` PLUS block-level gradient checkpointing on
the sprinter UNet/ControlNet. The audit counts each in-graph sprinter
UNet+CN pass **3x** (forward-no-save, recompute in backward, block-level
recompute inside that) plus the backward itself, and the VAE decode / CLIP
/ text encoders 2x.

Units: one sprinter UNet+CN forward = 1. Backward ~ 2 (standard VJP cost).
Per variation per sprinter inference step:

* in the graph (baseline, every variation):  3 (fwd + recomputes) + 2 (bwd) = **5**
* out of the graph (backsel no-grad pass):    **1**

Per outer step, guidance share only (visualisation/eval excluded):

    baseline  C_full = 5 N
    backsel   C_sel  = N (no-grad) + 5 k (selected, in graph)
    ratio     C_sel / C_full = (N + 5k) / (5N) = 1/5 + k/N

| N | k | ratio | graph memory (variation graphs) |
|---|---|---|---|
| 6 (defaults) | 2 | 0.53 | k/N = 1/3 |
| 6 | 1 | 0.37 | 1/6 |
| 100 (DPS runs, n_targets=100) | 8 | 0.28 | 0.08 |
| 100 | 4 | 0.24 | 0.04 |

The floor is 1/5 (the no-grad forward of every variation is kept: the MMD
geometry stays full-batch by design). Graph memory scales with the number
of variations IN the graph -- with checkpointing that is the saved block
boundaries and the recompute workspace per variation, so peak memory drops
by ~k/N, which is what makes large N (the 100-variation DPS runs) feasible
at all: at N=100 the baseline holds 100 variation graphs; backsel holds k.
The k-row regeneration could additionally be batched (`variation_batch_size
= k` instead of 1, AUDIT sec 6a) since the selected rows are known before
the graph pass.

Caveats: (i) the 5:1 ratio is the audit's accounting, not a measurement --
the audit's memory harness (`measure_dps_step_memory.ipynb`) has no saved
outputs and should be re-run with backsel to validate; (ii) the extra
kernel evaluation for `g_i` (CLIP-space MMD, n x m) and the selection are
negligible at SD scale; (iii) the quality side must be re-tested there with
the corrected protocol -- the synthetic screening [legacy protocol] shows
no collapse and graceful `n/k` degradation for the unbiased rules, and
`clust k>=4` as the most robust rule in the higher-dimensional settings,
which is the ordering to try first.
