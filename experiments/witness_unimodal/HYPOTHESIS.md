# Pre-registered hypothesis — mean-blindness & witness back-selection on unimodal targets

*Registered 2026-09-01, before any experiment runs (smoke or full). Agent W, branch
`claude/witness-function-simulation-analysis`.*

## Claim (user's, verbatim in substance)

1. **Mean-matching inverse design cannot distinguish targets that share a mean.**
   A 50/50 bimodal target at ±c (the toy analogue of a male/female mix) and a
   CONCENTRATED UNIMODAL target N(0, s²) at the midpoint (the "androgynous" target)
   have the same mean (0). A guidance loss of the form
   ‖mean(y_gen) − mean(y_target)‖² is therefore *identical* for the two targets and
   blind to the difference between them. MMD guidance is not blind: its loss separates
   the two targets through higher moments.

2. **Witness-function back-selection should help MOST for the concentrated unimodal
   target.** The witness function
   w(e) ≈ mean_i k(e, e_i) − mean_j k(e, y_j)
   (sign convention of `simulations/src/witness_utils.compute_witness_scores`: positive
   where the generated set has excess mass relative to the target) is largest in
   magnitude for generated samples far from the single target mode — the off-mode
   stragglers. Selecting the k-of-n backward subsample proportional to |w| concentrates
   the backward budget exactly on the samples that need to move. For a broad or bimodal
   target the mismatch is spread more evenly across samples, so |w| is flatter and
   witness selection degrades toward uniform selection.

## Concrete predictions (falsifiable)

Setting: the repo's `2D_cond_1D` toy (1-D y | 1-D x, pretrained `Diffusion_cond`
seed-42 checkpoint), guidance loop = `Optimization.optimize_LGD` and a mean-loss twin
of it; n = 32 conditional samples per step, back-selection k = 8; targets share mean 0:

- (a) bimodal 50/50 at ±c, component std 0.1, c ∈ {1, 2};
- (b) unimodal N(0, s²), s ∈ {0.1, 0.25, 0.5}.

Predictions, over ≥ 40 paired restarts (8 for the local smoke):

P1. **Mean arm aces the mean metric on both targets**: its final |mean(y_gen)| is
    comparable to (or better than) the MMD arms' on both (a) and (b).

P2. **Mean arm fails concentration on target (b)**: its generated std(y) stays far
    above s and its fraction-within-2s-of-the-mode stays far below the MMD arms',
    because nothing in the mean loss rewards concentrating at the midpoint. On
    target (a) the mean arm likewise fails to produce two modes at ±c — but the
    headline prediction is (b), where "same mean" is achievable *without* moving
    any mass.

P3. **MMD arms distinguish the targets**: full-MMD final MMD-to-target is materially
    lower than the mean arm's on target (b) (and (a)); the mean arm's MMD is
    approximately what an unguided/mean-only trajectory gives.

P4. **Witness > uniform back-selection, most at small s**: at matched (n=32, k=8),
    the paired difference (witness − uniform) in final MMD is negative (witness
    better), and the effect is largest at s = 0.1, shrinking as s grows to 0.5 and
    on the bimodal targets. Direction of the ordering matters more than any single
    cell's significance.

## What would falsify / complicate

- Witness ≈ uniform everywhere (flat |w|, ESS ≈ n): the heterogeneity the mechanism
  needs may not arise in this toy — a real negative result (see
  `witness_utils.witness_scenario_stats` rationale), not an implementation bug.
- Mean arm accidentally concentrating on (b): would mean the diffusion prior alone
  already concentrates y at x with E[y|x]=0, i.e. the task under-identifies the claim;
  we would then need a joint GMM whose conditionals at mean-0 x are wide/multimodal.
- Note: s = 0.1 is *narrower than any achievable conditional* of the 2D_cond_1D joint
  (conditional y|x std ≈ 0.35 for every component: 0.2 − 0.195²/0.5 ≈ 0.124). The
  concentrated targets are therefore (mildly) infeasible — deliberate: infeasibility
  is what creates persistent stragglers for the witness to exploit.

## Arms (fixed before running)

1. `mean` — ‖mean(y_gen) − 0‖² guidance (thin twin of `optimize_LGD`, only the loss
   differs; see README for the documented deviation).
2. `mmd` — full MMD (`optimize_LGD`, `backsel_k=None`).
3. `mmd_uniform` — MMD + uniform back-selection k=8 of n=32 (`backsel_rule='uniform'`).
4. `mmd_witness` — MMD + witness back-selection, same k (`backsel_rule='witness'`,
   `witness_floor=0.3`, the sweep scripts' canonical value).
5. *(dropped)* witness + trust-style step cap: `optimize_LGD` on this branch has no
   step-cap toggle (only `zeta` / `use_inv_sqrt_alpha_scale`), and porting one from
   elsewhere was explicitly out of scope.

**Amendment (2026-09-01, during the smoke run, before any mean-arm numbers were
recorded):** the mean arm's raw `zeta=1` update diverges to NaN (~10 steps in) because
the quadratic mean loss is unbounded, unlike the bounded RBF-MMD. The mean arm now
clips its gradient norm to 1.0 (direction unchanged). No prediction changes.

Metrics per restart (n_eval = 256 fresh conditional samples at the returned x):
final MMD to a fresh target draw, |mean error|, std(y_gen), fraction of y_gen within
2·(target mode scale) of the nearest mode (mode scale = s for (b), 0.1 for (a)).
Pairing: identical `set_run_seed(seed, i)` per restart index across all arms/targets;
identical eval seeds.

## Amendment 2 (2026-09-01, after the round-1 40-restart array; before any capped run)

Round-1 outcome (see VERDICTS.md / RESULTS.md):
- **P4 not supported**: witness − uniform is null in every cell (no 95% CI excludes
  0 on any metric); final-MMD point estimates even favor uniform in 4/5 targets.
- **mmd-vs-mean contrasts are INVALID as run**: the raw `optimize_LGD` update has no
  step cap (known legacy defect), and MMD-arm restarts frequently diverge off-manifold
  (per-restart |mean err| in the hundreds, medians 1-26), while the mean arm was
  norm-clipped — an asymmetric comparison. P1 (mean arm aces its own metric,
  median |mean err| ≈ 0.07) stands; P2/P3 are unresolved.

Rationale for the capped rerun (round 2): add an opt-in noise-level trust region
||Delta_t|| <= tau*sqrt(1-alphabar_t), tau = 1 (the trust_noise1 semantics of the
perf campaign's IMPROVEMENTS.md §1 on branch tfg-generalization-v2), applied
SYMMETRICALLY to all four arms (the mean arm's norm clip is turned off when the cap
is on), implemented entirely inside `exp_witness_unimodal.py` (`optimize_capped`,
a replicated minimal loop — Ori's files untouched). Predictions P1-P4 are unchanged
and will be re-read on the capped (tag `cap1`) results only.

## Round 2 (identifiability) — pre-registered 2026-09-01, before any round-2 run

*Supersedes the plain capped rerun of Amendment 2 (the cap work is kept and reused).*

**Construction.** A purpose-built analytic joint GMM over (x, y), no trained models:
components A = N((x_bi, +c), diag(σx², σy_bi²)), B = N((x_bi, −c), diag(σx², σy_bi²)),
C = N((x_uni, 0), diag(σx², s²)); weights (0.25, 0.25, 0.5); constants x_bi = −2,
x_uni = +2, σx = 0.7, c = 2, σy_bi = 0.25, s = 0.25. With diagonal covariances,
p(y|x) = Σ_k w_k(x) N(μ_yk, σ_yk²) with w_k(x) ∝ α_k N(x; m_k, σx²) — derived
explicitly and verified numerically in exp_identifiability.py (--verify). By the
A/B symmetry (±c at the same x location, equal weights), **E[y|x] ≡ 0 for EVERY x**:
the conditional mean carries literally zero information about x. At x_bi the
conditional is bimodal (±c, mean 0); at x_uni it is concentrated unimodal N(0, s²);
|x_bi − x_uni| = 4 ≫ σx so the shapes are clean at each design point.

**Oracle sampling (documented sampler).** Analytic conditional with reparameterised
differentiable sampling in the spirit of the perf campaign's exp5 oracle
(tfg-generalization-v2 history; that code is not on this branch) and of
optimize_LGD's _reference_loss_term: here a straight-through Gumbel-softmax over
the K=3 conditional weights (forward = exact hard mixture samples, backward = soft
relaxation gradient, tau = 0.5) mixed with the per-component reparameterised
Gaussians. Note the repo's generate_mog_samples blocks weight-gradients (hard
multinomial after the softmax); since in this construction ALL x-dependence of
p(y|x) flows through the weights, ST-Gumbel is required — a documented deviation.
The prior over x is also analytic: E[x0|x_t] for a 1-D GMM prior under the repo's
cosine schedule (replicated from Diffusion.py, s = 0.008) in closed form, driving
the same DDIM/LGD step structure; x_T ~ randn (not zeros).

**Step convention.** Protocol note: Ori's own simulations used optimize_LGD's
use_inv_sqrt_alpha_scale=True (the TFG line-9 1/√α_t convention); our round-1 array
ran with the default False (recorded in VERDICTS.md). Round 2 exposes
--inv_sqrt_alpha and runs the mmd arm under a small step-convention factor:
{cap only, inv_sqrt only, cap+inv_sqrt}; the mean arm and the witness/uniform
comparison run under cap+inv_sqrt, the verified-correct combination from the main
campaign. Prediction: inv_sqrt alone amplifies steps at large t (α_t small) and
still diverges without the cap; cap+inv_sqrt is stable.

**Design.** 2 targets (250 oracle samples from p(y|x_bi) and from p(y|x_uni)) ×
arms (mean, mmd, mmd_uniform k=8/32, mmd_witness) × 40 paired restarts; all arms
step-capped at τ = 1 (‖Δ_t‖ ≤ τ·√(1−ᾱ_t)).

**Predictions.**
- R1 (mean-arm degeneracy): the mean arm's landing distribution over x_hat is
  statistically IDENTICAL across the two targets (two-sample KS test between its
  landing sets; expected non-significant), because its loss is flat in x by
  construction. Its landings should follow the prior mass (≈ half near each design
  point), not the target.
- R2 (MMD identifiability): the mmd arm lands on the matching design point for each
  target — majority (> 50%) of restarts within 0.5 of x_bi for the bimodal target
  and within 0.5 of x_uni for the unimodal target — and its fresh-sample cross-MMD
  matrix is diagonal-dominant (lower MMD to the target it optimized).
- R3 (witness vs uniform, secondary): same P4 logic as round 1, now that steps are
  capped and the convention is correct: witness − uniform ≤ 0 on final MMD, largest
  gain on the concentrated unimodal target.
- R4 (step convention): mmd arm with inv_sqrt only diverges (off-manifold landings,
  huge |mean err|); cap only and cap+inv_sqrt are stable, cap+inv_sqrt ≥ cap only.

### Round-2 sampler note (2026-09-01, after the pre-run gradient probe, BEFORE any
### round-2 array run)

The originally specified ST-Gumbel oracle sampler was probed before running
(mean of dMMD/dx over 200 draws on a grid of x): its gradient is SIGN-BIASED near
the basin boundary — for the unimodal target at x = 0 it pushes toward x_bi
(+0.26) although the true expected-MMD landscape decreases toward x_uni (finite-
difference check). This construction is the ST estimator's worst case: ALL
x-dependence of p(y|x) is in the mixture weights. The sampler is therefore
replaced by an exact implicit-reparameterisation sampler (1-D inverse-CDF
pathwise gradient, y = y0 − (F(x,y0) − stop_grad F)/p(y0|x); forward pass
unchanged = exact hard mixture samples). ST kept behind --sampler st for
reference. Predictions R1-R4 unchanged; the smoke and array run with
--sampler implicit.

## Round 3 (concentration sweep) — pre-registered 2026-09-01, before any round-3 run

*User-requested: does the witness advantage GROW with target concentration?*

**Design.** Same analytic identifiability construction as Round 2 with ONE new
parameter: the unimodal component C's y-std, s ∈ {0.05, 0.1, 0.25, 0.5, 1.0}
(very concentrated → fat). The bimodal blobs stay fixed at ±2 with std 0.25 and
ALL x-structure (component locations, weights, σx, prior, schedule) is unchanged,
so the basin geometry and x_uni are identical across s, and E[y|x] ≡ 0 still holds
for every s (the A/B symmetry never involves s; re-verified numerically per s at
run time). Target = 250 oracle samples from p(y|x_uni) at each s. Arms: paired
mmd_uniform (k=8/32) vs mmd_witness (same k), plus full mmd as reference. Corrected
convention throughout: step cap τ=1 + 1/√α_t scaling, x_T ~ randn, implicit-
reparameterisation sampler. TWO seeds from the start (42 primary, 1042
confirmation), 40 restarts each → 80 pooled pairs per s (no seed-lottery round
trip this time).

**MMD bandwidth policy (stated up front).** Identical to Rounds 1-2 and NOT tuned
per s: the multi-bandwidth RBF (LossFunctions.RBF, 5 kernels, multipliers
2^{−2..+2}) with the adaptive base bandwidth Σ‖·‖²/(n²−n) recomputed per call on
the STACKED generated+target batch. Interaction to note: as s shrinks the
target's within-set distances vanish, but the generated set and the cross terms
keep the batch scale up, so the adaptive bandwidth stays O(gen scale) — tiny-s
targets are therefore resolved through the smallest multiplier rather than a
per-target bandwidth. Accepted as-is; any under-resolution at s=0.05 is part of
the fixed policy, not a knob.

**Predictions.**
- C1 (primary): the paired dist_uni difference (witness − uniform) becomes MORE
  NEGATIVE as s shrinks — monotone trend over s (sign of the slope of pooled
  diffs vs s positive, i.e. diff increasing with s toward 0), with the largest
  advantage at s = 0.05 and ≈ 0 by s = 1.0 (at s = 1.0 the "concentrated" target
  is as fat as the conditional itself — no stragglers to exploit).
- C2 (mechanism): witness selection over-samples off-mode generated points:
  fraction of SELECTED samples that are off-mode (|y| > 2s at selection time)
  exceeds the batch's overall off-mode fraction for the witness arm and not for
  the uniform arm, with the gap growing as s shrinks.
- C3 (sanity): full-mmd reference stays basin-correct at every s (the target's
  identifiability does not degrade with s).

### Round-3 close-out (2026-09-01, after the 15-cell array, both seeds, 80 pairs/s)

**C1 REFUTED in direction.** The witness advantage does NOT grow with
concentration. Pooled paired dist_uni diff (witness − uniform): s=0.05 +0.050
±0.183, s=0.1 +0.018 ±0.192 (nulls with ~4x the variance of other cells),
s=0.25 −0.047* ±0.047, s=0.5 −0.094* ±0.050, s=1.0 −0.046* ±0.032. The
significant advantage sits at MODERATE-to-FAT s and vanishes at the small s
where it was predicted to be largest. Post-hoc decomposition (VERDICTS.md):
the small-s variance blow-up is entirely basin-flip pairs — excluding pairs
where either arm flipped basin, s=0.05 is a tight null (−0.010 ± 0.035, n=68)
and s=0.1 likewise (+0.017 ± 0.045, n=71), while the moderate-s advantages
survive exclusion unchanged. Flips are a backsel cost, not witness-specific
(10-11/80 for BOTH backsel arms at s=0.05 vs ≤1/80 for full mmd).
**C2 NOT SUPPORTED anywhere**: witness frac_selected_offmode tracks the base
rate at every s (max |deviation| ≈ 0.004, sometimes below base). **C3 holds**
(full-mmd basin correctness ≥ 98.75% at every s).

The user's more-concentrated-more-help hypothesis is not supported. Proposed
decisive follow-up (NOT run): a witness-scores-SHUFFLED control arm — permute
the computed scores across the batch before selection (same probability
histogram and RNG consumption, zero sample-identity information) — plus
logging witness_scenario_stats (score dispersion / ess_raw, already in
witness_utils but not wired here) to separate informative selection from a
re-randomized-uniform artifact.

## Round 4 (selection sharpening + shuffled control) — pre-registered 2026-09-01,
## before any round-4 run

*User-requested: sharpen witness selection so top scores dominate; settle the
round-3 mechanism question in the same round.*

**Selection family.** p_i ∝ |witness_i|^β, floor 0.0, β ∈ {1, 2, 4}; plus a
deterministic top-k arm (β=∞). Protocol change vs round 3, stated up front: all
stochastic round-4 arms (INCLUDING the uniform reference) draw k=8 indices WITH
replacement and apply unbiased IPW gradient weights g_i = counts_i/(k·p_i)
(E[g_i]=1, so the expected gradient equals the full-batch gradient for any loss;
implemented by value-preserving gradient scaling, loss VALUE still uses all n
rows). The ONLY difference between stochastic arms is therefore the selection
distribution p. witness_b1_f0 is the β-family baseline — it differs from
round-3's witness arm in both floor (0.0 vs 0.3) and estimator (IPW
with-replacement vs plain detach without replacement). The top-k arm is
DETERMINISTIC AND BIASED — no IPW correction exists for deterministic selection
(inclusion probability 1 for the top k, 0 otherwise carries no importance
weights); documented as such, included because it is the practical limit.

**Shuffled-scores control (mechanism test).** witness_b4_shuffled: compute the
β=4 probabilities, then randomly permute them across the batch before sampling —
identical probability histogram, floor, IPW weighting, and RNG consumption;
zero information about which sample scored what.

**Diagnostics logged this round (the round-3 gap):** per-step, averaged per run:
normalized selection entropy H = −Σp ln p / ln n (top-k: ln k/ln n by
convention; uniform ≡ 1), witness-score dispersion max|w|/median|w| and std(w),
and frac_selected_offmode vs base rate.

**Design.** s ∈ {0.1, 0.25, 0.5} (round-3 null / effect / strongest-effect
points); arms mmd_uniform, witness_b1_f0, witness_b2_f0, witness_b4_f0,
witness_topk, witness_b4_shuffled; 2 seeds (42, 1042) × 40 paired restarts →
80 pooled pairs per (s, arm); corrected convention (cap τ=1 + 1/√α_t, x_T ~
randn, implicit sampler); MMD bandwidth policy unchanged (fixed rounds-1-3 rule).

**Predictions.**
- S1 (mechanical): selection entropy strictly decreases with β
  (1 > H_b1 > H_b2 > H_b4 > H_topk-convention) — sharpening actually happens.
- S2 (user's): at s = 0.5 (round 3's strongest cell), b4 and top-k beat b1:
  paired diff vs uniform more negative as β grows.
- S3 (mechanism, decisive): witness_b4_shuffled ≈ uniform (diff ≈ 0) while
  witness_b4_f0 < 0 ⇒ selection is informative; witness_b4_shuffled ≈
  witness_b4_f0 < 0 ⇒ the round-2/3 advantage is a selection-stream artifact.
- S4 (diagnostic, weakly held after round 3's C2 failure): frac_selected_offmode
  rises above base rate with β for the true-witness arms and not the shuffled arm.

### Round-4 close-out (2026-09-02, after the 18-cell array, both seeds, 80 pairs per (s, arm))

**S1 CONFIRMED (mechanical):** entropy fell as designed, 1.000 (uniform) → 0.951
(β=1) → 0.889 (β=2) → 0.794 (β=4) → 0.600 (top-k convention), stable across s.
**S2 SUPPORTED:** at s=0.5 the advantage over uniform grows monotonically with
sharpening: β=1 −0.069 n.s. → β=2 −0.233* → β=4 −0.319* → top-k −0.476*; top-k
is also the best arm at s=0.25 (−0.501*). At s=0.1 (round-3's flip regime) the
ordering breaks: β=2 −0.233* is best and top-k collapses to +0.033 ±0.338.
**S3 SPLIT:** β=4 beats its shuffled control at s=0.5 (−0.206* ±0.169 —
informative selection there) but NOT at s ≤ 0.25 (no separation; the shuffled
arm itself sits at −0.088..−0.110 n.s., so the witness effect at small/moderate
s is stream-artifact-compatible). **S4 FAILS everywhere:** sel_offmode tracks
(if anything, sits slightly BELOW) the base rate for every witness arm at every
s — the informative content at s=0.5 is NOT off-mode selection; the mechanism
remains unidentified (within-mode extremeness is the remaining candidate).
Top-k's wins carry the documented estimator bias (no IPW exists for
deterministic selection) — flagged for a bias-check cell (top-k vs full-mmd on
final quality) before any recommendation.

## Round 5 (replacement mode) — pre-registered 2026-09-02, before any round-5 run

*User-requested: round 4 sampled WITH replacement (duplicates possible, IPW
g_i = counts_i/(k p_i)); compare against WITHOUT-replacement sampling.*

**Design.** New arms, k=8 DISTINCT samples drawn by successive sampling
without replacement (torch.multinomial(p, k, replacement=False)):
uniform_wor (reference), witness_b{1,2,4}_wor (p ∝ |w|^β, floor 0). Targets
s ∈ {0.25, 0.5} (where round-4 effects live; s=0.1 skipped), 2 seeds × 40
paired restarts. The WITH-replacement side is NOT rerun: round-4 results are
reused (same seeds, same restarts, same construction), giving exact
(seed, restart)-paired with-vs-without contrasts per β.

**Weight correction (choice (a), documented honestly).** Horvitz-Thompson
gradient weights g_i = 1{i selected}/π_i, with inclusion probabilities
approximated as π_i ≈ 1 − (1 − p_i)^k. Derivation: this expression is EXACT
for k i.i.d. draws from p (with replacement); for successive without-
replacement sampling the true π_i are intractable (they depend on all
orderings), and the approximation understates π for large-p_i items (which
can only be "included once" but the i.i.d. formula credits repeat chances) —
so their weights 1/π_i are slightly too large: a small, sign-understood bias,
largest at high β where p is most peaked. Exceptions that are EXACT:
uniform_wor uses the true π_i = k/n (weights n/k — exactly unbiased), and
k ≥ n selects every row with weight 1 (full gradient; enforced as a special
case and covered by a selftest, --selftest, which also Monte-Carlo-checks the
π approximation error at k=8).

**Duplicate accounting.** Round-4 JSONs did not record duplicate rates, so the
report computes the EXPECTED duplicate share of the k=8 budget analytically
from each arm's logged selection entropy via the perplexity approximation
(treat p as uniform over m = n^H effective categories; E[#distinct] =
m(1−(1−1/m)^k); duplicate share = 1 − E[#distinct]/k), documented as an
estimate.

**Prediction R5.** Without replacement is equal-or-better at high β — with-
replacement wastes backward budget on duplicates exactly when sharpened
(expected duplicate share grows with β) — and indistinguishable at β=1:
the direct paired contrast (wor − wr) at matching β is ≤ 0 at β=4 and ≈ 0 at
β=1, most visibly at s=0.5 (the round-4 signal cell). Secondary: the HT
approximation bias does not dominate (uniform_wor ≈ mmd_uniform, since both
uniform references are exactly unbiased and differ only in duplicate noise).

### Round-5 close-out (2026-09-02, after the 8-cell array, both seeds, 80 pairs per (s, arm))

**R5 REFUTED.** Without-replacement sampling is equal-or-WORSE, not
equal-or-better. Direct paired contrasts (wor − wr, same β, same
(seed, restart)): +0.119* ±0.095 (s=0.25, β=2), +0.052/+0.076/+0.071 n.s.
(s=0.25 β=4, s=0.5 β=2, s=0.5 β=4), ≈0 at β=1 (−0.001/−0.014) — positive
where sharpened, never negative. Against the matching-mode uniform reference
the wor advantage shrinks everywhere (s=0.5 β=4: −0.190* vs wr's −0.319*;
s=0.25 β=2: +0.005 vs wr's −0.197*). The estimator sanity check passes: the
two exactly-unbiased uniform references agree (wor − wr = −0.083/−0.059 n.s.).
Mechanistic reading: with-replacement's duplicates are NOT wasted budget — a
row drawn c times carries gradient weight c/(k·p), i.e. the duplicates ARE the
sharpened emphasis on the highest-witness rows; without-replacement discards
that emphasis (each selected row counted once, HT-reweighted) and adds the
documented π-approximation bias (max error ≈ 0.12 at large p_i, exactly the
high-β regime). The ~12-20% expected "duplicate share" was emphasis, not waste.
**Recommendation: keep WITH-replacement sampling (+ IPW) for the witness rule.**

## Round 6 (pointwise scalar-target baseline) — pre-registered 2026-09-02,
## before any round-6 run

*User-reframed core experiment: the realistic practitioner baseline is not a
32-sample mean-matching arm but POINTWISE scalar-target inverse design — per
diffusion step, generate ONE conditional sample y_1 ~ p(y|x0_hat) and take
loss ||y_1 − a||² for a SCALAR target a. No target sample set at all.*

**Arms.** point_sq_a0 (a=0, the shared conditional mean), point_sq_a2 (a=+2,
where the bimodal conditional has an actual mode), point_abs_a2 (|y − a|,
included since it is cheap; see its separate prediction below). All: n=1
sample per step via the implicit-reparameterisation sampler, corrected
convention (cap τ=1 + 1/√α_t, x_T ~ randn). No extra gradient clip: the trust
cap already bounds the step (the round-1 mean-arm clip predates the cap).
References: the ROUND-2 mmd arm given the bimodal target and given the
unimodal target — reused from the existing identifiability cell JSONs
(cap1_inv, seeds 42/1042), no rerun. Same analytic construction (s = 0.25
default; E[y|x] ≡ 0 everywhere).

**Two competing predictions, both stated up front (either supports the paper's
claim that a scalar target does not determine the answer; we record which):**
- **(P-a) target-blind splitting (user's):** with a = 0 the landings split
  between the two basins target-blindly, like the round-2 mean arm — the n=1
  stochastic gradient is too noisy to integrate any systematic signal, so the
  prior decides.
- **(P-b) variance seeking:** E‖y − a‖² = Var(y|x) + (E[y|x] − a)², and
  E[y|x] ≡ 0 by construction, so the a-dependence is a CONSTANT in x — for
  EVERY scalar a (a = 0 and a = +2 alike) the expected pointwise squared loss
  is minimized by minimizing conditional VARIANCE (Var at x_bi ≈ 4.06 vs
  0.0625 at x_uni): both arms drift to x_uni, and the a=0 and a=+2 landing
  distributions are statistically IDENTICAL (KS n.s.) — even though a=+2 sits
  exactly on a bimodal mode.
- **point_abs_a2 (separate prediction):** E|y − a| at a=+2 is ≈ 2.10 at x_bi
  (half the mass at distance ≈0.2, half at ≈4) vs ≈ 2.00 at x_uni — the same
  x_uni preference but with a ~40x smaller margin than the squared loss
  (0.1 vs 4.0), so expect weak/mixed drift: landings noisier, possibly split.

**Decision criteria.** Basin fractions per arm; two-sample KS between the
point_sq_a0 and point_sq_a2 landing sets (identical under P-b); mean |x̂|;
cross-MMD of fresh samples at x̂ to both round-2 target sets (reused
machinery). 2 seeds × 40 restarts.

### Round-6 close-out (2026-09-02, after the 3-cell array, both seeds, 80 restarts per arm)

**Outcome: between P-a and P-b, dominated by P-b — squared scalar-target design
is VARIANCE-SEEKING.** point_sq_a0 lands in the tight-unimodal basin 97% of the
time; point_sq_a2 — a target value sitting EXACTLY on a bimodal mode — still
lands unimodal 71%. The target value nudges the landings (KS a0-vs-a2 pooled
p=0.0007, rejecting strict P-b identity; per-seed p=0.054/0.014) but does not
determine the answer. point_abs_a2 is a near coin-flip (42% bi / 57% uni), as
its pre-registered weak-margin prediction anticipated. Contrast: round-2 MMD
with the actual target sets is 99-100% basin-correct.

Honest scoring of the two pre-registered alternatives: **P-a refuted in its
literal form** (landings are not prior-following/target-blind — a=0 goes 97%
to one basin, and a shifts the distribution), **P-b refuted in its strict
form** (the a=0 and a=+2 landing sets are NOT identical). The
variance-domination reading — E‖y−a‖² = Var(y|x) + const-in-x drives the drift,
with a residual a-dependence through the stochastic n=1 gradient — is a
POST-HOC synthesis of the two pre-registered alternatives, recorded as such.

Paper-facing sentence: pointwise scalar-target inverse design under conditional
stochasticity implicitly optimizes mean-plus-variance, so it collapses toward
the lowest-variance conditional almost regardless of the requested value — it
cannot express, and barely responds to, WHICH conditional distribution the
practitioner wants; distributional (MMD) targets determine it essentially
perfectly.

## Round 7 (init sweep) — pre-registered 2026-09-02, before any round-7 run

*User-requested: separate objective pull from basin capture by sweeping the
initialization.*

**Design.** x_T ~ c + N(0, 1) with offset c ∈ {−3, −1.5, 0, +1.5, +3}
(σ = 1 kept; the cap τ=1 + 1/√α_t convention and everything else unchanged —
the offset is the ONLY new degree of freedom). Arms: point_sq_a0, point_sq_a2,
point_abs_a2 (round-6 pointwise arms), AND the two MMD references RE-RUN at
each offset (mmd given the bimodal target, mmd given the unimodal target —
round-2 results cannot be reused here because the init changed; the runs are
cheap). 2 seeds (42, 1042) × 40 restarts per (arm, c) → 80 per cell; 25 cells.
Note x_bi = −2, x_uni = +2, so c = ∓3 start deep inside one basin and c = 0 on
the ridge.

**Pre-registered predictions.**
- (i) point_sq_a0 lands unimodal from EVERY c (flat ≈ 100% uni curve) — the
  variance pull dominates any basin capture.
- (ii) point_sq_a2 and point_abs_a2 show a CAPTURE WINDOW: bi-basin landings
  appear only for negative-side starts (c < 0) and grow as c → −3; from
  positive-side starts they land unimodal.
- (iii) MMD escapes wrong-side inits: given the bimodal target it reaches x_bi
  even from c = +3 (and mmd-given-unimodal reaches x_uni from c = −3) — the
  strong distributional-robustness readout. If it fails at extreme c, the
  CAPTURE RADIUS is reported honestly (fraction correct vs c, and where it
  drops).

**Readouts.** Fraction landing in each basin vs c per arm (curves,
figures/fig9_init_sweep.png); mean distance-to-correct-design-point for the two
MMD arms per c with 95% CIs; INIT_SWEEP.md.

### Round-7 CORRECTION (2026-09-03, before any close-out; MMD cells invalidated)

A bug in the round-7 build: the `--init_offset` patch to
`exp_identifiability.optimize_arm` was a SILENT NO-OP (its string-replace
anchor did not match and the patch script reported success unconditionally),
while the pointwise path did receive the offset. Consequences, recorded
honestly:

- The 10 round-7 MMD cells (mmd_bi / mmd_uni × 5 offsets) actually ran at
  c = 0 five times over — their byte-identical rows across c (first x̂ values
  [-3.0312, -3.1073, -2.987, ...] in every file) are the fingerprint, since
  torch.manual_seed(run_seed) with no offset gives identical trajectories.
  Those JSONs are quarantined (*.INVALID_no_offset); INIT_SWEEP.md rebuilt
  from the 15 VALID pointwise cells only (their curves vary with c — offset
  verified applied on that path: a0 uni-fraction 0.20/0.68/0.97/1.00/1.00).
- The round-7 smoke "16/16 escapes from c=+3" reading is RETRACTED — it was
  the c=0 trajectory; the x̂ values matching round-2 exactly were the tell.
- The fix is now applied WITH verification (anchor asserted unique, grep
  check; exp_identifiability.py line ~203). Post-fix 3-restart check, mmd_bi
  at c=+3, realised x_T printed: x_T=+3.34/+2.35/+3.06 → x̂=+2.71/−2.41/+3.86 —
  only 1 of 3 escaped. Prediction (iii) is therefore genuinely OPEN and the
  early indication is a REAL capture radius for MMD; the rerun quantifies it.
- Cells to rerun: ONLY the 10 mmd cells (submit_init_sweep.sh array indices
  with arm ∈ {mmd_bi, mmd_uni}: 3,4,8,9,13,14,18,19,23,24). The 15 pointwise
  cells stand. Predictions (i)-(iii) unchanged.

### Round-7 close-out (2026-09-03, after the corrected 10-cell MMD rerun, job 46052413)

*(The correction saga — silent no-op offset patch on the MMD path, quarantined
first-array MMD cells, retracted smoke reading — is documented in the dated
Round-7 CORRECTION note above; the pointwise cells were valid throughout.)*

- **(i) SUPPORTED.** point_sq_a0's landing curve is init-dominated but
  variance-tilted: uni-fraction 0.20/0.68/0.97/1.00/1.00 across c — not the
  flat ≈100% predicted, but at every matched init it sits ABOVE the a=2 and
  |y−2| curves (the variance tilt), and from on-ridge or right-side starts it
  is ≈100% uni.
- **(ii) SUPPORTED.** The capture windows shift with the target exactly as
  predicted: at c=0, uni-fraction 0.97 (a=0) > 0.71 (a=+2) > 0.57 (|y−2|),
  and the 0.5-crossings move right in that order (≈ −2.3, −1.2, −0.2).
- **(iii) PARTIALLY REFUTED — MMD has a FINITE capture radius.** It recovers
  the correct basin from moderate wrong-side inits (mmd_bi 100/100/99% correct
  at c=−3/−1.5/0 and 77% at c=+1.5; mmd_uni 100% at c≥0, 85% at c=−1.5) but
  fails from deep wrong-side starts (mmd_bi 35% correct at c=+3; mmd_uni 54%
  at c=−3) — capture radius roughly 1.5-3 units past the ridge; mean
  dist-to-correct rises to 3.99±0.60 / 3.47±0.61 at the far ends.

Comparative sentence (recorded): MMD's basin selection is TARGET-DRIVEN within
its capture radius and INIT-DRIVEN beyond it, while the scalar arms are
init-driven everywhere with only a variance tilt. Practical implication:
restarts / annealed inits matter for distributional inverse design too — the
target does not rescue a sufficiently wrong initialization.
