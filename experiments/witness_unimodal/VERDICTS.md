# Round-1 verdicts (40 paired restarts x 5 targets x 4 arms, no step cap)

*Recorded 2026-09-01 from RESULTS.md built over `results/witness_unimodal_seed42_cell_*.json`.
Round-2 (step-cap tau=1, tag `cap1`) reruns are motivated by verdict V2; see
HYPOTHESIS.md Amendment 2.*

- **V1 — P4 (witness > uniform, most at small s): NOT SUPPORTED.** The paired
  witness − uniform diff is null in every cell: no 95% CI excludes 0 on any metric
  in any of the 5 targets. Final-MMD point estimates actually favor uniform in 4/5
  targets (bimodal_c1 +0.84, unimodal_s010 +0.94, s025 +0.80, s050 +0.03; only
  bimodal_c2 −0.24), i.e. the predicted small-s ordering did not appear.
- **V2 — mean-vs-MMD contrasts: INVALID as run.** All mmd − mean final-MMD diffs are
  positive with CIs excluding 0, but the MMD arms diverge: raw `optimize_LGD` applies
  `x_t = x_{t-1} − ζ·grad` with no step cap (known legacy defect), and restarts drift
  off-manifold (mean |mean err| 20-680 per arm; medians 0.3-26), while the mean arm
  alone was norm-clipped. The comparison is asymmetric and says nothing about
  mean-blindness. P2/P3 unresolved pending the capped rerun.
- **V3 — P1 (mean arm aces its own metric): SUPPORTED.** Mean arm median
  |mean err| ≈ 0.065 across targets, an order of magnitude below every MMD arm's
  median, with zero divergent restarts (clip on).
- **V4 — mean arm on unimodal targets:** it does NOT concentrate to the target scale
  (gen std median ≈ 0.45 vs s = 0.10/0.25), consistent with mean-blindness, but the
  concentration comparison against MMD arms is contaminated by V2.

**Protocol difference vs Ori's runs (recorded 2026-09-01):** the round-1 array ran
optimize_LGD with `use_inv_sqrt_alpha_scale` at its default `False` (constant ζ = 1
step scale), whereas Ori reports his own simulations used `True` (the TFG line-9
1/√α_t convention). Round-1 verdicts V1-V4 therefore apply to the ζ-constant
protocol only. Round 2 (identifiability, see HYPOTHESIS.md Round 2) exposes
`--inv_sqrt_alpha` and runs a step-convention factor on the mmd arm.

---

# FINAL verdict — Round 2 identifiability, both seeds (recorded 2026-09-01)

Primary array seed 42 + pre-registered confirmation array seed 1042, 40 paired
restarts each (pooled pairs keyed by (seed, restart), n=80). Full tables in
IDENTIFIABILITY.md; predictions R1-R4 in HYPOTHESIS.md "Round 2".

- **R1 — mean-blindness: CONFIRMED, both seeds.** The mean arm's landing
  distribution over x̂ is statistically identical across the bimodal and unimodal
  targets (KS: seed 42 p=0.99, seed 1042 p=0.77, pooled p=0.56, n=80/80), and it
  lands by prior mass (~40-50% per basin) regardless of target — as forced by the
  construction (E[y|x] ≡ 0 for every x). Every MMD variant separates the targets
  at pooled KS p < 1e-4. Mean-matching inverse design cannot distinguish targets
  that share a mean; MMD guidance can.
- **R2 — MMD identifiability: CONFIRMED, both seeds.** The full-MMD arm lands in
  the correct basin 40/40 per seed per target, with diagonal-dominant cross-MMD
  (seed 1042: 0.017 to bi vs 3.47 to uni on the bi target; 0.018 to uni vs 3.51
  to bi on the uni target). Caveat: the strict within-0.5 landing criterion
  under-counts (basin overshoot where MMD is float-flat); basin side + cross-MMD
  are the meaningful readouts.
- **R4 — step convention: CONFIRMED.** Without the cap, inv_sqrt-only overshoots
  (|x̂−x_uni| = 2.64±3.45 / 1.90±1.49 by seed vs ≈1.0±0.5 for both capped
  conventions); cap-only and cap+inv_sqrt are equivalent and stable. The
  trust-region cap, not the 1/√α_t scaling, is what stabilises this loop.
- **R3 — witness-vs-uniform on the concentrated target: DIRECTION-CONSISTENT,
  MAGNITUDE NOT REPRODUCED; pooled effect marginal.** dist_uni (uni target,
  witness − uniform): seed 42 −0.073 ± 0.066 (significant), seed 1042
  −0.020 ± 0.068 (same sign, null), pooled −0.047 ± 0.047 (CI excludes 0 by a
  hair; pooled dist_bi −0.058 ± 0.051 likewise). Bimodal-target contrasts stay
  null/mixed at both seeds (pooled dist_bi +0.093 ± 0.198). Read: the
  pre-registered asymmetry (witness helps where off-mode stragglers exist, i.e.
  the concentrated unimodal target, and not on the bimodal target) is directionally
  supported but the effect is small (≈0.05 in x-units against modes 4 apart) and
  seed-sensitive — suggestive, not conclusive.
- **Caveats (carry into any write-up):** (1) y-space eval metrics are basin-level
  only — evaluation is seed-paired and p(y|x̂) is float-exactly constant within a
  basin, so same-basin arms give byte-identical y-metrics by construction;
  x-space distances carry the within-basin signal. (2) The R3 effect is one
  borderline pooled cell. (3) Scope: a fully analytic 1-D toy with an
  ST-free implicit-reparameterisation oracle; round-1's model-based toy showed
  P4 null under the uncapped protocol. Next escalation is SD_PLAN.md's
  androgynous-scribble experiment: single androgynous prompt group vs M/F groups,
  arms mean/MMD/MMD+witness, fresh-sample MMD + gender-axis projection
  bimodality check — the image-space analogue of R1/R3.

---

# Round-3 verdicts — concentration sweep (recorded 2026-09-01)

5 s-values x 3 arms x 2 seeds x 40 restarts (80 pooled pairs per s); tables in
CONCENTRATION.md, money plot figures/fig4_witness_vs_s.png; pre-registration
HYPOTHESIS.md "Round 3".

- **C1 — "witness advantage grows with concentration": REFUTED in direction.**
  Pooled paired dist_uni diff (witness − uniform): +0.050 ±0.183 (s=0.05),
  +0.018 ±0.192 (s=0.1), −0.047* ±0.047 (s=0.25), −0.094* ±0.050 (s=0.5),
  −0.046* ±0.032 (s=1.0). The advantage is significant at moderate-to-fat s and
  is a high-variance null exactly where it was predicted largest.
- **Flip decomposition (why small-s variance blows up).** Basin flips (x̂ < 0)
  out of 80: backsel arms 10-11 (s=0.05), 7 (s=0.1), 4 (0.25), 1 (0.5), 5 (1.0);
  full mmd ≤1 everywhere. Excluding pairs where either backsel arm flipped:
  s=0.05 → −0.010 ±0.035 (n=68), s=0.1 → +0.017 ±0.045 (n=71) — tight nulls;
  s=0.25/0.5/1.0 → −0.050*/−0.095*/−0.049*, essentially unchanged. So (a) the
  small-s cells are flip-noise, not a hidden effect, and within-basin witness ≈
  uniform there; (b) flipping is a cost of backprop-subsampling itself (both
  backsel arms flip alike), not of witness scoring.
- **C2 — straggler-selection mechanism: NO SUPPORT at any s.** Witness
  frac_selected_offmode tracks its base rate everywhere (max |dev| ≈ 0.004,
  e.g. 0.204 vs 0.208 at s=0.05, 0.087 vs 0.090 at s=0.5) — witness selection
  is NOT over-sampling off-mode stragglers. The pre-registered mechanism for
  the user's hypothesis fails even where the advantage exists.
- **C3 — sanity: HOLDS.** Full-mmd basin correctness ≥ 98.75% at every s.
- **Honest mechanistic reading.** With the 0.3 uniform floor and (plausibly)
  low-dispersion witness scores, the witness arm's selection probabilities may
  be near-uniform — making it effectively a differently-seeded uniform arm.
  A pure re-randomization should average to zero paired diff, and the
  moderate-s effect is significant at both seeds independently (s=0.5:
  −0.074*/−0.113*), which argues against a pure RNG artifact — but with C2
  dead we have no positive mechanism: if selection is informative it is not
  via mode membership (perhaps via extremeness WITHIN the mode, which the
  binary |y|>2s probe cannot see), and an RNG-stream interaction is not
  excluded. Score dispersion was not logged in round 3, so this is
  undetermined on current data.
- **The one cheap decisive test (proposed, NOT run):** add a
  `witness_shuffled` control arm — compute the witness scores, randomly permute
  them across the batch, then select as usual. Identical probability histogram,
  floor, and RNG consumption; zero information about which sample scored what.
  If shuffled ≈ witness, the moderate-s advantage is a selection-stream
  artifact; if shuffled ≈ uniform while witness stays ahead, selection is
  genuinely informative. Wire witness_utils.witness_scenario_stats (ess_raw /
  score std, already implemented) into the same run to read dispersion directly.
- **Bottom line: the user's more-concentrated-more-help hypothesis is not
  supported.** Witness back-selection's benefit in this construction is a
  modest, real-looking x-space improvement at moderate-to-fat target widths
  (~0.05-0.09 against modes 4 apart), absent at high concentration, and its
  advertised mechanism (straggler selection) is contradicted by its own
  selection statistics.

---

# Round-4 verdicts — sharpened selection + shuffled control (recorded 2026-09-02)

3 s-values x 6 arms x 2 seeds x 40 restarts (80 pooled pairs per (s, arm));
tables in SHARPNESS.md, figure figures/fig5_sharpness.png; pre-registration
HYPOTHESIS.md "Round 4". All stochastic arms share the with-replacement + IPW
estimator; top-k is deterministic and BIASED (documented, no IPW correction
exists).

- **S1 — sharpening happened as designed: CONFIRMED.** Normalized selection
  entropy: 1.000 → 0.951 (β=1) → 0.889 (β=2) → 0.794 (β=4) → 0.600 (top-k, by
  the ln k/ln n convention), essentially identical across s. Score dispersion
  is modest and flat (max/med ≈ 2.0 at every s) — witness scores are never
  strongly peaked in this construction.
- **S2 — sharpening helps: SUPPORTED (with a small-s caveat).** Paired dist_uni
  diff vs uniform at s=0.5: β=1 −0.069 ±0.168 n.s. → β=2 −0.233* → β=4 −0.319*
  → top-k −0.476* — monotone in sharpness. Top-k is also best at s=0.25
  (−0.501* ±0.156). At s=0.1 the ordering breaks: β=2 (−0.233*) is the only
  significant arm and top-k collapses into flip-noise (+0.033 ±0.338) —
  deterministic concentration of the gradient is fragile exactly where round 3
  showed basin flips.
- **S3 — mechanism (shuffled control): SPLIT.** β=4 − shuffled: s=0.5 −0.206*
  ±0.169 (genuinely informative selection there); s=0.25 −0.107 ±0.187 and
  s=0.1 −0.047 ±0.236 (no separation). The shuffled arm itself runs at
  −0.088..−0.113 n.s. at every s, so at s ≤ 0.25 the witness-family advantage
  is compatible with a selection-stream artifact (non-uniform IPW weighting
  alone), and only s=0.5 carries demonstrated signal.
- **S4 — off-mode mechanism: FAILS everywhere (round-3 C2 stands).**
  frac_selected_offmode tracks — if anything sits slightly BELOW — the base
  rate for every witness arm at every s (e.g. s=0.5 β=4: 0.077 vs base 0.086;
  top-k 0.094 vs 0.113), while the shuffled arm matches base. **The mechanism
  of the s=0.5 informative signal remains unidentified**: it is not off-mode
  (straggler) selection; within-mode extremeness is the remaining candidate
  (untested — would need a finer positional probe than the binary |y|>2s).
- **Honest caveats.** (1) Top-k is the best arm where it works but carries the
  documented estimator bias — before recommending it, run a bias-check cell:
  top-k vs full-mmd (n=32, no subsampling) on final quality, to confirm its
  gradient bias is not steering x systematically. (2) The round-4 estimator
  (IPW, with replacement, floor 0) differs from rounds 2-3 (plain detach,
  floor 0.3), so magnitudes are not directly comparable across rounds — note
  the uniform reference moved too. (3) 1-D analytic toy; basin-level y-metric
  caveat unchanged.

---

# Round-5 verdicts — with- vs without-replacement sampling (recorded 2026-09-02)

2 s-values x 4 without-replacement arms x 2 seeds x 40 restarts (80 pooled
pairs per (s, arm)); with-replacement side reused from round 4 (same
seeds/restarts, exact pairing). Tables in REPLACEMENT.md, figure
figures/fig7_replacement.png; pre-registration HYPOTHESIS.md "Round 5".

- **R5 — "without-replacement equal-or-better at high β": REFUTED.** The direct
  paired contrast (wor − wr) is never negative where sharpened: +0.119* ±0.095
  at (s=0.25, β=2) — significantly WORSE — and +0.052/+0.076/+0.071 n.s. at the
  other sharpened cells; ≈0 at β=1 (−0.001/−0.014), as predicted only for the
  unsharpened case.
- **Same story vs the matching-mode uniform references:** the wor witness
  advantage shrinks at every cell (s=0.5: β=4 −0.190* vs wr −0.319*, β=2
  −0.098* vs −0.233*; s=0.25: β=2 +0.005 vs −0.197*, β=4 −0.061 n.s. vs
  −0.196*).
- **Estimator sanity check: PASSES.** uniform_wor (exact n/k HT) vs mmd_uniform
  (exact IPW) agree: −0.083 ±0.140 / −0.059 ±0.128 n.s. — so the wor deficit in
  the witness arms is not an artifact of the without-replacement machinery per
  se.
- **Honest mechanistic reading.** With-replacement's duplicates are not wasted
  budget: a row drawn c times enters the gradient with weight c/(k·p_i), so
  duplication IS the sharpened emphasis on the highest-|witness| rows — the
  very thing round 4 showed helps. Without-replacement forces 8 distinct rows,
  discarding that emphasis (each row counted once, HT-reweighted toward
  flatness), and layers on the documented π ≈ 1−(1−p)^k approximation bias
  (selftest-measured max error ≈ 0.12 at large p_i — precisely the rows that
  matter at high β). The analytically estimated 12-20% "duplicate share" of the
  k=8 budget was emphasis, not waste.
- **Recommendation: keep WITH-replacement sampling + IPW for the witness rule.**
  (Unchanged flags from round 4: top-k remains best-but-biased pending the
  proposed bias-check cell; the s=0.5 signal is informative per S3 but its
  mechanism is still unidentified; 1-D analytic toy scope.)

---

# Round-6 verdicts — pointwise scalar-target baseline (recorded 2026-09-02)

3 arms x 2 seeds x 40 restarts (80 per arm); round-2 mmd cells reused as
references (no rerun). Tables in POINTWISE.md, figure figures/fig8_pointwise.png;
pre-registration HYPOTHESIS.md "Round 6" (competing P-a vs P-b, both stated
up front).

- **Headline: squared scalar-target design is VARIANCE-SEEKING.** point_sq_a0
  (a=0, the shared conditional mean): 97% of landings in the tight-unimodal
  basin (mean MMD→uni 0.105 vs MMD→bi 3.42). point_sq_a2 (a=+2 — a value
  sitting EXACTLY on a mode of the bimodal conditional): still 71% unimodal.
  point_abs_a2: 42% bi / 57% uni, the near coin-flip its pre-registered
  weak-margin prediction (E|y−2| margin ≈ 0.1 vs the squared loss's ≈ 4.0)
  anticipated. References: MMD with the actual target sets is 99%/100%
  basin-correct.
- **Scoring the pre-registered alternatives, honestly.** P-a (target-blind,
  prior-following splitting) is refuted in its literal form — a=0 goes 97% to
  one basin and the target value measurably shifts the landing distribution.
  P-b (strict variance seeking, a0/a2 landings identical) is refuted in its
  strict form — KS a0-vs-a2 pooled p=0.0007 (per-seed 0.054 / 0.014). The
  recorded reading — variance domination with a residual target-value nudge —
  is a POST-HOC synthesis of the two pre-registered alternatives, not itself a
  pre-registered prediction. Either way the paper claim holds: the scalar
  target did not determine the answer.
- **Paper-facing sentence:** pointwise scalar-target inverse design under
  conditional stochasticity implicitly optimizes mean-plus-variance, so it
  collapses toward the lowest-variance conditional almost regardless of the
  requested value — it cannot express, and barely responds to, WHICH
  conditional distribution the practitioner wants; distributional (MMD)
  targets determine it essentially perfectly.
- **Caveats.** n=1-per-step is the practitioner-realistic but noisiest
  estimator (the a-nudge might grow with per-step batch size — untested);
  |y−a| behaves differently, as pre-registered (its E-loss margin between
  basins is ~40x smaller, and the data show the predicted near-split); 1-D
  analytic toy scope, basin-level y-metric caveat unchanged.

---

# Round-7 verdicts — init sweep: objective pull vs basin capture
# (recorded 2026-09-03; corrected MMD cells from job 46052413)

5 offsets x 5 arms x 2 seeds x 40 restarts (80 per (arm, c)); x_T ~ c + N(0,1).
Tables in INIT_SWEEP.md, figure figures/fig9_init_sweep.png; pre-registration
HYPOTHESIS.md "Round 7". The first array's 10 MMD cells ran without the offset
(silent no-op patch) and were quarantined and rerun; the retraction of the
interim "MMD escapes from c=+3" smoke reading is documented in the dated
Round-7 CORRECTION note. Pointwise cells were valid throughout.

- **(i) — variance pull vs init, point_sq_a0: SUPPORTED (init-dominated,
  variance-tilted).** Uni-fraction 0.20/0.68/0.97/1.00/1.00 across
  c=−3..+3 — not perfectly flat, but uniformly ABOVE the a=+2
  (0.11/0.41/0.71/0.93/1.00) and |y−2| (0.03/0.16/0.57/0.88/0.97) curves at
  every matched init: the variance tilt is always there, the init decides at
  deep left starts.
- **(ii) — capture windows shift with the target: SUPPORTED.** The three
  pointwise curves are ordered a=0 > a=+2 > |y−2| at every c, with
  0.5-crossings at roughly −2.3 / −1.2 / −0.2 — the scalar target and loss
  shape move the window exactly as pre-registered.
- **(iii) — MMD escapes wrong-side inits: PARTIALLY REFUTED — finite capture
  radius.** Correct-basin fractions: mmd_bi 1.00/1.00/0.99/0.77/0.35 across
  c=−3..+3; mmd_uni 0.54/0.85/1.00/1.00/1.00. So MMD recovers from moderate
  wrong-side inits (up to ~1.5 units past the ridge at 77-85%) but is
  init-captured at deep wrong-side starts (35%/54% correct at 5 units past the
  correct design point); mean dist-to-correct rises from ≈0.9-1.0 (c=0) to
  3.99±0.60 (mmd_bi, c=+3) and 3.47±0.61 (mmd_uni, c=−3). Capture radius
  roughly 1.5-3 units past the ridge under this cap/schedule.
- **Comparative sentence (paper-facing):** MMD's basin selection is
  TARGET-DRIVEN within its capture radius and INIT-DRIVEN beyond it; the
  scalar arms are init-driven everywhere, with only a variance tilt
  distinguishing their targets. Distributional targets buy a wide — but not
  unbounded — basin of attraction.
- **Practical implication:** restarts and/or annealed initializations matter
  for DISTRIBUTIONAL inverse design too — a sufficiently wrong init defeats
  the target; multiple restarts scored by final MMD (which is diagnostic:
  wrong-basin landings show MMD ≈ 3.4 vs ≈ 0.02-0.06 correct) recover the
  round-2 reliability cheaply.
- **Caveats:** capture radius is specific to this cap (τ=1) and cosine
  schedule — an uncapped or larger-τ run trades stability for reach
  (untested); asymmetry between mmd_bi (35%) and mmd_uni (54%) at the far end
  plausibly reflects the uni basin's flatter far side, not a target property;
  1-D analytic toy scope.
