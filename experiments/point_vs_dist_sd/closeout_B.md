
---

# CLOSE-OUT — Scenario B, all arms (2026-09-13)

*Scored against HYPOTHESIS.md (pre-registration + Amendments 1-3 + the P-C1 power note).
Full scoring tables live in HYPOTHESIS.md "CLOSE-OUT 3"; this is the reader's summary.*

## Headline

**No configuration we found makes SD-scale distributional guidance improve on the unguided
baseline over the full 250-step schedule. A 20-step (40/20) schedule does — the only
configuration that works — and the mechanism for the difference is measured.**

| | full schedule 250/125 | gate schedule 40/20 |
|---|---|---|
| unguided baseline (fresh MMD) | 0.6000 | 0.5938 |
| guided, zeta 12 + tau 0.2 | 0.6358 (**worse**) | **0.4911** (better by +0.103) |
| guidance objective trend (P-C1) | +0.0043, corr -0.031 → FAIL (6.3 SE) | -0.0388, corr -0.567 → PASS |
| fraction of applied guidance surviving to the final latent | **0.147** | **0.495** |

The two schedules share the same unguided baseline, so this is not an image-quality
artefact of the schedule: it is what guidance manages to keep. On the fine schedule each
correction is followed by 124 further denoising steps that re-project the latent toward the
model's manifold; the full run applies **5x more total correction** (452 vs 91) for a
*smaller* net displacement benefit and no improvement. The gate's success is a
short-horizon / coarse-step effect, and that is the finding rather than a caveat to it.

## What each arm shows

* **Arm 3 (distributional, MMD).** Uncalibrated it never entered the measured descent band
  (median step 1.84 vs band 5.9-23.5) and was flat. Calibrated (zeta 12, tau 0.2) it is
  still flat over 125 steps and still worse than doing nothing — but the same settings on
  20 steps descend and beat the baseline. Cap-throttling is refuted (the cap binds on 0-4 %
  of steps in Q1-Q4); sub-band stepping persists (in-band 4-28 %) because the single
  t = 497 probe does not transfer across the schedule.
* **Arm 2 (pointwise).** Both calibrated configurations rise rather than fall — but the
  zeta = 0 **drift control** rises by +0.0556 +- 0.0140 on its own, and the residuals
  (+0.057 +- 0.047, z = 1.2; +0.021 +- 0.040, z = 0.5) are **not resolvable**. Per the
  pre-registered reading the claim is *"the point arm does not improve on the drift
  baseline"*, not "guidance makes it worse". Separately, P-C1 is **underpowered** for this
  arm (its 1-sample loss has sd ~0.09, so -0.03 is 0.7-0.8 SE) — its FAIL is uninformative.
* **Arm 1 (AdvI2I-style PGD).** Six runs spanning **five distinct configurations**
  (eps 32/64/128 at lr = eps/10, plus lr = eps/100 and eps/3 at eps 32; eps 32 @ lr = eps/10
  was run twice) **all** end far above their starting point, and none ever drops below its
  own step-0 loss. Per the registered two-stage rule, the two lrs that passed stage (i) by a
  hair were fresh-evaluated (stage ii) and both confirm failure; lr = eps/100 failed stage (i)
  and was correctly not evaluated. Fresh 2000-sample point loss L(x*) is **0.79-0.88 against
  L(x0) = 0.4218** for every evaluated config, and fresh MMD 0.700-0.801 against the 0.3776
  do-nothing baseline.

## What is NOT established

1. That in-band stepping is *necessary*: the gate descended with steps mostly **below** its
   band (median 3.65, in-band 20 %). Only far-sub-band stepping (median 1.84) clearly failed.
2. That the point arm cannot descend: its criterion had no power, and its rise is drift.
3. Anything about the prior-vs-point and point-vs-distributional contrasts the experiment
   was built for — see the reachability defect below.

## The standing defect (independent of everything above)

G_bal's PC1 variance is ~97 % between-group, created by the two generating PROMPTS, while
guidance and evaluation both use one neutral prompt. The best PC1 variance ratio anything
achieved is the untouched reference f_phi(x0) = 0.300; the unguided trajectory alone drops
it to 0.046. The target is therefore close to unreachable in this setup, which is why every
arm's PC1 ratio is 0.005-0.02 and why P-B2 cannot be tested fairly here. **Scenario A**
(target generated with the same neutral prompt from x_f, reachable by construction) is the
clean test and is unaffected by this defect.
