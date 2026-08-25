# Agent P -- preconditioning: final screening report (round 3)

Cluster job 45935744, 90 cells, 40 paired restarts 0..39 (offset 0), arm
no-LGD/none, rng=tape, float32; results in `precond/runs/`, per-cell tables in
`precond_tables.md` (paired mean diff, bootstrap 95% CI, permutation p, from
`cells.py report`), rows in `precond_rows.csv`. Pre-registration:
`../hypotheses/agentP.yaml`; theory: `THEORY.md`; implementation and smoke:
`REPORT_PHASE1.md`. Every cell has cm_samples identical to baseline (all rules
are O(d) transforms of the already-computed gradient).

## 0. Controls -- PASS

In the 2D setting the spatial x is 1-dimensional, so the norm-preserving
direction rules (cov / diag / sign) are provably the identity. The screen
confirms it exactly: their 2D scores equal the baseline's to every printed
digit and the paired diff is +0.000 with p = 1.000 in all 9 control cells;
likewise every `*_trust` form equals trust_noise1 exactly in 2D. The plumbing
is end-to-end exact.

## 1. Result matrices (paired diff, + = candidate better; ** = p <= 0.05)

vs BASELINE (score in brackets where notable):

| candidate | 2D n=4 | 2D n=8 | 2D n=32 | 5D n=4 | 5D n=8 | 5D n=32 | 10D n=4 | 10D n=8 | 10D n=32 |
|---|---|---|---|---|---|---|---|---|---|
| trust_noise1 (champion) | **+.407** (.178) | **+.200** (.181) | +.021 (.231) | +.028 | +.034 | +.006 | -.009 | **+.094** (.522) | +.038 |
| precond_cov | 0 (identity) | 0 | 0 | +.005 | -.023 | -.007 | -.035 | -.024 | **-.095** |
| precond_cov_trust | = trust | = trust | = trust | **+.111** (.462) | +.049 | **+.019** (.418) | +.043 | +.014 | **-.067** |
| precond_diag | 0 | 0 | 0 | -.039 | -.042 | -.016 | -.075 | -.042 | -.009 |
| precond_diag_trust | = trust | = trust | = trust | +.042 | +.044 | **+.009** (.428) | +.002 | +.058 | +.016 |
| precond_sign | 0 | 0 | 0 | +.017 | -.034 | -.005 | -.073 | -.002 | -.032 |
| precond_sign_trust | = trust | = trust | = trust | +.069 | +.014 | +.002 | +.002 | +.031 | +.015 |
| precond_median | **+.368** (.217) | **+.254** (.126) | **+.079** (.173) | **-.121** | **-.125** | **-.054** | **-.095** | **-.098** | **-.138** |
| precond_median_trust | **+.364** (.220) | **+.250** (.131) | **+.079** (.173) | -.062 | **-.110** | **-.053** | +.026 | +.048 | **-.087** |

vs TRUST_NOISE1 (the promoted champion; this is the comparison that matters):

| candidate | 2D n=4 | 2D n=8 | 2D n=32 | 5D n=4 | 5D n=8 | 5D n=32 | 10D n=4 | 10D n=8 | 10D n=32 |
|---|---|---|---|---|---|---|---|---|---|
| precond_cov_trust | 0 | 0 | 0 | +.083 (p=.067) | +.016 | **+.013** | +.053 | **-.080** | **-.105** |
| precond_diag_trust | 0 | 0 | 0 | +.014 | +.010 | +.003 | +.011 | -.035 | -.021 |
| precond_sign_trust | 0 | 0 | 0 | +.041 | -.019 | -.004 | +.011 | **-.063** | -.022 |
| precond_median | -.039 | +.054 (p=.064) | **+.058** (.173 vs .231) | **-.148** | **-.158** | **-.060** | -.085 (p=.051) | **-.191** | **-.175** |
| precond_median_trust | -.043 | +.050 | **+.058** | **-.090** | **-.143** | **-.059** | +.036 | -.046 | **-.125** |

## 2. Verdicts against the pre-registered hypotheses (agentP.yaml)

* **P-1 `precond_cov[_trust]` -- REJECT.** Alone: significant 10D n=32 loss
  (-.095) and no win anywhere; the predicted whitening signal-suppression in
  a noise-dominated covariance is consistent with the 10D pattern. With trust:
  genuinely beats the champion in 5D (n=32 +.013 p<.001; n=4 +.083 p=.067,
  score 0.462 -- the best 5D n=4 seen in the campaign) but is significantly
  WORSE than the champion in 10D at n=8 (-.080) and n=32 (-.105). Regime-
  dependent with one constant -> the same failure class as norm_only.
  (5D-only conditional note: if a calibrated zeta_d run (exp5b) ever makes 5D
  the regime of interest, cov_trust is worth a re-test there.)
* **P-2 `precond_diag[_trust]` -- REJECT.** Alone it never wins and trends
  negative in 5D/10D: the diagonal DIRECTION part of Adam contributes nothing
  positive -- Adam's 2D benefit was the magnitude floor + tail cap, which
  settles the pre-registered attribution question. `diag_trust` is the safest
  candidate on the board (never significantly worse than either comparator in
  any of the 18 comparisons, one small vsB win at 5D n=32 +.009 p=.01) but
  has NO p<=0.05 win over trust_noise1 anywhere -> rejected by the
  pre-registered domination rule: it adds nothing over the champion.
* **P-3 `precond_sign[_trust]` -- REJECT.** No wins; sign_trust significantly
  worse than the champion at 10D n=8. Per-coordinate magnitudes are not pure
  noise; discarding them buys nothing.
* **P-4 `precond_median[_trust]` -- REJECT for promotion; strongest 2D result
  of the round.** In 2D it is the only rule that beats the CHAMPION: n=32
  +.058 vs trust_noise1 (p<.001), score 0.173 vs 0.231 -- also better than
  relclip2's screening 0.212 at that n -- and n=8 0.126 (vsT +.054, p=.064).
  But it loses significantly in 5D at EVERY n (-.05..-.16) and in 10D
  (median alone at every n; median_trust at n=32), exactly the pre-registered
  lag/rotation risk: with d>1 the 5-step window median lags a rotating
  gradient. Fails the "never significantly worse" bar and the 2+ scales bar.

## 3. Promotion recommendation

**Promote: nothing. trust_noise1 remains the champion.** No candidate beats
or matches it on 2+ scales without a significant regression:

| candidate | beats/matches champion on | significant regression | bar |
|---|---|---|---|
| precond_cov_trust | 2D (exact match), 5D (beats) | 10D n=8, n=32 vs champion | FAIL |
| precond_diag_trust | all 3 (matches only) | none | technically passes "matches", but zero wins vs champion anywhere -> rejected by pre-registered domination rule; promoting it would add a mechanism with no measurable benefit |
| precond_sign_trust | 2D (exact match) | 10D n=8 vs champion | FAIL |
| precond_median[_trust] | 2D (beats, incl. the champion at n=32) | 5D all n, 10D | FAIL |

Graveyard additions (do not retry without a new idea): norm-preserving
whitening / diagonal / sign direction fixes (direction is not the binding
constraint; the step-size rule was and remains it); temporal median-of-window
aggregation in d>1 (lag bias beats robustness). Open leads worth a future
pre-registration: (i) `precond_median` with a schedule-shrinking window
(w_t ~ 2 + 3*sqrt(1-alphabar_t)) to cut the late-step lag that kills d>1;
(ii) the within-step median-of-means over the n per-row gradients
(THEORY.md sec e) -- it has NO temporal lag, so the 5D/10D failure mode of
the window median does not apply, at the cost of k extra backward passes;
(iii) cov_trust under a calibrated zeta_d (5D evidence above). None of these
is claimed; all would need the full pre-register/screen/verify path.

No verifier hand-off is requested (nothing to promote).
