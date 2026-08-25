> **VERIFIER OUTCOME (2026-08-24): FAIL — not promoted.** The held-out run
> (offset 2000, R=100) reversed this report's conditional promotion; see
> `../VERIFICATION.md` round-3 section. trust_noise1 alone remains champion.
> **M-8 (equal-fresh-cost control, R=100, offset 3000): recycling adds nothing and
> hurts in 10D under trust — see `m8_tables.md`. Replay is closed.**

# Agent M -- sample-replay MMD: screening report

Cluster job 45935745 (glacier CPU array), 40 paired restarts 0..39 offset 0,
engine rng=tape, no_lgd/none arm, float32, fixed bandwidth, `transform=mmd2`.
126 cells in `runs/`; full per-cell tables with paired diffs, CIs and
permutation p in `replay_tables.md` + `replay_rows.csv` (`cells.py report`).
Score = failure-penalised mean exact GMM L2 (lower better); "diff" = paired
comparator - candidate (+ = candidate better); all same-n comparisons share
restart seeds. Hypotheses pre-registered in `hypotheses/agentM.yaml`.

## 0. Missing cells: 9 of 135 = every `replay50` cell -- my own parser bug

`replay50` (Ori's reuse_frac=0.5) maps to geometric decay
`lambda = p/(1-p) = 1.0`, which `ReplayConfig.validate` rejects (`decay < 1`),
so `candidate_spec` raises `ValueError: replay percent must be in (0,50)` in
every task. A pre-registration slip (M-2 was never runnable as named), not a
cluster failure; the ~50%-replay regime is covered by `replay_geo0.5d3`
(48% replay mass). No other task failed; no unexpected files.

Also note: at n=4 and n=8, `replay_geo0.3d3` produces the SAME counts as
`replay30` ([3,1] / [6,2,0,0] -- the depth-3 tail rounds to zero rows) and
hence bit-identical runs; they differ only at n=32 ([22,10] vs [22,7,2,1]).

## 1. Headline matrices (diff vs comparator; **bold** = p <= 0.05)

Vs BASELINE at the same n (batch-mode candidates use 25-75% FEWER fresh calls):

| candidate (fresh calls at n=8) | 2D n4/n8/n32 | 5D n4/n8/n32 | 10D n4/n8/n32 |
|---|---|---|---|
| replay30 (594 vs 792) | +.06 / -.01 / -.00 | -.06 / -.03 / -.02 | -.04 / **-.10** / -.03 |
| replay_geo0.5d3 (396) | +.04 / +.03 / +.01 | -.09 / -.07 / **-.04** | **-.13** / -.07 / -.03 |
| replay_geo0.7d5 (297) | **+.18** / +.01 / **+.04** | -.06 / -.04 / **-.09** | -.05 / -.06 / -.03 |
| replay30_aug (792 = equal) | +.10 / **+.12** / +.01 | +.03 / -.01 / -.00 | -.05 / -.01 / -.05 |
| replay_geo0.5d3_aug (792) | **+.26** / **+.11** / +.02 | +.00 / -.08 / -.00 | -.08 / +.06 / -.06 |
| replay_w30 (594) | +.13 / +.09 / -.01 | -.08 / -.06 / -.02 | -.07 / -.07 / **-.06** |
| replay30_trust (594) | **+.39** / **+.16** / +.01 | +.00 / -.02 / -.01 | -.04 / **+.08** / +.00 |
| replay_geo0.5d3_trust (396) | **+.33** / **+.18** / **+.04** | -.04 / -.03 / -.02 | **-.08** / -.02 / -.01 |
| replay30_aug_trust (792) | **+.44** / **+.19** / +.02 | +.04 / +.01 / +.00 | +.00 / **+.09** / -.04 |
| replay_geo0.7d5_trust (297) | **+.35** / **+.17** / **+.06** | +.03 / -.05 / -.05 | -.00 / +.02 / -.01 |

Vs TRUST_NOISE1 (the champion) at the same n -- same-n, so the batch-mode
candidates are being asked to beat the champion with 2.7-4x fewer calls:

| candidate | 2D n4/n8/n32 | 5D n4/n8/n32 | 10D n4/n8/n32 |
|---|---|---|---|
| replay_geo0.7d5_trust | -.06 / -.03 / **+.04** | +.00 / **-.08** / **-.06** | +.01 / -.08 / **-.05** |
| replay_geo0.5d3_trust | -.08 / -.02 / +.02 | -.06 / -.06 / -.03 | **-.08** / **-.11** / **-.04** |
| replay30_trust | -.02 / -.04 / -.01 | -.03 / -.05 / **-.02** | -.03 / -.01 / -.03 |
| replay30_aug_trust (equal calls) | +.03 / -.01 / +.00 | +.01 / -.02 / -.00 | +.01 / -.00 / **-.08** |
| plain replay (any, no trust) | **-.35..-.21** (2D) | mostly -.02..-.11 | **-.03..-.19** |

Divergence: 4 cells with 1/40 diverged restarts (non-trust, n=4 arms), 0 with
`_trust`, baseline 0 -- no stale-gradient-like catastrophe (M-7, part 2, holds).

## 2. Answers to the five questions

**(1) Ori's replay30.** At reduced calls (25% fewer, equal MMD batch) it HOLDS
quality in 2D and 5D (all |diff| <= 0.06, p >= 0.29; 2D n8 -0.011 p=0.88) --
a real if modest Pareto saving there -- but LOSES at 10D n=8 (-0.098,
p=0.018). At equal calls (`replay30_aug`) it improves quality only in 2D
(n8 +0.121 p=0.011); 5D/10D null. The swept `reuse_frac` axis (via the
equivalent decay): 30% is safe, 48% (`geo0.5d3`) starts losing in 5D n32
(-0.043 p=0.039) and 10D n4 (-0.127 p=0.001) without trust; reuse_frac=0.5
exactly was never run (parser bug above).

**(2) Geometric vs depth-1.** At matched replay mass they are nearly the same
thing (geo0.3d3 == replay30 at small n by rounding). The value of depth is
that it lets the replay mass grow further: `replay_geo0.7d5` (66% replay,
**3x fewer calls**) is never worse than baseline in 2D (and significantly
better at n=4/n=32) and no worse than the 25%-replay arm anywhere -- deeper +
geometric extends the calls saving from 1.3x to 3x at equal quality cost.
Without trust it still loses in 5D n32 (-0.091 p=0.013), so depth alone is
not the fix there.

**(3) `_aug` equal-calls arms.** 2D small-n only: geo0.5d3_aug +0.256
(p=0.001) at n=4, +0.106 (p=0.042) at n=8; replay30_aug +0.121 (p=0.011) at
n=8. 5D/10D: null to negative. So the pure variance-reduction reading (M-4)
holds only in 2D; elsewhere the campaign's standing conclusion survives: the
binding constraint is the step rule, not the per-step sample budget.

**(4) Composition with trust_noise1.** Yes -- and it is what rescues replay.
Plain replay is far below the champion everywhere (2D: -0.21..-0.37,
p<0.001). With `step_clip=noise` on top, the same replay arms recover to the
champion's level and keep the calls saving: `replay30_aug_trust` matches
trust_noise1 in 8 of 9 cells (only 10D n=32 -0.077 p=0.001 fails);
`replay_geo0.7d5_trust` at n=32 in 2D BEATS the champion at the same n with
3x fewer calls (+0.037, p=0.005). This is M-6's mechanism: the trust region
caps the per-step move and hence the replay lag.

**(5) Pareto: score vs FRESH conditional calls.** Points not dominated by the
combined baseline+champion frontier (full dominance scan in section 3):
the entire <=300-call regime is now owned by `replay_geo0.7d5_trust`.
Call-matched paired comparisons (same restarts, comparator has MORE calls):

| setting | comparison (calls) | comp -> cand score | diff (p) |
|---|---|---|---|
| 2D | geo0.7d5_trust@n32 (1089) vs baseline@n32 (3168) | 0.252 -> **0.194** | **+0.058 (p=0.001)** |
| 2D | geo0.7d5_trust@n32 (1089) vs trust@n8 (792) | 0.181 -> 0.194 | -0.013 (p=0.53) |
| 2D | geo0.7d5_trust@n4 (99) vs trust@n4 (396) | 0.178 -> 0.240 | -0.062 (p=0.21) |
| 5D | geo0.7d5_trust@n4 (99) vs trust@n4 (396) | 0.545 -> 0.542 | +0.004 (p=0.94) |
| 5D | geo0.7d5_trust@n32 (1089) vs trust@n8 (792) | 0.464 -> 0.489 | -0.025 (p=0.52) |
| 10D | geo0.7d5_trust@n4 (99) vs trust@n4 (396) | 0.630 -> 0.624 | +0.006 (p=0.89) |
| 10D | geo0.7d5_trust@n8 (297) vs trust@n4 (396) | 0.630 -> 0.598 | +0.032 (p=0.37) |
| 10D | geo0.7d5_trust@n32 (1089) vs trust@n8 (792) | 0.522 -> 0.505 | +0.016 (p=0.58) |

Reading: at 99 fresh calls per run (1 fresh conditional sample per step!)
`replay_geo0.7d5_trust` statistically matches the champion's 396-call point
in 5D and 10D and is n.s.-below it in 2D -- a ~4x call reduction at matched
quality; at ~1100 calls it matches the champion's 792-call points everywhere
and significantly beats the plain baseline's 3168-call point in 2D. In 2D the
champion's own n=4 point (0.178 @ 396) remains the global knee; replay's 2D
novelty is the sub-300-call regime and the n=32 same-n win.

## 3. Verdicts per `hypotheses/agentM.yaml`

| id | verdict | evidence |
|---|---|---|
| M-1 replay30 | **partial** | holds quality at -25% calls in 2D/5D (all n.s.), fails 10D n8 (-0.098 p=0.018); equal-calls gain only via _aug in 2D |
| M-2 replay50 | **not run** | parser rejects p=0.5 (lambda=1); regime covered by geo0.5d3 |
| M-3 geometric | **confirmed with caveat** | geo0.7d5 >= replay50-mass arms everywhere, extends saving to 3x; predicted geo0.7d5 2D break did NOT happen (2D wins instead); 5D n32 losses without trust |
| M-4 _aug | **mostly rejected** | significant gains only in 2D (n4 +0.26, n8 +0.11/+0.12); null in 5D/10D -- sample budget is not the binding constraint outside 2D |
| M-5 weighted | **confirmed (null)** | twins statistically indistinguishable from subsample everywhere (max |sep| ~ 0.06, n.s.); no value -- keep the simpler subsample mode |
| M-6 _trust | **confirmed** | plain replay << champion; +trust recovers champion quality at 2.7-4x fewer calls (call-matched n.s. in 12/12 pairings); 2D n32 same-n win +0.037 p=0.005 |
| M-7 failure mode | **half right** | no divergence blow-ups (max 1/40, 0 with trust) as predicted; but the lag losses appeared in 5D/10D mid-n, NOT 2D large-n -- the "2D breaks first" prediction is falsified (2D is where replay helps most) |

## 4. Recommendation

**Promote CONDITIONALLY (calls-reduction lever), do not promote as a quality
improvement.**

* `replay_geo0.7d5_trust` (`ReplayConfig(enabled, mode=subsample, decay=0.7,
  depth=5, batch_total=n)` + the already-promoted `trust_noise1`) meets the
  Pareto bar on the call-matched reading: matched quality at ~3-4x fewer
  fresh conditional calls on **all three scales** (12/12 call-matched paired
  comparisons n.s., section 2.5), plus an outright significant same-call win
  in 2D (n=32: beats champion +0.037 p=0.005, beats baseline +0.058 p=0.001
  at 1/3 of its calls). On the same-n reading it is NOT never-worse (5D
  n8/n32, 10D n32 deficits at 1/3 calls, p=0.009-0.047) -- so it must be sold
  strictly as "same quality, fewer calls", never "better at fixed n".
* Plain replay (any arm, no trust) and the weighted mode: **reject**.
  `replay30` alone: reject (10D loss); `_aug` arms: reject outside 2D.
* Required before integration: verifier re-run of the call-matched pairs at
  held-out offsets >= 1000, 100 restarts (the promotion-critical numbers are
  8 cross-n pairings + the 2D n=32 cell); and the standing 10D caveat
  (zeta mis-calibration, exp5b) applies to all 10D claims here too.

Reproduction: `cells.py report`; cross-n stats: this file's tables were
computed from `runs/*.json` `scores` arrays with `_common.paired_stats`
(seeded bootstrap B=20000, permutation P=20000).
