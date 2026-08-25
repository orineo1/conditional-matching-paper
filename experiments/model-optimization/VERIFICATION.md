# VERIFICATION -- independent verification and red-team report (Agent 6)

## CURRENT STATUS (2026-08-24, after round 5 + offset-7000 confirmation)

| status | configuration | evidence |
|---|---|---|
| **PASS under the corrected protocol** (x_T ~ N(0,I), per-arm calibrated zeta) | **trust_noise1** (`temporal.step_clip="noise", step_tau=1`, no momentum, no LGD) at zeta 16 / 8 / 4 (2D / 5D / 10D; zeta 8 equally good in 2D) | two independent 100-restart paired runs (offsets 6000, 7000) vs no-trust at ITS best zeta (2 / 0.25 / 1): pooled R=200 diff 2D +0.56/+0.36/+0.21/+0.08 (n=4/8/16/32, all p<0.001), 10D +0.10/+0.12/+0.06/+0.04 (p<0.001/<0.001/0.013/0.07), 5D +0.03/+0.05/+0.06/+0.05 (p=0.10/0.018/0.001/0.016); 0 divergences in 2800 trust runs; never significantly worse in 24 cells -- section 10 |
| Legacy-protocol PASS only (x_T = 0, zeta = 1); NOT re-tested under the corrected protocol, no promotion claim | sqrt_floor, sqrtfloor_clip0.5 (2D+5D conditional); replay_fifo16_trust / replay_cohort16_trust (10D-only, equal fresh cost) | sections 5, 9 |
| Rejected (held-out FAIL or not replicated) | clip0.5, relclip1 (=qclip0.5), relclip2, relclip_ema2 (2D-only / 10D regressions); replay_geo0.7d5_trust (calls-saving claim did not replicate); precond_* (clean reject); adaptive n, stale-k, recur2, crn, antithetic, bandwidth policies, norm_only, unit (screening) | sections 5, 8, 9; PHASE1.md |
| Exact-speed | fast_mmd cached target block: EXACT (promote as drop-in); batched LGD / generator seeding: EXACT; batched restarts: statistical equivalence only, ~10x throughput | section 4 |

Everything below section 10 is the historical record; sections 1-9 carry the
**[legacy protocol]** label.

Campaign `experiments/model-optimization/` (brief FABLE_CDM_PERFORMANCE_ORCHESTRATION),
commit `6af2081` + working tree of branch `tfg-generalization-v2`, 2026-08-23.
Verifier files: `verification/PHASE1.md` (code/invariant review, local checks),
`verification/check_fast_mmd.py|.log`, `verification/check_systems.py|.log`,
`verification/heldout_cells.py`, `verification/submit_heldout.sh`,
`verification/analyze_heldout.py`, results `verification/heldout_runs/*.json` (108),
`verification/heldout_tables.md` (all per-cell tables), `verification/heldout_rows.csv`.
Nothing outside `verification/` and this file was written; no implementing agent's
file was edited.

## 1. Protocol

**Invariants (Phase 1, local).** Full test suite; mutation tests; line-by-line read of
the engine/config diff; programmatic audit of all 519 screening JSONs (offset, restarts,
dtype, calls vs baseline, NaNs); local bit-identity of `_guided.run` vs engine-legacy.

**Exact-speed candidates (Phase 1, local).** Own float64 value+gradient check of
`exact_loss/fast_mmd.py` vs `simulations/src/LossFunctions.MMDLoss` (not the authors'
tests); re-run of `systems/runners.py` equivalence (bit identity, teacher-forced
per-step gradient agreement, end-to-end diffs) on 8 restarts and quiet-as-possible
re-timing with warm-up (median of 5 / 200).

**Estimator candidates (Phase 2, cluster).** Held-out confirmation of the round-2
survivors named by the boss -- `trust_noise1, relclip2, relclip_ema2, sqrt_floor,
clip0.5, relclip1, sqrtfloor_clip0.5` -- through the SAME engine path as the screening
(`estimator/engine_runner.py`, imported unchanged, `rng=tape`, float32), but on
**restarts 1000..1099 (offset 1000, never used by the screening, which used 0..39 at
offset 0), 100 restarts per cell**, settings 2D/5D/10D x n in {4,8,16,32}, arm
no-LGD/none, plus budget-matched Pareto baselines no-LGD/none at n in {64,96} and
LGD/none at n in {8,32}. 108 cells, **18 SLURM array tasks = groups**: one group = one
(setting, n, arm) baseline AND all its candidates run sequentially in ONE process on
ONE node (all 108 cells ran on `AMD EPYC 7662`, glacier; checked from the JSON
`verifier.cpu/host`), so every paired comparison is same-node (see red flag 1).
Budgets: identical `cm_samples` within every pair (asserted in the analysis).
Statistics: paired diff `base - cand` (+ = candidate better) of the failure-penalised
exact GMM L2 (cap 2.0; `_common.penalised_score`), bootstrap 95% CI and paired
permutation p (`_common.paired_stats`, B=P=20000), wins; the same for the secondary
metric `mmd2_eval` = MMD^2 of 256 fresh conditional draws at `x_hat` vs `S_G`
(fixed bandwidth, keyed on the restart only, cap 1.0 on divergence -- the optimisation
objective evaluated at the end point, independent of the guidance noise); success rate
(|x_hat - x*| < 0.5) and divergence counts. The screening's 40-restart estimate of the
same diff is printed next to the held-out one.

**Promotion rule applied (brief):** credible Pareto improvement on held-out seeds at
matched compute (p <= 0.05 and CI excluding 0 on the paired L2 diff, with calls equal),
surviving >= 2 task scales (settings), and no significant regression at any tested cell
of a promoted scale. Per setting: PASS = significant gain at >= 2 of the 4 n and no
significant loss; FAIL = a significant loss at some n; INCONCLUSIVE = neither.

## 2. Commands

```
# invariants
cd simulations && /Users/stolk/miniconda3/bin/python -m pytest tests -q                 # 331 passed, 1 skipped
# exact-speed checks (local)
cd simulations && /Users/stolk/miniconda3/bin/python -m pytest ../experiments/model-optimization/exact_loss -q   # 236 passed
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/check_fast_mmd.py
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/check_systems.py
# held-out run (cluster, job 45923971, glacier, 18 array tasks, no errors)
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
sbatch --array=0-17%18 ../experiments/model-optimization/verification/submit_heldout.sh
# analysis (local)
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/analyze_heldout.py
```

## 3. Invariants -- all hold

| invariant | result |
|---|---|
| default engine path == frozen reference | 331 passed / 1 skipped (pre-existing structural placeholder); `test_defaults_are_still_the_reference` atol 0.0; mutation tests pass |
| new mechanisms opt-in, off by default | all `config.py` defaults are the Algorithm-1 values; default path traced unchanged (`_preprocess`, `_step_clip`, stale, adaptive-n, recurrence all inert); `all_extensions_disabled()` also checks the legacy switches |
| NoiseTape keying | delta `(t,j)`; renoise `(t,r)` only when `r < N_recur`; eta `(t,i)` (opt-in `(t,j,i)` / frozen) |
| `Delta_t/sqrt(alpha_t)` | preserved; the opt-in `guidance_scaling="raw"` is the documented `_guided` convention, proven bit-for-bit |
| schedules rebuilt, not cast | `RepositorySchedule` recomputes the cosine formula per dtype; float32 bit-identical to the checkpoint |
| caches exact / target-only | fast_mmd caches Y-only terms; BatchedMMD caches YY; CMSampler cache = exact reuse of identical `(x, key)` within a step; stale gradient labelled approximate and rejected |
| held-out seeds disjoint | screening: all 519 cells offset 0, restarts 0..39; held-out: offset 1000 (asserted `>= 1000`) |
| budgets matched | within every screening pair except the labelled stale (1/k) / recur (2x); within every held-out pair exactly |
| selection vs evaluation metric | same metric (exact GMM L2: the paper's independent metric, never optimised); the held-out seeds remove the winner's-curse, `mmd2_eval` + success are secondary checks |

## 4. Exact-speed candidates -- verdicts

### 4.1 `exact_loss/fast_mmd.py` (cached target block, Agent 2) -- **EXACT**

Own float64 check (`check_fast_mmd.log`): value and dX-gradient vs `MMDLoss`, worst over
(n,m,d) in {(1,250,2),(4,250,2),(8,250,5),(32,250,10),(7,13,3),(250,8,2),(64,64,4)} x 2
seeds, fixed AND adaptive (stacked, gradient through the bandwidth) bandwidth, variants
cdist/mm x exp/powchain/loop, chunked XY (64, 7), `batched()` incl. perturbed sets,
`reattach_yy="autograd"`: **all <= 1.3e-14 relative**. Authors' 236 tests pass.
In float32 (production dtype) agreement is at round-off (|dgrad| <= 1.6e-7 on |g| 0.16),
i.e. not bit-identical, and the 99-step float32 loop then moves `x_hat` by up to 0.18
(their `end_to_end_results.csv`) -- the chaos of red flag 1, not an error.
Speed (own micro-timing, fwd+bwd, float32, m=250, loaded Mac): n=8 1.96 ms -> 0.42 ms
(fixed_cdist, 4.6x) / 0.28 ms (fixed_mm_powchain, 7.0x); n=32 2.39 -> 0.68 / 0.58 ms
(3.5x / 4.1x). Consistent with their 5-7x (n<=8) and the 1.6-2.6x end-to-end.
**Claim supported: mathematically identical loss and first-order gradient (float64 to
1e-14); a drop-in, opt-in replacement that changes only float32 summation order;
4-7x on the loss, ~1.6-2.6x per restart.** Caveat: the adaptive-bandwidth YY
re-attachment is first-order only (documented; irrelevant for the fixed-bandwidth
synthetic loop).

### 4.2 `systems/runners.py` (batched LGD / batched restarts / cached YY, Agent 5) -- **EXACT for the listed items, otherwise STATISTICAL EQUIVALENCE only**

Own re-run (`check_systems.log`, 2D, 8 restarts): generator seeding, batched LGD,
frozen params, lean DDIM: **0.0 end-to-end (EXACT)**. Cached-YY/batched MMD: per-step
teacher-forced |dg| ~1e-6 (REORDER). Batched restarts B>=2: per-step |dg| up to
**2.7e-3 absolute = 8% relative** at one step (2D n=32) -- larger than the authors'
"<= 2.2e-4". Root cause traced: batching the denoiser changes `pred_x0` by 4.8e-7
(GEMM path) and the CM is a ReLU MLP whose Jacobian jumps when a unit flips (a 5e-7
perturbation of `cond` changes `dMMD/dcond` by 8% at that point; 1e-7 does not); the
float32 reference itself agrees with float64 to 3e-7, so this is round-off amplified by
a discontinuous Jacobian, not an algorithmic difference. End-to-end trajectories differ
O(0.01-3) (chaotic). Timing (own, loaded machine): B=1 1.6-5x, B=8 4-15x per restart;
their B=32 14-25x is plausible on a quiet machine (not re-run).
**Claim supported: batched restarts are the SAME algorithm with the same per-restart
RNG draws; per-step gradients agree to round-off except at occasional ReLU-boundary
steps (up to ~10%); the distribution of outcomes over restarts is the only valid
equivalence statement ("statistical equivalence"), and the throughput gain (order 10x
at B>=8) holds. NOT supported: per-restart or per-step numerical equivalence.**
Any experiment using `run_batched_restarts` must be reported as a re-run, not a
reproduction, of the sequential numbers.

## 5. Estimator candidates -- held-out results

Baselines (held-out, no-LGD/none, score / mmd2_eval): 2D n=4/8/16/32/64/96:
0.597/0.418/0.282/0.247/0.249/0.259; 5D: 0.534/0.508/0.449/0.444/0.434/0.432;
10D: 0.667/0.658/0.564/0.477/0.456/0.442. LGD/none n=8/32: 2D 0.201/0.225, 5D
0.467/0.433, 10D 0.518/0.449. Held-out baselines agree with the screening's within the
seed-noise floor (e.g. 2D n=4 0.597 vs 0.585; 10D n=32 0.477 vs 0.494).

### 5.1 Paired L2 diff (base - cand, + = better), 100 held-out restarts; `*` = p <= 0.05

| candidate | 2D n=4 | n=8 | n=16 | n=32 | 5D n=4 | n=8 | n=16 | n=32 | 10D n=4 | n=8 | n=16 | n=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trust_noise1 | +0.401* | +0.250* | +0.090* | +0.024* | +0.026 | +0.036* | +0.008 | +0.010 (p=.06) | +0.053* | +0.123* | +0.075* | +0.019 |
| relclip2 | +0.444* | +0.279* | +0.114* | +0.026 (p=.06) | +0.010 | +0.010 | +0.007 | +0.015* | **-0.047*** | +0.018 | +0.049* | +0.000 |
| relclip_ema2 | +0.426* | +0.265* | +0.118* | +0.036* | -0.017 | -0.015 | +0.005 | +0.016* | +0.004 | -0.011 | +0.044 (p=.05) | +0.008 |
| sqrt_floor | +0.232* | +0.119* | +0.049 (p=.06) | +0.038* | +0.030 | +0.049* | -0.004 | +0.023* | -0.003 | +0.048 (p=.06) | +0.027 | **-0.038*** |
| clip0.5 | +0.358* | +0.193* | +0.054* | +0.009 | +0.032 | -0.021 | +0.004 | +0.033* | +0.018 | +0.056* | +0.022 | **-0.075*** |
| relclip1 (== qclip0.5) | +0.337* | +0.250* | +0.081* | -0.020 | -0.008 | +0.022 | -0.005 | +0.029* | -0.010 | +0.061* | +0.031 | **-0.076*** |
| sqrtfloor_clip0.5 | +0.360* | +0.212* | +0.066* | +0.036* | +0.033 | +0.059* | -0.009 | +0.035* | +0.002 | +0.073* | +0.015 | **-0.088*** |

CIs, wins, p, success, divergence, `mmd2_eval` diffs and the screening estimates for
every cell are in `verification/heldout_tables.md`. Divergences: baseline 2D n=4 had 2,
every candidate 0 everywhere. Success rate (|x_hat-x*|<0.5) rises with every candidate
in 2D (e.g. n=8: 40% -> 80-85% for relclip2/relclip1/relclip_ema2/trust_noise1) and is
0% for all arms in 5D/10D (the paper's success tolerance is not reachable there at
these n, baseline included).

Shrinkage: correlation of the screening (R=40) and held-out (R=100) paired diffs over
the 84 pairs is 0.96; mean |change| 0.023-0.025 per setting; 19/84 pairs flip sign --
all of them among effects |diff| < 0.05 in 5D/10D. The 2D effects are fully confirmed;
the 5D/10D "small wins" of the screening are mostly noise, and the 10D n=32 losses of
the absolute-clip family are confirmed.

### 5.2 Secondary metric `mmd2_eval` (objective at the end point)

Agrees with L2 in 2D (all candidates improve both at n<=16; at n=32 `relclip1` is
significantly WORSE on mmd2_eval, -0.049, while null on L2). In 5D and 10D **every
candidate improves `mmd2_eval` significantly at almost every n** (10D n=4: +0.07 to
+0.38 on a baseline of 0.55) even where its exact L2 is null or worse (10D n=32:
clip0.5/relclip1/sqrtfloor_clip0.5 L2 -0.075..-0.088 but mmd2_eval +0.06..+0.07).
Reading: in 10D the step-control rules do make the guidance reach a lower value of the
objective it optimises (fixed-bandwidth sample MMD), but a lower MMD at `x_hat` does not
translate into a lower exact GMM L2 there -- the objective and the paper's metric are
not aligned at these (n, d). That is a statement about the MMD objective (bandwidth /
finite n), not evidence for the candidates; promotion is judged on the exact L2 as the
brief prescribes.

### 5.3 Pareto (score vs conditional calls, held-out; full tables in `heldout_tables.md`)

* **2D**: frontier is `relclip2` at n=4 (396 calls, 0.153) and n=8 (792 calls, 0.139);
  every candidate at n=8 (0.14-0.30) beats the baseline at n=96 (9504 calls, 0.259) and
  LGD/none at n=32 (9504 calls, 0.225); `relclip2`/`relclip_ema2`/`trust_noise1`/
  `relclip1` at n=8 beat LGD/none at n=8 (2376 calls, 0.201). At n=32 the gains are
  0.01-0.04 (sqrt_floor, relclip_ema2, sqrtfloor_clip0.5, trust_noise1 significant).
  Matched-compute improvement is large and unambiguous.
* **5D**: frontier `sqrtfloor_clip0.5` at n=4/8/32 (0.500/0.450/0.409) and
  `trust_noise1` at n=16 (0.441); the baseline needs 9504 calls for 0.432 and LGD/none
  n=32 (9504) gives 0.433, so `sqrtfloor_clip0.5`/`clip0.5`/`relclip1`/`sqrt_floor` at
  n=32 (3168 calls, 0.409-0.421) are a genuine 3x-calls Pareto improvement (p < 0.001),
  but the absolute effect is 0.02-0.035 and at n <= 16 nothing is reliably better.
* **10D**: frontier `trust_noise1` at n=4/8/16/32 (0.615/0.535/0.489/0.457), then the
  baseline at n=64/96 (0.456/0.442). `trust_noise1` at n=16 (1584 calls, 0.489) beats
  LGD/none at n=8 (2376 calls, 0.518) and at n=32 (3168, 0.457) matches the baseline at
  n=64 (6336, 0.456) -- a 2x-calls Pareto gain at n<=16 (p <= 0.04), null at n=32.
  The absolute-clip family (clip0.5, relclip1, sqrtfloor_clip0.5) and sqrt_floor are
  significantly WORSE than the baseline at 10D n=32.

### 5.4 Verdict table

| candidate | 2D | 5D | 10D | overall (>= 2 scales, no regression) |
|---|---|---|---|---|
| **trust_noise1** (`step_clip=noise, tau=1`) | **PASS** (4/4 n significant, incl. n=32) | INCONCLUSIVE (1/4 significant, +0.036 at n=8; all + ; mmd2_eval + at every n) | **PASS** (n=4,8,16 significant, n=32 null +0.02; never negative) | **PASS -- promote.** Only rule that is a credible Pareto improvement at 2 scales with no significant regression at any of the 12 cells; 2x-3x calls at small n in 2D/10D, 0.01-0.04 at n=32. |
| relclip2 (`clip_rel median x2`) | **PASS** (n=4,8,16; n=32 p=.06) | INCONCLUSIVE (n=32 only, +0.015) | **FAIL** (n=4 -0.047, p=.048; n=16 +0.049) | **not promoted** (one scale; significant loss at 10D n=4). Best 2D rule at n<=16. |
| relclip_ema2 (`clip_rel ema x2`) | **PASS** (4/4 significant) | INCONCLUSIVE (n=32 only, +0.016) | INCONCLUSIVE (all null, never negative) | **not promoted** (one scale). Safe: no significant regression anywhere. |
| sqrt_floor (loss transform) | **PASS** (n=4,8,32; n=16 p=.06) | **PASS** (n=8 +0.049, n=32 +0.023; n=4 +0.030 n.s.) | **FAIL** (n=32 -0.038, p=.024) | **CONDITIONAL PASS** (2 scales) -- promote for 2D/5D only; do not use at 10D n>=32. |
| sqrtfloor_clip0.5 | **PASS** (4/4) | **PASS** (n=8 +0.059, n=32 +0.035) | **FAIL** (n=32 -0.088; n=8 +0.073) | **CONDITIONAL PASS** (2 scales) -- same caveat; the absolute clip is scale-dependent (pre-registered by Agent 4). |
| clip0.5 (absolute clip) | **PASS** (n=4,8,16; n=32 null) | INCONCLUSIVE (n=32 only) | **FAIL** (n=32 -0.075; n=8 +0.056) | **not promoted** (one scale; confirmed 10D regression). |
| relclip1 (= qclip0.5, median x1) | **PASS** (n=4,8,16; n=32 null L2 but mmd2_eval -0.049*) | INCONCLUSIVE (n=32 only) | **FAIL** (n=32 -0.076; n=8 +0.061) | **not promoted** (one scale; 10D regression; duplicate of qclip0.5). |

Where a rule "PASSes" in 2D, the effect sizes are large and fully held up on the new
seeds (+0.25 to +0.44 at n=4, +0.19 to +0.28 at n=8). Where a rule "PASSes" in 5D, the
effects are 0.02-0.06 -- real (p <= 0.05 with 100 paired restarts, 78-92/100 wins at
n=32) but small relative to the baseline 0.41-0.53 and with success rate stuck at 0%.

## 6. Red flags

1. **Float32 trajectories are chaotic across platforms, not only across variants.** The
   same restart lands in a different mode on the Mac and the cluster (restart 1, 2D
   n=8: -4.35 vs +6.24); "legacy rng reproduces the README numbers" holds for the
   distribution over restarts, not restart-for-restart across machines. Screening pairs
   ran as separate array tasks (extra pairing noise if nodes differ -- on this cluster
   all held-out cells ran on one CPU model, and the screening/held-out diffs correlate
   0.96, so the effect was small, but it is a design hazard). Held-out pairs were
   same-node/same-process by construction.
2. **Batched-restart per-step equivalence is weaker than reported** (ReLU Jacobian
   jumps, up to ~10% at some steps). Statistical equivalence is the defensible claim;
   report batched runs as re-runs.
3. `qclip0.5` and `relclip1` are the same rule (identical numbers in every cell); count
   once.
4. `mmd2_eval` and exact L2 disagree in 10D at small n: the step-control rules lower the
   optimised MMD at `x_hat` while the exact L2 gets worse for the absolute-clip family.
   The fixed-bandwidth sample MMD is an imperfect proxy for the paper's metric in 10D at
   n <= 32; any future candidate judged on the objective alone would be misjudged.
5. Screening `s/run` includes a 17 s/run contention outlier; Pareto must use calls (it
   does). `screening_rows.csv` stamps the report machine's architecture as `hardware`.
6. Engine instance state (clip history, Adam moments, stale cache) is not reset by
   `run()`; harmless in the runner (fresh engine per restart), a hazard for reuse.
7. `CMSampler(cache=True)` keys on `x` bytes and returns graph-attached rows of the first
   call; correct in the engine's agreement path (same tensor, `retain_graph=True`), a
   footgun elsewhere; with LGD (`n_mc>1`) the cache thrashes (halves are not free).
8. Success rate is 0% for every arm in 5D/10D at every n (baseline included): the
   |x_hat-x*| < 0.5 criterion is uninformative there; only L2 carries information.

## 7. Bottom line

* Exact-speed: `fast_mmd` cached-target MMD is exact (promote as a drop-in for the
  fixed-bandwidth loop); Agent 5's batched LGD / generator seeding are exact; batched
  restarts are statistically equivalent and ~10x faster -- usable for screening, not for
  bit-reproduction.
* Estimator: **`trust_noise1` (||Delta_t|| <= sqrt(1-alphabar_t)) is the one rule that
  passes the promotion rule** (2D and 10D, never worse, Pareto-improving at matched
  calls); `sqrt_floor` and `sqrtfloor_clip0.5` pass at 2D+5D but regress at 10D n=32
  (conditional); the absolute-clip family and `relclip1/relclip2` are 2D-only and
  regress or are mixed at 10D; `relclip_ema2` is safe but 2D-only. Adam-arm and LGD-arm
  interactions were not part of the held-out set.

---

# 8. Round 3 (2026-08-24): sample-replay MMD -- `replay_geo0.7d5_trust`

## 8.1 Protocol

Claim under test (replay/REPORT.md section 4, screening R=40 offset 0):
`replay_geo0.7d5_trust` (ReplayConfig subsample, decay 0.7, depth 5,
batch_total=n, composed with trust_noise1) achieves MATCHED quality at ~3-4x
fewer fresh conditional calls on 2D/5D/10D, plus a same-calls win at 2D n=32
(+0.037, p=0.005); implementer's evidence: 12/12 call-matched pairings n.s.

Phase-1 red-team (verification/PHASE1.md round-3 addendum) -- mechanism CLEAN:
no target leak (replay rows are detached past conditional samples); tape keys
`("replay", t, j, k)` disjoint from eta/delta/renoise (verified by logging all
tape requests); calls accounting honest (`cm_samples` = generator draws only:
99/297/1089 fresh at n=4/8/32, MMD batch ramps to n after a depth-5 warm-up);
bit-identity claims confirmed by own runs (replay off == baseline exact;
`replay_geo0.3d3` == `replay30` torch.equal at n=4,8, differs at n=32);
screening cells all offset 0/R=40, comparator cells score-identical to the
round-2 estimator runs. Precond round: spot-checked the 2D exact-zero controls
(precond_sign/diag/cov bit-identical to baseline at d_x=1) -- clean reject
stands, no further verification.

Held-out: **OFFSET 2000** (fully fresh; offset 1000 was not reused because
trust_noise1 was promoted partly on those seeds -- a non-inferiority
comparison against a seed-selected champion would be biased in the candidate's
favour), 100 paired restarts, 24 cells = 3 array tasks (one per setting, all 8
cells per setting in ONE process on one node): trust_noise1 and
replay_geo0.7d5_trust at n in {4,8,32}, baseline at n in {8,32}. Job 45936496,
no errors. Analysis `verification/analyze_r3.py` ->
`verification/heldout_r3_tables.md`. Decision standard (pre-registered in the
Phase-1 addendum): non-inferiority judged on the 95% CI of the paired L2 diff
(lower bound above ~-0.05 = "matched"), not on p > 0.05.

## 8.2 Results (paired diff = comparator - candidate, + = replay better; R=100)

Cell scores (fresh calls): 2D trust 0.178/0.161/0.216 at 396/792/3168, replay
0.300/0.170/0.194 at 99/297/1089, baseline 0.334/0.231 at 792/3168.
5D trust 0.510/0.515/0.427, replay 0.598/0.517/0.447, baseline 0.553/0.444.
10D trust 0.612/0.516/0.443, replay 0.656/0.622/0.524, baseline 0.673/0.469.

| comparison (cand calls vs comp calls) | 2D | 5D | 10D |
|---|---|---|---|
| same-n n=4 (99 vs 396, the "4x" claim) | **-0.122*** [-0.182,-0.062] | **-0.088*** [-0.151,-0.026] | -0.045 [-0.099,+0.011] |
| same-n n=8 (297 vs 792) | -0.009 [-0.050,+0.030] | -0.003 [-0.053,+0.047] | **-0.105*** [-0.146,-0.063] |
| same-n n=32 (1089 vs 3168) | +0.022 [-0.009,+0.050] p=0.154 | **-0.020*** [-0.040,-0.004] | **-0.080*** [-0.111,-0.051] |
| replay@n8 vs trust@n4 (297 vs 396, 1.3x fewer) | +0.008 [-0.038,+0.052] | -0.008 [-0.059,+0.043] | -0.010 [-0.055,+0.036] |
| replay@n32 vs trust@n8 (1089 vs 792, MORE calls) | -0.034 [-0.069,-0.000] p=0.054 | +0.068* | -0.008 [-0.044,+0.029] |
| replay@n32 vs plain baseline@n32 (1089 vs 3168) | **+0.037*** [+0.004,+0.067] | -0.004 [-0.026,+0.018] | **-0.054*** [-0.093,-0.016] |

(* = p <= 0.05; full tables with wins, mmd2_eval CIs in heldout_r3_tables.md.)
Secondary metric `mmd2_eval`: significantly WORSE for replay in 5D at every n
(p <= 0.007) and in 10D at every n (-0.06..-0.23, p < 0.001); never
significantly better except in the vs-plain-baseline@n8 pairings.

## 8.3 Verdict: **FAIL -- do not promote** (the promoted claim did not replicate)

| element of the claim | held-out outcome |
|---|---|
| "matched at ~4x fewer calls" (replay@n4 99 vs trust@n4 396) | **FAIL**: significantly worse in 2D (-0.122) and 5D (-0.088); 10D CI reaches -0.099 -- nowhere does the CI support "matched" |
| "matched at ~3x fewer calls, same n" (replay@n32 vs trust@n32) | **FAIL in 5D/10D** (-0.020*, -0.080*); 2D OK (+0.022 n.s.) |
| same-calls win at 2D n=32 vs champion (+0.037 p=0.005 in screening) | **NOT REPLICATED**: +0.022, p=0.154 (same-n vs trust; the +0.037* that does replicate is vs the PLAIN baseline, which trust@n8 already beats with 1/4 of the calls) |
| "12/12 call-matched pairings n.s." | the n.s. pairings were R=40 non-inferiority by absence of significance; at R=100 two of the honest-savings pairings become significant losses and one ("1089 vs 792") gives the candidate MORE calls and still trends worse in 2D (-0.034, p=0.054) |
| what SURVIVES | a ~1.33x saving (replay@n8, 297 calls, matches trust@n4, 396 calls, in all three settings: CIs within [-0.06,+0.05]); and 2D-only: replay@n32 beats the plain baseline at 1/3 calls |

Per setting: 2D **FAIL** (4x leg fails, n=32 win not replicated; frontier points
dominated by trust's own n=4/n=8 points), 5D **FAIL** (same-n inferior at n=32,
4x leg fails, mmd2_eval worse everywhere), 10D **FAIL** (inferior at nearly
every pairing incl. vs plain baseline at n=32). A 1.33x call saving in a
regime trust_noise1 already serves does not justify adding a buffer mechanism;
the "promote conditionally" recommendation of replay/REPORT.md is REVERSED.
`trust_noise1` alone remains the promoted configuration; the round-2 verdict
table is unchanged.

## 8.4 Round-3 red flags

1. **Winner's curse on non-inferiority screening.** The promotion rested on
   n.s. p-values at R=40 -- absence of evidence -- and every borderline
   pairing moved against the candidate at R=100 on fresh seeds (2D n=4:
   -0.062 n.s. -> -0.122***; 2D n=32 win +0.037** -> +0.022 n.s.; 5D n=32
   -0.06 -> -0.020*). Future non-inferiority claims must be pre-registered
   with a margin and judged on CIs at the confirmation sample size.
2. One of the implementer's "call-matched" pairings (replay@n32 1089 vs
   trust@n8 792) gives the CANDIDATE 37% more calls; it is a dominance check,
   not a savings claim, and indeed trust@n8 dominates replay@n32 in 2D and 10D.
3. `mmd2_eval` (the objective itself) is consistently worse for replay in
   5D/10D -- the replayed rows bias the guidance signal exactly as THEORY.md's
   trajectory-smoothing reading predicts; the trust region caps but does not
   remove the lag.
4. Mechanism-level caveats (benign, for the record): depth-5 warm-up ramp
   (first ~5 steps run smaller MMD batches); `replay50` never ran (validate
   rejects lambda=1); cross-n pairings share only the first min(n_t) eta draws
   per step.

---

# 9. Round 4 (2026-08-24) [legacy protocol]: progressive FIFO / cohort replay (M-10)

**PROTOCOL CAVEAT (applies to every number in sections 1-9).** An external audit
established that the whole campaign so far ran the LEGACY protocol
(`init="zeros"`: x_T = 0, and zeta = 1 / `guidance_scaling="raw"`). All
verdicts in this file, including this section, are therefore labelled
**[legacy protocol]**. A corrected-protocol re-test (round 5: x_T ~ N(0, I),
calibrated zeta) is REQUIRED before ANY candidate -- trust_noise1 included --
is promoted, regardless of the outcome below.

## 9.1 Protocol
Claim (replay/m10_tables.md, hypotheses M-10): `replay_fifo16_trust` (f fresh
rows + up to 14 recycled rows from the last 7 steps, gradient through the
fresh rows only, + trust_noise1) beats fresh-only `trust_noise1@f` at EQUAL
fresh cost: 2D f=2 +0.089 (p=0.007), 10D f=2 +0.084 (p=0.0005) and f=4 +0.082
(p=0.0007), 5D null; `replay_cohort16_trust` similar but weaker. The M-10
estimate is a selection among 10 policy x f arms on the M-9 seeds (offset
4000), with the comparators reused from a different SLURM job (same-node
pairing unverifiable from the artefacts -- PHASE1.md round-4 addendum).
Phase-1 red-team: plans, calls accounting (cm_samples = f x 99 exactly, MMD
batch ramps to 16), tape keys (fresh draws identical to trust@f, first step
bit-identical), detached buffer, cohort8@f4 == fifo8@f4 degeneracy, and the
M-10 statistics were all reproduced/confirmed.
Held-out: OFFSET 5000 (never used), R=100, 21 cells in 3 array tasks (one per
setting, all 7 cells in ONE process/node): trust_noise1@{2,4,8},
replay_fifo16_trust@{2,4}, replay_cohort16_trust@{2,4}. Job 45938379, no
errors. Analysis `verification/analyze_r4.py` -> `heldout_r4_tables.md`.

## 9.2 Results (paired diff = trust@f - candidate@f, + = candidate better; R=100)

| setting | f | fifo16 vs trust@f [CI] p | cohort16 vs trust@f [CI] p | M-10 estimate (fifo16 / cohort16) | fifo16 vs trust@8 (792 calls) |
|---|---|---|---|---|---|
| 2D | 2 | +0.038 [-0.030,+0.104] p=0.28 | +0.020 [-0.043,+0.084] p=0.54 | +0.089** / +0.066* | -0.048 (p=0.08) |
| 2D | 4 | -0.009 [-0.058,+0.040] p=0.73 | +0.007 [-0.046,+0.059] p=0.79 | -0.032 / -0.005 | -0.022 n.s. |
| 5D | 2 | -0.017 [-0.064,+0.029] p=0.48 | +0.026 [-0.022,+0.073] p=0.28 | +0.020 / +0.035 | -0.043 n.s. |
| 5D | 4 | +0.014 [-0.031,+0.060] p=0.54 | -0.002 [-0.050,+0.045] p=0.93 | -0.009 / -0.012 | -0.007 n.s. |
| 10D | 2 | **+0.103** [+0.055,+0.150] p<0.001 | **+0.080** [+0.036,+0.123] p<0.001 | +0.084*** / +0.050* | +0.008 n.s. (matches trust@8 at 1/4 calls) |
| 10D | 4 | **+0.074** [+0.025,+0.122] p=0.004 | **+0.083** [+0.039,+0.127] p<0.001 | +0.082*** / +0.080*** | **+0.044** (p=0.031); cohort16 +0.053 (p=0.007) |

Secondary `mmd2_eval` agrees in 10D (fifo16 f=2 +0.199***, cohort16 f=4
+0.064*) and is null elsewhere (5D fifo16 f=4 +0.027*** on mmd2_eval with a
null L2). Divergences 0 everywhere. Call-halving check (candidate@2, 198
calls, vs trust@4, 396): n.s. in every setting (10D fifo16 +0.038 [-0.013,
+0.089]); the 10D win is at equal fresh cost, not a call reduction beyond
what trust_noise1@f already gives.

## 9.3 Verdict [legacy protocol]

| candidate | 2D | 5D | 10D | overall |
|---|---|---|---|---|
| replay_fifo16_trust | **NOT REPLICATED** (M-10 +0.089** -> +0.038 n.s.; f=4 null) | null (as screened) | **PASS** (f=2 +0.103***, f=4 +0.074**; fifo16@2 matches trust@8 at 1/4 the calls; fifo16@4 beats trust@8) | **FAIL for promotion** under the >= 2-scale rule: a real, replicated 10D-only equal-cost gain |
| replay_cohort16_trust | null | null | **PASS** (f=2 +0.080***, f=4 +0.083***) | same: 10D-only |

The 10D effect is genuine (fresh seeds, same-process pairing, both policies,
both f, CIs well clear of zero, objective agrees) -- it is the first
recycling mechanism whose 10D benefit survives held-out confirmation, and it
is consistent with the round-2 finding that 10D at n <= 8 is the regime where
the guidance gradient is most noise-dominated (a 16-row MMD batch at 2-4 fresh
calls per step is what helps there). But the 2D claim shrank to null on fresh
seeds (the usual post-selection shrinkage, red flag 1 of section 8.4) and 5D
is null, so the candidate does not meet the campaign's promotion rule; it can
be recorded as a 10D-specific, equal-cost improvement pending round 5. No
promotion of anything until the corrected-protocol re-test.

## 9.4 Round-4 red flags
1. **Legacy protocol** (x_T = 0, zeta = 1) for the entire campaign: every
   verdict in this file is provisional until the round-5 corrected-protocol
   re-test; effects that depend on the zero start (e.g. the small-n 2D
   divergence tail that clipping/trust fix) may change size or sign.
2. Comparator reuse across SLURM jobs in M-10 (unverifiable same-node
   pairing); the held-out design re-ran every comparator in-process.
3. Post-selection shrinkage again: the 2D f=2 +0.089** became +0.038 n.s.;
   only the 10D effects, which were the largest and appeared for both
   policies and both f, replicated.
4. The 10D gain is an equal-fresh-cost quality gain, NOT a call saving:
   candidate@2 vs trust@4 is n.s. in all settings.

---

# 10. Round 5 (2026-08-24): corrected protocol -- trust_noise1 vs calibrated no-trust

## 10.1 What was tested
H-R5 (hypotheses/agent4.yaml): under the corrected protocol -- `x_T ~ N(0,I)`
(generator `0x5EED0000 ^ restart`, i.e. the same x_T for every arm of a
restart; verified in `RandnInitTape` / `_guided.py:103`) and a PER-ARM
calibrated guidance scale zeta -- does the noise-level trust region
(trust_noise1) still beat the no-trust estimator at equal conditional cost?
Arms: A = trust @ zeta*_trust (16 / 8 / 4 for 2D / 5D / 10D), B = no trust @
zeta*_notrust (2 / 0.25 / 1), C = no trust @ zeta*_trust. Single pre-specified
primary comparison A vs B, R=100 paired restarts at the fresh offset 6000, 36
cells (`protocol/cells_r5.py`, job a4r5; `r5_tables.md`). Calibration:
`protocol/calibrate_zeta.py`, engine path, n=128, 40 restarts at offset 5000,
grid {0.25..32} (`zeta_star.json/.md`).

## 10.2 Red-team of the calibration and protocol
* **Criterion amendment -- legitimate, with one open sensitivity.** The
  pre-registered exp5b "basin" rule (reached = ||x-x*|| < 0.5, a dim(x)=1
  construct) returned reached = 0% at every zeta in 5D (d_x=4) and 10D
  (d_x=9), so its fallback picked zeta = 0.25 for BOTH arms there -- a
  meaningless answer. The amendment (zeta* = argmin of the failure-penalised
  exact L2 over divergence-free zetas at n=128) was (i) written down before
  any round-5 cell ran, (ii) applied symmetrically to both arms and all
  settings, (iii) chosen on the paper's own metric, and (iv) changes the
  selection only where the basin rule had failed (5D/10D trust: fallback
  0.25 -> 8 / 4; notrust: identical under both rules in every setting) plus
  2D trust (8 -> 16, where the n=128 scores 0.2188 vs 0.2114 are within the
  40-restart noise floor; reach 75% vs 67.5%). I judge this a legitimate
  amendment, not a forking path: the alternative rule is inapplicable, and
  the calibration seeds (offset 5000) are disjoint from the test seeds
  (offset 6000). Residual risk: 2D A ran at zeta 16 while the basin rule
  says 8; the existing runs cannot test that (only zeta 16 exists at the
  compared n) -- the confirmatory re-run adds a 2D `A8` arm.
* **Calibration "divergence-free" does not transfer to small n.** Zeta was
  chosen divergence-free at n=128, but B diverges at 2D n=4/8/16 (9/2/1 of
  100) and 10D at every n (5/5/3/4), while A never diverges (0/1200). The
  penalised score charges each divergence 2.0, so part of A-B is
  divergence-driven. Restricted to restarts where neither arm diverged
  (own computation from the JSONs):
  2D n=4/8/16/32: +0.485/+0.335/+0.211/+0.087 (all p < 0.001, p=0.008 at
  n=32) -- robust; 10D n=4/8/16/32: +0.062 (p=0.019) / +0.052 (p=0.054) /
  +0.049 (p=0.047) / +0.004 (n.s.) vs the penalised +0.128/+0.116/+0.093/
  +0.059 -- roughly half of the 10D effect is B's divergences. Divergence
  IS a legitimate failure of the no-trust estimator (and the trust region is
  exactly the mechanism that prevents it), so the penalised numbers are the
  pre-registered primary, but the 10D effect on converged runs is small.
* **randn init seeding** OK: restart-only seed, independent of the tape
  (so A/B/C share x_T and all conditional draws), differs from the tape's
  own ("x_T",) key convention only by construction, recorded in every JSON
  (`protocol.x_init = "randn"`).
* **Protocol fields**: every runs_r5 JSON carries `protocol = {x_init:
  randn, zeta, step_clip, step_tau: 1.0, rng: tape, dtype: float32,
  loss_backend: reference}`, offset 6000, R=100; the zeta values match
  `zeta_star.json`; cm_samples = n x 99 in every cell (A and B equal cost).
* **Same-process pairing NOT satisfied**: `submit_r5.sh` runs one cell per
  array task (36 tasks), so A and B of a cell ran in different processes and
  possibly on different nodes; no host is recorded. Same seeds, but not
  same-node (the round-2 chaos flag). Unverifiable from the artefacts.
* Arm C diverged in 71-99% of restarts in every cell (score 1.45-2.0): the
  "no trust at the trust scale" result is unambiguous and needs no re-run.

## 10.3 Results (offset 6000, R=100; diff = B - A, + = trust better)

| setting | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---|---|
| 2D (A z=16 vs B z=2) | **+0.590*** [+0.483,+0.696] | **+0.360*** [+0.264,+0.456] | **+0.228*** [+0.143,+0.313] | **+0.087*** [+0.024,+0.150] p=0.008 |
| 5D (A z=8 vs B z=0.25) | -0.004 n.s. | +0.022 n.s. | **+0.065*** [+0.019,+0.113] p=0.008 | +0.029 n.s. (p=0.18) |
| 10D (A z=4 vs B z=1) | **+0.128*** [+0.055,+0.207] p<0.001 | **+0.116*** [+0.044,+0.193] p=0.003 | **+0.093*** [+0.028,+0.164] p=0.006 | +0.059 (p=0.09) |

A vs C: +0.87 to +1.52 everywhere, p < 1e-4 (C diverges). 2D success rates
A 52/74/85/75% vs B 11/30/46/53%. Prediction (i) of H-R5 -- that calibrating
zeta would shrink the trust effect far below the legacy +0.25..+0.40 at 2D
n<=8 -- is FALSIFIED in the other direction (+0.59/+0.36): with a random
start the uncapped estimator cannot use a large zeta at all (divergence-free
ceiling 2 vs 16), and that ceiling, not the tail cap per se, is most of the
2D gap.

## 10.4 Verdict under the corrected protocol and the 2-scale rule
**trust_noise1: PASS (provisional).** 2D: PASS (4/4 n, p <= 0.008). 10D:
PASS (3/4 n, p <= 0.006; n=32 +0.059 p=0.09; never negative; on converged
restarts only the gain is +0.05..+0.06 at n <= 16). 5D: INCONCLUSIVE (n=16
+0.065 p=0.008 only; never negative). Two scales pass, no significant loss
anywhere, both arms at their own best zeta, equal calls -> the promotion
rule is met on this run. The mechanism is now understood: the trust region's
value is that it makes large guidance scales usable (zeta 4-16 vs a
divergence-free ceiling of 0.25-2 without it), plus a modest tail-cap gain
on converged runs.

**Why "provisional" -- an independent re-run at offset 7000 IS warranted**,
and is authored: (1) A/B pairing in the round-5 run was cross-task (not
same-node), (2) the 2D zeta choice (16 vs the basin rule's 8) is untested at
the compared n, (3) roughly half of the 10D effect comes from B's
divergences at 4-5%, so the 10D margin (p=0.003-0.006) is thinner than it
looks and this is the campaign's final promotion decision, worth a second
independent draw. Design: `verification/heldout_r5_cells.py` +
`submit_heldout_r5.sh` -- offset 7000 (never used), R=100, one process per
setting (3 array tasks, 28 cells): A and B at n in {4,8,16,32} for
2D/5D/10D, plus the 2D sensitivity arm A8 (trust @ zeta 8); C is not re-run.
`analyze_r5.py` reports penalised and non-diverged-pair diffs, mmd2_eval,
A8 vs B and A8 vs A, and records the node. Status of trust_noise1 becomes
final PASS if 2D and 10D reproduce (p <= 0.05 at >= 2 n each, no significant
loss); if 10D drops to null on the re-run the corrected-protocol verdict
becomes "2D-only, INCONCLUSIVE overall".

## 10.5 Status summary of the campaign's promoted configuration
* Legacy protocol (x_T = 0, zeta = 1; rounds 2-4): trust_noise1 PASS
  (2D + 10D), all other candidates FAIL / conditional / 10D-only.
* Corrected protocol (round 5): trust_noise1 PASS provisional, pending the
  offset-7000 same-node confirmation; the other candidates have NOT been
  re-tested under the corrected protocol and keep no promotion claim.

## 10.6 Confirmatory re-run (offset 7000, job 45938702) and FINAL verdict

Protocol as authored in 10.4: 28 cells, R=100, one process per setting
(2D group on glacier-26, 5D/10D on glacier-34; all AMD EPYC 7662) -- every
A/B (and A8) pairing is same-process, same-node. Divergences: A 0 in all
1200 runs (A8 0/400); B 2D 9/1/2/0, 5D 1/0/0/0, 10D 1/4/3/1. cm_samples =
n x 99 in every cell. Full table `verification/heldout_r5_tables.md`.

Paired diff B - A (+ = trust better), offset 7000; second value = restricted
to pairs where neither arm diverged:

| setting | n=4 | n=8 | n=16 | n=32 |
|---|---|---|---|---|
| 2D | **+0.533*** / +0.416*** | **+0.357*** / +0.340*** | **+0.189*** / +0.154*** | **+0.064*** (p=0.016) / same |
| 5D | **+0.067*** (p=0.025) / +0.059 (p=0.043) | **+0.076*** (p=0.013) / same | **+0.056*** (p=0.021) / same | **+0.063*** (p=0.045) / same |
| 10D | **+0.068*** (p=0.024) / +0.058 (p=0.040) | **+0.119*** (p=0.001) / +0.069 (p=0.020) | +0.027 (p=0.44) / -0.016 n.s. | +0.021 (p=0.47) / +0.004 n.s. |

Pooled over both independent runs (offsets 6000 + 7000, R=200): 2D
+0.561/+0.358/+0.209/+0.075 (all p<0.001); 5D +0.032 (p=0.10)/+0.049
(p=0.018)/+0.060 (p=0.001)/+0.046 (p=0.016); 10D +0.098 (p<0.001)/+0.117
(p<0.001)/+0.060 (p=0.013)/+0.040 (p=0.071). `mmd2_eval` (the objective at
x_hat) is better for trust in every 5D/10D cell (p<0.001) and at 2D n<=8.

2D zeta sensitivity (basin rule 8 vs amended 16): A8 beats B at every n
(+0.59/+0.40/+0.21/+0.09, all p<0.001) and is not distinguishable from A
(A - A8 = +0.05/+0.04/+0.03/+0.03, p=0.12-0.40, slightly in favour of 8):
the 2D result does not depend on the calibration-rule choice.

**FINAL VERDICT -- trust_noise1 under the corrected protocol: PASS.**
2D: PASS on both runs (4/4 n). 10D: PASS (offset 6000: 3/4 n; offset 7000:
2/4 n at n<=8; pooled 3/4; never negative) -- the gain is real but
concentrated at n<=8 and about half of it, in the penalised metric, is the
no-trust arm's 3-5% divergence rate (converged-only +0.05..+0.07 at n<=8,
null at n>=16). 5D: PASS on the confirmation (4/4 at p<0.05, small
+0.06), INCONCLUSIVE on the first run, pooled 3/4 -- a small but consistent
gain. Two (arguably three) scales, equal conditional cost, both arms at their
own best zeta, zero divergences for trust, no significant loss in 24 cells
across two independent seed sets. The trust region is promoted as the
campaign's one confirmed estimator improvement; its mechanism is that it
makes guidance scales of 4-16 usable where the uncapped estimator diverges
above 0.25-2, plus a modest tail-cap gain on converged runs.
