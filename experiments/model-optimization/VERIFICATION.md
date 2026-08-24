# VERIFICATION -- independent verification and red-team report (Agent 6)

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
