# Agent 6 -- verification, Phase 1 (code/invariant review, local checks, held-out design)

Commit `6af2081` + the uncommitted working tree (branch `tfg-generalization-v2`), 2026-08-23.
Python `/Users/stolk/miniconda3/bin/python`, torch 2.12.0, CPU. Nothing outside
`experiments/model-optimization/verification/` was written. Scripts and logs in this
directory: `check_fast_mmd.py` (+`.log`), `check_systems.py` (+`.log`),
`heldout_cells.py`, `submit_heldout.sh`, `analyze_heldout.py` (Phase 2).

## A. Invariants

| # | invariant | finding | evidence |
|---|---|---|---|
| 1 | default engine path == frozen reference | **holds** | `cd simulations && python -m pytest tests -q`: **331 passed, 1 skipped** (the skip is the pre-existing structural C3 placeholder in `test_equivalence_is_not_vacuous.py:156`, documented as covered by M5). Mutation tests (`test_equivalence_is_not_vacuous.py`) and `test_agent4_candidates.py::test_defaults_are_still_the_reference` (atol 0.0 trace compare, `N_recur=2, n_mc=2`) pass. |
| 2 | every new mechanism opt-in / off by default | **holds** | `config.py` defaults: `init="randn"`, `guidance_scaling="tfg"`, `smoothing="tfg"`, `temporal.grad_norm="none"`, `step_clip="none"`, `n_schedule.type="constant"`, `eta_per_perturbation=False`, `eta_keying="per_step"`, `temporal_cache.enabled=False` (`implementation="gated"` still raises), `adaptive_recurrence.enabled=False` (`"gated"` still raises). Engine diff read line by line: with the defaults `_preprocess` returns `grad` unchanged, `_step_clip` returns `Delta_t`, `use_stale` is False, `n_recur_max = N_recur`, `stop` stays False, `eta_keys` are `("eta", t, i)`, the smoothing scale is `gamma_bar*sqrt(1-ab_t)`, `grad_norm_history` is only appended when a switch is on (so the traced key set is unchanged). `all_extensions_disabled()` now also requires the legacy switches to be at their defaults (tested). |
| 3 | NoiseTape keying | **holds** | delta `("delta", t, j)` (engine `_log_f_tilde`); renoise `("renoise", t, r)` drawn only when `r < n_recur_max and not stop` (== `r < N_recur` on the default path); eta `("eta", t, i)` by default, `("eta", t, j, i)` only with `eta_per_perturbation`, `("eta","frozen",...)` only with `eta_keying="frozen"` (labelled approximate). |
| 4 | `Delta_t / sqrt(alpha_t)` | **holds** | engine line 9: `guidance = Delta_applied / torch.sqrt(alpha_t)` unless `guidance_scaling="raw"` (the opt-in legacy `_guided` convention, where the division is deliberately absent -- and `test_engine_matches_guided.py` proves bit-for-bit agreement with `_guided.run` under it, 0.0 on 3 restarts locally). |
| 5 | schedules rebuilt, not cast | **holds** | `tfg/distributional.RepositorySchedule` recomputes the cosine formula of `Diffusion.py:25-29` in the requested dtype; `matches_model()` asserts bit identity with the float32 checkpoint attributes; float64 is a rebuild (1e-7 from the float32 values, stated). |
| 6 | caches: only exact/target statistics | **holds with two notes** | `fast_mmd.MMDFixedTarget` caches Y-only quantities (`Y_sq`, `D_yy`, `YY_fixed`); `systems/BatchedMMD` caches the YY kernel mean; `CMSampler(cache=True)` caches predictor rows keyed on `(key, x-bytes)` -- exact reuse of identical inputs within a step (used only by `adapt_agree`). Notes: (a) for the adaptive-bandwidth path `fast_mmd` re-attaches the YY term to the graph with a first-order expansion -- exact value and first derivative, documented as not valid for `create_graph=True`; (b) the `CMSampler` cache holds only the most recent `x`, so with `n_mc>1` (LGD) the "free" half batches are not free (the cache thrashes between perturbations); `adapt_agree` was only screened on no-LGD arms, so no reported number is affected. The `stale` gradient cache is labelled approximate and was rejected. |
| 7 | held-out seeds | **holds** | all 519 `estimator/runs/*.json` have `offset=0, restarts=40, dtype=float32` (checked programmatically); tape seed = restart index; my held-out runs use `--offset 1000` (asserted `>= 1000` in `heldout_cells.py`), i.e. tape seeds 1000..1099 and legacy keys `("cond", 1000+, t, j)` -- disjoint from the screening. |
| 8 | selection vs evaluation metric | **same metric, different seeds -- see below** | selection = failure-penalised mean exact GMM L2 (`_common.penalised_score`, cap 2.0) over restarts 0..39; my held-out uses the same score on restarts 1000..1099 PLUS an independent end-point metric (`mmd2_eval`: MMD^2 of 256 fresh conditional draws at `x_hat` vs `S_G`, keyed on the restart only) and the per-restart `abs_err`/success. |

On (8): exact GMM L2 is the paper's independent metric (it is never optimised -- the guidance optimises the sample MMD), so using it for selection is not a leak of the optimisation objective; the real risk is the winner's curse on the *seeds*, which the held-out run addresses. I add `mmd2_eval` anyway so a candidate that improves L2 but worsens the objective it is supposed to minimise (or vice versa) is visible.

Other checks from the brief:

* **Paired, not leaked.** Baseline and candidate of one cell share tape seed = restart index, so the same `("eta", t, j, i)`/`("delta", t, j)` draws; none of the held-out candidates changes the noise consumption (clipping / trust region / `sqrt_floor` act on the gradient or the loss only). Confirmed locally that the screening's tape-baseline restarts are distinct from the legacy-rng ones (different `x_hat`).
* **Budgets.** Across the 519 screening cells the only `cm_samples` mismatches versus the cell's baseline are the labelled `stale2/3` (1/2, 1/3) and `recur2` (2x) -- checked programmatically; adaptive-n cells match exactly (`cm_samples == n*99`; the engine's `conditional_calls` over-counts the requested halves, the tables use `cm_samples`, as stated). All held-out candidates have identical calls to their baseline by construction (asserted in `analyze_heldout.py`).
* **Stale caches / NaNs.** No NaN L2 in any screening cell; the runner rebuilds models, tape, sampler, loss and engine per restart.
* **Timings.** Screening `s/run` is a per-restart average over 40 restarts in one process (warm-up amortised) on shared cluster nodes: one outlier of 17.5 s/run (5D lgd n=32 relclip1; the arm's baseline is ~2 s) shows wall times there are contaminated by node contention -- use `calls`, not `s/run`, for the Pareto axis (the tables do). Agent 5's and Agent 2's timings were taken with warm-up and medians on a loaded Mac (stated by them; load average 8-10 during my re-timing too) -- ratios within one run are meaningful, absolute seconds are not.
* **Cherry-picking.** The round-2 cell list (`screen.py list --round 2`) is complete in `screening_tables.md` (every candidate x n x arm appears, including the negative ones), and the REPORT.md verdict tables report the negatives (clip0.1, norm_only, unit, trust_ddim, stale, adapt). One duplication to be aware of: `qclip0.5` and `relclip1` are the SAME rule (threshold = 1 x running median) and indeed have identical scores in every cell -- they should be counted once when judging how many "winners" there are.
* **Report label.** `estimator/report.py` stamps `hardware = platform.machine()+" cpu"` of the machine that *renders* the report (`arm64 cpu` in `screening_rows.csv`) although the runs were made on the x86 cluster; cosmetic, but the rows are mislabelled.

## Code-review findings (no fixes made; for the implementing agents)

1. **Engine state is per instance, not per run.** `GeneralizedTFG` keeps `_raw_norms`, `_ema_norm`, `_stale_grad`, `_stale_age` (new) and the Adam object / `_prev_used` (pre-existing) on `self`; a second `run()` on the same instance would continue the clip history/moments. `engine_runner.run_engine` builds a fresh engine per restart, so no reported number is affected, but the relative-clip rules are NOT reset by `run()`.
2. **`CMSampler(cache=True)` returns rows whose autograd graph hangs off the FIRST call's `x`.** Keying on `x` bytes means a later call with an equal-valued but *different* tensor would route gradients to the earlier tensor. In the engine the agreement halves pass the same `x0` object and use `retain_graph=True`, so it is correct there; it is a footgun for other users.
3. **`conditional_calls` (engine) vs `cm_samples` (sampler).** The engine counts requested batches (3x for `adapt_agree`); REPORT.md says so and uses `cm_samples` -- fine, but the CallCounter field name is now misleading for that candidate.
4. **Float32 trajectories are chaotic across platforms, not just across variants.** `_guided.run` / engine-legacy restart 0 (2D, n=8, no-LGD/none) gives `x_hat = -5.447157` on the Mac (bit-identical to `results/tfg/exp2_*`), while the cluster's `2D_n8_no_lgd_none_baseline_legacy.json` has `-5.447173`, and restart 1 is `-4.346` on the Mac vs `+6.239` on the cluster (a mode flip). So "legacy rng reproduces the README numbers" holds *statistically* (distribution over restarts), not restart-for-restart across machines. Consequence for the screening: baseline and candidate cells were separate SLURM array tasks, so if glacier is heterogeneous (different BLAS kernels) part of the pairing benefit is lost; my held-out design runs each baseline with all its candidates in ONE process (see C).
5. `DistributionalLoss(transform="sqrt_floor")` uses `k(0) = n_kernels` (=5) -- correct for the 5-RBF sum; `c = floor_frac*5*(1/n+1/m)`.
6. `engine_runner.cell()` applies the post-hoc divergence rule on the traced `x_prev`; with `adaptive_recurrence` it indexes by the LAST recurrence (`recurrence_history[T-t]`), correct.
7. `screen.py` cannot take a custom cell list without editing (only `--only` filtering of its fixed lists); hence the separate `verification/heldout_cells.py` that imports `estimator/engine_runner.py` unchanged.

## B. Exact-speed candidates -- my own numbers

### B(i) `exact_loss/fast_mmd.py` (Agent 2)

* Their tests: `python -m pytest ../experiments/model-optimization/exact_loss -q` -> **236 passed**.
* My independent check (`check_fast_mmd.py`, float64, relative errors of value and dX-gradient vs `LossFunctions.MMDLoss`, worst over `(n,m,d)` in {(1,250,2),(4,250,2),(8,250,5),(32,250,10),(7,13,3),(250,8,2),(64,64,4)} x 2 seeds, for `fixed` AND `adaptive` (stacked-rule, gradient through the bandwidth) bandwidth, variants cdist/mm x exp/powchain/loop, chunked XY (chunk 64 and 7), `batched()` on 3 sets incl. perturbed ones, `reattach_yy="autograd"`): **all <= 1.3e-14 relative** (value, gradient, batched value, batched gradient). `VERDICT float64: EXACT`.
* float32 (production default dtype, n=8, m=250): |dval| <= 9.5e-7, |dgrad|max <= 1.6e-7 on |grad|max 0.16 -- round-off-level, not bit-identical. Their `end_to_end_results.csv` shows the 99-step float32 loop then moves `x_hat` by up to 0.18 (2D n=32) -- consistent with finding 4 above (chaos), not with an error.
* Speed (my micro-timing, fwd+bwd, float32, m=250, median of 200 after 20 warm-up, loaded machine): n=8: reference 1.96 ms, `fixed_cdist` 0.42 ms (**4.6x**), `fixed_mm_powchain` 0.28 ms (**7.0x**); n=32: 2.39 ms -> 0.68 ms (3.5x) / 0.58 ms (4.1x). Their table claims 5.7x / 6.4x at n<=8 and ~3.4x/4.6x at n=32: **consistent**. End-to-end (their csv, 2D): 1.6-2.6x per restart, since the MMD is ~25-40% of a step.
* **Verdict: mathematically exact (float64 to 1e-14, value and first-order gradient); in float32 it is reorder-only (round-off), so the trajectory is not bit-reproducible; speedup claims hold (4-7x on the loss, ~1.6-2.6x on the loop).** Caveat: the adaptive-bandwidth YY re-attachment is first-order only (documented).

### B(ii) `systems/runners.py` (Agent 5)

`check_systems.py`, 2D, float32, restarts 0..7 (they used 0..3), threads 4, load avg 8-10:

| cell | run_single(flags off) vs `_guided.run` | batched_lgd e2e | batched_mmd per-step max abs dg (teacher-forced) / e2e dx | batched_restarts B=1 per-step / e2e | batched_restarts B=8 per-step (max rel) / e2e | timing ref -> B=1 -> B=8 (s/restart) |
|---|---|---|---|---|---|---|
| no_lgd/none n=8 | 0.0 (EXACT) | n/a | 1.6e-6 / 3.0 | 1.3e-6 / 1.3e-5 | 2.8e-4 (8e-4) / 3.0 | 0.53 -> 0.32 (1.6x) -> 0.068 (7.8x) |
| lgd/none n=8 | 0.0 (EXACT) | **0.0 (EXACT)** | 1.1e-6 / 0.62 | 1.0e-6 / 4.9e-2 | 9.4e-5 (5e-3) / 0.13 | 1.06-1.55 -> 0.22-0.39 (4-5x) -> 0.070-0.117 (13-15x) |
| no_lgd/adam n=8 | 0.0 (EXACT) | n/a | 6.7e-6 / 2.0 | 1.2e-6 / 7.2e-6 | 6.1e-5 (4e-3) / 1.7 | 0.40-0.56 -> 0.20-0.35 (1.6-2x) -> 0.044-0.072 (7.8-9x) |
| no_lgd/none n=32 | 0.0 (EXACT) | n/a | 2.1e-6 / 0.18 | 6.0e-7 / 3.6e-3 | **2.7e-3 (8.3e-2)** / 4.9e-2 | 0.46-0.66 -> 0.24-0.39 (1.7-1.9x) -> 0.128-0.130 (3.6-5.1x) |

(two timing runs; ranges given; the B=8 no_lgd n=32 column uses 8 restarts, i.e. 256 CM rows.)

* EXACT claims (generator seeding, batched LGD, `frozen_params`, `lean_ddim` relative to batched) **confirmed** (0.0).
* Cached-YY / batched MMD: per-step teacher-forced gradient agreement 1e-6 **confirmed**; e2e differences O(0.1-3) are the chaos (finding 4), as they state.
* Batched restarts B>=2: their "per-step <= 2.2e-4" **understates** it. I find up to **2.7e-3 absolute = 8% relative** at one step (2D n=32, restart 5, t=64). I traced it: batching the *denoiser* (batch 2 instead of 1) changes `pred_x0` by 4.8e-7 (GEMM path), and the CM is a **ReLU MLP** whose Jacobian jumps when a unit flips -- a 5e-7 perturbation of `cond` at that point changes `dMMD/dcond` from -0.01968 to -0.02132 (checked directly; 1e-7 does not). The float32 reference itself agrees with a float64 recomputation to 3e-7, so this is not conditioning of the reference -- it is round-off + a discontinuous Jacobian. So the right statement is: **statistically equivalent, round-off-triggered per-step gradient jumps of up to ~10% at a small fraction of steps, and e2e trajectories that differ O(1)**. It cannot be "numerically equivalent per step" in the 1e-6 sense; the distributional comparison over restarts (as their recommendation 1 says) is the only valid equivalence check, and my held-out run is exactly that kind of comparison for the engine path.
* Speedups: B=1 1.6-5x (mostly the MMD restructuring + batched LGD), B=8 **4-15x per restart** on this loaded machine; their 7.5-19x at B=8 and 14-25x at B=32 are plausible for a quiet machine (I did not run B=32). **Verdict: reorder-only (REORDER) except the EXACT items listed; speedup claims hold in magnitude.**

## C. Held-out confirmation -- design (authored, not run)

* Driver `verification/heldout_cells.py` imports `estimator/engine_runner.py` unchanged and calls `cell(...)` with `--offset 1000 --restarts 100`, `rng=tape`, float32 (as the screening). It adds `mmd2_eval` (256 fresh conditional draws at `x_hat`, keyed `("heldout_eval", restart, i)`, same for every candidate), host / CPU model / torch version per cell.
* Cells (108): 2D/5D/10D x n in {4,8,16,32} x no_lgd/none x {baseline, relclip2, relclip_ema2, trust_noise1, sqrt_floor, clip0.5, relclip1, sqrtfloor_clip0.5}; Pareto baselines no_lgd/none n in {64,96}; lgd/none baseline n in {8,32}.
* **18 array tasks = groups**, each group = one (setting, n, arm) baseline + its 7 candidates run sequentially in ONE process (same node -> clean pairing, finding 4); the 6 baseline-only groups hold the Pareto/LGD cells. Outputs `verification/heldout_runs/<cell>_off1000.json` (engine_runner summary + per-restart L2/abs_err/diverged/calls/seconds/mmd2_eval); existing files are skipped on resubmission.
* Phase-2 analysis `verification/analyze_heldout.py`: paired diff (base - cand), bootstrap 95% CI, permutation p, wins (`_common.paired_stats`, B=P=20000), the same for `mmd2_eval`, success/divergence deltas, a calls-match assertion, the screening (R=40, offset 0) estimate next to the held-out one (shrinkage), and a Pareto table + frontier per setting (score vs calls). Writes `heldout_tables.md`, `heldout_rows.csv`. Dry-run on a 3-restart smoke cell OK.
* Expected runtime per paired group: 800 restarts x 0.6-2 s ~ 10-30 min (2 CPUs); limit set to 6 h.

Command (after syncing `experiments/model-optimization/verification/` and the unchanged `estimator/engine_runner.py` to the cluster):

```
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
sbatch --array=0-17%18 ../experiments/model-optimization/verification/submit_heldout.sh
# pull back: experiments/model-optimization/verification/heldout_runs/*.json
# then locally: cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/analyze_heldout.py
```

## Red flags summary (for the boss)

1. Cross-platform chaos (finding 4): per-restart numbers are not comparable between Mac and cluster; paired comparisons must be same-node. Screening pairs were separate array tasks (possible node heterogeneity -> extra noise, not bias).
2. Batched-restart per-step equivalence is weaker than reported (ReLU Jacobian jumps, up to ~10% at some steps); statistically equivalent is the defensible claim.
3. `qclip0.5` == `relclip1` (duplicate rule); count once.
4. Screening wall times contain cluster-contention outliers (17 s/run); Pareto on calls only.
5. `screening_rows.csv` hardware label is the report machine, not the run machine.
6. Selection and evaluation metric coincide (exact GMM L2) -- fine since it is the paper's independent metric and is not the optimised objective; held-out seeds + `mmd2_eval` + success added.
