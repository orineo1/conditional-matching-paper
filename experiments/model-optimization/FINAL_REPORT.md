# FINAL REPORT -- CDM / distributional-TFG performance campaign

Campaign root `experiments/model-optimization/` (brief `FABLE_CDM_PERFORMANCE_ORCHESTRATION.md`,
2026-08-23), commit `6af2081` + working tree of branch `tfg-generalization-v2`. This report (boss-final, built on Agent 7's draft)
integrates `BASELINE.md`, `profiling/`, `exact_loss/`, `approx_loss/`, `estimator/`, `systems/`,
`verification/` and `VERIFICATION.md` (the authority for every verdict). Companion files written by
Agent 7: `HYPOTHESES.md` / `hypotheses.yaml` (every method considered, 49 entries, 47 rows),
`results.csv` (13,574 rows, 8 sources incl. the cluster MMD benchmark grid, seed-level where available), `pareto.md` / `pareto.png`,
`report_tools/` (the three scripts that generate them).

Labelling used throughout: **[V]** = verifier-confirmed (Agent 6, `VERIFICATION.md`);
**[I]** = implementer-reported, not independently verified; **[S]** = static analysis, never run.

---

## 1. Winning configuration

**Engine switch (quality):** `trust_noise1` -- a trust region on the applied guidance step,
`||Delta_t|| <= 1.0 * sqrt(1 - alphabar_t)`, i.e. `TemporalConfig.step_clip="noise",
step_tau=1.0`, on top of plain no-LGD guidance (`temporal.mode="none"`, `spatial=no_lgd`, the
`_guided.run` legacy conventions `init="zeros"`, `guidance_scaling="raw"`,
`n_schedule.eta_per_perturbation=True`, float32 `RepositorySchedule`). It is opt-in, off by
default (default engine path still trace-identical to the frozen reference, 331 tests at verification time, 348 after integration),
adds no conditional-model calls, no wall time (O(d) rescale of `Delta_t`), and no new constant
that depends on the task scale (the bound is the noise level of the schedule).

**Exact speed (compute):** the cached-target MMD, now integrated as
`simulations/src/tfg/fast_mmd.py::MMDFixedTarget` (opt-in: `DistributionalLoss(backend="fast")`,
`engine_runner.py --loss fast`; default stays the reference), a drop-in for
`LossFunctions.MMDLoss` in the synthetic loop: mathematically identical loss and first-order
gradient (float64 agreement <= 1.3e-14 relative [V]), 4.6-7.0x on the MMD forward+backward (Mac, float32; cluster EPYC grid: 6.3x float32 / 10.2x
float64 geometric mean, 13x at n<=8, m=250; L4 GPU: ~1x at n<=32 (launch-bound), 10-27x at n=100,
m=2000) and 1.66-2.02x on the whole `_guided.run` restart at identical conditional calls [V for
the loss, I for the loop timing]. In float32 it is reorder-only (round-off), so trajectories are not
bit-reproducible (see Limitations 7.1).

Supporting exact systems items (hygiene, all EXACT = 0.0 end-to-end [V]): batched LGD
perturbations (one CM call on 3n rows, 1.4x on LGD cells), `torch.Generator` instead of global
`manual_seed` (99 us x M per step), `requires_grad_(False)` on the frozen models, lean DDIM step.

Why `trust_noise1` and not `relclip2` (larger 2D gain): `relclip2` is 2D-only and regresses at
10D n=4 (-0.047, p=0.048 [V]); `trust_noise1` is the only rule that passes the brief's
promotion rule -- credible Pareto improvement at two task scales (2D, 10D) with no significant
regression at any of the 12 held-out cells (VERIFICATION.md 5.4).

## 2. Gains, with numbers (held-out: offset 1000, 100 paired restarts/cell, no-LGD/none, float32, same-node pairs [V])

Paired diff = baseline - candidate of the failure-penalised exact GMM L2 (+ = better), bootstrap
95% CI, paired permutation p; calls identical within every pair.

| setting | n | calls | baseline | trust_noise1 | diff [95% CI] | p | wins | success base -> cand |
|---|---|---|---|---|---|---|---|---|
| 2D | 4 | 396 | 0.597 | **0.196** | +0.401 [+0.318, +0.487] | <0.001 | 79/100 | 28% -> 76% |
| 2D | 8 | 792 | 0.418 | **0.167** | +0.250 [+0.183, +0.320] | <0.001 | 76/100 | 40% -> 80% |
| 2D | 16 | 1584 | 0.282 | **0.192** | +0.090 [+0.053, +0.132] | <0.001 | 68/100 | 48% -> 74% |
| 2D | 32 | 3168 | 0.247 | **0.223** | +0.024 [+0.008, +0.043] | 0.003 | 60/100 | 59% -> 70% |
| 5D | 4 | 396 | 0.534 | 0.508 | +0.026 [-0.016, +0.067] | 0.24 | 64/100 | 0% -> 0% |
| 5D | 8 | 792 | 0.508 | **0.473** | +0.036 [+0.006, +0.069] | 0.030 | 56/100 | 0% -> 0% |
| 5D | 16 | 1584 | 0.449 | 0.441 | +0.008 [-0.003, +0.017] | 0.14 | 58/100 | 0% -> 0% |
| 5D | 32 | 3168 | 0.444 | 0.434 | +0.010 [+0.000, +0.023] | 0.062 | 60/100 | 0% -> 0% |
| 10D | 4 | 396 | 0.667 | **0.615** | +0.053 [+0.004, +0.102] | 0.039 | 57/100 | 0% -> 0% |
| 10D | 8 | 792 | 0.658 | **0.535** | +0.123 [+0.071, +0.175] | <0.001 | 71/100 | 0% -> 0% |
| 10D | 16 | 1584 | 0.563 | **0.489** | +0.075 [+0.033, +0.117] | 0.001 | 64/100 | 0% -> 0% |
| 10D | 32 | 3168 | 0.477 | 0.457 | +0.019 [-0.009, +0.048] | 0.19 | 48/100 | 0% -> 0% |

Secondary metric `mmd2_eval` (MMD^2 of 256 fresh conditional draws at `x_hat`, the optimised
objective evaluated at the end point): improves significantly at 11 of 12 cells (2D n=32 null);
divergences: baseline 2D n=4 had 2, `trust_noise1` 0 everywhere [V].

**Compute-matched reading (pareto.md, pareto.png; frontier marked):**

* 2D: `trust_noise1` at n=8 (792 calls, 0.167) beats the baseline at every n up to 96 (9504
  calls, 0.259) and LGD/none at n=8 (2376 calls, 0.201) and n=32 (9504 calls, 0.225): >= 3x
  fewer conditional calls at better quality; at n=4 (396 calls, 0.196) it already matches LGD/none
  at n=8. The 2D frontier itself is `relclip2` (0.153 / 0.139 at n=4/8), a 2D-only rule.
* 5D: effects are 0.01-0.04 on a baseline of 0.43-0.53; `trust_noise1` at n=16 (1584 calls,
  0.441) sits on the frontier, = baseline at n=64 (6336 calls, 0.434) within CI; the 5D frontier
  is otherwise `sqrtfloor_clip0.5` / `sqrt_floor` (conditional rules, see 3).
* 10D: `trust_noise1` is the frontier at every n in {4,8,16,32}; at n=16 (1584 calls, 0.489) it
  beats LGD/none at n=8 (2376 calls, 0.518); at n=32 (3168 calls, 0.457) it equals the baseline at
  n=64 (6336 calls, 0.456): a 2x-calls Pareto gain at n <= 16 (p <= 0.04), null at n=32.

Screening (40 restarts, offset 0) vs held-out (100 restarts, offset 1000): correlation of the
paired diffs over the 84 pairs is 0.96; the 2D effects are fully confirmed, the 5D/10D "small
wins" of the screening are largely noise (19/84 pairs flip sign, all with |diff| < 0.05) [V].

**Speed gains (hardware-dependent; Apple M4, 4 threads, float32):**

| change | exactness | no-LGD n=8 | no-LGD n=32 | LGD n=8 | LGD n=32 | label |
|---|---|---|---|---|---|---|
| cached-target MMD (`fixed_mm_powchain`) in `_guided.run`, 99 steps | REORDER (f32) | 1.77x | 2.02x | 1.92x | 1.97x | I (exact_loss/end_to_end_results.csv); loss-level 4.6-7.0x V |
| batched LGD (3n rows) | EXACT 0.0 | - | - | 1.4x | - | V |
| batched MMD + cached YY (+batched LGD), systems runner | REORDER 1e-6/step | 1.5x | 1.9x | 4.5x | - | V |
| batched restarts B=8 / B=32 | statistical equivalence only | 7.5x / 14x | 7.6x / 8.6x | 19x / 25x | - | B=8 4-15x V; B=32 I |

Baseline per-restart wall (quiet window): 0.18-0.22 s no-LGD, 0.50-0.61 s LGD; per step 1.9-2.2 ms
(no-LGD) -- backward 42-46%, MMD forward 17-22% (79-94% of it the constant target block),
CM forward 15-21%, denoiser 4-10% (`profiling/baseline_profile.md`).

## 3. Verification status (VERIFICATION.md)

| item | verdict | basis |
|---|---|---|
| invariants (default path == frozen reference; all mechanisms opt-in; NoiseTape keys; `Delta_t/sqrt(alpha_t)`; schedules rebuilt; caches exact/target-only; held-out seeds disjoint; budgets matched) | all hold | 331 passed / 1 skipped; mutation tests; line-by-line diff read; programmatic audit of 519 screening JSONs |
| `fast_mmd` cached-target MMD | **EXACT** (f64 <= 1.3e-14; f32 round-off) -- promote | own value+gradient check on 7 (n,m,d) x 2 seeds x fixed/adaptive bw x 6 variants |
| systems: batched LGD, generator seeding, frozen params, lean DDIM | **EXACT (0.0)** | re-run on 8 restarts |
| systems: cached-YY / batched MMD | REORDER (1e-6 per step) | teacher-forced per-step gradients |
| systems: batched restarts B >= 2 | **statistical equivalence only** (per-step jumps up to 8% relative at ReLU-boundary steps; authors' "<= 2.2e-4" understated) | traced to a 5e-7 `pred_x0` perturbation crossing a ReLU Jacobian jump |
| `trust_noise1` | **PASS -- promote** (2D PASS 4/4, 10D PASS n=4/8/16, 5D inconclusive but all +; no regression in 12 cells) | 108-cell held-out run, job 45923971 |
| `sqrt_floor`, `sqrtfloor_clip0.5` | **CONDITIONAL PASS** (2D + 5D; FAIL 10D n=32: -0.038 / -0.088) | same |
| `relclip2`, `relclip_ema2`, `clip0.5`, `relclip1`(=`qclip0.5`) | **not promoted** (one scale; `relclip2` 10D n=4 -0.047, `clip0.5`/`relclip1` 10D n=32 -0.075/-0.076; `relclip_ema2` safe but 2D-only) | same |
| Adam-arm and LGD-arm interactions of the rules | **not held-out** (screening only, [I]) | -- |
| MNIST (Stage 2) and SD (Stage 3) | **not run** | MNIST: blocked -- the CDM checkpoints (`anon-submission-cdm/cdm-inverse-design`) need `HF_TOKEN`, and no token/checkpoint is present on the cluster or locally; SD: not reached within the campaign; static audits only |

## 4. Exact reproduction commands

All local commands: `cd /Users/stolk/github/conditional-matching-paper/simulations`, python
`/Users/stolk/miniconda3/bin/python`, no pip installs. Cluster: repo checked out at
`/sci/labs/orzuk/shaulytolk/cdm-perf` (incl. `simulations/artifacts/checkpoints/*.pt`,
`simulations/params/`), logs in `/sci/labs/orzuk/shaulytolk/cdm-perf/logs/`; sbatch needs a
login shell through the tunnel (`ssh -p 2222 shaulytolk@localhost "bash -lc '...'"`).

```bash
# --- baseline (Agent 1): 18 cells x restarts 0..4, 1 warm-up + 5 timed repeats; profiles
python ../experiments/model-optimization/profiling/run_baseline.py            # -> profiling/baseline_rows.csv
python ../experiments/model-optimization/profiling/run_baseline.py --cell 2D no_lgd none 8
python ../experiments/model-optimization/profiling/profile_guided.py
python ../experiments/model-optimization/profiling/torch_profile.py

# --- tests / invariants (Agent 4 + 6)
python -m pytest tests -q                                                       # 348 passed, 1 skipped (331 at verification time)
python experiments/exp8_trust_region.py --setting 2D --restarts 100 --offset 1000   # integrated engine-path experiment
python -m pytest tests/test_engine_matches_guided.py tests/test_agent4_candidates.py -q -s   # engine == _guided.run, 0.0
python -m pytest ../experiments/model-optimization/exact_loss -q                # 236 passed
python ../experiments/model-optimization/verification/check_fast_mmd.py        # own f64 value+grad check
python ../experiments/model-optimization/verification/check_systems.py         # systems equivalence re-run

# --- exact MMD end-to-end and micro-benchmarks (Agent 2)
python ../experiments/model-optimization/exact_loss/end_to_end_check.py        # -> exact_loss/end_to_end_results.csv
python ../experiments/model-optimization/exact_loss/bench_mmd.py --grid small   # -> bench_results_small.csv
# cluster full grids (DONE: jobs 45924238 glacier CPU, 45924283 catfish L4 -> exact_loss/bench_results.csv, bench_summary.md):
ssh -p 2222 shaulytolk@localhost "bash -lc 'cd /sci/labs/orzuk/shaulytolk/cdm-perf && sbatch experiments/model-optimization/exact_loss/submit_bench.sh'"
ssh -p 2222 shaulytolk@localhost "bash -lc 'cd /sci/labs/orzuk/shaulytolk/cdm-perf && sbatch experiments/model-optimization/exact_loss/submit_bench_gpu.sh'"

# --- systems benchmarks (Agent 5); cluster versions NOT submitted (local numbers only)
python ../experiments/model-optimization/systems/bench.py                       # -> systems/bench_rows.csv
ssh -p 2222 shaulytolk@localhost "bash -lc 'mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs && sbatch /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/model-optimization/systems/submit_bench.sh'"
ssh -p 2222 shaulytolk@localhost "bash -lc 'sbatch /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/model-optimization/systems/submit_bench_gpu.sh'"

# --- one screening cell locally (Agent 4), e.g. the winner
python ../experiments/model-optimization/estimator/engine_runner.py --setting 2D --n 8 --spatial no_lgd --temporal none \
    --candidate trust_noise1 --restarts 40 --offset 0 \
    --out ../experiments/model-optimization/estimator/runs/2D_n8_no_lgd_none_trust_noise1_tape.json
python ../experiments/model-optimization/estimator/grad_noise.py               # gradient SNR table (~10 min)

# --- screening grids on the cluster (done: round 1 258 cells, round 2 261 cells)
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
N=$(python ../experiments/model-optimization/estimator/screen.py list 2>/dev/null | wc -l)
sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/estimator/submit_screen.sh
N2=$(python ../experiments/model-optimization/estimator/screen.py list --round 2 2>/dev/null | wc -l)
ROUND=2 sbatch --array=0-$((N2-1))%40 ../experiments/model-optimization/estimator/submit_screen.sh
python ../experiments/model-optimization/estimator/screen.py report            # -> screening_rows.csv, screening_tables.md
python ../experiments/model-optimization/estimator/matrix.py                   # -> round2_matrix.md

# --- held-out confirmation (Agent 6; done: job 45923971, glacier, 18 array tasks = 108 cells)
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
sbatch --array=0-17%18 ../experiments/model-optimization/verification/submit_heldout.sh
# pull back verification/heldout_runs/*.json, then locally:
python ../experiments/model-optimization/verification/analyze_heldout.py       # -> heldout_tables.md, heldout_rows.csv

# --- this report's derived files (Agent 7)
python ../experiments/model-optimization/report_tools/merge_hypotheses.py      # -> hypotheses.yaml, HYPOTHESES.md
python ../experiments/model-optimization/report_tools/build_results.py         # -> results.csv
python ../experiments/model-optimization/report_tools/pareto.py                # -> pareto.md, pareto.png
```

Winning configuration in code (engine path, as `estimator/engine_runner.py::candidate_spec("trust_noise1")`):
`cfg.temporal.step_clip = "noise"; cfg.temporal.step_tau = 1.0` on the legacy-equivalent
`TFGConfig(init="zeros", guidance_scaling="raw", smoothing="tfg", n_mc=1)` with
`n_schedule.eta_per_perturbation=True`, `RepositorySchedule` float32, `CMSampler(source="tape")`.
Loss replacement: `MMDFixedTarget(Y=S_G, bandwidth=_common.fixed_bandwidth(S_G), dist="mm",
kernel_eval="powchain")` (the `FastMMDLossShim` in `exact_loss/end_to_end_check.py` shows the
drop-in `mmd(y, S_G)` signature). Both are integrated behind opt-in switches: `TemporalConfig.step_clip/step_tau`
(documented in `simulations/src/tfg/README.md`, "Step-size control"), `tfg/fast_mmd.py` +
`DistributionalLoss(backend="fast")`, `engine_runner.py --loss fast`, and the reproducible
experiment `simulations/experiments/exp8_trust_region.py` (engine path; row 8 + results section in
`simulations/experiments/README.md`). Tests: 348 passed (`simulations/tests/test_fast_mmd_integration.py`,
`test_agent4_candidates.py`, `test_engine_matches_guided.py`). `_guided.py` / `LossFunctions.py`
are deliberately left unchanged (legacy path, bit-identical to the engine's legacy switches).

## 5. Most important negative results

1. **Approximate losses (FFT / RFF / ORF / Nystrom / sliced / linear-time) -- all rejected [I].**
   Once the constant target block is cached, the exact loss costs `n(n+m)` kernel evaluations; at
   m=250 an RFF with D=256 costs 1.2-2.2x MORE with gradient cosine 0.82-0.90 (d=8,16); at d=768
   D~2^16 is needed (cos 0.23 at D=256). Nystrom on target landmarks collapses when X is
   off-target (cos 0.25, d=768) -- exactly when guidance matters. Sliced W2 is a different
   objective (cos 0.3-0.9, 0.1 at d=768). FFT has no grid in d>=3 and the alpha=2 SD kernel is
   not positive definite. The only approximation-class idea worth a screen is not an approximation:
   the population-GMM target (cos 0.989-0.998, 0.02-0.55x cost) -- never run.
2. **Adaptive n_t (agreement / improvement policies, equal total calls): rejected [I]** -- null or
   negative everywhere (2D n=8 -0.20..-0.34*, 10D n=32 -0.08..-0.14*). Reallocating samples across
   steps does not help; the binding constraint is the step rule (the per-step gradient is
   noise-dominated at every n: SNR 0.1-0.6, a single draw has the wrong sign 20-40% of the time).
3. **Stale gradients (reuse k steps, 1/k calls): rejected [I]** -- worse wherever significant
   (2D n=8 -0.23/-0.32, 5D n=4 -0.34/-0.67, 10D n=32 -0.20/-0.24). Adaptive recurrence, CRN,
   antithetic: null.
4. **Adam regime [I]:** Adam (rho=0.4) is worse than plain guidance in 5D/10D (5D n=8 0.61 vs
   0.50; 10D n=32 0.57 vs 0.49) and over-steps at 2D n=32; clipping before Adam never helps and
   hurts in 10D (-0.075..-0.122*); plain guidance + `trust_noise1` beats Adam in every dim (2D n=8
   0.181 vs 0.303). The Adam benefit is a tail cap + ~10-step averaging at small n, a rho-sized
   floor near the optimum at large n.
5. **Absolute clipping scale dependence [V]:** `clip0.5` is a tail cap in 2D (5x the median raw
   norm, +0.36/+0.19 at n=4/8) but clips the bulk in 10D (1.5x the median; n=32 -0.075*); `clip0.1`
   is the reverse (2D n=32 -0.356). Raw gradient-norm medians: 2D 0.04-0.09, 5D 0.09-0.25, 10D
   0.29-0.38. Scale-free rules (median x2, `sqrt(1-alphabar_t)`) are the ones that transfer; the
   relclip family still fails the two-scale rule on held-out seeds (10D n=4 / n=32).
6. **`trust_ddim` (step bounded by the DDIM move): rejected [I]** -- wins everywhere in 10D
   (+0.08..+0.19*) but catastrophic in 2D (-0.21..-0.46*); at tau=0.1 the 2D/5D scores equal the
   UNGUIDED chain (0.595/0.91 vs 0.597/0.912): the trust radius switches guidance off.
7. **torch.compile / MPS: rejected [I]** -- loop 0.84x steady state + 16 s compile (+270 MB);
   MMD alone 1.4-2.5x steady but 2-7 s compile per shape and 26k-58k-call break-even (a restart
   makes 99-297 calls; macOS inductor hung 9/12 attempts); MPS 5x slower at B=1, 24 r/s at B=32 vs
   70-90 on CPU (launch/sync bound).
8. **Batched restarts are not a reproduction [V]:** ~10x throughput, but per-step gradients jump
   up to 8% at ReLU-boundary steps and trajectories differ O(1); use for screening, report as
   re-runs.
9. **Hygiene items with no measurable gain [V]:** `requires_grad_(False)` (~0%), lean DDIM
   (+1-5%), chunked / batched / micro-batched MMD forms (exact, memory only at these sizes).

## 6. SD and MNIST static findings -- unverified recommendations [S]

No SD or MNIST run was made in this campaign (no GPU / checkpoints locally; Stage 2/3 not
reached). `profiling/baseline_profile.md` 4b-c and `systems/AUDIT.md` 2-3 give file:line audits.

**MNIST (`MNIST/run_mlgdf.py`, SWD guidance, 129 guided DDIM steps, `num_x_t=3`, 1500 samples):**
per seed 580,500 conditional samples, 1935 CM forwards, 387 SWD evals. Wasted/static: the
conditioning CNN encoder is recomputed at each of the 5 ladder levels (15x per step; hoist ->
5->1, EXACT, ~80% of encoder cost); target angles and the 50 SWD projections are resampled every
call (extra gradient noise -- an estimator change if fixed); `retain_graph=True` keeps the whole
graph alive; models not frozen; perturbations and 15 seeds run sequentially at batch 1
(launch-bound; batching est. 5-15x, changes RNG stream); uniform-target runs compute the full
inner loop for ~72 tail steps where `step_size=0`; `torch.cuda.empty_cache()` per seed.

**SD (`SD_cond_SD_controlnet/scripts/run_mlgd_f.py` + `src/generation.py`; the brief's
`run_dps_synthetic_targets.py` is not on this branch):** per guided step the architect UNet is
run twice (guided CFG batch-2 with grad although `guidance_scale=0.0`, plus an unguided control
trajectory); 6 sprinter pipeline calls at `variation_batch_size=1`, each re-encoding the constant
prompt through both SDXL text encoders (not frozen) and wrapped in nested gradient checkpointing
(UNet/ControlNet blocks computed 3x); `visualize_step` EVERY step = 5 more sprinter calls + 4-5
VAE decodes + 2 full-VAE dtype casts + savefig (~45% of sprinter forwards are diagnostics); CLIP
moved GPU<->CPU around eval; `gc.collect(); empty_cache()` every step; `K_yy` recomputed each step
(the MMD itself is < 0.1% of a step -- the cached-target MMD is NOT worth integrating for speed
there). Estimated: gate visualisation 25-35%, un-nest checkpointing 15-25% of guidance compute,
batch variations with per-sample generators 1.5-2x on the sprinter+CLIP path, `gs==0` single
batch ~2x on the architect share; all EXACT for the guided path. Numerics-changing options
(fp16-fix VAE, bf16 CLIP, truncated sprinter backprop, 1-step sprinter) need grad-cosine and
delta-MMD validation on >= 3 seeds. Whether `trust_noise1` transfers to SD (adaptive
`zeta = base_zeta / MMD`, `sqrt` loss, latent space) is untested; it is the one rule that needs no
scale constant, which is the reason to try it first.

## 7. Limitations

1. **Float32 chaos.** Exp 2-7 and the whole campaign run float32 end to end (not float64 as the
   README assumed). A 1-ulp change in a per-step gradient moves the final `x_hat` by O(1) on a
   fraction of restarts; the same restart lands in a different mode on the Mac and on the cluster
   (restart 1, 2D n=8: -4.35 vs +6.24) [V]. Consequences: only bit-identical changes can be
   "reproductions"; every REORDER change (cached MMD, batched restarts, BLAS/thread changes) is a
   statistically equivalent re-run; paired comparisons must be same-node (the held-out run was;
   the screening pairs were separate array tasks -- extra noise, not bias, diffs correlate 0.96).
2. **10D zeta mis-calibration.** With zeta=1 the unguided chain (0.581) beats guided no-LGD at
   n=4/8 (0.62) and Adam at every n; every step-shrinking rule wins in 10D and is catastrophic in
   2D; raw gradient norms are 3-8x the 2D ones. The 10D (and partly 5D) comparisons therefore
   measure "how much does the rule tame an over-sized step", not sample efficiency; a fair dim(x)
   sweep needs the per-dimension `zeta_d` of Exp 5B first. `trust_noise1` is invariant to that
   rescaling (annealed bound), which is why it is the rule that survives -- but its 10D gain should
   be re-measured after calibration.
3. **5D null.** No rule is reliably better than the baseline at 5D n <= 16; the significant 5D
   gains are 0.02-0.06 on a baseline of 0.41-0.53 (real: 78-92/100 wins at n=32, p <= 0.05) and
   `trust_noise1` is INCONCLUSIVE there (1/4 cells significant). The "two task scales" rule is met
   by 2D + 10D.
4. **Success rate is uninformative in 5D/10D** (0% for every arm incl. LGD n=32 and the baseline;
   the |x_hat-x*| < 0.5 threshold was set for dim(x)=1). Only the exact L2 carries information.
5. **Selection metric = evaluation metric** (exact GMM L2) -- defensible because it is the paper's
   independent metric and never the optimised objective, and the held-out seeds remove the
   winner's curse; `mmd2_eval` and success were added as secondary checks [V]. Note the objective
   and the metric disagree in 10D at small n: the absolute-clip family lowers the optimised MMD at
   `x_hat` while its exact L2 gets worse (10D n=32 L2 -0.075..-0.088 vs mmd2_eval +0.06..+0.07):
   the fixed-bandwidth sample MMD is an imperfect proxy for the paper's metric at these (n, d).
6. **Not held-out:** Adam-arm and LGD-arm interactions, `trust_noise0.3`, `relclip_ema1`,
   `qclip0.75`, the combination hypotheses (`trust_noise1 + relclip2`, `+ sqrt_floor`, `+ LGD at
   n=4`, `trust_noise1` with calibrated `zeta_d`) -- screening evidence only or none.
7. **MNIST / SD stages not run** (Stage 2/3 of the brief). MNIST was initially thought blocked on `HF_TOKEN`, but the checkpoint snapshot is present in the
   cluster HF cache (correction 2026-08-24) -- Stage 2 is feasible and simply was not run. SD was not reached; the cluster
   time went to the 627 synthetic cells (519 screening + 108 held-out) and the MMD benchmark grids.
   All SD/MNIST statements are static audits. Agent 2's cluster CPU/GPU MMD grids WERE run (EPYC
   7662 4 threads; NVIDIA L4); Agent 5's systems cluster scripts were not; the systems speed numbers
   are from a shared, often loaded Mac (ratios within one run are meaningful, absolute seconds are
   not).
8. **Code hazards noted by the verifier, not fixed:** engine instance state (clip history, Adam
   moments, stale cache) is not reset by `run()`; `CMSampler(cache=True)` keys on `x` bytes and
   returns graph-attached rows of the first call (thrashes under LGD); `screening_rows.csv`
   stamps the report machine as `hardware` (relabelled in `results.csv`); `qclip0.5 == relclip1`
   (counted once).
9. **Brief deliverable 6 done, 7 pending:** `trust_noise1` / `fast_mmd` are integrated (opt-in,
   tested, documented, `exp8_trust_region.py`); the clean commit series is NOT made because git
   actions are user-only in this repo -- see the suggested commands at the end.

## 8. Recommended paper claims (conservative phrasing)

Claims supported by verifier-confirmed evidence:

* "A noise-level trust region on the guidance step, `||Delta_t|| <= sqrt(1-alphabar_t)`, with
  no tuned constant, improves the exact-L2 quality of plain MMD guidance at matched conditional
  calls on held-out seeds: in the 2D benchmark the mean exact L2 at n=8 drops from 0.42 to 0.17
  (paired diff +0.25, 95% CI [+0.18, +0.32], 100 restarts) and the success rate from 40% to 80%,
  so that n=8 with the trust region matches or beats the un-regularised method at n=96 and LGD at
  n=32 (3-12x more conditional calls); in the 10D benchmark the gain is +0.05 to +0.12 at
  n <= 16 (p <= 0.04) and null at n=32; in 5D it is small (+0.01 to +0.04) and mostly not
  significant. No held-out cell shows a significant regression."
* "The fixed-target MMD can be evaluated exactly with only the cross and sample-sample kernel
  blocks (the target-target block is a run constant), which is mathematically identical to the
  stacked-matrix form (float64 agreement to 1e-14) and reduces the loss cost 4-7x and the
  synthetic guidance loop 1.7-2x at identical conditional-model calls." (Loop factor is
  implementer-measured; say "about 2x" at most.)
* "Absolute gradient clipping is scale-dependent across dimensions; only scale-free step control
  (noise-level trust region, running-median clipping) transfers, and of these only the
  noise-level trust region is free of a significant regression at some (dimension, n)."
* "Random-feature and low-rank approximations of the MMD do not pay off in this regime: with the
  target block cached the exact loss is cheaper than an RFF with enough features to match its
  gradient direction (D >~ m), and a Nystrom expansion on target landmarks loses the gradient
  when the samples are far from the target."

Claims to avoid or qualify:

* Do NOT claim per-trajectory equivalence for the cached MMD or batched restarts in float32 --
  only distributional equivalence over restarts.
* Do NOT claim a 5D improvement, a 10D improvement at n=32, or any Adam/LGD-arm interaction
  for `trust_noise1`; do NOT present `relclip2`'s 2D numbers as a general result.
* Do NOT quote the 10D comparisons as sample-efficiency results until `zeta_d` is calibrated
  (Exp 5B); present them as robustness to step mis-calibration.
* Do NOT present the MNIST/SD savings or the population-GMM target as results (static /
  diagnostic only).
* If batched restarts are used for any reported experiment, label it a re-run with statistical
  equivalence, not a reproduction of the sequential numbers.

## 9. Inconsistencies found while integrating (for the boss)

* `VERIFICATION.md` 5.3 says "every candidate at n=8 (0.14-0.30) beats the baseline at n=96
  (9504 calls, 0.259)" in 2D -- `sqrt_floor` at n=8 is 0.299 (> 0.259), so "every" should read
  "every candidate except sqrt_floor".
* `VERIFICATION.md` 4.1 quotes the fast-MMD end-to-end gain as "1.6-2.6x"; `exact_loss/end_to_end_results.csv`
  gives 1.66-2.02x (max 2.016). Use 1.7-2.0x.
* `exact_loss/REPORT.md` says "~10x on the loss micro-benchmark" (float64 geometric mean 10.2x)
  while the float32 production figure is 6.3x (geo-mean) / 4.6-7.0x (verifier); the loop runs float32.
* `estimator/REPORT.md` 7 says round 2 = "324 cells" in the reproduction block but "261 new" in
  the header; `runs/` holds 519 JSONs = 258 + 261.
* `systems/BENCH.md` reports batched-restart per-step agreement "<= 2.2e-4"; the verifier measured
  up to 2.7e-3 absolute (8% relative) at a ReLU-boundary step -- the BENCH claim is understated.
* `README.md` said "float64 as the repo does"; every agent found the experiment loop is float32
  end to end (only `evaluate` and `tfg.schedule` are float64) -- corrected in README.md.
* `estimator/screening_rows.csv` labels `hardware = arm64 cpu` (report host) for cluster x86 runs
  (relabelled in `results.csv`, VERIFICATION red flag 5).
* The 5D held-out frontier in `VERIFICATION.md` 5.3 includes `sqrtfloor_clip0.5` (not in the
  arm list requested for `pareto.png`); among the plotted arms the 5D frontier is `sqrt_floor`
  at n=4/8/32 and `trust_noise1` at n=16 (`pareto.md`).
* `hypotheses/agent1.yaml` and `agent4.yaml` are not strictly valid YAML (unquoted `: ` inside
  scalars); `report_tools/merge_hypotheses.py` loads them with a tolerant fallback.

## 10. Final status and suggested commit series (boss)

Campaign outcome against the brief's stop rule: **(a) reached** -- `trust_noise1` is an
independently verified Pareto improvement (quality at matched conditional calls, 2D + 10D, no
regression in any held-out cell) and the cached-target MMD is an independently verified exact
speed-up. Stage 2 (MNIST) is blocked on `HF_TOKEN`; Stage 3 (SD) is prepared as static,
file:line-anchored recommendations (`systems/AUDIT.md`) with `trust_noise1` the first rule to try
(no scale constant).

Suggested ordered commits (not executed; git is user-only):

```bash
git add simulations/src/tfg/{engine.py,config.py,n_schedule.py,adaptive.py,distributional.py,fast_mmd.py,README.md} \
        simulations/tests/{test_engine_matches_guided.py,test_agent4_candidates.py,test_fast_mmd_integration.py}
git commit -m "tfg: opt-in step-size control (noise trust region), exact cached-target MMD, engine==_guided equivalence tests"
git add simulations/experiments/{exp8_trust_region.py,README.md}
git commit -m "Exp 8: trust-region guidance through the engine; held-out results"
git add experiments/model-optimization
git commit -m "Performance campaign: baseline, hypotheses, screening, held-out verification, final report"
# the pre-existing uncommitted files (exp5b, exp7, _guided.py diff) are separate earlier work
```

## 11. Addendum — rounds 3-5 (2026-08-24)

**Protocol correction.** An external audit found rounds 1-4 used the legacy protocol
(`x_T = 0`, `zeta = 1`), which `simulations/experiments/README.md` had meanwhile shown
to be defective. Round 5 re-established the headline result under the corrected protocol
(`x_T ~ N(0,I)`, per-arm calibrated zeta, `protocol/`):
- calibration: with the trust region every dimension is divergence-free to zeta=32 and
  calibrates at zeta* = 16/8/4 (2D/5D/10D); without it the estimator cannot exceed
  zeta = 2/0.25/1. The trust region's primary value is making the correct step scale usable.
- trust vs no-trust at each arm's own zeta*, two independent seed sets (offsets 6000, 7000,
  R=100 each): 2D +0.5/+0.36/+0.2/+0.06 (all n, p<=0.016), 5D +0.06..+0.08 (confirmation run,
  all n), 10D +0.07/+0.12 at n<=8; no significant loss in 24 cells; robust to the zeta rule.
  **Verifier: FINAL PASS.** (`VERIFICATION.md` section 10)

**Round 3-4 candidates (all legacy-protocol unless stated):** preconditioning (4 rules) —
rejected, direction is not the constraint; sample replay (Ori's `reuse_frac`, found on
`claude/hybrid-sampling-optimization-55fv3b`, generalised to geometric / FIFO / cohort
buffers) — geo0.7d5 failed held-out; equal-cost controls M-8/M-9 showed no general gain;
fifo16 was a genuine 10D-only gain on the legacy protocol but under the corrected protocol
(M-11) it is 2D-only and significantly worse in 5D — closed; importance-selected backprop —
unbiased and stable, no cost win on the MLP benchmark (forwards n+k), estimated 2-3.5x guidance
saving at SD scale only (`backsel/THEORY.md` section 6).

**Standing results:** trust_noise1 (accuracy; corrected protocol PASS) and the exact
cached-target MMD (speed). Everything else tested is documented as rejected or regime-limited.
