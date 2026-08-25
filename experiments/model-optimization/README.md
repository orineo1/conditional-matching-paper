# CDM / Distributional-TFG performance campaign

Campaign root for the brief `FABLE_CDM_PERFORMANCE_ORCHESTRATION.md` (2026-08-23).
The brief's `research/performance/` deliverables live **here** instead
(`experiments/model-optimization/`), by user instruction.

Source of truth for the implemented baseline:

* engine: `simulations/src/tfg/` (`engine.py` = generalised TFG Algorithm 1,
  `reference.py` = frozen transcription, tests in `simulations/tests/`)
* synthetic experiment loop actually used by Exp 2-7:
  `simulations/experiments/_guided.py` (+ `_common.py`, `_models.py`)
* MMD: `simulations/src/LossFunctions.py` (`RBF`, 5 bandwidths, `mul_factor=2`,
  biased V-statistic on the stacked `(X;Y)` kernel)
* MNIST: `MNIST/`, Stable Diffusion: `SD_cond_SD_controlnet/`

Layout

| dir | owner | content |
|---|---|---|
| `profiling/` | Agent 1 | baseline reproduction, hierarchical profile, call accounting |
| `exact_loss/` | Agent 2 | mathematically equivalent MMD accelerations + tests + benchmarks |
| `approx_loss/` | Agent 3 | RFF / Nystrom / linear-time / sliced candidates + diagnostics |
| `estimator/` | Agent 4 | engine-vs-`_guided` equivalence, estimator/update-rule candidates |
| `systems/` | Agent 5 | autograd/systems audit and benchmarks |
| `verification/` | Agent 6 | independent re-runs and verdicts |
| `report_tools/` | Agent 7 | scripts that build `hypotheses.yaml`, `results.csv`, `pareto.*` |
| `hypotheses/` | all | one YAML per agent, merged into `hypotheses.yaml` |

Top-level deliverables: `BASELINE.md`, `HYPOTHESES.md`, `hypotheses.yaml`,
`results.csv`, `VERIFICATION.md`, `FINAL_REPORT.md`.

Conventions

* Python: `/Users/stolk/miniconda3/bin/python` (torch 2.12, CPU). NOTE: the synthetic
  experiment loop runs float32 end to end (only evaluation and `tfg.schedule` are float64). Run from `simulations/` with `sys.path` containing `simulations/src`
  and `simulations/experiments` (see `_common.py`).
* No `pip install`, no git add/commit/push (user-only).
* Timing: warm-up, `time.perf_counter()`, report median of >= 5 repeats, CPU
  peak RSS via `resource.getrusage`; report conditional-model calls as the
  hardware-independent cost.
* `results.csv` columns: commit,candidate,task,target,seed,config,hardware,dtype,
  wall_s,peak_mem_mb,score_calls,cond_calls,cond_samples,opt_loss,eval_metric,status

Cluster mirror used for all heavy runs: `/sci/labs/orzuk/shaulytolk/cdm-perf/` (simulations/ + experiments/model-optimization/), partition glacier (CPU) / catfish (L4). Final deliverables: `FINAL_REPORT.md` (start here), `VERIFICATION.md`, `HYPOTHESES.md`, `BASELINE.md`, `results.csv`, `pareto.md`/`pareto.png`.

## Campaign log — what we tried and what we got
> **PROTOCOL NOTE (2026-08-24, after external audit).** Every quality comparison in
> rounds 1-4 (trust region, preconditioning, replay, M-8/9/10, backsel) ran on the
> **legacy protocol**: `x_T = 0` and uncalibrated `zeta = 1` (engine_runner defaults,
> chosen to match `_guided.py` bit-for-bit). `simulations/experiments/README.md`
> ("PROTOCOL CORRECTION") shows that protocol is defective: zero init collapses all
> restarts into one basin and `zeta = 1` is ~8x under-scaled in 2D. Engineering results
> (exactness, speed, calls, equivalence) are unaffected. **All relative-quality
> verdicts below are [legacy protocol] and are being re-run under the corrected
> protocol (`x_T ~ N(0,I)`, per-arm/per-dim calibrated zeta, trust region as the
> divergence guard) in round 5.** Round 5 outcome: trust_noise1 PASSES under the corrected
> protocol (two independent seed sets); all other quality candidates remain legacy-only or rejected.


Details: `IMPROVEMENTS.md` (verdicts), `FINAL_REPORT.md` (round 1-2), `VERIFICATION.md` (all held-out verdicts).

| round | idea | outcome |
|---|---|---|
| 1-2 | exact MMD accelerations (cached target block, batched forms) | **PROMOTED** — exact, 4-7x loss / ~2x loop (`tfg/fast_mmd.py`) |
| 1-2 | step-size rules (clips, trust regions, sqrt transforms) | **PROMOTED: trust_noise1** `||Delta_t|| <= sqrt(1-abar_t)` — 2D n=8 L2 0.42->0.17, verified held-out; absolute clips don't transfer |
| 1-2 | approximations (RFF/ORF/Nystrom/FFT/sliced), adaptive n_t, stale gradients, CRN/antithetic, Adam variants, torch.compile/MPS | rejected |
| 3 | preconditioning (covariance whitening, diag RMS, sign, temporal median; +/- trust) | rejected — direction is not the constraint, step size is |
| 3 | sample replay, Ori's reuse_frac generalised (geometric decay, depth 5, + trust) | screening promised 3-4x fewer calls at matched quality; **held-out FAIL** (offset 2000, R=100) |
| 3 | M-8 control: fresh f vs fresh f + 30% recycled at EQUAL fresh cost (f=7,14,28) | recycling adds nothing; hurts 10D under trust |
| 3 | M-9: tiny budgets f=1,2,4, batch topped up from buffer (+ trust) | helps 2D at f<=2 (+0.30 at f=1), null 5D, **hurts 10D f=1** — regime-limited, not promoted |

Standing champion: plain no-LGD guidance + trust_noise1, with the fast MMD backend.
Standing caveat: 10D at zeta=1 is mis-calibrated (exp5b calibration is the prerequisite for 10D claims).

### Round 4 [legacy protocol]
- `backsel/` — importance-selected backpropagation: full forward batch without graphs -> full-batch MMD geometry -> select informative subset by output-space gradient -> replay exact samples via saved RNG keys -> backprop only through the subset (uniform / gradient-magnitude IS with inverse-probability weights / clustering).
- `cohort/` — progressive temporally weighted replay: few fresh samples + frozen recent cohorts, gradient only through fresh; explicit geometric decay vs fixed-size buffer vs capped cohort sizes 1,2,4,8,... progressively thinning older cohorts.

### Round 5 — corrected protocol (`protocol/`)
- Calibration (`calibrate_zeta.py`, n=128, x_T~N(0,I), 40 restarts): the basin-reach rule of the
  protocol correction gives 0% reach at every zeta in 5D/10D (dim(x)=1 construct), so the
  pre-registered amendment uses zeta* = argmin penalised exact L2 over divergence-free zeta:
  trust 16/8/4, no-trust 2/0.25/1 (2D/5D/10D). With trust: zero divergences up to zeta=32 in
  every dim; without: diverges above 2/0.25/1. `zeta_star.md`.
- Trust vs no-trust, each at its own zeta*, R=100 fresh seeds (offset 6000), `r5_tables.md`:
  2D +0.59/+0.36/+0.23/+0.09 (n=4..32, all p<=0.008); 10D +0.13/+0.12/+0.09 (p<=0.006), n=32
  +0.06 (p=0.09); 5D null, never negative. No-trust at the trust arm's zeta diverges.
  Independent confirmation at offset 7000 reproduced it (2D all n, 5D all n small, 10D n<=8):
  **verifier FINAL PASS under the corrected protocol.** Backsel: [legacy] no cost win on this benchmark (`backsel/REPORT.md`).
- M-11 (done, R=100, offset 8000, `replay/m11_tables.md`): fifo16/cohort16 replay vs fresh-only at equal
  fresh cost, corrected protocol. 2D: large wins (+0.25 at f=2, +0.16 at f=4 — fifo16@f=4 = 0.168 beats
  fresh n=8 = 0.251 at half the calls); 5D: significantly WORSE (-0.13/-0.07); 10D: null (the legacy 10D
  gain was zeta compensation). **Not promotable** (significant opposite sign in 5D) — a 2D-only,
  small-budget effect. Replay is closed for this campaign.

> Correction (2026-08-24): the MNIST CDM checkpoints are cached on the cluster (`/sci/labs/orzuk/shaulytolk/hf_cache/hub/models--anon-submission-cdm--cdm-inverse-design`), so the MNIST stage is NOT blocked on a token.

### Round 6 (in progress) — Stable Diffusion testbed (`sd/`)
The synthetic loop cannot test back-selection (cheap backward, shared Jacobians) or the trust region in latent space. Moving both into `SD_cond_SD_controlnet/scripts/run_mlgd_f.py` behind opt-in flags with per-step profiling; reduced-but-faithful dev config on L40S; final metric = fresh 2000-sample MMD, plus wall time, VRAM, differentiated variations.

### Round 7 (in progress) — Stable Diffusion runs (`sd/`)
Flags added to `SD_cond_SD_controlnet/scripts/run_mlgd_f.py` (all opt-in): `--mmd_cache_target` (exact cached-target MMD, `src/metrics.py::_target_stats`), `--no_vis`,
`--arch_single_batch`, `--trust_noise TAU`, `--backsel K --backsel_rule
{uniform,is,kcenter,strat} [--backsel_weighting soft]`, `--profile` (`src/profiling.py`), fixed
final-eval OOM (`src/metrics.py` chunked CLIP; final state persisted before eval), standalone
`sd/eval_final.py`.
**Unified code (2026-08-25):** the selection rules, soft weighting and the trust-region math
have ONE implementation in the shared framework — `simulations/src/tfg/backsel.py`
(`select_uniform/importance/kcenter/stratified/stratified_balanced`, `soft_tau` with
`tau_mode local|bandwidth`, `soft_aggregate`) and `simulations/src/tfg/trust.py`
(`noise_cap`, `clip_step`; the engine's `step_clip` calls it, new mode `noise_prev_rms` = the SD
latent convention). `SD_cond_SD_controlnet/src/backsel.py` and `src/trust.py` are thin adapters
(`src/_tfg_path.py` makes `tfg` importable without pip); `src/generation.py::variation_objective`
holds the SD-specific seeded regeneration / CLIP plumbing. Opt-in `--engine tfg`
(`src/tfg_engine_path.py`) runs the architect loop through `tfg.engine.GeneralizedTFG`
(NoiseTape-keyed noise, `x_init` SDEdit start, trust via config); differences from the legacy loop
are listed in that module's docstring and in `SD_cond_SD_controlnet/README.md` sec 5. Tests:
`sd/tests/` (26: adapters == tfg, toy engine-vs-legacy equivalence) and `simulations/tests/` (417). Dev config: 50/50 gender scribble task, 50 guided steps, 32 variations,
2000-sample fresh eval, 8 seeds, control = `novis`. Results: `sd/RESULTS.md`, `sd/REPORT_SEED1.md`,
`sd/BACKSEL_DIAG.md`.
- Cost (solid): back-selection k=8/32 -> 2.6x faster per step (43 -> 17 s), peak VRAM 34 -> 25 GB.
- Quality (7 seeds so far): kcenter k=8 loses +0.10 MMD (greedy k-center leaves one 11-19-member
  cluster whose summed gradient goes through ONE Jacobian -> coherent step inflation), uniform k=8
  loses +0.04, trust tau=0.25 neutral. Arms k=16, trust+uniform, stratified (balanced clusters,
  unbiased) ran; results pending pull.
- **Soft proximity reweighting** (user's idea, implemented as a config option on top of any
  selection rule): `--backsel_weighting soft [--backsel_soft_tau_scale S]` in
  `simulations/src/tfg/backsel.py::soft_tau` + `soft_aggregate` (SD adapter
  `SD_cond_SD_controlnet/src/backsel.py::soft_reweight`, dispatched from `select_backprop_set`;
  plumbed through `src/generation.py::variation_objective`). Each non-differentiated sample hands
  its CLIP-space gradient to the differentiated ones with weights softmax_i(-||e_j-e_i||^2/tau),
  tau = LOCAL scale by default (median squared distance of skipped samples to their nearest representative; the MMD bandwidth is the target spread and ~100x too large, see backsel/REPORT.md sec 7) x S, `--backsel_soft_tau_mode bandwidth` keeps the global option; representative i backprops
  g_i + sum_j a_ji g_j. Default `ht` (unbiased weights) unchanged. Tests:
  `sd/tests/test_sd_flags.py` (identity at k>=N, mass conservation, tau->inf / tau->0 limits).
  Dev-script arms: `backsel_k8_soft`, `backsel_k8_strat_soft`, `trust_backsel_soft`
  (`SOFT_TAU_SCALE` env). Synthetic counterpart + gradient-fidelity comparison: `backsel/` (B-R7).

#### Why greedy k-center failed on SD (working hypothesis, evidence in `sd/BACKSEL_DIAG.md`)
Inference from the per-step records of the two worst seeds, not a direct Jacobian measurement:
1. The losses are single catastrophic steps, not drift: correction norms of 10-15 at steps 40-49
   (control 2.5-3), MMD jumps 0.45 -> 0.58 that never recover because late steps cannot be undone.
   Uniform selection never produces such spikes.
2. Those steps coincide with one representative owning a huge cluster: greedy k-center seeds on
   outliers, leaving one cluster of 11-19 of 32 variations (up to 60%) behind a single sample.
3. It is not the weights: per-variation output-gradient norms are flat (p90/median 1.2). The heavy
   tail must come from the Jacobians -> many near-parallel gradients pushed through ONE sample's
   Jacobian add coherently instead of averaging (k-center correction norms 2x control, max 22 vs 11;
   uniform's N/k=4 weights carry the same mass but produce max 17).
4. The SD zeta is normalised by the LOSS, not the gradient, so gradient inflation goes 1:1 into the
   latent step.
Not yet done: measuring the individual variation Jacobians on SD (costs the backward we are saving).
The pending arms are the test of this hypothesis: soft reweighting (spread the mass), trust region
(cap the step), stratified clusters (bound cluster size). If they remove the spikes and the quality
loss, the story holds; if not, the substitution-bias explanation is wrong.
