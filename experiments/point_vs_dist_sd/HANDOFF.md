# Handoff — point-vs-distribution SD ablation (for Ori)

**Status 2026-09-15:** Scenario A complete at your gender-run backbone; Scenario B running.

## What this compares

Three ways of doing inverse design on the same variable (the scribble `x`), all sharing
your hyperparameters:

| arm | prior on x | loss | selection |
|---|---|---|---|
| 1. AdvI2I-style PGD (arXiv 2410.21471) | none (eps-ball on x) | `‖CLIP(y) − y*‖²` | n/a |
| 2. MLGD-F point | diffusion (Alg. 1) | `‖CLIP(y) − y*‖²` | witness k=50 |
| 3. MLGD-F distributional (yours) | diffusion (Alg. 1) | `MMD(S_cond, S_G)` | witness k=50 |

1→2 isolates the diffusion prior; 2→3 isolates the distributional target.

## Backbone (your gender config, used by every arm)

```
--n_steps 250 --start_step 125 --num_variations 100 \
--backsel_k 50 --backsel_rule witness --witness_floor 0.0 --witness_temperature 0.3 \
--base_zeta 5.0 --guidance_scale 0.0 --controlnet_scale 0.5 --loss_fn mmd --seed 1
```

The witness code is a **verbatim port of `upstream/main`** (`_witness_select_indices`,
`_log_witness_selection`, `compute_witness_scores`). `tests/test_witness_port.py` asserts
bitwise-identical selections and probabilities against main's functions across
k ∈ {1,5,20,40} × floor ∈ {0,0.3,1} × T ∈ {0.3,1,2} × replacement, incl. your exact
k=50/floor 0.0/T 0.3 case (88 tests, also re-run on the cluster's torch 2.10).

## Where the code is

Branch `tfg-generalization-v2`, directory `experiments/point_vs_dist_sd/`:

| file | role |
|---|---|
| `README.md` | entry point |
| `HYPOTHESIS.md` | pre-registration + dated amendments (incl. what was superseded and why) |
| `setup_targets.py` | builds the shared, byte-identical targets/`y*`/scribble caches |
| `run_pgd_arm.py` | arm 1 |
| `submit_setup.sh`, `submit_arm.sh` | the two commands below |
| `build_report.py` | arms × metrics table + PC1 histograms |
| `RESULTS.md`, `DEBUG_B.md` | results and the debugging record |

SD flags live in `SD_cond_SD_controlnet/scripts/run_mlgd_f.py` and `src/{generation,metrics,backsel,trust}.py`.

## How to run (cluster)

```bash
cd /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd
sbatch submit_setup.sh A            # or B
sbatch submit_arm.sh mmd   A        # arm 3 (yours)
sbatch submit_arm.sh point A        # arm 2
sbatch submit_arm.sh pgd   A        # arm 1, eps sweep
sbatch submit_report.sh             # builds RESULTS.md (must run on a compute node)
```
Env knobs (defaults = the backbone above): `ZETA`, `TRUST_TAU`, `N_COND`, `BACKSEL`, `SMOKE=1`, `GATE=1`, `SEED`.

## Where the results are (cluster)

```
/sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd/runs/{A,B}/<arm>_witness50/
    metrics.json          # per-step trace + final 2000-sample eval
    profile.json          # timing / VRAM
    npy/                  # CLIP embeddings used by the eval
    final_scribble_*.png  # guided vs unguided
/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_arm_<jobid>.{log,err}
```

## Scenario A results (2000 fresh samples, common unguided reference 0.2059)

| arm | job | final MMD | vs unguided |
|---|---|---|---|
| **mmd + witness** | 46160179 | **0.1557** | **−0.050 (better)** |
| point + witness | 46160180 | 0.2408 | +0.035 (worse) |
| point, n_cond=1 | 46160182 | 0.5994 | much worse |
| PGD eps 32/64/128 | 46160181 | own-loss 0.507 / 0.814 / 0.806 | never beats its start (0.42) |

Scenario B (G_bal 50/50, `y*` = its CLIP centroid): setup 46169136, arms 46169137 (mmd),
46169138 (point), 46169139 (pgd).

## Two caveats we are not hiding

1. The mmd arm's **final** MMD improves by 0.050 while its **guidance-loss trace** is flat
   (0.2378 → 0.2369). Under investigation — the trace and the eval are different estimands
   (100 in-graph variations at the current noisy latent vs 2000 fresh samples from the final
   scribble), but we have not yet demonstrated which explanation holds.
2. Our pre-registered trend criterion ("P-C1") fails the arm that wins and passes the arm
   that loses, so it is being recorded as a criterion defect; the primary measure for the
   paper is the final fresh-sample MMD against a common unguided reference.

An earlier Scenario B was run **without** witness back-selection and is superseded; its
numbers are kept and labelled in `HYPOTHESIS.md`/`RESULTS.md`, along with findings that do
not depend on the selection rule (step-survival 0.147 vs 0.495 over 125 vs 20 steps, a
zeta=0 drift control, a target-reachability defect, and the PGD failure).
