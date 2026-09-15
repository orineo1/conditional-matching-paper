# Point vs distributional targets on the SD pipeline (paper experiment)

**Question.** MLGD-F differs from adversarial input optimization in two confounded
ways: the **diffusion prior** on the scribble x, and the **distributional target**
(a set matched by MMD instead of one CLIP vector matched by squared distance).
Three arms unconfound them — arm 1→2 isolates the prior, arm 2→3 the
distributional term. Pre-registered predictions: **[HYPOTHESIS.md](HYPOTHESIS.md)**
(read it first; the headline is P-B1: point-target arms collapse to one
androgynous mode — p(male)≈50% but PC1 variance far below the target's —
because `E‖CLIP(y)−a‖² = Var + ‖mean−a‖²` is variance-seeking).

| arm | prior | target | code |
|---|---|---|---|
| 1 `pgd` | none — Adam in an L∞ eps-ball around x0 | point y* | [`run_pgd_arm.py`](run_pgd_arm.py) |
| 2 `point` | SDXL-base DDIM (SDEdit) | point y*, `--loss_fn point`, same backbone + witness k=50 (witness scored against y*) | `run_mlgd_f.py` |
| 3 `mmd` | SDXL-base DDIM (SDEdit) | S_G, MMD, `--backsel_rule witness --backsel_k 50` = Ori's config | `run_mlgd_f.py` |

**Scenarios — A FIRST.** A (oracle recovery, reachable by construction): y* = CLIP(y_f) of
one female portrait, x_f = HED(y_f) is a zero-loss solution, S_G = 100 samples of
f_phi(x_f,·); the oracle floor L(x_f) and ‖x*−x_f‖ (pixel + VAE-latent) are reported.
B (the gender case, after A): G = G_bal (10 Man + 10 Woman on the shared scribble),
y* = CLIP centroid of G_bal.

## How to run (in order) — SCENARIO A FIRST

**Backbone for every arm (Ori's working configuration, re-spec of 2026-09-13):**
`n_steps 250, start_step 125, num_variations 100, base_zeta 5.0, guidance_scale 0,
controlnet_scale 0.5, seed 1, 10 targets per group, witness back-selection k=50
(floor 0.0, temperature 0.3), trust region OFF`. See HYPOTHESIS.md "AMENDMENT 4" for why
the earlier Scenario-B results do not test this method.

```bash
cd /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd

# 0. smoke (proves the witness path runs end to end; ~30 min)
SMOKE=1 sbatch submit_setup.sh A
SMOKE=1 sbatch submit_arm.sh mmd A        # after setup finishes
SMOKE=1 sbatch submit_arm.sh point A
SMOKE=1 sbatch submit_arm.sh pgd A

# 1. SCENARIO A (first): oracle recovery, target reachable by construction
sbatch submit_setup.sh A                  # y_f, x_f = HED(y_f), S_G = 100 x f_phi(x_f), y* = CLIP(y_f)
sbatch submit_arm.sh mmd A                # arm 3: MMD + witness k=50   -> runs/A/mmd_witness50
sbatch submit_arm.sh point A              # arm 2: point loss + witness -> runs/A/point_witness50
sbatch submit_arm.sh pgd A                # arm 1: AdvI2I-style eps sweep
N_COND=1 sbatch submit_arm.sh point A     # arm 2 variant: original delta_{y*}, backsel is a no-op

# 2. SCENARIO B (after A)
sbatch submit_setup.sh B                  # G_bal 10+10, y* = centroid
sbatch submit_arm.sh mmd B
sbatch submit_arm.sh point B
sbatch submit_arm.sh pgd B
N_COND=1 sbatch submit_arm.sh point B

# 3. deliverable (CPU job — the login node OOMs on the 2000x2000 kernels)
sbatch submit_report.sh                   # -> RESULTS.md + figures/pc1_<scenario>.png
```

Optional labelled arms: `BACKSEL_RULE=uniform` (main's uniform rule), `BACKSEL_K=0`
(no back-selection at all), `ZETA=12 TRUST_TAU=0.2 BACKSEL_K=0` (reproduces the superseded
configuration). Output directories always carry their back-selection descriptor
(`_witness50`, `_uniform50`, `_nobacksel`, ...), so no run can overwrite another.

## Layout

```
HYPOTHESIS.md      pre-registered predictions + spec-vs-pipeline audit (n_MC absent, cn 0.5 flag)
common.py          seeds / projection / CIs / PC1 stats / cache IO (CPU-tested)
setup_targets.py   builds cache/<scenario>/ (GPU, once per scenario)
run_pgd_arm.py     arm 1
submit_setup.sh    sbatch wrapper for setup_targets.py
submit_arm.sh      sbatch wrapper, one arm per job; per-arm config block at the top
build_report.py    THE deliverable: arms x scenarios table + PC1 histograms -> RESULTS.md
tests/             CPU tests (pytest; 10)
cache/ runs/ figures/ RESULTS.md    (generated)
```

## Arm-by-arm configuration (single source of truth = `submit_arm.sh` top block)

Shared backbone (Ori's working configuration): **seed 1**, `--seeded_rng`, f_phi =
SDXL-Turbo + ControlNet-Scribble, 2 steps, CFG 0, **ControlNet scale 0.5** (via
`--variation_cn_scale`; the pipeline historically hard-coded 0.8), neutral sprinter prompt,
`n_steps 250 / start_step 125`, `num_variations 100`, `base_zeta 5.0`, **trust region OFF**,
eval N=2000 batch 8, 10 targets per group.

- **mmd (arm 3)**: `--loss_fn mmd --num_variations 100 --backsel_k 50 --backsel_rule witness
  --witness_floor 0.0 --witness_temperature 0.3` — Ori's configuration verbatim (the witness
  path is ported from `upstream/main` and asserted bitwise-equivalent in
  `tests/test_witness_port.py`).
- **point (arm 2)**: identical, with `--loss_fn point --point_target_pt <cache>/point_target.pt`.
  Back-selection applies normally at 100 samples; the witness is scored against **y\***
  (1 row) rather than a target set the point loss never sees — `witness_target` in
  `variation_objective`, set automatically for `--loss_fn point`. The original
  `G = delta_{y*}` variant (`N_COND=1`) is kept as a secondary arm, where back-selection is
  a no-op by definition.
- **pgd (arm 1)**: Adam on the pixel scribble, L∞ eps ∈ {32,64,128}/255 (AdvI2I's grid) with
  per-step clipping to the ball ∩ [0,1], lr = eps/10, 200 steps, ONE seeded inner sample per
  step. No conditional batch, so back-selection does not apply.
- **secondary (labelled)**: `BACKSEL_RULE=uniform`, `BACKSEL_K=0`, or
  `ZETA=12 TRUST_TAU=0.2 BACKSEL_K=0` (the superseded configuration, for continuity).

## AdvI2I mapping (arm 1 is an *adaptation*, cite as "AdvI2I-style")

AdvI2I: Zeng, Cao, Cao, Chang, Chen, Lin — arXiv 2410.21471.

| | AdvI2I (paper) | our arm 1 |
|---|---|---|
| optimization variable | pre-trained-VAE generator g_ψ(x) | the scribble pixels x directly (no generator) |
| constraint | ‖g_ψ(x)−x‖_∞ ≤ eps, clipping each update, eps ∈ {32,64,128}/255 | same ball / clipping / eps grid on x−x0 |
| loss | ‖f_θ^t(g_ψ(x), τ(p)) − f_θ^t(x, τ_shifted)‖² (UNet latent features toward a concept-shifted prompt embedding, t=1) | ‖CLIP(f_φ(x)) − y*‖² (image-embedding point target through OUR conditional generator) |
| victim model | InstructPix2Pix / SD-inpainting | SDXL-Turbo + ControlNet-Scribble (the pipeline's f_φ) |
| optimizer / steps | unspecified in the paper | Adam, lr = eps/10, 200 steps, one stochastic sample per step |

Kept: input-space adversarial optimization, eps-ball + clipping, embedding-space
point target. Changed: no generator network, CLIP embeddings instead of UNet
features, our f_φ. The comparison the experiment needs is "point target without a
diffusion prior", which the adaptation preserves.

## Status (2026-09-13): Scenario B superseded, witness path ported

The earlier Scenario-B results were obtained **without** the MMD-witness back-selection that
Ori's working configuration uses (it lives on `upstream/main`; this branch had a different
back-selection). Those results are re-labelled *"our-configuration (no witness)"* and are
superseded pending the re-run; the step-survival, drift-control, reachability and
P-C1-power findings survive as characterisations of the runs that were made. The witness
path is now ported here and asserted equivalent to main's, bitwise, in
`tests/test_witness_port.py`. Details: HYPOTHESIS.md "AMENDMENT 4".

## Two references (important when reading any delta)

Arms do not share an "unguided" path: the PGD arm's reference is **f_phi(x0)** (the source
scribble, no diffusion; MMD -> G = 0.3776) while the diffusion arms' reference is
**f_phi(unguided DDIM scribble)** (MMD -> G = 0.6000). They are different sample sets
(MMD between them 0.4772). `build_report.py` therefore scores every arm against ONE
common reference, f_phi(x0) — the do-nothing baseline all arms start from — and keeps
"delta vs own ref" only for completeness. The gap between the two references is itself a
result: the unguided SDEdit round trip drops the CLIP-PC1 variance ratio from 0.300 to
0.046.

## Superseded results (kept for the record)

Everything below the current Status was produced **without** witness back-selection and is
labelled accordingly; see HYPOTHESIS.md "AMENDMENT 4".

- **Scenario B (no-witness configuration): COMPLETE** — 11 runs, `RESULTS.md`, scored in
  HYPOTHESIS.md "CLOSE-OUT 3". No arm beat the do-nothing reference on the full schedule;
  the 20-step gate did (+0.103). PC1 variance ratios 0.005-0.046 (P-B1 partly confirmed,
  P-B2 falsified, P-B3 ordering only, **P-B4 confirmed and configuration-independent**).
- **Diagnosis (`DEBUG_B.md`)** — step-size probe (job 46134134), the zeta 12 + trust 0.2
  correction, the round-2 finding that only 15 % of applied guidance survives the 125-step
  schedule vs 49 % on 20 steps, and the zeta = 0 drift control. These characterise the runs
  that were made and are unaffected by the witness question.
- **Scenario A: not started** — now the FIRST scenario to run, per the 2026-09-13 re-spec.

## Deviations log

- 2026-09-06, pre-run: arm-1 eps grid aligned to AdvI2I's {32,64,128}/255 (see HYPOTHESIS.md).
- 2026-09-06: the first full `mmd` job (46103120) died silently at step 98/125 (SLURM
  reported COMPLETED 0:0; the log stopped mid-step and `profile.json` was left 0 bytes,
  with a "no space left on device" error from wandb on the 96 %-full `/sci/home`). Its
  directory is archived as `runs/B/mmd_incomplete_46103120` and is NOT in any table. The
  arm was rerun to completion as job 46124426 (125/125 steps, 4 h 35). `build_report.py`
  now refuses any run whose target set does not hash-match the scenario cache or whose
  eval count differs from the modal one, so a smoke or partial run can never enter a table
  silently.
- 2026-09-10: no parameters tuned after seeing Scenario B results (honesty clause). The
  step-size correction registered the same day is licensed by a MEASURED descent band from
  the probe, not by a configuration search, and is scored against the pre-committed P-C1
  criterion; original rows are never overwritten (new runs carry a `_zeta.._tau..` suffix).
- 2026-09-10: `submit_arm.sh` gained `ZETA` / `TRUST_TAU` / `GATE` / `TAG` env knobs
  (defaults unchanged = Appendix E) and calls `gate_check.py` after every diffusion arm.
