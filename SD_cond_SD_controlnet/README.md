# MLGD-F — distribution-guided diffusion with importance-selected backpropagation

**Marginal-distribution-guided Diffusion via Flow** steers a diffusion model (the
*architect*, SDXL-base) so that the distribution of images a fast conditional
generator (the *sprinter*, SDXL-Turbo + ControlNet-Scribble) produces from its
output matches a target distribution, measured by MMD in CLIP space.  The
expensive part is the backward pass through the sprinter for every generated
observation.  This README leads with the mechanism that removes most of that
cost — **back-selection**: forward every observation, backpropagate through a
chosen few — because that is what a collaborator most likely wants to reuse.

## 1. Back-selection: importance sampling of the generated observations

### What it does

At every architect step the guidance loss is `L = MMD(E, S)` between the `N`
sprinter observations `e_i = E_i(x)` (CLIP embeddings of `N` sprinter samples
conditioned on the architect's current clean-image estimate) and the fixed target
set `S`.  The gradient decomposes exactly as

    dL/dx = sum_{i=1}^{N} J_i^T g_i,     J_i = de_i/dx  (sprinter+VAE+CLIP Jacobian),
                                         g_i = dL/de_i  (kernel-only, cheap).

`g_i` for all `N` costs one kernel evaluation; each `J_i^T g_i` costs a full
backward pass through the sprinter.  Back-selection therefore

1. runs all `N` sprinter observations **without graphs** (`torch.no_grad`, one
   plain forward each),
2. computes the **full-batch** MMD value and all `g_i` from those embeddings
   (the loss geometry is never subsampled),
3. **selects `k << N`** observations and per-observation vectors `G_i`,
4. **regenerates exactly those `k`** with autograd graphs from their saved
   per-observation seeds (bit-identical: `regen_max_abs_err = 0.0` on every
   logged step), and
5. backpropagates the surrogate `sum_{i in S} <G_i, e_i>`, whose gradient is
   `sum_{i in S} J_i^T G_i`.

The value reported / used for the adaptive step size is always the full-batch
`L`; only the differentiated set shrinks from `N` to `k`.

### Selection rules (`--backsel_rule`) and weighting (`--backsel_weighting`)

| rule | `G_i` | estimator | status on SD (8 seeds, N=32, k=8, control = full backprop) |
|---|---|---|---|
| `uniform` | `(N/k) g_i`, k of N without replacement | **unbiased** (Horvitz-Thompson) | slight loss: +0.04 MMD [+0.005, +0.09] |
| `strat` | balanced k-center strata (capacity `ceil(N/k)`), one uniform member per stratum, `|C_c| g_r` | **unbiased**, weights `<= ceil(N/k)`, variance `<=` uniform | pending |
| `is` | `c_i/(k p_i) g_i`, `p_i ∝ 0.75‖g_i‖/Σ‖g‖ + 0.25/N` | **unbiased** | not run (‖g_i‖ is flat on SD, so it behaves like uniform) |
| `kcenter` | greedy k-center, `G_{r_c} = Σ_{i∈C_c} g_i` through the center's Jacobian | biased | **FAILED**: +0.10 MMD [+0.04, +0.18], see below |
| any + `--backsel_weighting soft` | `G_i = g_i + Σ_j a_ji g_j`, `a_ji = softmax_i(−‖e_j−e_i‖²/τ)` | biased (mass-conserving proximity reweighting) | pending (`τ` local by default, `--backsel_soft_tau_mode bandwidth` for the global scale) |

**Why greedy k-center failed on SD** (`experiments/model-optimization/README.md`,
"Why greedy k-center failed", evidence in `experiments/model-optimization/sd/BACKSEL_DIAG.md`):
farthest-point centers sit on outliers, so one cluster holds 11–19 of the 32
observations; pushing that cluster's summed `g` through ONE Jacobian adds the
near-parallel contributions coherently instead of letting the Jacobians
average, the step size doubles (max correction 22 vs 11 for full backprop) and,
because the step is normalised by the *loss* not the gradient, the inflation
lands 1:1 in the latent; late-trajectory spikes then cannot be undone.  `strat`
bounds the mass behind any Jacobian by `ceil(N/k)`; the trust region caps the
spikes.

### Copy-paste

```bash
cd <repo>   # run from the repository root
python SD_cond_SD_controlnet/scripts/run_mlgd_f.py \
    --output_dir output/my_run --mode gender \
    --target_prompts "Man:a superrealistic portrait photograph of a man, studio lighting:50" \
                     "Woman:a superrealistic portrait photograph of a woman, studio lighting:50" \
    --n_steps 100 --start_step 50 --num_variations 32 --base_zeta 5.0 \
    --seed 1 --seeded_rng --no_vis --arch_single_batch --profile \
    --backsel 8 --backsel_rule strat            # or: uniform | is | kcenter
    # optional:  --backsel_weighting soft --backsel_soft_tau_scale 1.0 --backsel_soft_tau_mode local
    # optional:  --trust_noise 0.25             # latent trust region (sec 2)
    # eval:      --eval_n 2000 --eval_batch_size 8
```

`--backsel K` forces `--seeded_rng` (per-observation `torch.Generator` seeds are
what make the regeneration exact).  Every flag is opt-in; with none of them the
script is the original pipeline.

### Where the code is

| piece | file : function |
|---|---|
| selection rules, soft weighting (ONE implementation, shared with the synthetic engine) | `simulations/src/tfg/backsel.py` : `select_uniform`, `select_importance`, `select_kcenter`, `select_stratified_balanced`, `soft_tau`, `soft_aggregate` |
| SD adapter (CLI names → tfg rules, generator → tape shim, logging dict) | `SD_cond_SD_controlnet/src/backsel.py` : `select_backprop_set`, `soft_reweight`, `GeneratorTape` |
| no-grad pass, full-batch `g`, seeded regeneration, surrogate | `SD_cond_SD_controlnet/src/generation.py` : `variation_objective` (backsel branch), `run_dps_step_clip` (legacy grad wrapper) |
| per-step records (selected indices, cluster sizes, weights, `g_norms`, `regen_max_abs_err`, forwards / differentiated counts) | `metrics.json["steps"][*]["backsel"]`, `profile.json` |
| theory, unbiasedness proofs, cost accounting | `experiments/model-optimization/backsel/THEORY.md` |
| tests (CPU) | `experiments/model-optimization/sd/tests/test_sd_flags.py` (unbiasedness, k≥N identity, adapter == tfg), `simulations/tests/test_backsel.py` |

### Measured cost (L40S, N=32, 50 guided steps; `experiments/model-optimization/sd/RESULTS.md`)

| | s / step | backward s | peak VRAM | sprinter forwards / differentiated |
|---|---|---|---|---|
| full backprop (`novis`) | 43.1 | 43.6 | 34.2 GB | 32 / 32 |
| back-selection k=8 | **16.9** (2.6×) | 10.2 | **24.9 GB** | 40 / 8 |

One in-graph observation costs ≈1.8 s (forward + checkpoint recompute + VJP), a
no-grad observation ≈0.2 s, so a step costs ≈ `0.2 N + 1.8 k` s: k=16 ≈ 31 s
(1.4×), k=4 ≈ 11 s (3.9×).  The cost result is independent of the rule.

### Current quality status (honest)

Paired-by-seed final MMD vs full backprop, same seeds, 2000-sample fresh eval,
single-run noise floor ≈0.05: `kcenter` k=8 **loses** (+0.10), `uniform` k=8 a
slight loss (+0.04), trust region alone neutral (−0.01, n.s.); `strat`, `soft`,
`trust+uniform`, k=16 arms are pending.  Nothing in the quality column is a
recommendation yet; the cost column is.

## 2. Trust region on the latent step (`--trust_noise TAU`)

The correction `Δ = −ζ_i ∇L` (with `ζ_i = base_zeta / L`) is rescaled so that
`‖Δ‖₂ ≤ TAU · sqrt(1 − ᾱ_{t_prev}) · sqrt(numel)` — its per-element RMS may not
exceed `TAU` noise standard deviations of the state it is added to.  Direction
is preserved; it only ever shrinks a step.  Shared implementation
`simulations/src/tfg/trust.py` (`noise_cap`, `clip_step`), SD adapter
`src/trust.py` (`prev_alpha_bar` reads ᾱ exactly as `DDIMScheduler.step`).
Every run logs `correction_norm_raw` and `trust_cap_tau1` per step, so a
baseline run tells you which `TAU` would bind and how often (on the dev task the
ratio has median 0.03 and a heavy tail up to 1.4; `TAU = 0.25` clips 5–8 % of
steps, the spikes only).

## 3. Profiling (`--profile`) and the other flags

`--profile` writes `<output_dir>/profile.json` (rewritten every step): CUDA-synced
per-step sections `architect`, `sprinter_fwd`/`vae`/`clip` (in-graph forward),
`nograd_*` (back-selection pass 1), `mmd`, `select`, `backward` (VJP incl.
checkpoint recomputes), `vis`, `eval_intermediate`, plus
`max_memory_allocated_mb`; `summary` holds the per-section means.
`--no_vis` skips the per-step figure (exact), `--arch_single_batch` drops the
CFG double batch at `guidance_scale 0` (exact up to fp16 batch round-off),
`--seeded_rng` derives every noise source from `--seed` (init, observations,
selection, eval), `--variation_batch_size B` batches the sprinter,
`--target_cache DIR` reuses generated targets/scribble across runs.
`src/profiling.py::StepProfiler`.

## 4. Evaluation

Final metric: `evaluate_distribution_mmd` decodes the final architect latent to
the scribble, generates `--eval_n` **fresh** sprinter photos (default 10, the
dev protocol uses 2000, batches of `--eval_batch_size`), CLIP-encodes them in
chunks, and reports the unbiased MMD to the targets for both the guided and the
unguided (twin, same init) trajectories.  Eval seeds are `seed·1000003 + 7000000 + j`
— disjoint from every guidance seed and identical across arms with the same
`--seed`.  The final state (`final_latents.pt`, `final_scribble_*.png`,
`target_clip_embeddings.pt`, `metrics_partial.json`) is written **before** the
eval, so `experiments/model-optimization/sd/eval_final.py --run_dir <dir>`
can (re)evaluate any run standalone with sprinter + CLIP only.
Multi-seed tables: `experiments/model-optimization/sd/analyze_sd.py`.

## 5. `--engine tfg` and the shared framework

The synthetic experiments (`simulations/`) run on the generalised TFG engine
`simulations/src/tfg/engine.py::GeneralizedTFG`.  The SD pipeline now shares its
primitives (`tfg.backsel`, `tfg.trust`; `src/_tfg_path.py` makes `tfg`
importable without pip) and can optionally run its whole architect loop through
the engine: `--engine tfg` (`src/tfg_engine_path.py`).  Mapping: `eps_theta` =
architect UNet, `log_f(x0) = −ζ_i · L` with the sprinter+CLIP+MMD objective
(`generation.variation_objective`, incl. back-selection), `SDSchedule` = the
DDIM ᾱ table of the guided steps, `NoiseTape(seed)` keys every noise source,
trust via `TemporalConfig(step_clip="noise_prev_rms")`, `guidance_scaling="raw"`,
`rho = 1`.  What it does **not** reproduce bit-for-bit: fp16 round-off of the
DDIM arithmetic (engine scalars vs diffusers' table), different (equally
reproducible) seed derivation for observations and selection, no per-step
visualisation / intermediate eval; the module docstring lists every difference.
A toy CPU test (`sd/tests/test_engine_path.py`) checks the engine trajectory
against a hand-rolled legacy loop to 1e-4 with and without the trust cap; the
GPU smoke comparison is pending.  The legacy path (`--engine legacy`, default)
is unchanged.

---

## 6. Pipeline overview (paper experiments)

Two diffusion models work together:

- **Architect** (SDXL Base): generates a scribble sketch via an N-step DDIM loop,
  initialised from a HED-scribble latent (SDEdit-style).
- **Sprinter** (SDXL Turbo + ControlNet-Scribble): takes the scribble and produces a
  realistic portrait in 2 steps.

At each architect step: decode `pred_x0` (VAE) → `num_variations` sprinter passes →
CLIP → MMD (or SWD) to the target embeddings → gradient on the architect latent →
correction `−ζ·∇` after the DDIM step.  Step-by-step accounting with the cost
expression: `experiments/model-optimization/sd/PIPELINE.md`.

### Repository structure

```
SD_cond_SD_controlnet/
├── src/
│   ├── models.py            Architect + Sprinter loading, LoRA support
│   ├── generation.py        noise prediction, pred_x0, variation_objective / run_dps_step_clip
│   ├── backsel.py           back-selection adapter -> simulations/src/tfg/backsel.py
│   ├── trust.py             trust-region adapter  -> simulations/src/tfg/trust.py
│   ├── tfg_engine_path.py   --engine tfg (GeneralizedTFG wrapper)
│   ├── profiling.py         --profile
│   ├── metrics.py           MMD, SWD, evaluate_distribution_mmd
│   ├── clip_utils.py        CLIP loading and differentiable encoding
│   ├── image_utils.py       Sobel, VAE decode, base image
│   ├── visualization.py     per-step grids, heatmap
│   └── analysis.py          offline PCA/t-SNE/KDE/boxplot plots
├── scripts/
│   ├── run_mlgd_f.py        main entry point
│   └── eval_baselines.py    baseline scribble generation for comparison
├── notebooks/               evaluation notebooks (see below)
├── experiments/             per-experiment scribbles and baseline outputs
├── age_submit_mlgd_f.sh, gender_submit_mlgd_f.sh   SLURM templates
└── requirements.txt
```

Campaign scripts (dev/smoke/eval SLURM submits, analysis):
`experiments/model-optimization/sd/`.

### Quickstart (paper configuration)

```bash
pip install -r requirements.txt

# Gender mode (man / woman target, or custom multi-group)
python scripts/run_mlgd_f.py --output_dir output/run_001 --mode gender \
    --n_steps 30 --start_step 15 --num_variations 6 --base_zeta 5.0 --seed 1

# Age mode (continuous age sweep)
python scripts/run_mlgd_f.py --output_dir output/run_001_age --mode age \
    --age_min 40 --age_max 80 --age_step 1 --seed 1
```

On a SLURM cluster: `export ENV_PATH=/path/to/env; sbatch gender_submit_mlgd_f.sh`
(or `age_submit_mlgd_f.sh`).

### Custom target groups

`--target_prompts "NAME:PROMPT:N" ...` defines any number of groups, e.g. the
4-class gender interpolation:

```bash
python scripts/run_mlgd_f.py --mode gender --target_prompts \
    "Woman:a superrealistic portrait photograph of a woman, extremely feminine features, studio lighting:25" \
    "Woman with masculine features:a superrealistic portrait photograph of a woman with masculine features, heavy brow ridge, studio lighting:25" \
    "Man with feminine features:a superrealistic portrait photograph of a man with extremely feminine features, soft delicate face, high cheekbones, studio lighting:25" \
    "Man:a superrealistic portrait photograph of a man, extremely masculine features, studio lighting:25"
```

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--mode` | `gender` | `gender` (prompt groups) or `age` (continuous sweep) |
| `--n_steps` / `--start_step` | 30 / 15 | architect DDIM steps / SDEdit start |
| `--num_variations` | 6 | sprinter observations per step (N) |
| `--base_zeta` | 5.0 | adaptive guidance strength (`ζ_i = base_zeta / L`) |
| `--loss_fn` | `mmd` | `mmd` or `swd` |
| `--bandwidth_scale`, `--kernel_alpha`, `--loss_scale` | 1.0 | MMD kernel / loss scaling |
| `--controlnet_scale` | 0.5 | ControlNet conditioning strength (target generation) |
| `--age_min/max/step` | 10/80/1 | age range for age mode |
| `--backsel`, `--backsel_rule`, `--backsel_weighting`, `--trust_noise`, `--profile`, `--engine`, `--eval_n` | off | section 1–5 |

### Experiments directory

Each experiment folder (`experiments/BalancedTarget/`, `SkewedTarget/`,
`GenderInterpolation/`, `AgeInterpolation/`) contains the scribbles of all
compared methods: `scribble_source.png` (SDEdit init), `scribble_mlgdd.png`
(MLGD-F), `scribble_avg.png` (weighted average of per-group HED scribbles),
`scribble_sdedit.png` (guided SDEdit with a descriptive prompt),
`scribble_sdedit_best.png` (best unguided SDEdit candidate, time-matched search),
`scribbles_all.png`, `source_portrait.png`, `sdedit_search.png`,
`baselines_meta.json` (timing, MMD, seeds).

### Baseline evaluation

```bash
sbatch slurm/run_eval_baselines.sh BalancedTarget 241   # 241 = MLGD-F runtime in minutes
python scripts/eval_baselines.py --experiment BalancedTarget --lgd_cm_minutes 241
```

generates `avg`, `sdedit`, `sdedit_best` under `experiments/<Name>/baselines/`
with the same wall-clock budget as the MLGD-F run.

### Offline analysis and notebooks

`python src/analysis.py --run_dir output/run_001` regenerates PCA, t-SNE, KDE,
boxplots, portrait grids and the scribble heatmap under `output/run_001/plots/`.
Notebooks (`notebooks/`, relative paths): `eval_all_experiments.ipynb` (N=2000
MMD + gender classification, all experiments), `eval_scribbl_interpolation.ipynb`
/ `_age.ipynb` (interpolation evaluations), `gender_saliency_eval.ipynb`,
`eps_g_experiment.ipynb`, `measure_dps_step_memory.ipynb`.  Cached results:
`notebooks/results/eval_all_results.json`.

### wandb

Entity = whoever is logged in via `wandb login`; override with `--wandb_project`
/ `--wandb_entity`.  The campaign scripts default to `WANDB_MODE=offline`.
