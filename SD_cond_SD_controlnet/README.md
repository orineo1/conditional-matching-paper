`# MLGD-F

**Marginal-distribution-guided Diffusion via Flow** — steers a diffusion model toward
a target distribution using MMD loss in CLIP embedding space.

---

## Repository structure

```
SD_cond_SD_controlnet/
│
├── src/                                      # core library
│   ├── models.py                             # Architect + Sprinter loading, LoRA support
│   ├── generation.py                         # noise prediction, pred_x0, DPS gradient steps
│   ├── metrics.py                            # MMD, SWD, evaluate_distribution_mmd
│   ├── clip_utils.py                         # CLIP loading and differentiable encoding
│   ├── image_utils.py                        # Sobel, VAE decode, base image
│   ├── visualization.py                      # per-step grids, heatmap
│   └── analysis.py                           # offline PCA/t-SNE/KDE/boxplot plots
│
├── scripts/
│   ├── run_mlgd_f.py                         # main MLGD-F entry point
│   ├── run_pgd.py                            # PGD competitor entry point
│   └── eval_baselines.py                     # baseline scribble generation for comparison
│
├── notebooks/
│   ├── results/
│   │   ├── eval_all_results.json             # cached N=2000 MMD results across all experiments
│   │   └── vjp_results_lightning.csv         # VJP memory/speed benchmark results
│   ├── eval_all_experiments.ipynb            # N=2000 MMD + gender evaluation across all experiments
│   ├── eval_pgd_experiments.ipynb            # short N=2000 eval, source/lgd_cm/pgd only
│   ├── eval_scribbl_interpolation.ipynb      # gender interpolation scribble evaluation
│   ├── eval_scribbl_interpolation_age.ipynb  # age interpolation scribble evaluation
│   ├── gender_saliency_eval.ipynb            # CLIP gender saliency + scribble diff analysis
│   ├── eps_g_experiment.ipynb                # ε_g ablation experiment
│   └── measure_dps_step_memory.ipynb         # GPU memory profiling per DPS step
│
├── experiments/                              # per-experiment scribbles and baseline outputs
│   ├── SkewedTarget/                         # 25% man / 75% woman target
│   ├── BalancedTarget/                       # 50% man / 50% woman target
│   ├── GenderInterpolation/                  # 4-class gender interpolation target
│   ├── AgeInterpolation/                     # age sweep target (40–79 yo)
│   └── eval_all_results.json                 # MMD evaluation results across all experiments
│
├── age_submit_mlgd_f.sh                      # SLURM submit script — age mode
├── gender_submit_mlgd_f.sh                   # SLURM submit script — gender mode
├── pgd_submit.sh                             # SLURM submit script — PGD, takes experiment/minutes/seed/loss_fn/opt_lr
├── pgd_opt_lr_sweep.sh                       # bash driver — submits the opt_lr sweep via pgd_submit.sh
├── requirements.txt
└── README.md
```

### Experiments directory

Each experiment folder (e.g. `experiments/BalancedTarget/`) contains the scribbles from all compared methods:

| File | Description |
|------|-------------|
| `scribble_source.png` | Source oval scribble used as SDEdit initialisation |
| `scribble_mlgdd.png` | **MLGD-F (distilled)** output |
| `scribble_avg.png` | Weighted average of per-group HED scribbles (baseline) |
| `scribble_sdedit.png` | Guided SDEdit scribble with a descriptive prompt (baseline) |
| `scribble_sdedit_best.png` | Best unguided SDEdit candidate over a time-matched search budget (baseline) |
| `scribbles_all.png` | Side-by-side grid of all scribbles |
| `source_portrait.png` | Source portrait used for HED scribble extraction |
| `sdedit_search.png` | MMD curve over the SDEdit candidate search |
| `baselines_meta.json` | Timing, MMD scores, and seed info for all baseline candidates |

---

## How it works

Two diffusion models work together:

- **Architect** (SDXL Base): generates a scribble sketch via a N-step denoising loop, initialised from a HED-scribble latent (SDEdit-style).
- **Sprinter** (SDXL Turbo + ControlNet-Scribble): takes the scribble and produces a realistic portrait in 2 steps.

At each Architect denoising step, MLGD-F:
1. Decodes the predicted clean image (`pred_x0`) to pixels via VAE.
2. Runs `num_variations` Sprinter passes to sample from the conditional distribution.
3. Encodes each Sprinter output through CLIP → 768-dim embeddings.
4. Computes MMD (or SWD) between the generated embeddings and the target distribution.
5. Backpropagates through the entire chain to get a gradient on the Architect latent.
6. Applies a correction `-ζ · ∇` before the scheduler step.

---

## Quickstart

```bash
pip install -r requirements.txt

# Gender mode (man / woman target, or custom multi-group)
python scripts/run_mlgd_f.py \
    --output_dir output/run_001 \
    --mode gender \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --base_zeta 5.0 \
    --seed 1

# Age mode (continuous age sweep)
python scripts/run_mlgd_f.py \
    --output_dir output/run_001_age \
    --mode age \
    --age_min 40 \
    --age_max 80 \
    --age_step 1 \
    --seed 1
```

On a SLURM cluster:

```bash
export ENV_PATH=/path/to/your/env
sbatch gender_submit_mlgd_f.sh   # gender mode
sbatch age_submit_mlgd_f.sh      # age mode
```

### Custom target groups

Use `--target_prompts` to define any number of groups with format `"NAME:PROMPT:N"`:

```bash
python scripts/run_mlgd_f.py \
    --mode gender \
    --target_prompts \
        "Woman:a superrealistic portrait photograph of a woman, extremely feminine features, studio lighting:25" \
        "Woman with masculine features:a superrealistic portrait photograph of a woman with masculine features, heavy brow ridge, studio lighting:25" \
        "Man with feminine features:a superrealistic portrait photograph of a man with extremely feminine features, soft delicate face, high cheekbones, studio lighting:25" \
        "Man:a superrealistic portrait photograph of a man, extremely masculine features, studio lighting:25"
```

---

## PGD (projected gradient descent) — competitor

`scripts/run_pgd.py` is an alternative to MLGD-F's DPS-style guidance, following
Shah & Hegde-style PGD for generative priors: alternate an unconstrained gradient
step in pixel space with a projection back onto a generator's range (found by
Adam-searching its input space, matching the paper's own
`P_G(w) = G(argmin_z ‖w - G(z)‖)`). Sprinter+CLIP play the same role as in
MLGD-F — the measurement operator the loss is computed on.

**This branch (`claude/pgd-generative-projection`) makes `G` a real generative
model**, unlike `claude/pgd-competitor`'s VAE-only `G`. The PGD paper assumes
sampling `z` from a bounded prior and decoding it yields a realistic image — a
raw VAE decoder doesn't have that property (arbitrary/optimized latents can
decode to unrealistic images; the actual generative prior in this stack lives
in the Architect's diffusion UNet, not its VAE). So here, `G(z)` = a short
*unconditional* denoising rollout of the Architect's own UNet+scheduler,
starting from the current latent partially noised with `z` — literally
MLGD-F's own "regular" (unguided) path, run standalone, with `z` as the noise
input being searched over. This is a genuinely generative sampling process
(unlike a bare VAE decode), at the cost of each Adam iteration now backpropping
through a multi-step UNet rollout instead of one VAE decode — `--proj_adam_steps`
defaults much lower here (20 vs. 100) to stay tractable, and `--proj_n_steps`/
`--proj_start_step` control how many actual denoising steps that rollout runs
(default: 3, out of a 30-step schedule — lower `--proj_start_step` for a more
faithful generative `G` at higher cost). Everything else — the optimization
step, target-building, experiment presets, budget matching, CLI surface — is
unchanged from `claude/pgd-competitor`.

When `--experiment` names
one with an existing `experiments/<Experiment>/` folder (`SkewedTarget`,
`BalancedTarget`, `GenderInterpolation`, `AgeInterpolation`), PGD reuses that
run's exact `scribble_source.png` as its own starting scribble instead of
generating a fresh one — so it starts from the same point as the MLGD-F run
it's being compared to, and skips the extra generation cost. `GenderTarget100`/
`GenderTarget1` have no matching MLGD-F run, so they still generate fresh.

On a SLURM cluster, `pgd_submit.sh` takes the experiment, the time budget, and
optionally a seed override and loss function as positional arguments, mirroring
`run_eval_baselines.sh`'s `<experiment> <minutes>` pattern:

```bash
export ENV_PATH=/path/to/your/env
sbatch pgd_submit.sh <EXPERIMENT> <TARGET_MINUTES> [SEED] [LOSS_FN] [NUM_VARIATIONS] [OPT_LR]

sbatch pgd_submit.sh GenderTarget100 241            # man scribble -> 100-sample woman target
sbatch pgd_submit.sh GenderTarget1   241 1    l2    # single-sample target, l2 loss, seed override
sbatch pgd_submit.sh SkewedTarget    241            # 25% male / 75% female (vs. experiments/SkewedTarget)
sbatch pgd_submit.sh BalancedTarget  241            # 50% male / 50% female (vs. experiments/BalancedTarget)
sbatch pgd_submit.sh GenderInterpolation 241        # 4-class (vs. experiments/GenderInterpolation)
sbatch pgd_submit.sh AgeInterpolation    177        # age sweep (vs. experiments/AgeInterpolation)
sbatch pgd_submit.sh GenderInterpolation 168 1 mmd 100 0.5   # opt_lr override
```

`pgd_opt_lr_sweep.sh` submits GenderInterpolation/SkewedTarget/BalancedTarget/
AgeInterpolation at three `--opt_lr` values each (`0.05`/`0.5`/`2.0` by
default — edit the `OPT_LRS` array to change) via repeated `sbatch
pgd_submit.sh` calls. It's a plain bash driver, run directly (not itself
submitted via `sbatch`):

```bash
export ENV_PATH=/path/to/your/env
bash pgd_opt_lr_sweep.sh
```

Every hyperparameter, `opt_lr` included, is logged to each run's wandb
config (`vars(args)`), so runs stay distinguishable regardless of how many
you launch.

`EXPERIMENT` selects a preset from `EXPERIMENT_PRESETS` in `scripts/run_pgd.py`
(target prompts/mode/controlnet_scale/age-range, matching the corresponding
MLGD-F/`eval_baselines.py` experiment); its default seed comes from the same
table `eval_baselines.py` uses, overridable via the `SEED` argument.

`TARGET_MINUTES` sets the wall-clock budget: pass the matching MLGD-F run's
measured runtime so both methods get the same time budget (round 1 is timed,
then the number of rounds is computed to hit that budget — the same pattern
`eval_baselines.py` uses for its SDEdit search). `--opt_steps`/`--proj_adam_steps`
(set inside `pgd_submit.sh`) control the optimization-vs-projection split
within that fixed budget.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--experiment` | *(none)* | Preset name from `EXPERIMENT_PRESETS` — fills in target/mode/seed/controlnet_scale |
| `--seed` | *(preset default)* | Overrides the preset's seed |
| `--loss_fn` | `mmd` | `mmd` (full distribution) or `l2` (mean-CLIP-embedding matching only) |
| `--opt_steps` | 3 | Ambient/pixel-space gradient steps per round |
| `--opt_lr` | 0.05 | Step size for the ambient gradient step |
| `--proj_adam_steps` | 20 | Adam iterations for the projection search (lower than the VAE-only variant's 100 -- each iteration now backprops through a UNet rollout) |
| `--proj_lr` | 0.1 | Adam learning rate for the projection search (paper's CelebA setting) |
| `--proj_n_steps` | 30 | Total denoising schedule length G's rollout is a tail slice of (matches MLGD-F's `--n_steps`) |
| `--proj_start_step` | 27 | G runs `timesteps[proj_start_step:]` -- i.e. `proj_n_steps - proj_start_step` actual UNet steps per Adam iteration (default: 3) |
| `--guidance_scale` | 0.0 | CFG scale for G's internal UNet calls (unconditional, matches MLGD-F's default) |
| `--prompt`/`--negative_prompt` | `""` | Prompt for G's internal UNet calls (matches MLGD-F's own unguided path) |
| `--target_minutes` | *(required)* | Wall-clock budget, matched to the compared MLGD-F run |
| `--n_eval` | 10 | Sprinter samples for the quick init/per-round MMD check |
| `--n_eval_final` | 250 | Sprinter samples for the final, higher-fidelity MMD (`final_pgd_mmd_250`) |
| `--n_photos_per_round` | 5 | Conditioned Sprinter photos logged to wandb per logged round |
| `--log_image_every` | 1 | Log the scribble + conditioned photos every N rounds |
| `--source_scribble_path` | *(preset default)* | Reuse an existing `experiments/<Experiment>/scribble_source.png` instead of generating a fresh one |

The PGD paper reports two different projection setups: MNIST (VAE, k=20) used 200 Adam steps @ lr=0.03; CelebA (DCGAN, k=100) — the closer analogue to our face-portrait task — used 100 steps @ lr=0.1. We default to the CelebA setting.

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `gender` | `gender` (prompt-based groups) or `age` (continuous sweep) |
| `--n_steps` | 30 | Architect denoising steps |
| `--start_step` | 15 | SDEdit start — MLGD-F guidance runs from here |
| `--num_variations` | 6 | Sprinter variations per step |
| `--base_zeta` | 5.0 | Adaptive guidance strength |
| `--loss_fn` | `mmd` | `mmd` or `swd` |
| `--bandwidth_scale` | 1.0 | MMD kernel bandwidth scale |
| `--kernel_alpha` | 1.0 | MMD RBF exponent (>1 = sharper falloff) |
| `--loss_scale` | 1.0 | Loss multiplier before grad |
| `--controlnet_scale` | 0.5 | ControlNet conditioning strength |
| `--age_min/max/step` | 10/80/1 | Age range for age mode |

---

## Baseline evaluation

After running MLGD-F and placing results in `experiments/<ExperimentName>/`, generate
competing baseline scribbles with a fair time-matched budget:

```bash
export ENV_PATH=/path/to/your/env
sbatch slurm/run_eval_baselines.sh SkewedTarget 241      # 241 = MLGD-F runtime in minutes
sbatch slurm/run_eval_baselines.sh BalancedTarget 241
sbatch slurm/run_eval_baselines.sh GenderInterpolation 241
sbatch slurm/run_eval_baselines.sh AgeInterpolation 177
```

Or run directly:

```bash
python scripts/eval_baselines.py \
    --experiment BalancedTarget \
    --lgd_cm_minutes 241
```

This generates `avg`, `sdedit`, and `sdedit_best` scribbles and saves them under
`experiments/<ExperimentName>/baselines/`. The SDEdit search is given the same
wall-clock budget as the MLGD-F run.

---

## Offline analysis

After a run completes, regenerate all plots without GPU:

```bash
python src/analysis.py --run_dir output/run_001
```

Produces PCA, t-SNE, KDE, boxplots, portrait grids, and scribble heatmap
under `output/run_001/plots/`.

---

## Notebooks

All notebooks live in `notebooks/` and run on the cluster via Jupyter.
They use relative paths — no path configuration needed.

| Notebook | Purpose |
|----------|---------|
| `eval_all_experiments.ipynb` | Full N=2000 MMD + gender classification across all 4 experiments and all methods |
| `eval_pgd_experiments.ipynb` | Short version of the above, scoped to `source`/`lgd_cm`/`pgd` only — for evaluating a new PGD result placed at `experiments/<Experiment>/scribble_pgd.png` |
| `eval_scribbl_interpolation.ipynb` | Evaluate MLGD-F scribble on the gender interpolation experiment; 5-class cosine softmax classification and PCA |
| `eval_scribbl_interpolation_age.ipynb` | Evaluate MLGD-F scribble on the age interpolation experiment; fit age axis via PCA on age-40 vs age-79 embeddings, pick photos evenly along the axis |
| `gender_saliency_eval.ipynb` | CLIP gender saliency heatmaps, scribble pixel diff visualisation, confidence boxplots |
| `eps_g_experiment.ipynb` | ε_g ablation |
| `measure_dps_step_memory.ipynb` | GPU memory and runtime profiling per DPS step |

Cached evaluation results are in `notebooks/results/eval_all_results.json`.

---

## wandb

 The entity is taken from whoever is
logged in via `wandb login`. Override with `--wandb_project` and `--wandb_entity`
if needed.
