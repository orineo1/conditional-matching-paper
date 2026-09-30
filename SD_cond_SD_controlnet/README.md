`# MLGD-F

**Matching-Loss Guided Diffusion with a Fast inner sampler** — steers a diffusion model toward
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
│   └── eval_baselines.py                     # baseline scribble generation for comparison
│
├── notebooks/
│   ├── results/
│   │   ├── eval_all_results.json             # cached N=2000 MMD results across all experiments
│   │   └── vjp_results_lightning.csv         # VJP memory/speed benchmark results
│   ├── eval_all_experiments.ipynb            # N=2000 MMD + gender evaluation across all experiments
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
| `scribble_avg_latent.png` | Weighted average of per-group VAE latents, decoded once, then HED (baseline) |
| `scribble_avg_prompt.png` | Single image from the frac-weighted average of the groups' CLIP text embeddings, then HED (baseline) |
| `scribble_sdedit.png` | Guided SDEdit scribble with a descriptive prompt (baseline) |
| `scribble_sdedit_best.png` | Best unguided SDEdit candidate over a time-matched search budget (baseline) |
| `scribbles_all.png` | Side-by-side grid of all scribbles |
| `source_portrait.png` | Source portrait used for HED scribble extraction |
| `sdedit_search.png` | MMD curve over the SDEdit candidate search |
| `baselines_meta.json` | Timing, MMD scores, and seed info for all baseline candidates |
| `eval/` | N-sample evaluation of every method (`scripts/eval_all_experiments.py`): `scribbles_all.png`, `target_*.png`, `preview_<method>.png`, `results_table.csv`, `eval_results.json` |

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

Two steps, both on the cluster. Step 1 builds the competing scribbles with a fair
time-matched budget; step 2 scores every scribble on N generated images.

**Step 1 — build baseline scribbles** (`avg`, `avg_latent`, `avg_prompt`, `sdedit`, `sdedit_best`):

```bash
export ENV_PATH=/path/to/your/env
sbatch run_eval_baselines.sh SkewedTarget 241      # 241 = MLGD-F runtime in minutes
sbatch run_eval_baselines.sh BalancedTarget 241
sbatch run_eval_baselines.sh GenderInterpolation 241
sbatch run_eval_baselines.sh AgeInterpolation 177
```

Or run directly:

```bash
python scripts/eval_baselines.py \
    --experiment BalancedTarget \
    --lgd_cm_minutes 241
```

This saves the scribbles under `experiments/<ExperimentName>/baselines/`. The SDEdit
search is given the same wall-clock budget as the MLGD-F run.

**Step 2 — evaluate all methods on N images** (target distribution, MMD / root-MMD / SWD,
p(male) with 95% CI, previews, results JSON). Scripted port of
`notebooks/eval_all_experiments.ipynb`:

```bash
sbatch run_eval_all_experiments.sh BalancedTarget 2000   # one experiment, N=2000
sbatch run_eval_all_experiments.sh all 2000              # all experiments -> experiments/eval_all_results.json
sbatch run_eval_all_experiments.sh BalancedTarget 100    # quick smoke test
```

Or run directly:

```bash
python scripts/eval_all_experiments.py --experiment BalancedTarget --n_eval 2000
```

Scribbles are read from `experiments/<Exp>/baselines/` first, then `experiments/<Exp>/`.
Outputs go to `experiments/<Exp>/eval/` (override with `--output_dir`). Pass
`--skip_missing` to evaluate whatever scribbles exist instead of failing on a missing one.

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
