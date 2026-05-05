# MLGD-F

**Marginal-distribution-guided Diffusion via Flow** — steers a diffusion model toward
a target distribution using MMD loss in CLIP embedding space.

## Repository structure

```
mlgd_f/
├── src/                   # core library
│   ├── models.py          # Architect + Sprinter loading, LoRA support
│   ├── generation.py      # noise prediction, pred_x0, DPS gradient steps
│   ├── metrics.py         # MMD, SWD, evaluate_distribution_mmd
│   ├── clip_utils.py      # CLIP loading and differentiable encoding
│   ├── image_utils.py     # Sobel, VAE decode, base image
│   ├── visualization.py   # per-step grids, heatmap
│   └── analysis.py        # offline PCA/t-SNE/KDE/boxplot plots
│
├── scripts/
│   └── run_mlgd_f.py      # main entry point
│
├── slurm/
│   └── submit_mlgd_f.sh   # SLURM submit script (salmon partition)
│
├── requirements.txt
└── README.md
```

## How it works

Two diffusion models work together:

- **Architect** (SDXL Base): generates a scribble sketch via a 30-step denoising loop,
  initialised from a HED-scribble latent (SDEdit-style).
- **Sprinter** (SDXL Turbo + ControlNet-Scribble): takes the scribble and produces a
  realistic portrait in 2 steps.

At each Architect denoising step, MLGD-F:
1. Decodes the predicted clean image (`pred_x0`) to pixels via VAE.
2. Runs `num_variations` Sprinter passes to sample from the conditional distribution.
3. Encodes each Sprinter output through CLIP → 768-dim embeddings.
4. Computes MMD (or SWD) between the generated embeddings and the target distribution.
5. Backpropagates through the entire chain to get a gradient on the Architect latent.
6. Applies a correction `-ζ · ∇` before the scheduler step.

## Quickstart

```bash
pip install -r requirements.txt

python scripts/run_mlgd_f.py \
    --output_dir output/run_001 \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --n_targets 20 \
    --base_zeta 5.0 \
    --seed 1
```

On a SLURM cluster (salmon partition, L40S 48 GB):

```bash
sbatch slurm/submit_mlgd_f.sh
```

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | 30 | Architect denoising steps |
| `start_step` | 15 | SDEdit start (MLGD-F runs from here) |
| `num_variations` | 6 | Sprinter variations per step |
| `n_targets` | 20 | Target portraits (10 man + 10 woman) |
| `base_zeta` | 5.0 | Adaptive guidance strength |
| `loss_fn` | `mmd` | `mmd` or `swd` |
| `bandwidth_scale` | 1.0 | MMD kernel bandwidth scale |
| `kernel_alpha` | 1.0 | MMD RBF exponent (>1 = sharper) |
| `loss_scale` | 1.0 | Loss multiplier before grad |

## Offline analysis

After a run completes, regenerate all plots without GPU:

```bash
python src/analysis.py --run_dir output/run_001
```

Produces PCA, t-SNE, KDE, boxplots, portrait grids, and scribble heatmap
under `output/run_001/plots/`.

## wandb

Runs log to project `MLGDF-EXP` by default. The entity (team/username) is taken
from whoever is logged in via `wandb login`. Override with `--wandb_project` and
`--wandb_entity` if needed.
