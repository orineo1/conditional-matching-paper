# MLGD-F: Manifold-Guided Diffusion with Distributional Feedback

This repository contains the implementation of **MLGD-F**, a Diffusion Posterior
Sampling pipeline that steers a diffusion model's generation toward a target
*distribution* using MMD or SWD loss in CLIP embedding space.

## Overview

Two diffusion models work together:

- **Architect** (SDXL): generates scribble sketches via a multi-step denoising loop
- **Sprinter** (SDXL + ControlNet-Scribble): takes a scribble and renders a
  realistic portrait in 2 steps

At each denoising step, the architect's predicted clean image is:
1. Decoded to pixels (VAE)
2. Passed through the Sprinter `num_variations` times → pixel samples
3. Encoded into CLIP embeddings
4. Compared to a fixed target distribution via MMD (or SWD)
5. The gradient flows back through the full chain to the architect's latent,
   correcting the next denoising step

This steers the generated scribble toward one that, when fed to the Sprinter,
produces images matching the target distribution.

## Repository Structure

```
├── run_gender.py          # Entry point: distributional attribute interpolation
├── run_age.py             # Entry point: age distribution interpolation
├── dps_loop.py            # Shared MLGD-F denoising loop
├── analysis.py            # Offline plot generation from saved runs
├── requirements.txt
│
├── src/                   # Core modules
│   ├── clip_utils.py      # CLIP loading and differentiable encoding
│   ├── generation.py      # Noise prediction, pred_x0, DPS gradient step
│   ├── image_utils.py     # Sobel/HED edge detection, VAE decoding
│   ├── metrics.py         # MMD, SWD, evaluate_distribution_mmd
│   ├── models.py          # Load Architect + Sprinter, LoRA support
│   └── visualization.py   # Per-step grids, PCA scatter, heatmaps
│
└── experiments/
    ├── gender/
    │   └── submit.sh      # Example SLURM script (fill in your paths)
    └── age/
        └── submit.sh      # Example SLURM script (fill in your paths)
```

## Installation

```bash
pip install -r requirements.txt
pip install controlnet_aux
```

## Usage

### Gender / Attribute Interpolation

```bash
python run_gender.py \
    --output_dir output/gender_run \
    --wandb_project mlgdf-gender \
    --n_targets 100 \
    --groups \
        "Woman:a superrealistic portrait photograph of a woman, studio lighting:50" \
        "Man:a superrealistic portrait photograph of a man, studio lighting:50" \
    --base_zeta 5.0 \
    --num_variations 6 \
    --n_steps 30 \
    --start_step 15 \
    --seed 1
```

Groups are defined as `name:prompt:percentage`. Percentages must sum to 100.
Any number of groups ≥ 2 is supported — e.g. a 4-group gender spectrum:

```bash
    --groups \
        "Woman:a portrait of a woman:25" \
        "Fem-androgynous:a portrait of an androgynous person, feminine:25" \
        "Masc-androgynous:a portrait of an androgynous person, masculine:25" \
        "Man:a portrait of a man:25"
```

### Age Interpolation

```bash
python run_age.py \
    --output_dir output/age_run \
    --wandb_project mlgdf-age \
    --age_min 10 \
    --age_max 80 \
    --age_step 1 \
    --age_gender man \
    --base_zeta 5.0 \
    --num_variations 6 \
    --n_steps 30 \
    --start_step 15 \
    --seed 1
```

### Key Arguments (both scripts)

| Argument | Default | Description |
|---|---|---|
| `--base_zeta` | 5.0 | DPS step size. Adaptive: `ζ = base_zeta / loss` |
| `--num_variations` | 6 | Sprinter samples per step for MMD estimate |
| `--n_steps` | 30 | Architect denoising steps |
| `--start_step` | 15 | SDEdit start — guidance runs from here |
| `--loss_fn` | mmd | `mmd` or `swd` |
| `--bandwidth_scale` | 1.0 | RBF kernel bandwidth multiplier |
| `--kernel_alpha` | 1.0 | RBF exponent (>1 = sharper falloff) |
| `--loss_scale` | 1.0 | Multiply loss before gradient computation |
| `--guidance_scale` | 0.0 | CFG scale for architect (0 = unconditional) |
| `--controlnet_scale` | 0.5 | ControlNet conditioning strength |

## Gradient Flow

```
latents_t (requires_grad=True)
  → UNet (frozen) → noise_pred
  → diffusion formula → pred_x0
  → VAE decode → pixel image
  → Sprinter × num_variations (frozen) → variation images
  → VAE decode → pixels
  → CLIP encode (frozen) → [num_variations, 768] embeddings
  → MMD / SWD vs target embeddings
  → autograd.grad → gradient on latents_t
  → correction: −ζ · grad  (applied before scheduler step)
```

## Output Structure

Each run saves to `--output_dir`:

```
output/my_run/
├── metrics.json               # Per-step gradients, final MMD/SWD, label stats
├── final_scribble_mlgdf.png   # Final MLGD-F scribble
├── final_scribble_unguided.png
├── scribble_heatmap.png       # Pixel diff between MLGD-F and unguided
├── final_photos_mlgdf.png
├── final_photos_unguided.png
├── target_clip_pca.png
├── source_portrait.png
├── scribble.png
├── steps/                     # Per-step visualization grids
│   ├── step_baseline.png
│   ├── step_000.png ...
├── photos_mlgdf/              # Individual eval photos
├── photos_unguided/
└── npy/                       # Saved numpy arrays for offline analysis
```

## Offline Analysis

```bash
python analysis.py --run_dir output/my_run --plots_dir output/my_run/plots
```

Regenerates all plots (PCA, t-SNE, KDE, boxplots, portrait grids) without GPU.

## Requirements

See `requirements.txt`. Tested with PyTorch 2.x + CUDA 12.x on an L40S (48 GB).
