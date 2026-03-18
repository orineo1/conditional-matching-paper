# Conditional Matching via Diffusion Posterior Sampling

Steer diffusion model generation so the output distribution matches a target distribution, using MMD-based guidance in CLIP embedding space.

## Overview

We use **Diffusion Posterior Sampling (DPS)** to guide an SDXL-based image generation pipeline. A two-model architecture generates diverse images from a single source:

1. **Architect** (SDXL Base 1.0) — 30-step denoising loop that produces edge-map scribbles
2. **Sprinter** (SDXL Turbo + ControlNet-Scribble) — fast 2-step generator that converts scribbles to photorealistic images

DPS guidance minimizes the **MMD (Maximum Mean Discrepancy)** between generated and target distributions in CLIP ViT-L/14 embedding space, steering the architect's latents so the sprinter's output distribution matches a target set.

## Repository Structure

```
SD_cond_SD_controlnet/       # Main DPS pipeline
  run_dps.py                 # Full DPS pipeline script
  models.py                  # Model loading (SDXL + ControlNet)
  generation.py              # Noise prediction, DPS gradient steps
  metrics.py                 # MMD with RBF kernel
  clip_utils.py              # CLIP encoding (ViT-L/14, 768-dim)
  image_utils.py             # Sobel edges, VAE decode
  visualization.py           # PCA & CLIP visualizations

scripts/                     # Evaluation scripts
  evaluate_gender_balance.py # Gender balance CLI (photo_*.png in a dir)
  reeval_gender.py           # Re-eval all runs in autoresearch_output/
  fairface/                  # FairFace gender classifier
    gender_classifier.py     # ResNet-34 multi-task classifier (fine-tuned)
    finetune_gender.py       # Fine-tune FC head on generated portraits
    submit_finetune.sh       # SLURM job for fine-tuning
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Models

- **SDXL Base 1.0**: `stabilityai/stable-diffusion-xl-base-1.0` (auto-downloaded from HuggingFace)
- **SDXL Turbo**: `stabilityai/sdxl-turbo` (auto-downloaded)
- **ControlNet Scribble**: `xinsir/controlnet-scribble-sdxl-1.0` (auto-downloaded)
- **CLIP**: `openai/clip-vit-large-patch14` (auto-downloaded)
- **FairFace** (for gender eval): [Download weights](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu) — `res34_fair_align_multi_7_20190809.pt`

## Usage

### Run DPS Pipeline

```bash
python SD_cond_SD_controlnet/run_dps.py \
    --num_steps 250 \
    --start_step 125 \
    --num_variations 100 \
    --output_dir output/my_run/
```

### Gender Balance Evaluation

Classify gender of generated portraits using a fine-tuned FairFace ResNet-34.
Fine-tuned weights (`models/fairface/res34_fairface_finetuned.pt`) are auto-selected if present.

**Evaluate a single folder** (looks for `photo_*.png`):

```bash
python scripts/evaluate_gender_balance.py \
    --image_dir output/my_run/ \
    --run_name my_eval
```

**Re-evaluate all runs** in `autoresearch_output/` (looks for `eval_photos_guided/` and `eval_photos_unguided/`):

```bash
python scripts/reeval_gender.py --base_dir autoresearch_output
# Or a single run:
python scripts/reeval_gender.py --base_dir autoresearch_output --run run2_zeta5
```

**Fine-tune the gender classifier** on new portrait data (freeze ResNet, train FC head only):

```bash
# Prepare data: data_dir/man/*.png + data_dir/woman/*.png
python scripts/fairface/finetune_gender.py \
    --data_dir scripts/fairface/finetune_data \
    --weights_path models/fairface/res34_fair_align_multi_7_20190809.pt \
    --output_path models/fairface/res34_fairface_finetuned.pt
```

## Key Results

| Configuration | Regular MMD | DPS MMD | MRI (%) |
|---|---|---|---|
| Base SDXL (250 steps, 100 var) | 0.295 | 0.110 | 62.8 |

**MRI** = MMD Relative Improvement = (MMD_regular - MMD_dps) / MMD_regular

## Tracking

- **wandb team**: `conditional-matching`
- **Projects**: `conditional-flow`, `gender-classifier`
