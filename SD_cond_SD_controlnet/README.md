# SD_cond_SD_controlnet — DPS with CLIP-MMD Guidance

Diffusion Posterior Sampling (DPS) pipeline that steers diffusion-generated images toward a **target distribution** using MMD loss in CLIP embedding space.

## Core Idea

Two diffusion models work together:
- **Architect** (SDXL Turbo): generates scribble sketches via a 30-step denoising loop
- **Sprinter** (SDXL Turbo + ControlNet-Scribble): takes a scribble and produces a realistic portrait in 2 steps

At each denoising step, the architect's predicted clean image is:
1. Decoded to pixels (VAE)
2. Encoded to CLIP embeddings (768-dim, differentiable)
3. Compared to a target distribution via MMD with RBF kernel
4. The MMD gradient flows back to the latent, correcting the next denoising step

This pushes the generated scribble toward one that, when fed to the sprinter, produces photos matching the target distribution (e.g., portraits of men and women).

## Original Files (by Ori Meidler)

| File | Purpose |
|------|---------|
| `main.ipynb` | Primary interactive notebook — full pipeline from setup to results |
| `models.py` | Loads architect + sprinter pipelines, freezes weights, enables gradient checkpointing |
| `generation.py` | Noise prediction with CFG, pred_x0 computation, scheduler steps, DPS gradient steps (latent-space and CLIP-space) |
| `metrics.py` | MMD computation with RBF kernel — gradient flows only through the generated samples |
| `image_utils.py` | Sobel edge detection, VAE latent-to-PIL decoding, base image (oval stick figure) construction |
| `clip_utils.py` | CLIP ViT-L/14 loading and differentiable image encoding (`[B,3,H,W]` -> `[B,768]`) |
| `visualization.py` | Per-step 2x7 visualization grids with PCA scatter of CLIP embeddings |
| `example.ipynb` | Legacy notebook using latent-space MMD (not maintained) |

## Files Edited by Shaul Tolkowsky

### `models.py` — LoRA support
Added `architect_lora_path` parameter to `load_models()`. When provided, loads LoRA weights onto the architect's U-Net via `PeftModel.from_pretrained()`. This teaches the architect to generate cleaner scribbles that the sprinter can better interpret.

### `generation.py` — Scheduler bug fixes
- Added `compute_pred_x0_direct()`: computes predicted clean image using the diffusion formula directly, without calling `scheduler.step()`. This avoids advancing the scheduler's internal `step_index`, which caused state corruption in newer versions of diffusers.
- Applied `.abs()` before `sqrt()` on MMD values in `run_dps_step_clip()` to handle slightly-negative unbiased estimates.

### `metrics.py` — MMD edge cases
- Fixed division by zero when `n=1`: the unbiased estimator term `(K_xx.sum() - trace) / (n*(n-1))` produces `0/0` for single samples. Now skips that term.
- Removed `clamp(min=0)` on MMD squared values: clamping killed gradients (grad=0 when clamped), blocking DPS guidance entirely.

### `visualization.py` — Headless rendering
Added optional `save_path` parameter to `plot_row()` and `visualize_step()`. When provided, saves to file instead of calling `plt.show()`, enabling use in cluster scripts without a display.

## Files Added by Shaul Tolkowsky

### `run_dps.py` — Script version of `main.ipynb`
CLI script that runs the full DPS pipeline from pure noise. Mirrors the notebook's 12 sections with argparse, wandb logging, and file output. Designed for headless cluster execution.

### `run_dps_experiment.py` — Redesigned experiment
A more meaningful experiment that starts from **real portraits** instead of pure noise:

1. Generate a target distribution (20 portraits: 10 man + 10 woman) and encode to CLIP
2. Generate N source male faces via sprinter
3. For each face: extract Sobel scribble, encode to latent, then for each noise strength (0.25, 0.5, 0.75):
   - Add noise at the corresponding timestep (partial noising, img2img-style)
   - Denoise **with** DPS CLIP-MMD guidance
   - Denoise **without** guidance (baseline)
   - Generate 10 conditioned photos from each denoised scribble
4. Produce a composite grid image per face showing all comparisons

This tests whether DPS guidance can steer a recognizable scribble toward the target distribution, rather than starting from random noise.

### SLURM Submit Scripts
- `submit_dps.sh` — submits `run_dps.py` (LoRA, zeta=0.2)
- `submit_experiment.sh` — submits `run_dps_experiment.py` (LoRA, zeta=0.2, 3 faces)
- `submit_experiment_strong.sh` — stronger guidance (LoRA, zeta=1.0, 3 faces)
- `submit_experiment_nolora.sh` — ablation without LoRA (zeta=0.2, 1 face)

All target the `salmon` partition (L40S 48GB GPU), 2-hour time limit.

## Files Edited by Ori Meidler (07/03/2026)

### `generation.py` — AMP compatibility fix
- `run_dps_step_clip()`: replaced deprecated `torch.cuda.amp.autocast` with
  `torch.amp.autocast('cuda', enabled=False)`. Disabling autocast around VAE
  decode prevents fp16/fp32 dtype mismatches that caused silent gradient corruption.

### `models.py` — HuggingFace Hub LoRA loading
- Added `_resolve_lora_path()`: transparently resolves `hf://<repo_id>/<file>`
  paths by downloading from HuggingFace Hub via `hf_hub_download`, extracting
  zip archives, and locating `adapter_config.json`. Local paths pass through
  unchanged.
- `load_models()` now calls `_resolve_lora_path()` before loading LoRA weights,
  so `architect_lora_path` can be either a local path or an `hf://` URL.

### `main.ipynb` — SDEdit-style init + HED scribbles
- **HED scribble conditioning**: replaced Sobel edge detection with
  `controlnet_aux.HEDdetector` in scribble mode for cleaner, sparser outlines
  that better match the ControlNet's training distribution.
- **SDEdit-style init**: instead of starting from pure Gaussian noise, the
  notebook now VAE-encodes the HED scribble, adds noise at `start_step=15`
  (halfway through the 30-step schedule), and runs DPS only from that point.
  This preserves the scribble's structural content while still allowing
  CLIP-MMD guidance to steer the generation.
- **Stronger guidance**: `base_zeta_prime` increased from 0.2 → 1.0;
  `guidance_scale` set to 0.0 (unconditional architect denoising).
- **Scheduler fix**: added `scheduler_regular = copy.deepcopy(architect.scheduler)`
  so the guided and unguided paths each have independent scheduler state,
  preventing step-index corruption.
- **`compute_pred_x0_direct`**: switched from `compute_pred_x0` (which calls
  `scheduler.step()` internally) to `compute_pred_x0_direct` for both paths.
- **HuggingFace auth**: notebook now reads `HF` and `GITHUB` tokens from Colab
  secrets and logs in automatically; falls back to manual input if absent.

## Gradient Flow Path

```
latents_t (requires_grad=True)
  -> UNet noise prediction (frozen)
  -> reverse diffusion formula -> pred_x0
  -> VAE decode -> pixels (float32)
  -> resize 224x224 + CLIP normalize
  -> CLIP vision encoder (frozen) -> 768-dim embedding
  -> MMD loss (RBF kernel, adaptive bandwidth)
  -> autograd.grad -> gradient on latents
  -> correction: -zeta * grad applied before scheduler step
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_steps` | 30 | Architect denoising steps |
| `guidance_scale` | 7.5 | Classifier-free guidance |
| `base_zeta` | 0.2 (or 1.0) | DPS correction strength, adaptive: `zeta = base_zeta / \|\|MMD\|\|` |
| `num_variations` | 20 | Sprinter variations per DPS step |
| `controlnet_scale` | 0.5 | Soft shape constraint |
| Target samples | 20 | 10 man + 10 woman portraits |

## LoRA Fine-Tuning

The architect can optionally be fine-tuned with LoRA on QuickDraw scribbles to produce cleaner sketches. See [`../scribble_tune/README.md`](../scribble_tune/README.md) for details.

## wandb

All runs log to team `conditional-matching`, project `conditional-flow`.
