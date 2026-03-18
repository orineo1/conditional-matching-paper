# DPS Gradient Flow Diagnostic Report

**Date:** 2026-03-04
**Job:** 44221193 (salmon-01)
**wandb run:** [`logical-moon-32`](https://wandb.ai/conditional-matching/conditional-flow/runs/cnld9518)
**Script:** `SD_cond_SD_controlnet/debug_gradient.py`
**Branch:** `scribble_tune`

## Configuration

| Parameter | Value |
|-----------|-------|
| n_steps | 10 |
| strength | 0.5 (→ 5 denoising steps) |
| base_zeta | 0.2 |
| guidance_scale | 7.5 |
| num_variations | 20 |
| n_targets | 20 (10 man + 10 woman) |
| edge_method | hed_scribble |
| LoRA | checkpoint-50000 |
| seed | 42 |
| detailed_step | 5 (bug: not triggered, see below) |

## Background

DPS CLIP-MMD guidance produces non-zero gradients and real MMD values (~0.4), but guided output is visually identical to unguided output across **all** tested configurations:
- Sobel vs HED scribbles
- zeta 0.2 / 1.0 / 2.0
- CFG 5.0 / 7.5
- 20 / 100 targets, 20 / 100 variations
- With / without LoRA, with / without prompt

This diagnostic script was created to identify **why** the correction doesn't steer.

## Results

### Category 1: Correction Magnitudes

| Step | Timestep | Latent Norm | Correction Norm | Correction / Latent | Scheduler Step Norm | Correction / Step |
|------|----------|-------------|-----------------|---------------------|--------------------:|------------------:|
| 0 | 499 | 274.00 | 0.6812 | 0.00249 (0.25%) | 152.50 | 0.00447 (0.45%) |
| 1 | 399 | 230.13 | 0.0507 | 0.00022 (0.02%) | 105.38 | 0.00048 (0.05%) |
| 2 | 299 | 197.00 | 0.0704 | 0.00036 (0.04%) | 77.88 | 0.00090 (0.09%) |
| 3 | 199 | 174.88 | 0.1366 | 0.00078 (0.08%) | 57.44 | 0.00238 (0.24%) |
| 4 | 99 | 158.50 | 0.2085 | 0.00132 (0.13%) | 43.31 | 0.00481 (0.48%) |

**Key insight:** The scheduler moves the latent by 43–152 units per step. The DPS correction moves it by 0.05–0.68 units. The correction is **0.05–0.48% of the scheduler step** — completely drowned out by the denoising process.

### Category 2: Gradient Details

| Step | Timestep | Grad Norm | Grad Max | MMD Loss | MMD² | Zeta |
|------|----------|-----------|----------|----------|------|------|
| 0 | 499 | 0.5664 | 0.2378 | 0.1663 | — | — |
| 1 | 399 | 0.1257 | 0.0316 | 0.4960 | — | — |
| 2 | 299 | 0.1360 | 0.0403 | 0.3859 | — | — |
| 3 | 199 | 0.3015 | 0.0864 | 0.4415 | — | — |
| 4 | 99 | 0.4202 | 0.1418 | 0.4030 | — | — |

Note: Step 0 has the largest gradient (0.57) but also the largest scheduler step (152.5), so the ratio is still small. MMD stays in the range 0.17–0.50 — it **never decreases**, confirming that the guidance has no cumulative effect.

### Category 3: Chain Decomposition

**NOT TRIGGERED.** The `--detailed_step 5` parameter was set, but with `strength=0.5` and `n_steps=10`, there are only 5 denoising steps (indices 0–4). Step 5 doesn't exist. This is a bug in the submitted configuration — should have been `--detailed_step 4` or lower.

A follow-up run with `--detailed_step 4` would provide:
- `grad_clip_emb_norm`: ∂MMD/∂clip_embedding
- `grad_pixel_x0_norm`: ∂MMD/∂pixel_x0 (CLIP attenuation)
- `grad_pred_x0_norm`: ∂MMD/∂pred_x0 (VAE decode attenuation)
- `grad_latents_step_norm`: ∂MMD/∂x_t (UNet attenuation)
- Attenuation ratios between consecutive stages
- Finite-difference verification of autograd correctness

### Category 4: Direction Consistency

| Step Pair | Cosine Similarity |
|-----------|------------------:|
| 0 → 1 | -0.0031 |
| 1 → 2 | -0.0125 |
| 2 → 3 | +0.1268 |
| 3 → 4 | -0.0015 |

| Statistic | Value |
|-----------|------:|
| Mean | 0.0274 |
| Std | 0.0575 |
| Min | -0.0125 |
| Max | 0.1268 |

**Key insight:** Mean cosine similarity ≈ 0.03 — effectively **zero**. Each step's gradient points in an essentially random direction in latent space. This means corrections cancel out over the trajectory, even if each individual correction were made larger.

For reference, random unit vectors in ℝ^(4×64×64) = ℝ^16384 have expected cosine similarity 0 with std ≈ 1/√16384 ≈ 0.008. Our observed std of 0.058 is slightly above pure random, but the mean of 0.027 shows no coherent direction.

### Category 5: Zeta Ablation

| Multiplier | Effective Zeta | L2 Distance from Unguided |
|:----------:|:--------------:|:-------------------------:|
| 1× | 0.20 | 46.69 |
| 10× | 2.00 | 47.38 |
| 100× | 20.00 | 50.44 |
| **1000×** | **200.00** | **257.50** |

**Key insight:** At 1× and 10× zeta, the guided trajectory barely diverges from unguided (L2 ≈ 47, which is likely just the inherent noise from the stochastic correction). At 100× there's a slight increase (50.4). Only at **1000× (zeta=200)** does the output meaningfully diverge (L2=257.5).

This means the correction needs to be approximately **1000× stronger** to have any visible effect. But given the random direction problem, even a 1000× increase would produce random distortions rather than coherent steering toward the target distribution.

## Diagnosis

Two problems compound to make DPS guidance ineffective:

### Problem A: Correction Magnitude Too Small

The adaptive zeta formula `ζ = base_zeta / ||MMD||` produces corrections that are 0.05–0.48% of the scheduler step norm. The denoising process completely dominates.

**Why:** The gradient must backpropagate through a long chain: `latents → UNet → pred_x0 → VAE decode → pixels → CLIP → MMD`. Each stage attenuates the gradient. The chain decomposition (Category 3) was not triggered in this run, so we don't know exactly where the bottleneck is — but the end-to-end attenuation is clear.

### Problem B: Gradient Direction Inconsistent Across Steps

Consecutive steps' gradients have near-zero cosine similarity (mean=0.027). The gradient direction is essentially random, so corrections cancel out over the trajectory even if each is made larger.

**Why (likely):** At each step, the gradient is computed from a **single** pred_x0 image encoded through CLIP — a highly nonlinear pipeline. Small changes in the latent at different timesteps produce very different pred_x0 images, which map to very different CLIP embeddings. The MMD gradient in CLIP space has no reason to be consistent when mapped back through this nonlinear chain to the shared latent space.

### Combined Effect

Even if we increase zeta by 1000× (Problem A), the random directions (Problem B) mean the output gets randomly distorted rather than coherently steered. The two problems must be solved together.

## Possible Solutions

| Approach | Addresses | Description |
|----------|-----------|-------------|
| **Latent-space MMD** | A + B | Skip CLIP + VAE decode entirely. Compute MMD between pred_x0 latent and target latents. Shorter chain → stronger gradient. Latent space is more structured → more consistent direction. |
| **Gradient momentum** | B | Accumulate gradient direction across steps: `m_t = β·m_{t-1} + (1-β)·g_t`. Smooths out random oscillations. Only helps if there's a weak consistent signal buried in noise. |
| **Multi-sample pred_x0** | B | Instead of 1 pred_x0, use multiple noise samples to get multiple pred_x0 estimates. Average the gradient to reduce variance. |
| **Massive zeta** | A | Increase base_zeta to 200+. Brute force — works for magnitude but doesn't fix direction. |
| **Guide the sprinter** | A + B | Apply guidance during sprinter's 2-step generation (closer to final output, shorter backprop chain). |
| **Pixel-space MMD** | A (partial) | Skip CLIP but keep VAE decode. MMD in pixel space — shorter chain than CLIP but longer than latent. |

**Most promising:** Latent-space MMD (shortest chain, addresses both problems) + gradient momentum (insurance against remaining directional noise).

## Fix Applied

**Root cause identified:** The CLIP version was a regression from the latent version. In the latent version (`run_dps_step`), gradient flows through **20 variations** via the sprinter — measuring how the scribble affects the output distribution. When moving to CLIP space, this was "simplified" to flow through a **single pred_x0 CLIP embedding** while variations were detached. This caused:

1. **n=1 MMD** — extreme variance, random gradient direction
2. **No sprinter in gradient path** — gradient didn't know how the scribble affects actual photos, only how pred_x0 looks in CLIP space (a scribble vs photos comparison, which is nonsensical)
3. **Detached variations were wasted** — increasing from 20 to 100 didn't help because gradient only flowed through the single pred_x0

**Fix (`generation.py`):** Restored the original design — gradient flows through all 20 variations (sprinter → VAE decode → CLIP → MMD). Variations carry gradient, targets are detached. Uses `torch.utils.checkpoint.checkpoint` + batch_size=1 for memory efficiency. Old approach commented out for reference.

## Files & References

- **Diagnostic script:** [`SD_cond_SD_controlnet/debug_gradient.py`](debug_gradient.py)
- **SLURM submit:** [`SD_cond_SD_controlnet/submit_debug.sh`](submit_debug.sh)
- **wandb run:** [logical-moon-32](https://wandb.ai/conditional-matching/conditional-flow/runs/cnld9518)
- **Output directory:** `SD_cond_SD_controlnet/output/debug_gradient_44221193/`
  - `diagnostics.json` — full per-step metrics
  - `source_face.png`, `scribble.png` — input images
  - `guided_result.png` — DPS-guided output
  - `unguided.png` — baseline output
  - `zeta_mult_{1,10,100,1000}.png` — zeta ablation outputs
- **Previous experiment runs** (for context):
  - [`avid-haze-24`](https://wandb.ai/conditional-matching/conditional-flow) — LoRA + zeta=0.2
  - [`solar-wave-25`](https://wandb.ai/conditional-matching/conditional-flow) — LoRA + zeta=1.0
  - [`firm-darkness-31`](https://wandb.ai/conditional-matching/conditional-flow) — HED + zeta=2.0 + CFG=5.0
