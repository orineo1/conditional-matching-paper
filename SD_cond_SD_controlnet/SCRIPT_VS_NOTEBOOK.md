# run_dps.py — Implementation & Deviations from main.ipynb

## Section Mapping

| Notebook Cell | Script Location (`run_dps.py`) | Notes |
|---|---|---|
| 1. Environment Setup | Lines 9-27 | `sys.path` setup, `matplotlib.use("Agg")` for headless |
| 2. Imports | Lines 28-39 | Same modules |
| 3. Load Models | Lines 87-90 | Adds `--lora_path` arg for architect LoRA |
| 4. Load CLIP | Line 89 | Same |
| 5. Base Image & Sobel | Lines 93-99 | Saves to disk instead of displaying |
| 6. Generate Targets | Lines 101-119 | Same logic, saves target images to disk |
| 7. Explore Targets (PCA) | Lines 139-148 | Saves PCA plot to file |
| 8. Encode Targets to CLIP | Lines 122-136 | Same |
| 9. Config & Seed Scribble | Lines 160-179 | Params via argparse instead of hardcoded |
| 10. Prepare DPS | Lines 182-213 | Same |
| 11. DPS Loop | Lines 220-337 | **Deviations below** |
| 12. Final Results | Lines 340-385 | Adds metrics.json, wandb summary |

## Deviations from Notebook

### 1. CLIP CPU Offloading (VRAM constraint)
- **Notebook**: CLIP stays on GPU throughout.
- **Script**: CLIP moved to CPU after target encoding (line 155), moved back to GPU per step (line 255), returned to CPU after (line 271).
- **Reason**: Cluster GPU (even L40S 48GB) can't hold SDXL + ControlNet + CLIP simultaneously during the DPS loop.

### 2. Separate Scheduler for Regular Path
- **Notebook**: Single `architect.scheduler` shared between DPS and regular paths.
- **Script**: `scheduler_regular = copy.deepcopy(architect.scheduler)` (line 199). Regular path uses its own scheduler instance.
- **Reason**: Newer diffusers (cluster: v1.x) tracks `step_index` internally. Sharing one scheduler between two `denoise_step()` calls per iteration corrupts the counter. Colab uses older diffusers where this wasn't tracked.

### 3. `compute_pred_x0_direct()` Instead of `compute_pred_x0()`
- **Notebook**: Uses `compute_pred_x0()` which calls `scheduler.step()` to extract `pred_original_sample`.
- **Script**: Uses `compute_pred_x0_direct()` (added to `generation.py`) which computes pred_x0 from the diffusion formula directly: `x0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)`.
- **Reason**: `scheduler.step()` advances `step_index` as a side effect. With 4 calls per iteration (2x pred_x0 + 2x denoise), the scheduler overflows. The direct formula avoids touching scheduler state.

### 4. MMD n=1 Fixes (two bugs, both in `metrics.py` / `generation.py`)
- **Notebook**: Same bugs exist but weren't caught (CLIP path not fully run on Colab).
- **Bug A — 0/0 NaN**: Unbiased MMD estimator has `(K_xx.sum() - K_xx.trace()) / (n*(n-1))`. When `n=1` (single current sample in CLIP-MMD), this is `0/0 = NaN`.
  - **Fix** (`metrics.py`): Skip K_xx term when `n=1` (it's a constant `exp(0)=1`, contributes no gradient).
- **Bug B — clamp kills gradient**: `torch.clamp(mmd_sq, min=0.0)` zeros out the MMD when the unbiased estimate is slightly negative (common with n=1). Clamp gradient is 0 when active, so `autograd.grad()` returns zero.
  - **Symptom**: `MMD=0.0001` (= `sqrt(1e-8)`), `grad_norm=0.0` every step.
  - **Fix** (`metrics.py`): Removed the clamp entirely. (`generation.py`): Changed `sqrt(mmd_sq + 1e-8)` to `sqrt(mmd_sq.abs() + 1e-8)` to safely handle negative estimates.

### 5. Output & Logging
- **Notebook**: Inline display, `step_vis_data` list in memory.
- **Script**: Saves per-step PNGs to `steps/`, logs to wandb, writes `metrics.json` with all step data.

### 6. Key Name Mismatch (Fixed)
- **Notebook**: `variation_latents_flat` in step data dict.
- **Script**: `variation_clip_flat` — correct name since values are CLIP embeddings, not raw latents.
