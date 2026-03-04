# scribble_tune — LoRA Fine-Tuning for Scribble Generation

LoRA fine-tuning of SDXL Turbo's U-Net on QuickDraw hand-drawn scribbles. The goal is to teach the **architect** model to generate clean, simple scribbles that the sprinter (ControlNet) can reliably interpret as portrait outlines.

## Why This Is Needed

Vanilla SDXL Turbo generates scribbles with noise, color artifacts, and complex textures. When the sprinter uses these as ControlNet conditioning, it misinterprets artifacts as content, producing distorted outputs. LoRA fine-tuning on real hand-drawn data teaches the architect to produce clean line art.

## Pipeline

```
QuickDraw API (345 categories x 1000 drawings)
  -> prepare_data.py: render strokes to 512x512 PNGs
  -> train_lora.py: LoRA fine-tune U-Net (rank 4, attention layers only)
  -> validate.py: noise-denoise test + pure generation comparison
  -> Integration: architect.unet = PeftModel.from_pretrained(unet, checkpoint_path)
```

## Files

| File | Purpose |
|------|---------|
| `prepare_data.py` | Downloads QuickDraw stroke data via `quickdraw` library, renders to 512x512 PNGs with PIL, writes `metadata.jsonl` |
| `train_lora.py` | LoRA training loop: VAE encode -> add noise -> U-Net predict -> MSE loss. Uses HuggingFace Accelerate for mixed precision |
| `validate.py` | Three-part validation: noise-denoise at multiple strengths, pure generation from scratch, gradient flow verification |
| `config.yaml` | All hyperparameters: model name, LoRA rank, learning rate, batch size, data paths |
| `submit_train.sh` | SLURM script for training (12h, single GPU) |
| `submit_prepare.sh` | SLURM script for data preparation |
| `submit_validate.sh` | SLURM script for validation |
| `setup_env.sh` | Conda environment creation (Python 3.10, PyTorch, diffusers, peft) |

## Key Packages

| Package | Version | Role |
|---------|---------|------|
| `diffusers` | latest | SDXL Turbo pipeline, schedulers, U-Net |
| `peft` | latest | LoRA injection via `LoraConfig` + `get_peft_model` |
| `accelerate` | latest | Mixed-precision training, gradient accumulation |
| `transformers` | latest | CLIP text encoders (both ViT-L and ViT-bigG for SDXL) |
| `quickdraw` | latest | QuickDraw dataset API — downloads stroke data by category |
| `wandb` | latest | Experiment tracking, composite grid images |

## Training Details

### Data Preparation (`prepare_data.py`)
- Uses `QuickDrawDataGroup(category, max_drawings=N, recognized=True)` to fetch drawings
- **Important**: the library returns generators, must call `list()` to materialize
- Strokes are lists of `(x, y)` tuples in [0, 255]; scaled to 512x512 with 10px margin
- All 345 categories x 1000 samples = ~345,000 training images
- Output: PNG files + `metadata.jsonl` with fixed caption per image

### LoRA Configuration (`train_lora.py`)
```python
LoraConfig(
    r=4,                    # rank 4 (low — minimal extra params)
    lora_alpha=4,           # alpha = rank (standard scaling)
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # attention layers only
    lora_dropout=0.0,
)
```

### Training Loop
1. **VAE encode**: images -> latents (float32 for stability, then cast to fp16)
2. **Add noise**: `DDPMScheduler.add_noise()` at random timesteps
3. **U-Net predict**: noise prediction with SDXL conditioning (`text_embeds` + `time_ids`)
4. **Loss**: MSE between predicted and actual noise (or velocity, depending on `prediction_type`)
5. **Optimizer**: AdamW, lr=1e-4, cosine schedule with 5% warmup, gradient clipping at 1.0

Text encoders are used once to pre-compute prompt embeddings, then offloaded to CPU to save ~3-4GB VRAM. LoRA parameters are cast to float32 for training stability while the rest of the U-Net stays in fp16.

### Training Config
- Batch size: 4 (max for 22GB GPU; 16 causes OOM even with text encoder offload)
- Epochs: ~2.3 completed before 12h SLURM timeout
- Checkpoints: every 5,000 steps (10 saved, best at `checkpoint-50000`)
- Fixed caption: `"a simple hand-drawn scribble"` for all images

## Validation (`validate.py`)

Three tests comparing base SDXL Turbo vs LoRA:

1. **Noise-denoise**: take held-out QuickDraw face scribbles (indices 500-504), add noise at varying strengths [0.1-0.5], denoise with real pipeline prompt + CFG 7.5. Same seed/noise for fair comparison.

2. **Pure generation**: generate scribbles from scratch with 5 different seeds. Clear LoRA win — base produces colorful chaos, LoRA produces clean simple scribbles.

3. **Gradient flow check**: backward pass through U-Net to verify LoRA parameters receive gradients (required for DPS guidance to work).

Outputs composite grid images logged to wandb (team: `conditional-matching`, project: `conditional-flow`).

## Integration with DPS Pipeline

One line in `models.py`:
```python
architect.unet = PeftModel.from_pretrained(architect.unet, lora_checkpoint_path)
```

The DPS experiment scripts accept `--lora_path` to enable this. See [`../SD_cond_SD_controlnet/README.md`](../SD_cond_SD_controlnet/README.md).

## Cluster

- Environment: `scribble_env` (conda, at `/sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env`)
- Training partition: `catfish` (L4 22GB) or `salmon` (L40S 48GB)
- Data stored at: `scribble_tune/data/quickdraw_512/` on cluster
