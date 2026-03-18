"""Gender saliency heatmap over scribble space.

Backpropagates the CLIP gender score through the differentiable pipeline
(scribble -> sprinter -> VAE decode -> CLIP) to produce a per-pixel saliency
map showing which spatial regions influence gender classification.

Usage:
    python scripts/gender_saliency.py --scribble_path path/to/scribble.png --output_dir /tmp/saliency
    python scripts/gender_saliency.py --scribble_path path/to/scribble.png --n_seeds 20 --sigma 2.5
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent dir so we can import from SD_cond_SD_controlnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SD_cond_SD_controlnet"))
from clip_utils import load_clip_model, encode_images_clip
from models import load_models, freeze_module


MALE_PROMPT = "a photo of a man"
FEMALE_PROMPT = "a photo of a woman"
VARIATION_PROMPT = "a superrealistic professional photograph of"


def compute_saliency(scribble_path, n_seeds, device, sigma):
    # ── Load models ──────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    _, sprinter = load_models(device)
    clip_model, clip_processor = load_clip_model(device)

    # Enable gradient checkpointing, freeze all weights
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    freeze_module(sprinter.unet)
    freeze_module(sprinter.controlnet)
    freeze_module(sprinter.vae)

    vae = sprinter.vae
    vae_scaling_factor = sprinter.vae.config.scaling_factor

    # ── Precompute text embeddings ───────────────────────────────────────
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    text_inputs = processor(text=[MALE_PROMPT, FEMALE_PROMPT],
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = clip_model.get_text_features(**text_inputs)
        text_embs = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)  # [2, 768]

    # ── Load scribble ────────────────────────────────────────────────────
    scribble_pil = Image.open(scribble_path).convert("RGB").resize((512, 512))
    scribble_np = np.array(scribble_pil).astype(np.float32) / 255.0  # [H, W, 3]

    # ── Accumulate gradients across seeds ────────────────────────────────
    grad_accum = np.zeros((512, 512), dtype=np.float64)

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

        # Fresh tensor each seed (needs its own grad)
        scribble_tensor = torch.tensor(scribble_np, device=device, dtype=torch.float32)
        scribble_tensor = scribble_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]
        scribble_tensor.requires_grad_(True)

        # Differentiable forward: sprinter -> VAE decode -> CLIP
        def sprinter_vae_clip_forward(ctrl):
            var_latents = sprinter(
                prompt=[VARIATION_PROMPT],
                image=ctrl,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.5,
                output_type="latent",
                return_dict=True,
            ).images
            var_pixels = vae.decode(
                (var_latents.float() / vae_scaling_factor).to(vae.dtype)
            ).sample
            var_pixels = torch.clamp((var_pixels.float() + 1.0) / 2.0, 0.0, 1.0)
            with torch.cuda.amp.autocast(enabled=False):
                return encode_images_clip(var_pixels.float(), clip_model, clip_processor)

        img_emb = torch.utils.checkpoint.checkpoint(
            sprinter_vae_clip_forward, scribble_tensor, use_reentrant=False
        )  # [1, 768]

        # Gender score: male - female cosine similarity
        score = (img_emb @ text_embs[0]) - (img_emb @ text_embs[1])
        score.backward()

        grad = scribble_tensor.grad.detach()  # [1, 3, 512, 512]
        grad_magnitude = grad.abs().sum(dim=1).squeeze(0)  # [512, 512]
        grad_norm = grad_magnitude.cpu().numpy().astype(np.float64)

        print(f"  Seed {seed:3d}: grad_norm={grad_magnitude.sum().item():.6f}, "
              f"score={score.item():.4f}", flush=True)

        grad_accum += grad_norm

    # ── Average and smooth ───────────────────────────────────────────────
    grad_avg = grad_accum / n_seeds
    grad_smooth = gaussian_filter(grad_avg, sigma=sigma)

    return grad_smooth, scribble_np


def plot_saliency(heatmap, scribble_np, output_dir, scribble_path):
    os.makedirs(output_dir, exist_ok=True)

    # Save raw numpy
    npy_path = os.path.join(output_dir, "saliency.npy")
    np.save(npy_path, heatmap)
    print(f"Raw saliency saved to: {npy_path}", flush=True)

    # Plot overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(scribble_np)
    axes[0].set_title("Input Scribble")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="hot")
    axes[1].set_title("Gender Saliency")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(scribble_np)
    axes[2].imshow(heatmap, cmap="hot", alpha=0.6)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    scribble_name = os.path.splitext(os.path.basename(scribble_path))[0]
    fig.suptitle(f"Gender Saliency — {scribble_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"saliency_{scribble_name}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to: {png_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Gender saliency heatmap over scribble space")
    parser.add_argument("--scribble_path", type=str, required=True,
                        help="Path to a scribble PNG")
    parser.add_argument("--n_seeds", type=int, default=20,
                        help="Number of seeds to average over (default: 20)")
    parser.add_argument("--output_dir", type=str, default="output/saliency",
                        help="Output directory for heatmap")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not set)")
    parser.add_argument("--sigma", type=float, default=2.5,
                        help="Gaussian smoothing sigma (default: 2.5)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    heatmap, scribble_np = compute_saliency(
        args.scribble_path, args.n_seeds, device, args.sigma
    )
    plot_saliency(heatmap, scribble_np, args.output_dir, args.scribble_path)


if __name__ == "__main__":
    main()
