"""
Interpolate between two anchor images in SDXL Turbo VAE latent space,
then project to CLIP (768D) and Latent-CLIP (512D) for visualization.

Flow:
  1. Encode anchors through VAE → z_A, z_B (4×64×64)
  2. Interpolate: z_i = α·z_A + (1-α)·z_B for N evenly spaced α
  3. Standard CLIP: VAE decode z_i → pixels → CLIP ViT-L/14 (768D)
  4. Latent-CLIP: z_i → LatentCLIPEmbedder (512D) — no pixel decoding

Requires: ssa_env (latent_clip_torch, diffusers, transformers, torch)
Run on cluster with GPU.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def load_vae(device):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-turbo", subfolder="vae"
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def load_standard_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def encode_pixels_clip(pixel_tensor, model, device):
    """pixel_tensor: (B, 3, H, W) in [0,1] → (B, 768) L2-normalized."""
    resized = F.interpolate(pixel_tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    normalized = (resized - mean) / std
    emb = model.vision_model(pixel_values=normalized).pooler_output
    emb = model.visual_projection(emb)
    return emb / emb.norm(dim=-1, keepdim=True)


class LatentCLIPEmbedder:
    def __init__(self, device):
        import latent_clip
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            "wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt"
        )
        self.model, _, _ = latent_clip.create_model_and_transforms(
            "Latent-ViT-B-8-512", pretrained=checkpoint_path, device=device,
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def embed(self, latents):
        """latents: (N, 4, 64, 64) scaled VAE latents → (N, 512)."""
        VAE_SCALE = 0.13025
        z = latents.to(self.device) / VAE_SCALE
        emb = self.model.encode_image(z)
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_a_path", type=str, required=True)
    parser.add_argument("--anchor_b_path", type=str, required=True)
    parser.add_argument("--n_targets", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="output/target_embeddings")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    N = args.n_targets

    # Load anchors
    anchor_a = Image.open(args.anchor_a_path).convert("RGB").resize((512, 512))
    anchor_b = Image.open(args.anchor_b_path).convert("RGB").resize((512, 512))
    print(f"Loaded anchors: {args.anchor_a_path}, {args.anchor_b_path}")

    # ── 1. Encode anchors to VAE latent space ────────────────────────────────
    print("\nEncoding anchors through VAE...")
    vae = load_vae(device)
    with torch.no_grad():
        imgs = torch.stack([TF.to_tensor(anchor_a), TF.to_tensor(anchor_b)]).to(device)
        imgs = (imgs * 2.0) - 1.0
        latents = vae.encode(imgs).latent_dist.mean * vae.config.scaling_factor
    z_a = latents[0:1]  # (1, 4, 64, 64)
    z_b = latents[1:2]

    # ── 2. Interpolate in VAE latent space ────────────────────────────────────
    print(f"Interpolating {N} points in VAE latent space...")
    alphas = torch.linspace(0.0, 1.0, N, device=device)
    # z_i = α·z_A + (1-α)·z_B
    z_interp = alphas.view(N, 1, 1, 1) * z_a + (1.0 - alphas.view(N, 1, 1, 1)) * z_b
    print(f"  Interpolated latents shape: {z_interp.shape}")

    # ── 3. Latent-CLIP: direct embedding (no pixel decode) ───────────────────
    print("\n=== Latent-CLIP (512D) — direct from VAE latents ===")
    lclip = LatentCLIPEmbedder(device)
    vae_interp_lclip = lclip.embed(z_interp)
    print(f"  Shape: {vae_interp_lclip.shape}")

    # Also embed the pure anchors for reference
    anchor_lclip = lclip.embed(latents)
    print(f"  Anchor cosine sim: {(anchor_lclip[0:1] @ anchor_lclip[1:2].T).item():.4f}")
    del lclip
    torch.cuda.empty_cache()

    # ── 4. Standard CLIP: decode to pixels first ─────────────────────────────
    print("\n=== Standard CLIP (768D) — VAE decode → pixels → CLIP ===")
    clip_model, clip_proc = load_standard_clip(device)

    batch_size = 10
    clip_embs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        with torch.no_grad():
            decoded = vae.decode(
                z_interp[start:end] / vae.config.scaling_factor
            ).sample
            pixels = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            emb = encode_pixels_clip(pixels, clip_model, device)
            clip_embs.append(emb.cpu())
        print(f"  Batch {start}-{end} done")

    vae_interp_clip = torch.cat(clip_embs, dim=0)
    print(f"  Shape: {vae_interp_clip.shape}")

    # Anchor CLIP embeddings (from pixels directly)
    with torch.no_grad():
        anchor_pixels = torch.stack([TF.to_tensor(anchor_a), TF.to_tensor(anchor_b)]).to(device)
        anchor_clip = encode_pixels_clip(anchor_pixels, clip_model, device)
    print(f"  Anchor cosine sim: {(anchor_clip[0:1] @ anchor_clip[1:2].T).item():.4f}")

    del clip_model, clip_proc, vae
    torch.cuda.empty_cache()

    # ── 5. Save ───────────────────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "vae_interp_embeddings.npz")
    np.savez(
        save_path,
        vae_interp_clip=vae_interp_clip.numpy(),
        vae_interp_lclip=vae_interp_lclip.numpy(),
        anchor_clip=anchor_clip.cpu().numpy(),
        anchor_lclip=anchor_lclip.numpy(),
        clip_anchor_sim=(anchor_clip[0:1] @ anchor_clip[1:2].T).item(),
        lclip_anchor_sim=(anchor_lclip[0:1] @ anchor_lclip[1:2].T).item(),
        alphas=alphas.cpu().numpy(),
        n_targets=N,
    )
    print(f"\nSaved to {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
