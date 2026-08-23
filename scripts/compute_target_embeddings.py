"""
Compute CLIP (768D) and Latent-CLIP (512D) embeddings of anchor images,
construct binary and interpolated synthetic target distributions,
and save embeddings + PCA plots.

Requires: ssa_env (latent_clip_torch, open_clip_torch, diffusers, torch)
Run on cluster with GPU.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ── Standard CLIP (ViT-L/14, 768D) ──────────────────────────────────────────

def load_standard_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def encode_standard_clip(pil_images, model, processor, device):
    """Encode PIL images → (N, 768) L2-normalized CLIP embeddings."""
    tensors = torch.stack([TF.to_tensor(img) for img in pil_images]).to(device)
    resized = F.interpolate(tensors, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    normalized = (resized - mean) / std
    emb = model.vision_model(pixel_values=normalized).pooler_output
    emb = model.visual_projection(emb)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # (N, 768)


# ── Latent-CLIP (ViT-B-8-512, 512D) ─────────────────────────────────────────

class LatentCLIPEmbedder:
    """Maps VAE latents (4x64x64) to CLIP space (512D) without pixel decoding."""

    def __init__(self, device, checkpoint_path=None):
        import latent_clip

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                "wendlerc/latent-clip-b-8-512-34b-80k", "checkpoints/epoch_34.pt"
            )

        self.model, _, _ = latent_clip.create_model_and_transforms(
            "Latent-ViT-B-8-512", pretrained=checkpoint_path, device=device,
        )
        self.model.eval()
        self.device = device
        self.embed_dim = 512
        print(f"  LatentCLIPEmbedder loaded: embed_dim={self.embed_dim}")

    @torch.no_grad()
    def embed_latents(self, latents, batch_size=64):
        """Embed VAE latents (N, 4, 64, 64) into CLIP space (N, 512).

        Args:
            latents: (N, 4, 64, 64) scaled VAE latents (already multiplied by scaling_factor)
        Returns:
            (N, 512) L2-normalized CLIP embeddings
        """
        N = latents.shape[0]
        VAE_SCALE = 0.13025
        embeddings = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z = latents[start:end].to(self.device) / VAE_SCALE
            emb = self.model.encode_image(z)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)


def load_vae(device):
    """Load SDXL Turbo VAE for encoding images to latents."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-turbo", subfolder="vae"
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def encode_images_to_latents(pil_images, vae, device):
    """Encode PIL images → VAE latents (N, 4, 64, 64)."""
    tensors = torch.stack([TF.to_tensor(img) for img in pil_images]).to(device)
    tensors = (tensors * 2.0) - 1.0  # [0,1] → [-1,1]
    with torch.no_grad():
        latents = vae.encode(tensors).latent_dist.mean
        latents = latents * vae.config.scaling_factor
    return latents  # (N, 4, 64, 64)


# ── Target construction ──────────────────────────────────────────────────────

def construct_binary(e_a, e_b, n_targets):
    """Bimodal: n/2 copies of each anchor."""
    n_a = n_targets // 2
    n_b = n_targets - n_a
    return torch.cat([e_a.repeat(n_a, 1), e_b.repeat(n_b, 1)], dim=0), n_a, n_b


def construct_interpolated(e_a, e_b, n_targets):
    """Geodesic interpolation: normalize(α·A + (1-α)·B) for evenly spaced α."""
    device = e_a.device
    alphas = torch.linspace(0.0, 1.0, n_targets, device=device)
    interp = alphas.unsqueeze(1) * e_a + (1.0 - alphas.unsqueeze(1)) * e_b
    return interp / interp.norm(dim=-1, keepdim=True), alphas


# ── Main ─────────────────────────────────────────────────────────────────────

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

    # Load anchor images
    anchor_a = Image.open(args.anchor_a_path).convert("RGB").resize((512, 512))
    anchor_b = Image.open(args.anchor_b_path).convert("RGB").resize((512, 512))
    print(f"Loaded anchors: {args.anchor_a_path}, {args.anchor_b_path}")

    N = args.n_targets

    # ── Standard CLIP (768D) ──────────────────────────────────────────────────
    print("\n=== Standard CLIP (ViT-L/14, 768D) ===")
    clip_model, clip_proc = load_standard_clip(device)
    with torch.no_grad():
        clip_anchors = encode_standard_clip([anchor_a, anchor_b], clip_model, clip_proc, device)
    e_a_clip = clip_anchors[0:1]  # (1, 768)
    e_b_clip = clip_anchors[1:2]  # (1, 768)
    print(f"  Anchor cosine sim: {(e_a_clip @ e_b_clip.T).item():.4f}")

    binary_clip, n_a, n_b = construct_binary(e_a_clip, e_b_clip, N)
    interp_clip, alphas_clip = construct_interpolated(e_a_clip, e_b_clip, N)
    print(f"  Binary: {n_a}×A + {n_b}×B = {binary_clip.shape}")
    print(f"  Interpolated: {interp_clip.shape}")

    # Free CLIP model
    del clip_model, clip_proc
    torch.cuda.empty_cache()

    # ── Latent-CLIP (512D) ────────────────────────────────────────────────────
    print("\n=== Latent-CLIP (ViT-B-8-512, 512D) ===")
    vae = load_vae(device)
    latents = encode_images_to_latents([anchor_a, anchor_b], vae, device)
    print(f"  VAE latents shape: {latents.shape}")

    # Free VAE
    del vae
    torch.cuda.empty_cache()

    lclip = LatentCLIPEmbedder(device)
    lclip_anchors = lclip.embed_latents(latents)  # (2, 512)
    e_a_lclip = lclip_anchors[0:1]  # (1, 512)
    e_b_lclip = lclip_anchors[1:2]  # (1, 512)
    print(f"  Anchor cosine sim: {(e_a_lclip @ e_b_lclip.T).item():.4f}")

    binary_lclip, _, _ = construct_binary(e_a_lclip, e_b_lclip, N)
    interp_lclip, alphas_lclip = construct_interpolated(e_a_lclip, e_b_lclip, N)
    print(f"  Binary: {binary_lclip.shape}")
    print(f"  Interpolated: {interp_lclip.shape}")

    del lclip
    torch.cuda.empty_cache()

    # ── Save all embeddings ───────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "target_embeddings.npz")
    np.savez(
        save_path,
        # Standard CLIP
        e_a_clip=e_a_clip.cpu().numpy(),
        e_b_clip=e_b_clip.cpu().numpy(),
        binary_clip=binary_clip.cpu().numpy(),
        interp_clip=interp_clip.cpu().numpy(),
        clip_anchor_sim=(e_a_clip @ e_b_clip.T).item(),
        # Latent-CLIP
        e_a_lclip=e_a_lclip.cpu().numpy(),
        e_b_lclip=e_b_lclip.cpu().numpy(),
        binary_lclip=binary_lclip.cpu().numpy(),
        interp_lclip=interp_lclip.cpu().numpy(),
        lclip_anchor_sim=(e_a_lclip @ e_b_lclip.T).item(),
        # Metadata
        n_targets=N,
        alphas=alphas_clip.cpu().numpy(),
    )
    print(f"\nSaved embeddings to {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
