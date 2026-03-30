"""
generate_targets.py
====================
Stage 1 of the interpolation experiment.

Generates:
  - ONE canonical masculine man portrait  → VAE latent duplicated 50 times
  - ONE canonical feminine woman portrait → VAE latent duplicated 50 times
  - 100-step linear interpolation in VAE latent space between the two latents
  - Decodes and saves all 100 interpolation frames for visual inspection

At DPS time (run_interpolation_dps.py), the target distribution for
interpolation step i is simply 50 identical copies of interp_latents[i].

Usage:
    python generate_targets.py \
        --output_dir SD_cond_SD_controlnet/output/interpolation_experiment \
        --n_copies 50 \
        --n_interp 100 \
        --seed 42

Outputs (in output_dir):
    man_canonical.png           — the single man portrait
    woman_canonical.png         — the single woman portrait
    man_vae_latents.pt          — [50, 4, 64, 64]  (50 identical copies)
    woman_vae_latents.pt        — [50, 4, 64, 64]  (50 identical copies)
    interp_vae_latents.pt       — [100, 4, 64, 64] linearly interpolated
    interp_decoded/             — decoded PNG for each of the 100 interp latents
    interp_contact_sheet.png    — 10-frame strip showing the path
    interp_viz_latent_pca.png   — PCA of flattened VAE latents
    interp_viz_clip_pca.png     — PCA of CLIP embeddings (perceptual view)
    pca_latent.pkl              — fitted PCA model (reused by run_interpolation_dps.py)
    pca_clip.pkl                — fitted PCA model (reused by run_interpolation_dps.py)
    metadata.json
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(SCRIPT_DIR, "SD_cond_SD_controlnet"),
    os.path.join(SCRIPT_DIR, "..", "SD_cond_SD_controlnet"),
    os.path.join(os.getcwd(), "SD_cond_SD_controlnet"),
]
for c in CANDIDATES:
    if os.path.isdir(c) and c not in sys.path:
        sys.path.insert(0, c)
        break

# ── Prompts ───────────────────────────────────────────────────────────────────
MAN_PROMPT = (
    "a hyperrealistic studio portrait photograph of a very masculine man, "
    "strong jawline, short hair, formal attire, sharp features, "
    "professional photography, 8k"
)
WOMAN_PROMPT = (
    "a hyperrealistic studio portrait photograph of a very feminine woman, "
    "long hair, soft features, elegant attire, beautiful, "
    "professional photography, 8k"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="One canonical man + one canonical woman, "
                    "duplicate 50x each, interpolate VAE latents over 100 steps."
    )
    p.add_argument("--output_dir", type=str,
                   default="SD_cond_SD_controlnet/output/interpolation_experiment")
    p.add_argument("--n_copies",  type=int, default=50,
                   help="Number of identical copies per canonical latent")
    p.add_argument("--n_interp",  type=int, default=100,
                   help="Number of interpolation steps (including endpoints)")
    p.add_argument("--controlnet_scale", type=float, default=0.4)
    p.add_argument("--man_seed",   type=int, default=0,
                   help="Generator seed for the canonical man image")
    p.add_argument("--woman_seed", type=int, default=1,
                   help="Generator seed for the canonical woman image")
    p.add_argument("--seed",  type=int, default=42,
                   help="Global numpy/torch seed")
    p.add_argument("--controlnet_model_id", type=str,
                   default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--sprinter_model_id",   type=str,
                   default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id",  type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    return p.parse_args()


# ── Generate exactly ONE image + its VAE latent ───────────────────────────────
def generate_one(pipe, prompt, cond_pil, cn_scale, seed):
    """
    Returns (PIL image, latent tensor [4,64,64] float32, pipeline-scaled).
    The latent is what the pipeline outputs in output_type='latent' — i.e.
    already multiplied by vae.config.scaling_factor.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe.vae.to(dtype=torch.float16)

    with torch.no_grad():
        result = pipe(
            prompt=[prompt],
            image=[cond_pil],
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=cn_scale,
            output_type="latent",
            return_dict=True,
            generator=generator,
        )
        lat = result.images   # [1, 4, 64, 64] fp16, pipeline-scaled

        # Decode to PIL for visual check
        decoded = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
        decoded = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0)
        pil     = TF.to_pil_image(decoded[0].cpu())

    pipe.vae.to(dtype=torch.float32)
    return pil, lat[0].float().cpu()   # PIL, [4, 64, 64]


# ── Decode a stack of latents → list of PIL images ───────────────────────────
def decode_latents(latents_batch, vae, batch_size=4):
    """
    latents_batch : [N, 4, 64, 64] float32, pipeline-scaled.
    Returns        : list of N PIL images.
    """
    device = next(vae.parameters()).device
    images = []
    vae.to(dtype=torch.float16)
    with torch.no_grad():
        for i in range(0, len(latents_batch), batch_size):
            batch   = latents_batch[i:i + batch_size].to(device).half()
            decoded = vae.decode(batch / vae.config.scaling_factor).sample
            decoded = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0)
            for j in range(decoded.shape[0]):
                images.append(TF.to_pil_image(decoded[j].cpu()))
    vae.to(dtype=torch.float32)
    return images


# ── Encode PIL list → CLIP embeddings (visualization only) ───────────────────
def encode_pil_to_clip(pil_list, clip_model, clip_processor, device, batch_size=8):
    from clip_utils import encode_images_clip
    all_embs = []
    clip_model.to(device)
    with torch.no_grad():
        for i in range(0, len(pil_list), batch_size):
            batch   = pil_list[i:i + batch_size]
            tensors = torch.cat(
                [TF.to_tensor(img).unsqueeze(0) for img in batch], dim=0
            ).to(device)
            embs = encode_images_clip(tensors, clip_model, clip_processor)
            all_embs.append(embs.cpu())
    clip_model.to("cpu")
    return torch.cat(all_embs, dim=0).float()


# ── Shared PCA scatter helper ─────────────────────────────────────────────────
def plot_interp_pca(coords, alphas_np, endpoint_labels, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(len(coords)), cmap="RdBu_r",
        s=60, edgecolors="black", linewidths=0.3, zorder=3
    )
    plt.colorbar(sc, ax=ax, label="Interpolation step (0=man, N-1=woman)")

    # Annotate every 10th step
    for idx in range(0, len(coords), max(1, len(coords) // 10)):
        ax.annotate(f"{idx}", coords[idx],
                    textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)

    ax.scatter(*coords[0],  s=250, c="royalblue", marker="*", zorder=5,
               label=endpoint_labels[0])
    ax.scatter(*coords[-1], s=250, c="crimson",   marker="*", zorder=5,
               label=endpoint_labels[1])
    ax.plot(coords[:, 0], coords[:, 1], "k--", alpha=0.25, linewidth=1, zorder=2)

    ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Plot saved → {save_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load models ────────────────────────────────────────────────────────
    print("\n[1/6] Loading models...", flush=True)
    from models      import load_models
    from clip_utils  import load_clip_model
    from image_utils import build_base_image, sobel_proxy
    import torchvision.transforms as T

    _, sprinter = load_models(
        device,
        controlnet_model_id=args.controlnet_model_id,
        sprinter_model_id=args.sprinter_model_id,
        architect_model_id=args.architect_model_id,
    )
    clip_model, clip_processor = load_clip_model(device)
    clip_model.to("cpu")
    print("  ✅ Models loaded.", flush=True)

    # ── 2. Build neutral oval scribble ────────────────────────────────────────
    print("\n[2/6] Building base scribble...", flush=True)
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_tensor = sobel_proxy(base_tensor, device)
        cond_pil     = T.ToPILImage()(sobel_tensor.squeeze(0).cpu())
    cond_pil.save(os.path.join(args.output_dir, "base_scribble.png"))
    print("  ✅ Base scribble saved.", flush=True)

    # ── 3. Generate the two canonical portraits ───────────────────────────────
    print(f"\n[3/6] Generating canonical man  (seed={args.man_seed})...", flush=True)
    man_pil, man_latent = generate_one(
        sprinter, MAN_PROMPT, cond_pil, args.controlnet_scale, seed=args.man_seed
    )
    man_pil.save(os.path.join(args.output_dir, "man_canonical.png"))
    print(f"  Latent shape: {man_latent.shape}  norm: {man_latent.norm():.3f}", flush=True)

    print(f"\n[3/6] Generating canonical woman (seed={args.woman_seed})...", flush=True)
    woman_pil, woman_latent = generate_one(
        sprinter, WOMAN_PROMPT, cond_pil, args.controlnet_scale, seed=args.woman_seed
    )
    woman_pil.save(os.path.join(args.output_dir, "woman_canonical.png"))
    print(f"  Latent shape: {woman_latent.shape}  norm: {woman_latent.norm():.3f}", flush=True)

    # Distance stats between the two latents
    man_flat   = man_latent.reshape(-1).float()
    woman_flat = woman_latent.reshape(-1).float()
    cos_sim    = torch.dot(man_flat / man_flat.norm(),
                           woman_flat / woman_flat.norm()).item()
    l2_dist    = (man_latent - woman_latent).norm().item()
    print(f"\n  Man↔Woman cosine sim (VAE latent): {cos_sim:.4f}")
    print(f"  Man↔Woman L2 distance (VAE latent): {l2_dist:.3f}", flush=True)

    # ── 4. Duplicate each latent n_copies times ───────────────────────────────
    print(f"\n[4/6] Duplicating each latent {args.n_copies}×...", flush=True)

    # Shape: [n_copies, 4, 64, 64] — all rows identical
    man_latents   = man_latent.unsqueeze(0).expand(args.n_copies, -1, -1, -1).clone()
    woman_latents = woman_latent.unsqueeze(0).expand(args.n_copies, -1, -1, -1).clone()

    torch.save(man_latents,   os.path.join(args.output_dir, "man_vae_latents.pt"))
    torch.save(woman_latents, os.path.join(args.output_dir, "woman_vae_latents.pt"))
    print(f"  man_vae_latents.pt   {tuple(man_latents.shape)}")
    print(f"  woman_vae_latents.pt {tuple(woman_latents.shape)}", flush=True)

    # ── 5. Linear interpolation in VAE latent space ───────────────────────────
    print(f"\n[5/6] Building {args.n_interp}-step linear interpolation...", flush=True)

    alphas = torch.linspace(0.0, 1.0, args.n_interp)   # [n_interp]

    # Simple lerp: (1-α)*man + α*woman  — no renormalization needed here
    # because VAE latents are not constrained to a sphere
    interp_latents = torch.stack(
        [(1.0 - a) * man_latent + a * woman_latent for a in alphas],
        dim=0
    )   # [n_interp, 4, 64, 64]

    torch.save(interp_latents, os.path.join(args.output_dir, "interp_vae_latents.pt"))
    print(f"  interp_vae_latents.pt  {tuple(interp_latents.shape)}", flush=True)

    # Sanity check endpoints
    err_start = (interp_latents[0]  - man_latent).abs().max().item()
    err_end   = (interp_latents[-1] - woman_latent).abs().max().item()
    print(f"  Endpoint error — step 0 ↔ man:   {err_start:.2e}")
    print(f"  Endpoint error — step N ↔ woman: {err_end:.2e}", flush=True)

    # ── 6. Decode + visualize ─────────────────────────────────────────────────
    print(f"\n[6/6] Decoding all {args.n_interp} interpolation latents...", flush=True)

    interp_pil = decode_latents(interp_latents, sprinter.vae, batch_size=4)

    # Save individual decoded frames
    decoded_dir = os.path.join(args.output_dir, "interp_decoded")
    os.makedirs(decoded_dir, exist_ok=True)
    for i, img in enumerate(tqdm(interp_pil, desc="Saving interp frames")):
        alpha_str = f"{alphas[i].item():.3f}".replace(".", "p")
        img.save(os.path.join(decoded_dir, f"interp_{i:03d}_a{alpha_str}.png"))
    print(f"  ✅ {len(interp_pil)} decoded frames → {decoded_dir}", flush=True)

    # Contact sheet: 10 evenly-spaced keyframes
    keyframe_idx = np.linspace(0, args.n_interp - 1, 10, dtype=int)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Interpolation Contact Sheet: Masculine Man → Feminine Woman",
                 fontsize=13, fontweight="bold")
    for j, idx in enumerate(keyframe_idx):
        ax = axes[j // 5][j % 5]
        ax.imshow(interp_pil[idx])
        ax.set_title(f"step {idx}  α={alphas[idx]:.2f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    contact_path = os.path.join(args.output_dir, "interp_contact_sheet.png")
    fig.savefig(contact_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ Contact sheet → {contact_path}", flush=True)

    # PCA of flattened VAE latents
    print("  PCA of VAE latents...", flush=True)
    all_flat = interp_latents.reshape(args.n_interp, -1).numpy()
    pca_lat  = PCA(n_components=2)
    lat_coords = pca_lat.fit_transform(all_flat)
    var_lat    = pca_lat.explained_variance_ratio_.sum()
    plot_interp_pca(
        lat_coords, alphas.numpy(),
        endpoint_labels=["Man (α=0)", "Woman (α=1)"],
        title=f"VAE Latent PCA — {args.n_interp} Interpolation Steps\n"
              f"Var explained: {var_lat:.1%}",
        save_path=os.path.join(args.output_dir, "interp_viz_latent_pca.png"),
    )

    # CLIP PCA — perceptual check
    print("  CLIP PCA (perceptual)...", flush=True)
    clip_embs  = encode_pil_to_clip(interp_pil, clip_model, clip_processor, device)
    pca_clip   = PCA(n_components=2)
    clip_coords = pca_clip.fit_transform(clip_embs.numpy())
    var_clip    = pca_clip.explained_variance_ratio_.sum()
    plot_interp_pca(
        clip_coords, alphas.numpy(),
        endpoint_labels=["Man (α=0)", "Woman (α=1)"],
        title=f"CLIP PCA (perceptual) — {args.n_interp} Interpolation Steps\n"
              f"Var explained: {var_clip:.1%}",
        save_path=os.path.join(args.output_dir, "interp_viz_clip_pca.png"),
    )

    # Save fitted PCA models so run_interpolation_dps.py can reuse them
    with open(os.path.join(args.output_dir, "pca_latent.pkl"), "wb") as f:
        pickle.dump(pca_lat, f)
    with open(os.path.join(args.output_dir, "pca_clip.pkl"), "wb") as f:
        pickle.dump(pca_clip, f)

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "args":                         vars(args),
        "device":                       device,
        "man_latent_shape":             list(man_latent.shape),
        "woman_latent_shape":           list(woman_latent.shape),
        "man_latent_norm":              man_latent.norm().item(),
        "woman_latent_norm":            woman_latent.norm().item(),
        "man_woman_cosine_sim_latent":  cos_sim,
        "man_woman_l2_dist_latent":     l2_dist,
        "interp_latents_shape":         list(interp_latents.shape),
        "pca_latent_var_explained":     float(var_lat),
        "pca_clip_var_explained":       float(var_clip),
        "man_prompt":                   MAN_PROMPT,
        "woman_prompt":                 WOMAN_PROMPT,
        "outputs": {
            "man_canonical":      "man_canonical.png",
            "woman_canonical":    "woman_canonical.png",
            "man_vae_latents":    "man_vae_latents.pt",
            "woman_vae_latents":  "woman_vae_latents.pt",
            "interp_vae_latents": "interp_vae_latents.pt",
            "interp_decoded":     "interp_decoded/",
            "contact_sheet":      "interp_contact_sheet.png",
            "latent_pca":         "interp_viz_latent_pca.png",
            "clip_pca":           "interp_viz_clip_pca.png",
            "base_scribble":      "base_scribble.png",
        }
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"  Canonical man latent:   {tuple(man_latent.shape)}  norm={man_latent.norm():.3f}")
    print(f"  Canonical woman latent: {tuple(woman_latent.shape)}  norm={woman_latent.norm():.3f}")
    print(f"  Copies per class:       {args.n_copies}  → [{args.n_copies}, 4, 64, 64] each")
    print(f"  Interpolation steps:    {args.n_interp}")
    print(f"  Man↔Woman L2 (latent):  {l2_dist:.3f}")
    print(f"  Man↔Woman cos-sim:      {cos_sim:.4f}")
    print(f"  Output dir:             {args.output_dir}")
    print("="*65)
    print("\n✅ Done! Run run_interpolation_dps.py next.\n")


if __name__ == "__main__":
    main()
