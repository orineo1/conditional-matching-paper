"""
Explore which semantic axes have the strongest variance in CLIP image space.

For each attribute axis, generates N images at each pole using SDXL Turbo,
encodes to CLIP, and computes:
  - Cosine distance between pole means
  - Fisher ratio: between-class variance / within-class variance
  - PC1 variance explained when combining both poles

Outputs a ranked bar chart + per-axis PCA scatter plots.

Usage:
    python explore_clip_axes.py --n_per_pole 20 --output_dir clip_axes_exploration
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model

# ── Define attribute axes ─────────────────────────────────────────────────────
AXES = {
    "gender":        ("a superrealistic professional photograph of a man",
                      "a superrealistic professional photograph of a woman"),
    "age":           ("a superrealistic professional photograph of a 10-year-old person",
                      "a superrealistic professional photograph of an 80-year-old person"),
    "era":           ("a superrealistic portrait of a person from prehistoric times, 8000 BC",
                      "a superrealistic portrait of a person from the far future, year 2150"),
    "expression":    ("a superrealistic professional photograph of a person with a wide happy smile",
                      "a superrealistic professional photograph of a person with a sad, tearful expression"),
    "ethnicity":     ("a superrealistic professional photograph of a white person",
                      "a superrealistic professional photograph of a black person"),
    "hair_color":    ("a superrealistic professional photograph of a blonde person",
                      "a superrealistic professional photograph of a person with jet black hair"),
    "formality":     ("a superrealistic professional photograph of a person in a formal business suit",
                      "a superrealistic professional photograph of a person in casual streetwear"),
    "setting":       ("a superrealistic professional photograph of a person outdoors in a forest",
                      "a superrealistic professional photograph of a person in a modern office"),
    "lighting":      ("a superrealistic professional photograph of a person in bright daylight",
                      "a superrealistic professional photograph of a person in dark dramatic low-key lighting"),
    "body_build":    ("a superrealistic professional photograph of a very muscular athletic person",
                      "a superrealistic professional photograph of a slim slender person"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_pole", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="clip_axes_exploration")
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--axes", type=str, nargs="*", default=None,
                   help="Subset of axes to run (default: all). E.g. --axes gender age era")
    return p.parse_args()


def generate_images(pipe, prompt, n, seed, device):
    imgs = []
    for i in range(n):
        g = torch.Generator(device=device).manual_seed(seed * 1000 + i)
        img = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=512, width=512,
            generator=g,
        ).images[0]
        imgs.append(img)
    return imgs


def encode_pil_list(pil_imgs, clip_model, clip_processor, device):
    tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in pil_imgs], dim=0).to(device)
    with torch.no_grad():
        return encode_images_clip(tensors, clip_model, clip_processor).cpu().numpy()


def fisher_ratio(emb_a, emb_b):
    """Between-class variance / mean within-class variance (scalar)."""
    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    mean_all = np.vstack([emb_a, emb_b]).mean(axis=0)
    between = (np.linalg.norm(mean_a - mean_all)**2 + np.linalg.norm(mean_b - mean_all)**2) / 2
    within_a = np.mean(np.linalg.norm(emb_a - mean_a, axis=1)**2)
    within_b = np.mean(np.linalg.norm(emb_b - mean_b, axis=1)**2)
    within = (within_a + within_b) / 2
    return between / (within + 1e-8)


def pole_cosine_dist(emb_a, emb_b):
    """Cosine distance between mean embeddings of two poles."""
    ma = emb_a.mean(axis=0); ma /= np.linalg.norm(ma)
    mb = emb_b.mean(axis=0); mb /= np.linalg.norm(mb)
    return 1.0 - float(ma @ mb)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    axes_to_run = {k: v for k, v in AXES.items()
                   if args.axes is None or k in args.axes}

    print(f"Running {len(axes_to_run)} axes × 2 poles × {args.n_per_pole} images each", flush=True)

    from diffusers import StableDiffusionXLPipeline
    print(f"Loading {args.model_id}...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16,
        use_safetensors=True, variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    clip_model, clip_processor = load_clip_model(device)

    results = {}
    imgs_dir = os.path.join(args.output_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)

    for axis_name, (prompt_a, prompt_b) in axes_to_run.items():
        print(f"\n[{axis_name}]", flush=True)
        print(f"  A: {prompt_a}", flush=True)
        print(f"  B: {prompt_b}", flush=True)

        imgs_a = generate_images(pipe, prompt_a, args.n_per_pole, args.seed, device)
        imgs_b = generate_images(pipe, prompt_b, args.n_per_pole, args.seed + 999, device)

        # Save sample images
        for i, img in enumerate(imgs_a[:4]):
            img.save(os.path.join(imgs_dir, f"{axis_name}_A_{i:02d}.png"))
        for i, img in enumerate(imgs_b[:4]):
            img.save(os.path.join(imgs_dir, f"{axis_name}_B_{i:02d}.png"))

        emb_a = encode_pil_list(imgs_a, clip_model, clip_processor, device)
        emb_b = encode_pil_list(imgs_b, clip_model, clip_processor, device)

        fr   = fisher_ratio(emb_a, emb_b)
        cdist = pole_cosine_dist(emb_a, emb_b)

        combined = np.vstack([emb_a, emb_b])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(combined)
        pc1_var = pca.explained_variance_ratio_[0]
        n = args.n_per_pole

        results[axis_name] = {
            "fisher_ratio": float(fr),
            "cosine_dist":  float(cdist),
            "pc1_var":      float(pc1_var),
            "prompt_a":     prompt_a,
            "prompt_b":     prompt_b,
        }
        print(f"  Fisher={fr:.3f}  cosine_dist={cdist:.4f}  PC1_var={pc1_var*100:.1f}%", flush=True)

        # Per-axis PCA scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:n, 0], coords[:n, 1], c='dodgerblue', alpha=0.7, s=50, label='A')
        ax.scatter(coords[n:, 0], coords[n:, 1], c='crimson',    alpha=0.7, s=50, label='B')
        ax.set_title(f"{axis_name}\nFisher={fr:.2f}  cosine_dist={cdist:.3f}  PC1={pc1_var*100:.1f}%",
                     fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"pca_{axis_name}.png"), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # ── Summary ranking plot ──────────────────────────────────────────────────
    sorted_axes = sorted(results.items(), key=lambda x: x[1]["fisher_ratio"], reverse=True)

    fig, axes_plot = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CLIP Axis Hierarchy", fontsize=14, fontweight="bold")

    names  = [a for a, _ in sorted_axes]
    fishers = [v["fisher_ratio"] for _, v in sorted_axes]
    cdists  = [v["cosine_dist"]  for _, v in sorted_axes]
    pc1vars = [v["pc1_var"]*100  for _, v in sorted_axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))

    for ax_plot, vals, ylabel, title in [
        (axes_plot[0], fishers,  "Fisher ratio",         "Fisher ratio (between/within variance)"),
        (axes_plot[1], cdists,   "Cosine distance",      "Cosine distance between pole means"),
        (axes_plot[2], pc1vars,  "PC1 variance (%)",     "PC1 variance explained"),
    ]:
        bars = ax_plot.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax_plot.set_xlabel(ylabel)
        ax_plot.set_title(title, fontsize=10)
        ax_plot.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, vals[::-1]):
            ax_plot.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                         f"{val:.2f}", va='center', fontsize=8)

    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, "clip_axis_hierarchy.png")
    fig.savefig(summary_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(sorted_axes, f, indent=2)

    print("\n" + "="*60)
    print("CLIP AXIS HIERARCHY (ranked by Fisher ratio):")
    print(f"{'Axis':<16} {'Fisher':>8} {'CosDist':>9} {'PC1%':>7}")
    print("-"*44)
    for name, v in sorted_axes:
        print(f"{name:<16} {v['fisher_ratio']:>8.3f} {v['cosine_dist']:>9.4f} {v['pc1_var']*100:>6.1f}%")
    print(f"\nSummary plot: {summary_path}")


if __name__ == "__main__":
    main()
