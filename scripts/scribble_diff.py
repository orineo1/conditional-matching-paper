"""Heatmap of pixel differences between guided and unguided scribbles.

Usage:
    python scripts/scribble_diff.py \
        --guided scripts/assets/zeta5_final_guided.png \
        --unguided scripts/assets/zeta5_final_unguided.png \
        --output_dir output/scribble_diff
"""

import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Scribble difference heatmap")
    parser.add_argument("--guided", type=str, required=True, help="Guided scribble PNG")
    parser.add_argument("--unguided", type=str, required=True, help="Unguided scribble PNG")
    parser.add_argument("--output_dir", type=str, default="output/scribble_diff")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    guided = np.array(Image.open(args.guided).convert("RGB")).astype(np.float32) / 255.0
    unguided = np.array(Image.open(args.unguided).convert("RGB")).astype(np.float32) / 255.0

    # Resize to match if needed
    if guided.shape != unguided.shape:
        h, w = min(guided.shape[0], unguided.shape[0]), min(guided.shape[1], unguided.shape[1])
        guided = np.array(Image.open(args.guided).convert("RGB").resize((w, h))) / 255.0
        unguided = np.array(Image.open(args.unguided).convert("RGB").resize((w, h))) / 255.0

    # Per-pixel L2 difference across channels
    diff = np.sqrt(((guided - unguided) ** 2).sum(axis=2))  # [H, W]
    diff_smooth = gaussian_filter(diff, sigma=args.sigma)

    # Save raw
    np.save(os.path.join(args.output_dir, "scribble_diff.npy"), diff)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(unguided)
    axes[0].set_title("Unguided Scribble")
    axes[0].axis("off")

    axes[1].imshow(guided)
    axes[1].set_title("DPS Guided Scribble")
    axes[1].axis("off")

    im = axes[2].imshow(diff_smooth, cmap="hot")
    axes[2].set_title(f"Difference (σ={args.sigma})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(unguided)
    axes[3].imshow(diff_smooth, cmap="hot", alpha=0.6)
    axes[3].set_title("Overlay on Unguided")
    axes[3].axis("off")

    fig.suptitle("Scribble Difference: Guided vs Unguided", fontsize=14, fontweight="bold")
    fig.tight_layout()

    png_path = os.path.join(args.output_dir, "scribble_diff.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.4f}")
    print(f"Saved to: {png_path}")


if __name__ == "__main__":
    main()
