"""
Build a PDF comparing synthetic target distributions in CLIP and Latent-CLIP space.

Layout (3×2 when VAE input provided):
  Row 1: Binary (50×A + 50×B) — CLIP-space construction
  Row 2: CLIP-space interpolated (geodesic in embedding space)
  Row 3: VAE-space interpolated (linear in VAE latent space, then projected)
  Col 1: Standard CLIP (ViT-L/14, 768D)
  Col 2: Latent-CLIP (ViT-B-8-512, 512D)

Usage:
  python scripts/plot_target_distributions.py \
    --input output/target_embeddings/target_embeddings.npz \
    --vae_input output/target_embeddings/vae_interp_embeddings.npz
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import PCA


def plot_binary(ax, coords, n_a, title, var_explained):
    ax.scatter(coords[:n_a, 0], coords[:n_a, 1],
               c="dodgerblue", s=50, alpha=0.7, label=f"Anchor A ({n_a})")
    ax.scatter(coords[n_a:, 0], coords[n_a:, 1],
               c="crimson", s=50, alpha=0.7, label=f"Anchor B ({coords.shape[0] - n_a})")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_interpolated(ax, coords, alphas, title, var_explained):
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=alphas, cmap="coolwarm", s=50, alpha=0.7)
    plt.colorbar(sc, ax=ax, label=r"$\alpha$ (A$\to$B)", shrink=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to target_embeddings.npz (CLIP-space constructions)")
    parser.add_argument("--vae_input", type=str, default=None,
                        help="Path to vae_interp_embeddings.npz (VAE-space interpolation)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDF path (default: same dir as input)")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)

    binary_clip = data["binary_clip"]
    interp_clip = data["interp_clip"]
    binary_lclip = data["binary_lclip"]
    interp_lclip = data["interp_lclip"]
    alphas = data["alphas"]
    N = int(data["n_targets"])
    n_a = N // 2
    clip_sim = float(data["clip_anchor_sim"])
    lclip_sim = float(data["lclip_anchor_sim"])

    has_vae = args.vae_input is not None
    if has_vae:
        vdata = np.load(args.vae_input, allow_pickle=True)
        vae_interp_clip = vdata["vae_interp_clip"]
        vae_interp_lclip = vdata["vae_interp_lclip"]
        vae_alphas = vdata["alphas"]

    n_rows = 3 if has_vae else 2

    # PCA for each distribution
    pca_bc = PCA(n_components=2).fit(binary_clip)
    pca_ic = PCA(n_components=2).fit(interp_clip)
    pca_bl = PCA(n_components=2).fit(binary_lclip)
    pca_il = PCA(n_components=2).fit(interp_lclip)

    bc_coords = pca_bc.transform(binary_clip)
    ic_coords = pca_ic.transform(interp_clip)
    bl_coords = pca_bl.transform(binary_lclip)
    il_coords = pca_il.transform(interp_lclip)

    if has_vae:
        pca_vc = PCA(n_components=2).fit(vae_interp_clip)
        pca_vl = PCA(n_components=2).fit(vae_interp_lclip)
        vc_coords = pca_vc.transform(vae_interp_clip)
        vl_coords = pca_vl.transform(vae_interp_lclip)

    # Build PDF
    output_path = args.output or os.path.join(
        os.path.dirname(args.input), "target_distributions.pdf")

    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
        fig.suptitle(
            f"Synthetic Target Distributions — {N} points each\n"
            f"CLIP anchor sim: {clip_sim:.4f}  |  Latent-CLIP anchor sim: {lclip_sim:.4f}",
            fontsize=13, fontweight="bold", y=0.98)

        # Row 1: Binary
        plot_binary(axes[0, 0], bc_coords, n_a,
                    "Binary — Standard CLIP (768D)",
                    pca_bc.explained_variance_ratio_)
        plot_binary(axes[0, 1], bl_coords, n_a,
                    "Binary — Latent-CLIP (512D)",
                    pca_bl.explained_variance_ratio_)

        # Row 2: CLIP-space interpolated
        plot_interpolated(axes[1, 0], ic_coords, alphas,
                          "CLIP-space Interpolated — Standard CLIP (768D)",
                          pca_ic.explained_variance_ratio_)
        plot_interpolated(axes[1, 1], il_coords, alphas,
                          "CLIP-space Interpolated — Latent-CLIP (512D)",
                          pca_il.explained_variance_ratio_)

        # Row 3: VAE-space interpolated
        if has_vae:
            plot_interpolated(axes[2, 0], vc_coords, vae_alphas,
                              "VAE-space Interpolated — Standard CLIP (768D)",
                              pca_vc.explained_variance_ratio_)
            plot_interpolated(axes[2, 1], vl_coords, vae_alphas,
                              "VAE-space Interpolated — Latent-CLIP (512D)",
                              pca_vl.explained_variance_ratio_)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"PDF saved to {output_path}")


if __name__ == "__main__":
    main()
