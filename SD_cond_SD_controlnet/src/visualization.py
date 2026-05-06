"""
visualization.py — Step-level and final visualizations for MLGD-F.

Functions:
    plot_row                  Horizontal strip of PIL images with a title.
    visualize_step            Per-DPS-step 2×(2+num_cond+1) grid + wandb log.
    compare_scribbles_heatmap Pixel-difference heatmap between two scribbles.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.decomposition import PCA

from image_utils import latent_to_pil


def plot_row(images, title, count=5, save_path=None):
    """
    Display a horizontal strip of up to `count` PIL images.

    Args:
        images:    list of PIL images.
        title:     figure suptitle.
        count:     max images to show.
        save_path: if given, saves to file instead of plt.show().
    """
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for i in range(min(count, len(images))):
        axes[i].imshow(images[i])
        axes[i].axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_step(
    sd, architect, sprinter, target_clip_np,
    num_cond=4, save_path=None, pca_fixed=None,
    group_names=None, group_sizes=None,
):
    """
    Generate the per-step 2×(2+num_cond+1) visualization grid.

    Row 0: Regular path  — x_t, pred_x0, blank columns.
    Row 1: MLGD-F path  — x_t, pred_x0, num_cond sprinter samples, CLIP PCA.

    Logs to wandb and optionally saves to disk.

    Args:
        sd:             step data dict from the MLGD-F loop.
        architect:      Architect pipeline (for VAE decode).
        sprinter:       Sprinter pipeline.
        target_clip_np: [N, 768] numpy array of target CLIP embeddings.
        num_cond:       number of Sprinter conditioning samples to show.
        save_path:      optional file path.
        pca_fixed:      optional fitted PCA for consistent projection across steps.
    """
    i = sd["step"]
    # fallback: treat all targets as one group if metadata not provided
    if group_names is None:
        group_names = ["Target"]
    if group_sizes is None:
        group_sizes = [target_clip_np.shape[0]]
    with torch.no_grad():
        img_xt_reg = latent_to_pil(
            sd["latents_step_regular_cpu"].to(architect.device),
            architect.vae, architect.image_processor,
        )
        img_x0_reg = latent_to_pil(
            sd["pred_x0_regular_cpu"].to(architect.device),
            architect.vae, architect.image_processor,
        )
        img_xt_dps = latent_to_pil(
            sd["latents_step_cpu"].to(architect.device),
            architect.vae, architect.image_processor,
        )
        img_x0_dps = latent_to_pil(
            sd["pred_x0_cpu"].to(architect.device),
            architect.vae, architect.image_processor,
        )

        pred_x0_dev = sd["pred_x0_cpu"].to(architect.device)
        px = architect.vae.decode(
            pred_x0_dev.to(architect.vae.dtype)
            / architect.vae.config.scaling_factor
        ).sample
        px_norm = torch.clamp((px + 1.0) / 2.0, 0.0, 1.0)

        sprinter.vae.to(dtype=torch.float16)
        cond_imgs = [
            sprinter(
                prompt="a superrealistic professional photograph of",
                image=px_norm,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pil",
            ).images[0]
            for _ in range(num_cond)
        ]
        sprinter.vae.to(dtype=torch.float32)

        combined = np.vstack([target_clip_np, sd["variation_clip_flat"]])
        if pca_fixed is not None:
            pca_coords = pca_fixed.transform(combined)
            pca_var = pca_fixed.explained_variance_ratio_.sum()
        else:
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(combined)
            pca_var = pca.explained_variance_ratio_.sum()

        target_pca = pca_coords[: target_clip_np.shape[0]]
        gen_pca    = pca_coords[target_clip_np.shape[0]:]

        # split target_pca back into per-group slices using group_sizes
        group_pca_slices = []
        offset = 0
        for sz in group_sizes:
            group_pca_slices.append(target_pca[offset: offset + sz])
            offset += sz

    _COLORS  = ["royalblue", "crimson", "limegreen", "orange",
                "mediumpurple", "gold", "deepskyblue", "hotpink"]
    _MARKERS = ["o", "x", "^", "s", "D", "P", "v", "<"]

    n_cols = 2 + num_cond + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle(
        f"Step {i + 1}  (t={sd['timestep']:.0f})", fontsize=14, fontweight="bold"
    )

    axes[0, 0].imshow(img_xt_reg)
    axes[0, 0].set_title("Regular x_t")
    axes[0, 1].imshow(img_x0_reg)
    axes[0, 1].set_title("Regular pred x_0")
    for j in range(2, n_cols):
        axes[0, j].text(
            0.5, 0.5, "N/A", ha="center", va="center",
            transform=axes[0, j].transAxes,
        )
        axes[0, j].set_facecolor("#f0f0f0")

    axes[1, 0].imshow(img_xt_dps)
    axes[1, 0].set_title(f"MLGD-F x_t  ζ={sd['zeta_i']:.4f}")
    axes[1, 1].imshow(img_x0_dps)
    axes[1, 1].set_title(f"MLGD-F x_0  MMD={sd['mmd_loss']:.6f}")
    for j, ci in enumerate(cond_imgs):
        axes[1, j + 2].imshow(ci)
        axes[1, j + 2].set_title(f"Cond {j + 1}")

    ax = axes[1, n_cols - 1]
    for g_idx, (g_pca, g_name) in enumerate(zip(group_pca_slices, group_names)):
        ax.scatter(g_pca[:, 0], g_pca[:, 1],
                   c=_COLORS[g_idx % len(_COLORS)],
                   marker=_MARKERS[g_idx % len(_MARKERS)],
                   alpha=0.6, s=40, label=g_name)
    ax.scatter(gen_pca[:, 0],  gen_pca[:, 1],
               c="limegreen", alpha=0.8, s=50, marker="x", label="Generated")
    ax.set_title(f"CLIP PCA  Var={pca_var:.1%}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    for row in axes:
        for ax_ in row:
            ax_.axis("off")
    axes[1, n_cols - 1].axis("on")

    plt.tight_layout()
    wandb.log({"step_visualization": wandb.Image(fig)}, step=i + 1, commit=True)

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)


def compare_scribbles_heatmap(mlgd_f_pil, regular_pil, save_path=None):
    """
    Pixel-difference heatmap between the MLGD-F and regular scribbles.

    Renders: [Unguided | MLGD-F | Difference heatmap].

    Args:
        mlgd_f_pil:  PIL image from the MLGD-F path.
        regular_pil: PIL image from the regular (unguided) path.
        save_path:   optional file path.

    Returns:
        matplotlib Figure.
    """
    mlgd_f_np  = np.array(mlgd_f_pil).astype(float)
    regular_np = np.array(regular_pil).astype(float)

    diff      = np.abs(mlgd_f_np - regular_np).mean(axis=2)
    diff_norm = diff / (diff.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(regular_pil); axes[0].set_title("Unguided"); axes[0].axis("off")
    axes[1].imshow(mlgd_f_pil);  axes[1].set_title("MLGD-F");  axes[1].axis("off")
    im = axes[2].imshow(diff_norm, cmap="hot")
    axes[2].set_title("Difference"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle(
        f"Max diff: {diff.max():.1f}  Mean diff: {diff.mean():.2f}", fontsize=12
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig
