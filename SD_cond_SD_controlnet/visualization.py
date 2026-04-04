import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from image_utils import latent_to_pil
import wandb
import matplotlib.cm as cm

def plot_row(images, title, count=5, save_path=None):
    fig, axes = plt.subplots(1, count, figsize=(4*count, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(min(count, len(images))):
        axes[i].imshow(images[i]); axes[i].axis('off')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight'); plt.close(fig)
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from image_utils import latent_to_pil
import wandb

def plot_row(images, title, count=5, save_path=None):
    fig, axes = plt.subplots(1, count, figsize=(4*count, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(min(count, len(images))):
        axes[i].imshow(images[i]); axes[i].axis('off')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight'); plt.close(fig)
    else:
        plt.show()

def visualize_step(sd, architect, sprinter, target_clip_np, num_cond=4,
                   save_path=None, pca_fixed=None,
                   n_groups=4,
                   group_names=("woman", "man", "andro_masc", "mostly_masc"),
                   group_colors=("#4169E1", "#9B59B6", "#E67E22", "#E74C3C"),
                   group_markers=("o", "D", "^", "s")):
    i = sd['step']
    with torch.no_grad():
        img_xt_reg  = latent_to_pil(sd['latents_step_regular_cpu'].to(architect.device), architect.vae, architect.image_processor)
        img_x0_reg  = latent_to_pil(sd['pred_x0_regular_cpu'].to(architect.device),      architect.vae, architect.image_processor)
        img_xt_dps  = latent_to_pil(sd['latents_step_cpu'].to(architect.device),          architect.vae, architect.image_processor)
        img_x0_dps  = latent_to_pil(sd['pred_x0_cpu'].to(architect.device),               architect.vae, architect.image_processor)

        pred_x0_dev = sd['pred_x0_cpu'].to(architect.device)
        px = architect.vae.decode(pred_x0_dev.to(architect.vae.dtype) / architect.vae.config.scaling_factor).sample
        px_norm = torch.clamp((px + 1.0) / 2.0, 0.0, 1.0)

        sprinter.vae.to(dtype=torch.float16)
        cond_imgs = [
            sprinter(prompt="a superrealistic professional photograph of", image=px_norm,
                     num_inference_steps=2, guidance_scale=0.0,
                     controlnet_conditioning_scale=0.8, output_type="pil").images[0]
            for _ in range(num_cond)
        ]
        sprinter.vae.to(dtype=torch.float32)

        combined = np.vstack([target_clip_np, sd['variation_clip_flat']])
        if pca_fixed is not None:
            pca_coords = pca_fixed.transform(combined)
            pca_var = pca_fixed.explained_variance_ratio_.sum()
        else:
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(combined)
            pca_var = pca.explained_variance_ratio_.sum()

        target_pca = pca_coords[:target_clip_np.shape[0]]
        gen_pca    = pca_coords[target_clip_np.shape[0]:]
        n_per_group = target_clip_np.shape[0] // n_groups

    n_cols = 2 + num_cond + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle(f"Step {i+1}  (t={sd['timestep']:.0f})", fontsize=14, fontweight='bold')

    axes[0,0].imshow(img_xt_reg); axes[0,0].set_title("Regular x_t")
    axes[0,1].imshow(img_x0_reg); axes[0,1].set_title("Regular pred x_0")
    for j in range(2, n_cols):
        axes[0,j].text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=axes[0,j].transAxes)
        axes[0,j].set_facecolor('#f0f0f0')

    axes[1,0].imshow(img_xt_dps); axes[1,0].set_title(f"DPS x_t  ζ={sd['zeta_i']:.4f}")
    axes[1,1].imshow(img_x0_dps); axes[1,1].set_title(f"DPS x_0  MMD={sd['mmd_loss']:.6f}")
    for j, ci in enumerate(cond_imgs):
        axes[1, j+2].imshow(ci); axes[1, j+2].set_title(f"Cond {j+1}")

    # ── PCA scatter ───────────────────────────────────────────────────────────
    ax = axes[1, n_cols - 1]
    for g in range(n_groups):
        start = g * n_per_group
        end   = start + n_per_group
        ax.scatter(target_pca[start:end, 0], target_pca[start:end, 1],
                   c=group_colors[g], marker=group_markers[g],
                   alpha=0.7, s=50, label=group_names[g])
        cx = target_pca[start:end, 0].mean()
        cy = target_pca[start:end, 1].mean()
        ax.scatter(cx, cy, c=group_colors[g], marker='*', s=180,
                   edgecolors='black', linewidths=0.6, zorder=5)

    ax.scatter(gen_pca[:, 0], gen_pca[:, 1],
               c='limegreen', alpha=0.9, s=60, marker='x',
               linewidths=1.5, label='Generated', zorder=6)

    ax.set_title(f"CLIP PCA  Var={pca_var:.1%}")
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    for row in axes:
        for ax_ in row:
            ax_.axis("off")
    axes[1, n_cols - 1].axis("on")

    plt.tight_layout()
    wandb.log({"step_visualization": wandb.Image(fig)}, step=i+1, commit=True)
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def compare_scribbles_heatmap(dps_pil, regular_pil, save_path=None):
    dps_np     = np.array(dps_pil).astype(float)
    regular_np = np.array(regular_pil).astype(float)

    diff       = np.abs(dps_np - regular_np).mean(axis=2)  # [H, W]
    diff_norm  = diff / (diff.max() + 1e-8)                 # normalize to [0,1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(regular_pil);                   axes[0].set_title("Unguided");   axes[0].axis("off")
    axes[1].imshow(dps_pil);                       axes[1].set_title("DPS");        axes[1].axis("off")
    im = axes[2].imshow(diff_norm, cmap="hot");    axes[2].set_title("Difference"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle(f"Max diff: {diff.max():.1f}  Mean diff: {diff.mean():.2f}", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig