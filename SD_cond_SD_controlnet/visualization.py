import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from image_utils import latent_to_pil

def plot_row(images, title, count=5):
    fig, axes = plt.subplots(1, count, figsize=(4*count, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i in range(min(count, len(images))):
        axes[i].imshow(images[i]); axes[i].axis('off')
    plt.tight_layout(); plt.show()

def visualize_step(sd, architect, sprinter, all_latents_flat, num_cond=4):
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

        combined = np.vstack([all_latents_flat, sd['variation_latents_flat']])
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(combined)
        target_pca = pca_coords[:all_latents_flat.shape[0]]
        gen_pca    = pca_coords[all_latents_flat.shape[0]:]

    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    fig.suptitle(f"Step {i+1}  (t={sd['timestep']:.0f})", fontsize=14, fontweight='bold')

    axes[0,0].imshow(img_xt_reg); axes[0,0].set_title("Regular x_t")
    axes[0,1].imshow(img_x0_reg); axes[0,1].set_title("Regular pred x_0")
    for j in range(2, 7):
        axes[0,j].text(0.5,0.5,'N/A',ha='center',va='center',transform=axes[0,j].transAxes)
        axes[0,j].set_facecolor('#f0f0f0')

    axes[1,0].imshow(img_xt_dps); axes[1,0].set_title(f"DPS x_t  ζ={sd['zeta_i']:.4f}")
    axes[1,1].imshow(img_x0_dps); axes[1,1].set_title(f"DPS x_0  MMD={sd['mmd_loss']:.6f}")
    for j, ci in enumerate(cond_imgs):
        axes[1,j+2].imshow(ci); axes[1,j+2].set_title(f"Cond {j+1}")

    axes[1,6].scatter(target_pca[:,0], target_pca[:,1], c='blue', alpha=0.5, s=20, label='Target')
    axes[1,6].scatter(gen_pca[:,0],    gen_pca[:,1],    c='red',  alpha=0.6, s=30, marker='x', label='Generated')
    axes[1,6].set_title(f"PCA  Var={pca.explained_variance_ratio_.sum():.1%}")
    axes[1,6].legend(fontsize=8); axes[1,6].grid(True, alpha=0.3)

    for row in axes:
        for ax in row: ax.axis("off")
    axes[1,6].axis("on")
    plt.tight_layout(); plt.show()
```

---

**`main.ipynb`** — clean cell-by-cell orchestration:
```
Cell 1: Imports & device
Cell 2: Load models (models.py)
Cell 3: Build base image + Sobel (image_utils.py)
Cell 4: Generate target distributions + PCA plot (generation.py, visualization.py)
Cell 5: Config (n_steps, zeta, prompts, etc.)
Cell 6: Encode prompts, prepare latents
Cell 7: DPS denoising loop
Cell 8: Final results & summary plots