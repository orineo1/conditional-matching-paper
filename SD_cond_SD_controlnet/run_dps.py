"""
DPS CLIP-MMD Pipeline — Script version of main.ipynb.

Runs the full Diffusion Posterior Sampling loop with CLIP-space MMD guidance
using SDXL Turbo architect + ControlNet sprinter. Optionally loads LoRA weights
on the architect. Logs to wandb and saves outputs to disk.
"""

import argparse
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Ensure SD_cond_SD_controlnet/ is on the path (for cluster runs from repo root)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
    run_dps_step_clip,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd
from models import load_models, setup_gradient_checkpointing
from visualization import plot_row, visualize_step


def parse_args():
    p = argparse.ArgumentParser(description="DPS CLIP-MMD Pipeline")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/dps_run")
    p.add_argument("--lora_path", type=str, default=None,
                   help="Path to LoRA checkpoint for architect (None = no LoRA)")
    p.add_argument("--wandb_project", type=str, default="conditional-flow")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--num_variations", type=int, default=10)
    p.add_argument("--base_zeta", type=float, default=0.2)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ── Output dirs ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    steps_dir = os.path.join(args.output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    # ── Seed ───────────────────────────────────────────────────────────────────
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
    )

    # ── 1. Load models ─────────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    architect, sprinter = load_models(device, architect_lora_path=args.lora_path)
    clip_model, clip_processor = load_clip_model(device)
    print("Models loaded.", flush=True)

    # ── 2. Base image + Sobel ──────────────────────────────────────────────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    base_image_pil.save(os.path.join(args.output_dir, "base_image.png"))
    sobel_cond_pil.save(os.path.join(args.output_dir, "sobel_conditioning.png"))

    # ── 3. Generate targets ────────────────────────────────────────────────────
    N_TARGETS = 20  # always 20 targets for good distribution coverage
    n_total = N_TARGETS // 2
    print(f"Generating {N_TARGETS} target images ({n_total} man + {n_total} woman)...", flush=True)

    with torch.no_grad():
        man_images, man_latents = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, n_total, batch_size=2, cn_scale=args.controlnet_scale,
        )
        woman_images, woman_latents = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a woman, studio lighting",
            sobel_cond_pil, n_total, batch_size=2, cn_scale=args.controlnet_scale,
        )

    plot_row(man_images, "Man Portrait Samples",
             save_path=os.path.join(args.output_dir, "target_samples_man.png"))
    plot_row(woman_images, "Woman Portrait Samples",
             save_path=os.path.join(args.output_dir, "target_samples_woman.png"))

    # ── 4. Encode targets to CLIP ──────────────────────────────────────────────
    with torch.no_grad():
        man_clip_embs = encode_images_clip(
            pil_images_to_tensor(man_images, device), clip_model, clip_processor)
        woman_clip_embs = encode_images_clip(
            pil_images_to_tensor(woman_images, device), clip_model, clip_processor)
    all_clip_embeddings = torch.cat([man_clip_embs, woman_clip_embs], dim=0)
    print(f"Target CLIP embeddings shape: {all_clip_embeddings.shape}", flush=True)

    # Sanity checks
    norms = all_clip_embeddings.norm(dim=-1)
    intra_sim = (man_clip_embs @ man_clip_embs.T).mean().item()
    inter_sim = (man_clip_embs @ woman_clip_embs.T).mean().item()
    print(f"  Norms min/max: {norms.min():.4f} / {norms.max():.4f}")
    print(f"  Intra-class sim (man): {intra_sim:.4f}")
    print(f"  Inter-class sim: {inter_sim:.4f}")

    # Target CLIP PCA
    from sklearn.decomposition import PCA
    _pca = PCA(n_components=2)
    _coords = _pca.fit_transform(all_clip_embeddings.cpu().numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(_coords[:n_total, 0], _coords[:n_total, 1], c='dodgerblue', label='Man', alpha=0.7)
    ax.scatter(_coords[n_total:, 0], _coords[n_total:, 1], c='crimson', label='Woman', alpha=0.7)
    ax.set_title("PCA of Target CLIP Embeddings"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "target_clip_pca.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)
    del _pca, _coords

    # Free target images from GPU/CPU, keep only CLIP embeddings
    del man_images, woman_images, man_clip_embs, woman_clip_embs
    del man_latents, woman_latents

    # Move CLIP to CPU — it'll be moved back per-step inside run_dps_step_clip
    clip_model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
    print("CLIP model offloaded to CPU to save VRAM.", flush=True)

    # ── 5. Initial scribble ────────────────────────────────────────────────────
    prompt = "rough pencil scribble outline, loose sketch, minimal line art"
    negative_prompt = "detailed, realistic, photograph, complex, colored, shading"
    height, width = 512, 512
    n_steps = args.n_steps
    num_variations = args.num_variations
    variation_batch_size = 1
    base_zeta_prime = args.base_zeta
    guidance_scale = args.guidance_scale

    print(f"Generating seed scribble (prompt: {prompt})...", flush=True)
    scribble_latents = architect(
        prompt=prompt, negative_prompt=negative_prompt,
        num_inference_steps=50, guidance_scale=guidance_scale,
        output_type="latent",
    ).images

    with torch.no_grad():
        pixels = architect.vae.decode(scribble_latents.to(torch.float32) / 0.13025).sample
        control_image = architect.image_processor.postprocess(pixels, output_type="pil")[0]
    control_image.save(os.path.join(args.output_dir, "initial_scribble.png"))

    # ── 6. Prepare DPS ─────────────────────────────────────────────────────────
    sprinter.vae.to(dtype=torch.float32)
    setup_gradient_checkpointing(architect, sprinter)

    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
        )

    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps

    latents = architect.prepare_latents(
        1, architect.unet.config.in_channels,
        height, width, prompt_embeds.dtype, device, None,
    )
    latents_regular = latents.detach().clone()

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device)
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids": add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    step_gradients = []
    target_clip_np = all_clip_embeddings.cpu().numpy()

    print(f"Ready for DPS loop ({n_steps} steps, {num_variations} variations/step).", flush=True)

    # ── 7. DPS loop ────────────────────────────────────────────────────────────
    for i, t in enumerate(timesteps):
        print(f"\n{'='*60}", flush=True)
        print(f"Step {i+1}/{n_steps}  (t={t})", flush=True)

        latents_step = latents.detach().requires_grad_(True)
        latents_step_regular = latents_regular.detach()

        # A. Noise prediction
        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, guidance_scale,
        )
        with torch.no_grad():
            noise_pred_regular = predict_noise_cfg(
                architect.unet, architect.scheduler,
                latents_step_regular, t, cfg_encoder_states, added_cond_kwargs, guidance_scale,
            )

        # B. pred_x0 (save/restore scheduler step_index — compute_pred_x0 calls
        #    scheduler.step() internally which advances the counter)
        saved_step_index = architect.scheduler.step_index
        pred_x0 = compute_pred_x0(architect.scheduler, noise_pred, t, latents_step)
        architect.scheduler._step_index = saved_step_index
        with torch.no_grad():
            pred_x0_regular = compute_pred_x0(
                architect.scheduler, noise_pred_regular, t, latents_step_regular)
            architect.scheduler._step_index = saved_step_index

        # C. Decode pred_x0 -> pixel space (keep grad)
        pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

        def vae_decode_checkpoint(lat):
            return architect.vae.decode(lat.to(architect.vae.dtype)).sample

        pixel_x0 = torch.utils.checkpoint.checkpoint(
            vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
        pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

        # D. CLIP-MMD + gradient (move CLIP to GPU for this step)
        clip_model.to(device)
        grad, mmd_loss, zeta_i, loss_norm, vl_clip_flat = run_dps_step_clip(
            latents=latents,
            latents_step=latents_step,
            noise_pred=noise_pred,
            pixel_x0_norm=pixel_x0_norm,
            sprinter=sprinter,
            all_clip_embeddings=all_clip_embeddings,
            num_variations=num_variations,
            variation_batch_size=variation_batch_size,
            base_zeta_prime=base_zeta_prime,
            clip_model=clip_model,
            clip_processor=clip_processor,
            vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
        )
        clip_model.to("cpu"); torch.cuda.empty_cache()

        grad_norm = grad.norm().item()
        zeta_val = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i
        print(f"  MMD={mmd_loss.item():.6f}  zeta={zeta_val:.4f}  "
              f"grad_norm={grad_norm:.6f}  correction={zeta_val * grad_norm:.6f}", flush=True)

        # NaN guard
        if torch.isnan(grad).any():
            print(f"  WARNING: NaN in gradient at step {i} — skipping correction", flush=True)
            correction = torch.zeros_like(latents_step)
        else:
            correction = -zeta_i * grad

        step_data = {
            'step': i,
            'timestep': t.item(),
            'gradient_norm': grad_norm,
            'mmd_loss': mmd_loss.item(),
            'zeta_i': zeta_val,
            'loss_norm': loss_norm.item(),
            'correction_norm': zeta_val * grad_norm,
        }
        step_gradients.append(step_data)

        # Log to wandb
        wandb.log({
            "step": i,
            "mmd_loss": mmd_loss.item(),
            "gradient_norm": grad_norm,
            "zeta": zeta_val,
            "correction_norm": zeta_val * grad_norm,
        })

        # E. Visualize & save
        with torch.no_grad():
            sd = {
                'step': i,
                'timestep': t.item(),
                'mmd_loss': mmd_loss.item(),
                'zeta_i': zeta_val,
                'latents_step_cpu': latents_step.detach().cpu(),
                'latents_step_regular_cpu': latents_step_regular.detach().cpu(),
                'pred_x0_cpu': pred_x0.detach().cpu(),
                'pred_x0_regular_cpu': pred_x0_regular.detach().cpu(),
                'variation_clip_flat': vl_clip_flat,
            }

        step_save_path = os.path.join(steps_dir, f"step_{i:03d}.png")
        visualize_step(sd, architect, sprinter, target_clip_np, num_cond=2, save_path=step_save_path)
        wandb.log({"step_visualization": wandb.Image(step_save_path)})

        # F. Scheduler step (only one call should advance step_index)
        latents = denoise_step(architect.scheduler, noise_pred, t, latents_step, correction=correction)
        saved_step_index = architect.scheduler.step_index
        with torch.no_grad():
            latents_regular = denoise_step(
                architect.scheduler, noise_pred_regular, t, latents_step_regular)
        architect.scheduler._step_index = saved_step_index

        # Cleanup
        del grad, mmd_loss, loss_norm, zeta_i, correction
        del pixel_x0, pixel_x0_norm, pred_x0, pred_x0_regular
        del latents_step_regular, noise_pred_regular
        gc.collect(); torch.cuda.empty_cache()

    del latents_step, noise_pred
    torch.cuda.empty_cache()
    print(f"\nDPS Complete! {len(step_gradients)} steps.", flush=True)

    # ── 8. Final results ───────────────────────────────────────────────────────
    with torch.no_grad():
        final_image = latent_to_pil(latents.cpu().to(device), architect.vae, architect.image_processor)

    # Final comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(base_image_pil); axes[0].set_title("Input Shape"); axes[0].axis('off')
    axes[1].imshow(final_image);    axes[1].set_title("DPS Output");  axes[1].axis('off')
    fig.suptitle("CLIP-MMD DPS Result", fontsize=16, fontweight='bold')
    final_path = os.path.join(args.output_dir, "final_image.png")
    fig.savefig(final_path, dpi=100, bbox_inches='tight'); plt.close(fig)

    # Training curves
    steps_list = [d['step'] for d in step_gradients]
    mmd_vals   = [d['mmd_loss'] for d in step_gradients]
    grad_norms = [d['gradient_norm'] for d in step_gradients]
    zetas      = [d['zeta_i'] for d in step_gradients]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CLIP-MMD DPS Training Curves", fontsize=14, fontweight='bold')
    axes[0].plot(steps_list, mmd_vals, color='royalblue');   axes[0].set_title("MMD Loss");      axes[0].set_xlabel("Step"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps_list, grad_norms, color='crimson');   axes[1].set_title("Gradient Norm"); axes[1].set_xlabel("Step"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps_list, zetas, color='seagreen');       axes[2].set_title("Zeta");          axes[2].set_xlabel("Step"); axes[2].grid(True, alpha=0.3)
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=100, bbox_inches='tight'); plt.close(fig)

    # Metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "args": vars(args),
            "steps": step_gradients,
            "final_mmd": step_gradients[-1]["mmd_loss"],
        }, f, indent=2)

    # ── 9. wandb final logs ────────────────────────────────────────────────────
    wandb.log({
        "final_image": wandb.Image(final_path),
        "training_curves": wandb.Image(curves_path),
    })
    wandb.summary["final_mmd"] = step_gradients[-1]["mmd_loss"]
    wandb.summary["final_grad_norm"] = step_gradients[-1]["gradient_norm"]
    wandb.finish()

    print(f"All outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
