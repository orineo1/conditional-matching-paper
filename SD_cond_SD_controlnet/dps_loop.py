"""
dps_loop.py — shared MLGD-F training loop used by run_gender.py and run_age.py.

Both entry-points build `all_clip_embeddings`, `pca_fixed`, and group metadata,
then call `run_mlgdf_loop()` which handles the full denoising loop, eval, and logging.
"""

import copy
import gc
import json
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import wandb

from src.clip_utils import encode_images_clip
from src.generation import (
    compute_pred_x0_direct,
    denoise_step,
    predict_noise_cfg,
    run_dps_step_clip,
)
from src.image_utils import latent_to_pil
from src.metrics import compute_mmd, compute_swd, evaluate_distribution_mmd
from src.models import setup_gradient_checkpointing
from src.visualization import compare_scribbles_heatmap, plot_row, visualize_step


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)


def save_image_list_npy(pil_list, path):
    """Save a list of PIL images as [N, H, W, 3] uint8 numpy array."""
    arr = np.stack([np.array(img) for img in pil_list], axis=0)
    np.save(path, arr)


def compute_clip_softmax(pil_list, clip_model, clip_processor,
                         prompt_a, prompt_b, device):
    """
    Compute two-class CLIP softmax probabilities for each image.

    Returns:
        results: list of {"p_a": float, "p_b": float, "label": "a"|"b"}
        image_features_np: [N, 768] numpy array
    """
    import torch.nn.functional as F

    text_inputs = clip_processor(
        text=[prompt_a, prompt_b], return_tensors="pt", padding=True,
    ).to(device)

    clip_model.to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_img_feats = []
    for start in range(0, len(pil_list), 8):
        batch = pil_list[start:start + 8]
        tensors = torch.cat(
            [TF.to_tensor(img).unsqueeze(0) for img in batch], dim=0
        ).to(device)
        with torch.no_grad():
            feats = encode_images_clip(tensors, clip_model, clip_processor)
        all_img_feats.append(feats)

    image_features = torch.cat(all_img_feats, dim=0)
    logits = (image_features @ text_features.T) * 100.0
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    results = [
        {"p_a": float(p[0]), "p_b": float(p[1]),
         "label": "a" if p[0] > 0.5 else "b"}
        for p in probs
    ]
    clip_model.to("cpu")
    return results, image_features.cpu().numpy()


# ── Main MLGD-F loop ──────────────────────────────────────────────────────────

def run_mlgdf_loop(
    *,
    # Models
    architect,
    sprinter,
    clip_model,
    clip_processor,
    # Data
    scribble_pil,
    all_clip_embeddings,    # [N, 768] tensor on device
    pca_fixed,              # fitted sklearn PCA for visualization
    target_clip_np,         # all_clip_embeddings as numpy
    group_names_list,       # list[str]
    group_colors,           # list[str or tuple]
    group_markers,          # list[str]
    n_groups,               # int
    # Config
    args,                   # argparse.Namespace
    output_dir,
    steps_dir,
    device,
    # Softmax evaluation prompts (two extremes of the target distribution)
    softmax_prompt_a,
    softmax_prompt_b,
    # Extra npy saves: dict[key -> (pil_list, rel_path)]
    extra_npy_saves=None,
):
    """
    Full MLGD-F denoising loop shared by run_gender.py and run_age.py.

    Steps:
      1. SDEdit-style init: encode scribble → latent, noise to start_step
      2. Baseline visualization (before any correction)
      3. DPS loop: noise pred → pred_x0 → VAE → sprinter → CLIP → MMD → grad
      4. Final MMD evaluation for MLGD-F and unguided paths
      5. Save all outputs, logs, npy arrays

    Returns: dict of final metrics
    """
    height, width = 512, 512
    n_steps = args.n_steps
    start_step = args.start_step

    sprinter.vae.to(dtype=torch.float32)
    setup_gradient_checkpointing(architect, sprinter)

    # Encode prompts
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            device=device,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
        )

    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps
    scheduler_regular = copy.deepcopy(architect.scheduler)

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        dtype=prompt_embeds.dtype, device=device,
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        ),
        "time_ids": add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # SDEdit-style init: encode scribble → latent, add noise at start_step
    with torch.no_grad():
        scribble_tensor = TF.to_tensor(scribble_pil).unsqueeze(0).to(device).float()
        scribble_tensor = (scribble_tensor * 2.0) - 1.0
        scribble_latent = architect.vae.encode(scribble_tensor).latent_dist.mean
        scribble_latent = scribble_latent * architect.vae.config.scaling_factor

    t_start = timesteps[start_step]
    alphas_cumprod = architect.scheduler.alphas_cumprod.to(device)
    alpha = alphas_cumprod[t_start.long()].float()
    noise = torch.randn_like(scribble_latent)
    latents = (
        (alpha ** 0.5) * scribble_latent + ((1 - alpha) ** 0.5) * noise
    ).to(torch.float16)
    latents_regular = latents.detach().clone()

    timesteps_to_run = timesteps[start_step:]
    eval_interval = (
        args.eval_interval if args.eval_interval > 0
        else max(1, len(timesteps_to_run) // 5)
    )

    print(f"Ready. Starting from step {start_step}/{n_steps}  "
          f"(t={t_start.item():.0f})", flush=True)
    print(f"Running {len(timesteps_to_run)} MLGD-F steps...", flush=True)

    # Loss function
    if args.loss_fn == "mmd":
        loss_fn = partial(
            compute_mmd,
            bandwidth_scale=args.bandwidth_scale,
            kernel_alpha=args.kernel_alpha,
        )
    else:
        loss_fn = compute_swd

    step_gradients = []
    step_vis_data = []

    # ── Baseline visualization (step 0, before any correction) ────────────────
    with torch.no_grad():
        bl_noise = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents.detach(), timesteps_to_run[0],
            cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        bl_pred_x0 = compute_pred_x0_direct(
            architect.scheduler, bl_noise, timesteps_to_run[0], latents.detach()
        )
        bl_px = architect.vae.decode(
            (bl_pred_x0 / architect.vae.config.scaling_factor).to(architect.vae.dtype)
        ).sample
        bl_px_norm = torch.clamp((bl_px + 1.0) / 2.0, 0.0, 1.0)

        sprinter.vae.to(dtype=torch.float16)
        bl_var_imgs = [
            sprinter(
                prompt=args.sprinter_variation_prompt,
                image=bl_px_norm,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=args.controlnet_scale,
                output_type="pil",
            ).images[0]
            for _ in range(args.n_eval)
        ]
        sprinter.vae.to(dtype=torch.float32)

        var_tensors = torch.cat(
            [TF.to_tensor(img).unsqueeze(0) for img in bl_var_imgs], dim=0
        ).to(device)
        clip_model.to(device)
        bl_clip_flat = encode_images_clip(
            var_tensors, clip_model, clip_processor
        ).cpu().numpy()
        clip_model.to("cpu")

        sd_baseline = {
            "step": 0,
            "timestep": timesteps_to_run[0].item(),
            "mmd_loss": 0.0,
            "zeta_i": 0.0,
            "latents_step_cpu": latents.detach().cpu(),
            "latents_step_regular_cpu": latents.detach().cpu(),
            "pred_x0_cpu": bl_pred_x0.detach().cpu(),
            "pred_x0_regular_cpu": bl_pred_x0.detach().cpu(),
            "variation_clip_flat": bl_clip_flat,
        }

    visualize_step(
        sd_baseline, architect, sprinter, target_clip_np,
        num_cond=4,
        save_path=os.path.join(steps_dir, "step_baseline.png"),
        pca_fixed=pca_fixed,
        n_groups=n_groups,
        group_names=group_names_list,
        group_colors=group_colors,
        group_markers=group_markers,
    )

    # ── MLGD-F denoising loop ──────────────────────────────────────────────────
    dps_start_time = time.time()
    for i, t in enumerate(timesteps_to_run):
        print(f"\n{'='*60}", flush=True)
        print(f"Step {i+1}/{len(timesteps_to_run)}  (t={t})", flush=True)
        print(f"{'='*60}", flush=True)

        latents_step = latents.detach().requires_grad_(True)
        latents_step_regular = latents_regular.detach()

        # Noise prediction (MLGD-F path keeps grad; regular path detached)
        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        with torch.no_grad():
            noise_pred_regular = predict_noise_cfg(
                architect.unet, scheduler_regular,
                latents_step_regular, t, cfg_encoder_states, added_cond_kwargs,
                args.guidance_scale,
            )

        # pred_x0 via direct formula (no scheduler side effects)
        pred_x0 = compute_pred_x0_direct(
            architect.scheduler, noise_pred, t, latents_step
        )
        with torch.no_grad():
            pred_x0_regular = compute_pred_x0_direct(
                scheduler_regular, noise_pred_regular, t, latents_step_regular
            )

        # Decode pred_x0 → pixel space (keep grad for MLGD-F path).
        # VAE must be float32 for gradient to flow — cast it temporarily.
        architect.vae.to(dtype=torch.float32)
        pred_x0_scaled = pred_x0.float() / architect.vae.config.scaling_factor

        def vae_decode_checkpoint(lat):
            return architect.vae.decode(lat).sample

        pixel_x0 = torch.utils.checkpoint.checkpoint(
            vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False
        )
        pixel_x0_norm = (pixel_x0 + 1.0) / 2.0
        pixel_x0_norm = pixel_x0_norm.clamp(0.0, 1.0)

        # CLIP-MMD gradient
        grad, loss_value, zeta_i, loss_norm, vl_clip_flat = run_dps_step_clip(
            latents_step=latents_step,
            pixel_x0_norm=pixel_x0_norm,
            sprinter=sprinter,
            all_clip_embeddings=all_clip_embeddings,
            num_variations=args.num_variations,
            variation_batch_size=1,
            base_zeta_prime=args.base_zeta,
            clip_model=clip_model,
            clip_processor=clip_processor,
            vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=args.sprinter_variation_prompt,
            loss_fn=loss_fn,
            loss_scale=args.loss_scale,
        )

        grad_norm = grad.norm().item()
        zeta_val = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i
        print(f"  loss={loss_value.item():.6f}  "
              f"ζi={zeta_val:.4f}  ∥∇∥={grad_norm:.6f}", flush=True)

        # Restore architect VAE to float16 now that gradient has been computed
        architect.vae.to(dtype=torch.float16)

        if torch.isnan(grad).any():
            print(f"  WARNING: NaN in gradient at step {i} — skipping correction",
                  flush=True)
            correction = torch.zeros_like(latents_step)
        else:
            correction = -zeta_i * grad

        step_gradients.append({
            "step": i + 1,
            "timestep": t.item(),
            "gradient_norm": grad_norm,
            "loss": loss_value.item(),
            "zeta_i": zeta_val,
            "loss_norm": loss_norm.item(),
            "correction_norm": zeta_val * grad_norm,
        })

        wandb_log = {
            "step": i + 1,
            "loss": loss_value.item(),
            "gradient_norm": grad_norm,
            "zeta": zeta_val,
            "correction_norm": zeta_val * grad_norm,
        }

        # Intermediate MMD evaluation
        if i % eval_interval == 0:
            unguided_mmd, _, _ = evaluate_distribution_mmd(
                pred_x0_regular.detach(), architect.vae, architect.image_processor,
                sprinter, clip_model, clip_processor,
                all_clip_embeddings, args.sprinter_eval_prompt,
                n_eval=args.n_eval, device=device,
            )
            wandb_log["intermediate/unguided_mmd"] = unguided_mmd
            wandb_log["intermediate/mlgdf_loss"] = loss_value.item()
            wandb_log["intermediate/loss_delta"] = loss_value.item() - unguided_mmd
            print(f"  [eval] mlgdf={loss_value.item():.6f}  "
                  f"unguided={unguided_mmd:.6f}  "
                  f"delta={loss_value.item()-unguided_mmd:.6f}", flush=True)

        wandb.log(wandb_log, commit=False)

        with torch.no_grad():
            sd = {
                "step": i + 1,
                "timestep": t.item(),
                "mmd_loss": loss_value.item(),
                "zeta_i": zeta_val,
                "latents_step_cpu": latents_step.detach().cpu(),
                "latents_step_regular_cpu": latents_step_regular.detach().cpu(),
                "pred_x0_cpu": pred_x0.detach().cpu(),
                "pred_x0_regular_cpu": pred_x0_regular.detach().cpu(),
                "variation_clip_flat": vl_clip_flat,
            }
            step_vis_data.append(sd)

        visualize_step(
            sd, architect, sprinter, target_clip_np,
            num_cond=5,
            save_path=os.path.join(steps_dir, f"step_{i:03d}.png"),
            pca_fixed=pca_fixed,
            n_groups=n_groups,
            group_names=group_names_list,
            group_colors=group_colors,
            group_markers=group_markers,
        )

        latents = denoise_step(
            architect.scheduler, noise_pred, t, latents_step,
            correction=correction,
        )
        with torch.no_grad():
            latents_regular = denoise_step(
                scheduler_regular, noise_pred_regular, t, latents_step_regular,
            )

        del grad, loss_value, loss_norm, zeta_i, correction
        del pixel_x0, pixel_x0_norm, pred_x0, pred_x0_regular
        del latents_step_regular, noise_pred_regular
        gc.collect()
        torch.cuda.empty_cache()

    del latents_step, noise_pred
    torch.cuda.empty_cache()
    print(f"\nMLGD-F complete. {len(step_vis_data)} steps stored.", flush=True)

    # ── Final MMD evaluation ───────────────────────────────────────────────────
    print("Computing final MMD (unguided)...", flush=True)
    unguided_mmd, unguided_eval_photos, _ = evaluate_distribution_mmd(
        latents_regular, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=args.n_eval, device=device,
    )

    print("Computing final MMD (MLGD-F)...", flush=True)
    mlgdf_mmd, mlgdf_eval_photos, _ = evaluate_distribution_mmd(
        latents, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=args.n_eval, device=device,
    )

    print(f"Unguided MMD : {unguided_mmd:.6f}", flush=True)
    print(f"MLGD-F MMD   : {mlgdf_mmd:.6f}", flush=True)
    print(f"Delta (lower is better for MLGD-F): "
          f"{unguided_mmd - mlgdf_mmd:.6f}", flush=True)

    # ── Save final scribbles & heatmap ─────────────────────────────────────────
    with torch.no_grad():
        final_mlgdf_pil = latent_to_pil(
            latents, architect.vae, architect.image_processor
        )
        final_unguided_pil = latent_to_pil(
            latents_regular, architect.vae, architect.image_processor
        )

    final_mlgdf_pil.save(os.path.join(output_dir, "final_scribble_mlgdf.png"))
    final_unguided_pil.save(os.path.join(output_dir, "final_scribble_unguided.png"))

    heatmap_path = os.path.join(output_dir, "scribble_heatmap.png")
    compare_scribbles_heatmap(final_mlgdf_pil, final_unguided_pil,
                               save_path=heatmap_path)

    plot_row(unguided_eval_photos,
             f"Unguided final photos  (MMD={unguided_mmd:.4f})",
             save_path=os.path.join(output_dir, "final_photos_unguided.png"))
    plot_row(mlgdf_eval_photos,
             f"MLGD-F final photos    (MMD={mlgdf_mmd:.4f})",
             save_path=os.path.join(output_dir, "final_photos_mlgdf.png"))

    # Save individual eval photos
    for folder, photos in [("photos_unguided", unguided_eval_photos),
                            ("photos_mlgdf", mlgdf_eval_photos)]:
        photo_dir = os.path.join(output_dir, folder)
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(photos):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))

    # ── CLIP softmax probabilities ─────────────────────────────────────────────
    print("Computing CLIP softmax probabilities...", flush=True)
    clip_model.to(device)
    mlgdf_softmax, mlgdf_clip_embs = compute_clip_softmax(
        mlgdf_eval_photos, clip_model, clip_processor,
        softmax_prompt_a, softmax_prompt_b, device,
    )
    unguided_softmax, unguided_clip_embs = compute_clip_softmax(
        unguided_eval_photos, clip_model, clip_processor,
        softmax_prompt_a, softmax_prompt_b, device,
    )
    clip_model.to("cpu")
    print("CLIP softmax done.", flush=True)

    # ── SWD to target ──────────────────────────────────────────────────────────
    print("Computing SWD to target...", flush=True)
    target_clip_t = torch.from_numpy(target_clip_np).float()
    with torch.no_grad():
        swd_mlgdf = compute_swd(
            torch.from_numpy(mlgdf_clip_embs).float(), target_clip_t
        ).item()
        swd_unguided = compute_swd(
            torch.from_numpy(unguided_clip_embs).float(), target_clip_t
        ).item()
    print(f"  SWD MLGD-F  : {swd_mlgdf:.6f}", flush=True)
    print(f"  SWD Unguided: {swd_unguided:.6f}", flush=True)

    # ── wandb final logs ───────────────────────────────────────────────────────
    wandb.log({
        "final_mlgdf_mmd": mlgdf_mmd,
        "final_unguided_mmd": unguided_mmd,
        "mmd_delta": unguided_mmd - mlgdf_mmd,
        "mmd_relative_improvement": (unguided_mmd - mlgdf_mmd) / (unguided_mmd + 1e-8),
        "final_scribble_mlgdf": wandb.Image(final_mlgdf_pil),
        "final_scribble_unguided": wandb.Image(final_unguided_pil),
        "mlgdf_eval_photos": [wandb.Image(p) for p in mlgdf_eval_photos],
        "unguided_eval_photos": [wandb.Image(p) for p in unguided_eval_photos],
        "scribble_heatmap": wandb.Image(heatmap_path),
    })
    wandb.summary["final_mlgdf_mmd"] = mlgdf_mmd
    wandb.summary["final_unguided_mmd"] = unguided_mmd
    wandb.summary["mmd_delta"] = unguided_mmd - mlgdf_mmd
    wandb.summary["final_grad_norm"] = step_gradients[-1]["gradient_norm"]

    # ── Save npy arrays ────────────────────────────────────────────────────────
    npy_dir = os.path.join(output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    save_image_list_npy(mlgdf_eval_photos,
                        os.path.join(npy_dir, "photos_mlgdf.npy"))
    save_image_list_npy(unguided_eval_photos,
                        os.path.join(npy_dir, "photos_unguided.npy"))
    save_image_list_npy([final_mlgdf_pil],
                        os.path.join(npy_dir, "final_scribble_mlgdf.npy"))
    save_image_list_npy([final_unguided_pil],
                        os.path.join(npy_dir, "final_scribble_unguided.npy"))

    if extra_npy_saves:
        for key, (pil_list, rel_path) in extra_npy_saves.items():
            save_image_list_npy(pil_list, os.path.join(npy_dir, rel_path))

    np.save(os.path.join(npy_dir, "clip_mlgdf.npy"), mlgdf_clip_embs)
    np.save(os.path.join(npy_dir, "clip_unguided.npy"), unguided_clip_embs)

    # ── Gender/label stats helper ──────────────────────────────────────────────
    def label_stats(softmax_list):
        a_items = [x for x in softmax_list if x["label"] == "a"]
        b_items = [x for x in softmax_list if x["label"] == "b"]
        return {
            "n_a": len(a_items),
            "n_b": len(b_items),
            "mean_conf_a": float(np.mean([x["p_a"] for x in a_items])) if a_items else None,
            "mean_conf_b": float(np.mean([x["p_b"] for x in b_items])) if b_items else None,
            "per_image": softmax_list,
        }

    mlgdf_stats = label_stats(mlgdf_softmax)
    unguided_stats = label_stats(unguided_softmax)

    optimization_time_sec = time.time() - dps_start_time

    # ── metrics.json ──────────────────────────────────────────────────────────
    metrics = {
        "args": vars(args),
        "steps": step_gradients,
        "final_mlgdf_mmd": mlgdf_mmd,
        "final_unguided_mmd": unguided_mmd,
        "mmd_delta": unguided_mmd - mlgdf_mmd,
        "final_mlgdf_swd": swd_mlgdf,
        "final_unguided_swd": swd_unguided,
        "swd_delta": swd_unguided - swd_mlgdf,
        "optimization_time_sec": optimization_time_sec,
        "mlgdf_label_stats": mlgdf_stats,
        "unguided_label_stats": unguided_stats,
        "npy": {
            "photos_mlgdf": "npy/photos_mlgdf.npy",
            "photos_unguided": "npy/photos_unguided.npy",
            "final_scribble_mlgdf": "npy/final_scribble_mlgdf.npy",
            "final_scribble_unguided": "npy/final_scribble_unguided.npy",
            "clip_mlgdf": "npy/clip_mlgdf.npy",
            "clip_unguided": "npy/clip_unguided.npy",
        },
    }
    if extra_npy_saves:
        for key, (_, rel_path) in extra_npy_saves.items():
            metrics["npy"][key] = f"npy/{rel_path}"

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics.json saved.", flush=True)
    print(f"Optimization time: {optimization_time_sec / 60:.1f} min", flush=True)

    wandb.finish()
    print(f"All outputs saved to {output_dir}", flush=True)
    return metrics
