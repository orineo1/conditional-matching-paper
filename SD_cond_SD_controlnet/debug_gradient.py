"""
Debug DPS Gradient Flow — Diagnostic script.

Runs 1 face, 1 strength, ~10 steps and logs detailed gradient diagnostics
at each step to understand why DPS CLIP-MMD guidance doesn't visually steer output.

Diagnostics:
  1. Magnitudes: correction vs latent vs scheduler step
  2. Gradient chain decomposition: where does attenuation happen?
  3. Finite-difference verification: is autograd correct?
  4. Direction consistency: do gradients cancel across steps?
  5. Zeta ablation: at what zeta does output actually diverge?
"""

import argparse
import copy
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0_direct,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd
from models import load_models, setup_gradient_checkpointing


def parse_args():
    p = argparse.ArgumentParser(description="Debug DPS Gradient Flow")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/debug_gradient")
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="conditional-flow")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--num_variations", type=int, default=20)
    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--base_zeta", type=float, default=0.2)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str,
                   default="rough pencil scribble outline, loose sketch, minimal line art")
    p.add_argument("--negative_prompt", type=str,
                   default="detailed, realistic, photograph, complex, colored, shading")
    p.add_argument("--edge_method", type=str, default="hed_scribble",
                   choices=["sobel", "hed_scribble"])
    p.add_argument("--detailed_step", type=int, default=5,
                   help="Which step to do full chain decomposition (0-indexed)")
    p.add_argument("--zeta_multipliers", type=str, default="1,10,100,1000",
                   help="Comma-separated zeta multipliers for ablation")
    return p.parse_args()


def pil_to_tensor(pil_img, device):
    return TF.to_tensor(pil_img).unsqueeze(0).to(device).float()


def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)


def encode_to_latent(vae, image_tensor):
    scaled = image_tensor * 2.0 - 1.0
    latent_dist = vae.encode(scaled.to(vae.dtype)).latent_dist
    return latent_dist.sample().float() * vae.config.scaling_factor


def extract_scribble_hed(pil_image):
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_image, scribble=True)


def compute_dps_with_diagnostics(
    architect, scheduler, latents_step, t, cfg_encoder_states, added_cond_kwargs,
    guidance_scale, sprinter, clip_model, clip_processor, all_clip_embeddings,
    num_variations, base_zeta_prime, device, detailed=False,
):
    """Single DPS step with full diagnostic instrumentation.

    Returns dict with grad, correction, and all diagnostic metrics.
    If detailed=True, also does chain decomposition and finite-diff check.
    """
    diag = {}

    # ── Forward: noise pred → pred_x0 → pixels → CLIP → MMD ──
    noise_pred = predict_noise_cfg(
        architect.unet, scheduler, latents_step, t,
        cfg_encoder_states, added_cond_kwargs, guidance_scale,
    )

    pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents_step)

    # Decode to pixels (keep grad)
    pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

    def vae_decode_checkpoint(lat):
        return architect.vae.decode(lat.to(architect.vae.dtype)).sample

    pixel_x0 = torch.utils.checkpoint.checkpoint(
        vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
    pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

    # Retain grads on intermediates for chain decomposition
    if detailed:
        pred_x0.retain_grad()
        pixel_x0_norm.retain_grad()

    # CLIP encode (carries gradient)
    clip_model.to(device)
    current_clip_emb = encode_images_clip(pixel_x0_norm, clip_model, clip_processor)

    if detailed:
        current_clip_emb.retain_grad()

    # Generate variations (detached)
    variation_clip_list = []
    with torch.no_grad():
        for start_idx in range(0, num_variations, 1):
            end_idx = min(start_idx + 1, num_variations)
            bs = end_idx - start_idx
            ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1).detach()

            var_latents = sprinter(
                prompt=["a superrealistic professional photograph of"] * bs,
                image=ctrl_batch,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="latent", return_dict=True,
            ).images

            var_pixels = sprinter.vae.decode(
                (var_latents.float() / sprinter.vae.config.scaling_factor).to(sprinter.vae.dtype)
            ).sample
            var_pixels = torch.clamp((var_pixels.float() + 1.0) / 2.0, 0.0, 1.0)
            var_clip = encode_images_clip(var_pixels, clip_model, clip_processor)
            variation_clip_list.append(var_clip)

    variation_clip_embs = torch.cat(variation_clip_list, dim=0).detach()
    full_target = torch.cat([variation_clip_embs, all_clip_embeddings], dim=0).detach()
    torch.cuda.empty_cache()

    # MMD
    mmd_squared = compute_mmd(current_clip_emb, full_target)
    mmd_loss = torch.sqrt(mmd_squared.abs() + 1e-8)
    loss_norm = mmd_loss.detach()
    zeta_i = base_zeta_prime / loss_norm

    diag["mmd_loss"] = mmd_loss.item()
    diag["mmd_squared"] = mmd_squared.item()
    diag["zeta"] = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i

    # ── Backward ──
    # retain_graph=True always, since we need noise_pred for denoise_step after
    mmd_loss.backward(retain_graph=True)

    grad = latents_step.grad.clone()
    diag["grad_norm"] = grad.norm().item()
    diag["grad_max"] = grad.abs().max().item()
    diag["grad_mean"] = grad.abs().mean().item()

    # ── Chain decomposition (detailed step only) ──
    if detailed:
        if current_clip_emb.grad is not None:
            diag["grad_clip_emb_norm"] = current_clip_emb.grad.norm().item()
        else:
            diag["grad_clip_emb_norm"] = None

        if pixel_x0_norm.grad is not None:
            diag["grad_pixel_x0_norm"] = pixel_x0_norm.grad.norm().item()
        else:
            diag["grad_pixel_x0_norm"] = None

        if pred_x0.grad is not None:
            diag["grad_pred_x0_norm"] = pred_x0.grad.norm().item()
        else:
            diag["grad_pred_x0_norm"] = None

        diag["grad_latents_step_norm"] = grad.norm().item()

        # Attenuation ratios
        chain = []
        labels = []
        for name, val in [
            ("clip_emb", diag.get("grad_clip_emb_norm")),
            ("pixel_x0", diag.get("grad_pixel_x0_norm")),
            ("pred_x0", diag.get("grad_pred_x0_norm")),
            ("latents", diag["grad_latents_step_norm"]),
        ]:
            if val is not None:
                chain.append(val)
                labels.append(name)

        for i in range(len(chain) - 1):
            ratio = chain[i + 1] / (chain[i] + 1e-12)
            diag[f"attenuation_{labels[i]}_to_{labels[i+1]}"] = ratio

    # Correction
    correction = -zeta_i * grad
    diag["correction_norm"] = correction.norm().item()

    # ── Category 1: Magnitudes ──
    diag["latent_norm"] = latents_step.detach().norm().item()
    diag["correction_ratio"] = diag["correction_norm"] / (diag["latent_norm"] + 1e-12)

    # Scheduler step magnitude (how much does the scheduler move the latent?)
    with torch.no_grad():
        scheduler_copy = copy.deepcopy(scheduler)
        x_prev = scheduler_copy.step(noise_pred.detach(), t, latents_step.detach(),
                                      return_dict=True).prev_sample
        scheduler_step_delta = (x_prev - latents_step.detach()).norm().item()
        del scheduler_copy, x_prev
    diag["scheduler_step_norm"] = scheduler_step_delta
    diag["correction_vs_step"] = diag["correction_norm"] / (scheduler_step_delta + 1e-12)

    # ── Category 3: Finite-difference verification (detailed step only) ──
    if detailed:
        eps = 0.01
        grad_dir = grad / (grad.norm() + 1e-12)

        # MMD at original point (already computed)
        mmd_orig = mmd_loss.item()

        # MMD at perturbed point
        with torch.no_grad():
            perturbed = latents_step.detach() + eps * grad_dir

        perturbed.requires_grad_(False)
        # Re-run forward to get MMD at perturbed point (use scheduler copy to avoid side effects)
        scheduler_fd = copy.deepcopy(scheduler)
        scheduler_fd._step_index = scheduler._step_index - 1  # rewind to before this step
        with torch.no_grad():
            noise_pred_p = predict_noise_cfg(
                architect.unet, scheduler_fd, perturbed, t,
                cfg_encoder_states, added_cond_kwargs, guidance_scale,
            )
            pred_x0_p = compute_pred_x0_direct(scheduler_fd, noise_pred_p, t, perturbed)
            pred_x0_p_scaled = pred_x0_p / architect.vae.config.scaling_factor
            pixel_x0_p = architect.vae.decode(pred_x0_p_scaled.to(architect.vae.dtype)).sample
            pixel_x0_p_norm = torch.clamp((pixel_x0_p + 1.0) / 2.0, 0.0, 1.0)
            clip_emb_p = encode_images_clip(pixel_x0_p_norm, clip_model, clip_processor)
            mmd_sq_p = compute_mmd(clip_emb_p, full_target)
            mmd_p = torch.sqrt(mmd_sq_p.abs() + 1e-8).item()

        finite_diff = (mmd_p - mmd_orig) / eps
        autograd_directional = (grad * grad_dir).sum().item()
        diag["finite_diff_dMMD_deps"] = finite_diff
        diag["autograd_directional_deriv"] = autograd_directional
        diag["fd_vs_autograd_ratio"] = finite_diff / (autograd_directional + 1e-12)

    clip_model.to("cpu")
    torch.cuda.empty_cache()

    # Return noise_pred so caller can do denoise_step without re-computing
    noise_pred_detached = noise_pred.detach()

    # Clean up
    del pred_x0, pred_x0_scaled, pixel_x0, pixel_x0_norm
    del current_clip_emb, variation_clip_list, variation_clip_embs, full_target
    del mmd_squared, mmd_loss, loss_norm, noise_pred

    return grad, correction, zeta_i, diag, noise_pred_detached


def run_zeta_ablation(
    architect, base_scheduler, scribble_latent, timesteps_partial, start_step,
    cfg_encoder_states, added_cond_kwargs, guidance_scale,
    sprinter, clip_model, clip_processor, all_clip_embeddings,
    num_variations, base_zeta, zeta_multipliers, noise, prompt_embeds_dtype, device,
):
    """Run the same trajectory with different zeta multipliers.

    Returns dict mapping multiplier → final latent L2 distance from unguided.
    """
    t_start = timesteps_partial[0]
    noised_latent = architect.scheduler.add_noise(scribble_latent, noise, t_start.unsqueeze(0))
    noised_latent = noised_latent.to(prompt_embeds_dtype)

    # First: unguided trajectory
    print("\n  Zeta ablation: running unguided baseline...", flush=True)
    scheduler_base = copy.deepcopy(architect.scheduler)
    scheduler_base.set_timesteps(len(timesteps_partial) + start_step, device=device)
    scheduler_base._step_index = start_step

    latents = noised_latent.detach().clone()
    for t in timesteps_partial:
        with torch.no_grad():
            noise_pred = predict_noise_cfg(
                architect.unet, scheduler_base, latents, t,
                cfg_encoder_states, added_cond_kwargs, guidance_scale,
            )
            latents = denoise_step(scheduler_base, noise_pred, t, latents)
    unguided_latent = latents.detach()
    del scheduler_base

    results = {}
    for mult in zeta_multipliers:
        effective_zeta = base_zeta * mult
        print(f"\n  Zeta ablation: multiplier={mult} (zeta={effective_zeta:.2f})...", flush=True)

        scheduler_abl = copy.deepcopy(architect.scheduler)
        scheduler_abl.set_timesteps(len(timesteps_partial) + start_step, device=device)
        scheduler_abl._step_index = start_step

        latents = noised_latent.detach().clone()
        for i, t in enumerate(timesteps_partial):
            latents_step = latents.detach().requires_grad_(True)

            noise_pred = predict_noise_cfg(
                architect.unet, scheduler_abl, latents_step, t,
                cfg_encoder_states, added_cond_kwargs, guidance_scale,
            )
            pred_x0 = compute_pred_x0_direct(scheduler_abl, noise_pred, t, latents_step)
            pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

            def vae_decode_fn(lat):
                return architect.vae.decode(lat.to(architect.vae.dtype)).sample

            pixel_x0 = torch.utils.checkpoint.checkpoint(
                vae_decode_fn, pred_x0_scaled, use_reentrant=False)
            pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

            clip_model.to(device)
            current_clip_emb = encode_images_clip(pixel_x0_norm, clip_model, clip_processor)

            with torch.no_grad():
                var_clip_list = []
                for si in range(0, num_variations, 1):
                    ctrl = pixel_x0_norm[0].unsqueeze(0).detach()
                    vl = sprinter(
                        prompt=["a superrealistic professional photograph of"],
                        image=ctrl, num_inference_steps=2, guidance_scale=0.0,
                        controlnet_conditioning_scale=0.8, output_type="latent", return_dict=True,
                    ).images
                    vp = sprinter.vae.decode(
                        (vl.float() / sprinter.vae.config.scaling_factor).to(sprinter.vae.dtype)
                    ).sample
                    vp = torch.clamp((vp.float() + 1.0) / 2.0, 0.0, 1.0)
                    vc = encode_images_clip(vp, clip_model, clip_processor)
                    var_clip_list.append(vc)
                var_clip = torch.cat(var_clip_list, dim=0).detach()
                full_target = torch.cat([var_clip, all_clip_embeddings], dim=0).detach()

            mmd_sq = compute_mmd(current_clip_emb, full_target)
            mmd_loss = torch.sqrt(mmd_sq.abs() + 1e-8)
            loss_norm = mmd_loss.detach()
            adaptive_zeta = effective_zeta / loss_norm

            grad = torch.autograd.grad(mmd_loss, latents_step, retain_graph=False)[0]

            clip_model.to("cpu"); torch.cuda.empty_cache()

            if torch.isnan(grad).any():
                correction = torch.zeros_like(latents_step)
            else:
                correction = -adaptive_zeta * grad

            latents = denoise_step(scheduler_abl, noise_pred.detach(), t, latents_step, correction=correction)

            del noise_pred, pred_x0, pred_x0_scaled, pixel_x0, pixel_x0_norm
            del current_clip_emb, var_clip_list, var_clip, full_target
            del mmd_sq, mmd_loss, loss_norm, grad, correction
            gc.collect(); torch.cuda.empty_cache()

        guided_latent = latents.detach()
        l2_dist = (guided_latent - unguided_latent).norm().item()
        results[mult] = {
            "effective_zeta": effective_zeta,
            "l2_dist_from_unguided": l2_dist,
        }
        print(f"    L2 distance from unguided: {l2_dist:.6f}", flush=True)

        # Save final image
        with torch.no_grad():
            pil = latent_to_pil(guided_latent, architect.vae, architect.image_processor)
            pil.save(os.path.join(args.output_dir, f"zeta_mult_{mult}.png"))

        del scheduler_abl, guided_latent
        gc.collect(); torch.cuda.empty_cache()

    # Save unguided for comparison
    with torch.no_grad():
        pil = latent_to_pil(unguided_latent, architect.vae, architect.image_processor)
        pil.save(os.path.join(args.output_dir, "unguided.png"))

    del unguided_latent
    return results


def main():
    global args
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zeta_multipliers = [float(x) for x in args.zeta_multipliers.split(",")]
    print(f"Device: {device}", flush=True)
    print(f"Zeta multipliers: {zeta_multipliers}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──
    import wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        tags=["debug-gradient"],
    )

    # ── 1. Load models ──
    print("Loading models...", flush=True)
    architect, sprinter = load_models(device, architect_lora_path=args.lora_path)
    clip_model, clip_processor = load_clip_model(device)
    setup_gradient_checkpointing(architect, sprinter)
    print("Models loaded.", flush=True)

    # ── 2. Base image + targets ──
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    n_half = args.n_targets // 2
    print(f"Generating {args.n_targets} targets...", flush=True)
    with torch.no_grad():
        man_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale)
        woman_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a woman, studio lighting",
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale)

    with torch.no_grad():
        man_clip = encode_images_clip(pil_images_to_tensor(man_images, device), clip_model, clip_processor)
        woman_clip = encode_images_clip(pil_images_to_tensor(woman_images, device), clip_model, clip_processor)
    all_clip_embeddings = torch.cat([man_clip, woman_clip], dim=0)
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    del man_images, woman_images, man_clip, woman_clip
    clip_model.to("cpu"); gc.collect(); torch.cuda.empty_cache()

    # ── 3. Generate 1 source face ──
    print("Generating source face...", flush=True)
    with torch.no_grad():
        face_imgs, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, 1, batch_size=1, cn_scale=args.controlnet_scale)
    face_pil = face_imgs[0]
    face_pil.save(os.path.join(args.output_dir, "source_face.png"))

    # Extract scribble
    if args.edge_method == "hed_scribble":
        scribble_pil = extract_scribble_hed(face_pil)
        scribble_tensor = pil_to_tensor(scribble_pil, device)
    else:
        face_tensor = pil_to_tensor(face_pil, device)
        with torch.no_grad():
            scribble_tensor = sobel_proxy(face_tensor, device)
        scribble_pil = T.ToPILImage()(scribble_tensor.squeeze(0).cpu())
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))

    # Encode scribble to latent
    with torch.no_grad():
        scribble_latent = encode_to_latent(architect.vae, scribble_tensor)
    print(f"Scribble latent shape: {scribble_latent.shape}", flush=True)

    # ── 4. Prepare DPS components ──
    prompt = args.prompt if args.prompt else ""
    negative_prompt = args.negative_prompt if args.negative_prompt else ""
    height, width = 512, 512

    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
        )

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device)
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids": add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # ── 5. Set up partial denoising schedule ──
    n_steps = args.n_steps
    strength = args.strength
    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps
    start_step = int(n_steps * (1 - strength))
    timesteps_partial = timesteps[start_step:]
    print(f"Denoising {len(timesteps_partial)} steps (strength={strength}, start_step={start_step})",
          flush=True)

    # Add noise
    noise = torch.randn_like(scribble_latent)
    t_start = timesteps_partial[0]
    noised_latent = architect.scheduler.add_noise(scribble_latent, noise, t_start.unsqueeze(0))
    noised_latent = noised_latent.to(prompt_embeds.dtype)

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 1–4: Per-step diagnostics with chain decomposition
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("RUNNING DIAGNOSTIC DENOISING LOOP", flush=True)
    print(f"{'='*70}", flush=True)

    scheduler_diag = copy.deepcopy(architect.scheduler)
    scheduler_diag._step_index = start_step

    sprinter.vae.to(dtype=torch.float32)

    latents = noised_latent.detach().clone()
    all_diags = []
    grad_directions = []  # for direction consistency

    for i, t in enumerate(timesteps_partial):
        is_detailed = (i == args.detailed_step)
        if is_detailed:
            print(f"\n  *** DETAILED STEP {i} (t={t.item():.0f}) ***", flush=True)
        else:
            print(f"\n  Step {i}/{len(timesteps_partial)} (t={t.item():.0f})", flush=True)

        latents_step = latents.detach().requires_grad_(True)

        grad, correction, zeta_i, diag, noise_pred = compute_dps_with_diagnostics(
            architect, scheduler_diag, latents_step, t,
            cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
            sprinter, clip_model, clip_processor, all_clip_embeddings,
            args.num_variations, args.base_zeta, device,
            detailed=is_detailed,
        )

        diag["step"] = i
        diag["timestep"] = t.item()

        # Category 4: direction consistency
        grad_dir = (grad / (grad.norm() + 1e-12)).detach().cpu()
        if len(grad_directions) > 0:
            cosine_sim = (grad_dir * grad_directions[-1]).sum().item()
            diag["cosine_sim_with_prev"] = cosine_sim
        grad_directions.append(grad_dir)

        all_diags.append(diag)

        # Print summary
        print(f"    latent_norm={diag['latent_norm']:.4f}  "
              f"correction_norm={diag['correction_norm']:.6f}  "
              f"correction_ratio={diag['correction_ratio']:.8f}", flush=True)
        print(f"    scheduler_step_norm={diag['scheduler_step_norm']:.4f}  "
              f"correction_vs_step={diag['correction_vs_step']:.8f}", flush=True)
        print(f"    grad_norm={diag['grad_norm']:.8f}  "
              f"grad_max={diag['grad_max']:.8f}  "
              f"mmd_loss={diag['mmd_loss']:.6f}", flush=True)

        if is_detailed:
            print(f"    --- Chain decomposition ---", flush=True)
            for k in ["grad_clip_emb_norm", "grad_pixel_x0_norm",
                       "grad_pred_x0_norm", "grad_latents_step_norm"]:
                v = diag.get(k)
                print(f"    {k}: {v}", flush=True)
            for k, v in diag.items():
                if k.startswith("attenuation_"):
                    print(f"    {k}: {v:.6f}", flush=True)
            print(f"    --- Finite-difference check ---", flush=True)
            print(f"    finite_diff={diag.get('finite_diff_dMMD_deps', 'N/A')}", flush=True)
            print(f"    autograd={diag.get('autograd_directional_deriv', 'N/A')}", flush=True)
            print(f"    fd/autograd ratio={diag.get('fd_vs_autograd_ratio', 'N/A')}", flush=True)

        if "cosine_sim_with_prev" in diag:
            print(f"    cosine_sim_with_prev={diag['cosine_sim_with_prev']:.4f}", flush=True)

        # Log to wandb
        wandb.log({f"diag/{k}": v for k, v in diag.items()
                    if isinstance(v, (int, float)) and v is not None})

        # Apply correction and advance — use noise_pred from diagnostic forward pass
        if torch.isnan(grad).any():
            correction = torch.zeros_like(latents_step)
        latents = denoise_step(scheduler_diag, noise_pred, t,
                               latents_step.detach(), correction=correction.detach())

        del grad, correction, latents_step, noise_pred
        gc.collect(); torch.cuda.empty_cache()

    # Save guided result
    with torch.no_grad():
        guided_pil = latent_to_pil(latents, architect.vae, architect.image_processor)
        guided_pil.save(os.path.join(args.output_dir, "guided_result.png"))
        wandb.log({"guided_result": wandb.Image(os.path.join(args.output_dir, "guided_result.png"))})

    # ── Direction consistency summary ──
    cosines = [d.get("cosine_sim_with_prev") for d in all_diags if "cosine_sim_with_prev" in d]
    if cosines:
        print(f"\n  Direction consistency: mean cosine={np.mean(cosines):.4f}, "
              f"std={np.std(cosines):.4f}, min={np.min(cosines):.4f}, max={np.max(cosines):.4f}",
              flush=True)
        wandb.log({
            "direction/mean_cosine": np.mean(cosines),
            "direction/std_cosine": np.std(cosines),
            "direction/min_cosine": np.min(cosines),
            "direction/max_cosine": np.max(cosines),
        })

    del scheduler_diag, latents, grad_directions
    gc.collect(); torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 5: Zeta ablation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("ZETA ABLATION", flush=True)
    print(f"{'='*70}", flush=True)

    architect.scheduler.set_timesteps(n_steps, device=device)
    ablation_results = run_zeta_ablation(
        architect, architect.scheduler, scribble_latent, timesteps_partial, start_step,
        cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        sprinter, clip_model, clip_processor, all_clip_embeddings,
        args.num_variations, args.base_zeta, zeta_multipliers,
        noise, prompt_embeds.dtype, device,
    )

    print("\n  Zeta ablation results:", flush=True)
    for mult, res in ablation_results.items():
        print(f"    mult={mult}: zeta={res['effective_zeta']:.2f}, "
              f"L2_from_unguided={res['l2_dist_from_unguided']:.6f}", flush=True)
        wandb.log({
            f"zeta_ablation/mult_{mult}_l2": res["l2_dist_from_unguided"],
            f"zeta_ablation/mult_{mult}_zeta": res["effective_zeta"],
        })

    # ── Save all diagnostics ──
    output = {
        "args": vars(args),
        "per_step_diagnostics": all_diags,
        "zeta_ablation": {str(k): v for k, v in ablation_results.items()},
    }
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary table to wandb ──
    import wandb
    table = wandb.Table(columns=[
        "step", "timestep", "latent_norm", "correction_norm", "correction_ratio",
        "scheduler_step_norm", "correction_vs_step", "grad_norm", "mmd_loss",
        "cosine_sim_with_prev"
    ])
    for d in all_diags:
        table.add_data(
            d["step"], d["timestep"], d["latent_norm"],
            d["correction_norm"], d["correction_ratio"],
            d["scheduler_step_norm"], d["correction_vs_step"],
            d["grad_norm"], d["mmd_loss"],
            d.get("cosine_sim_with_prev", None),
        )
    wandb.log({"diagnostics_table": table})

    wandb.finish()
    print(f"\nDiagnostics complete. Outputs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
