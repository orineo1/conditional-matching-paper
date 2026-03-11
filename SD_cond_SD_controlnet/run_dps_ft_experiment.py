"""
DPS Experiment — Base vs Fine-Tuned Architect (6 Variants).

For each of N source faces:
  1. Generate a male portrait via sprinter conditioned on face outline
  2. Extract inverted HED scribble from portrait
  3. Encode scribble to latent space (VAE encode)
  4. For each noise strength, run 6 denoising variants:
     - base_prompt_regular:  Base U-Net, scribble prompt, no DPS
     - base_prompt_dps:      Base U-Net, scribble prompt, DPS guidance
     - ft_prompt_regular:    FT U-Net, scribble prompt, no DPS
     - ft_prompt_dps:        FT U-Net, scribble prompt, DPS guidance
     - ft_noprompt_regular:  FT U-Net, unconditional, no DPS
     - ft_noprompt_dps:      FT U-Net, unconditional, DPS guidance
  5. Generate conditioned photos from each denoised scribble via sprinter
  6. Produce one composite image per face (all strengths × 6 variants)
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
from PIL import Image, ImageDraw, ImageFont, ImageOps

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0_direct,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
    run_dps_step_clip,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd
from models import load_models, setup_gradient_checkpointing
from visualization import plot_row


# Variant definitions: (label, unet_key, use_prompt, dps)
VARIANTS = [
    ("base_prompt_regular",  "base", True,  False),
    ("base_prompt_dps",      "base", True,  True),
    ("ft_prompt_regular",    "ft",   True,  False),
    ("ft_prompt_dps",        "ft",   True,  True),
    ("ft_noprompt_regular",  "ft",   False, False),
    ("ft_noprompt_dps",      "ft",   False, True),
]


def parse_args():
    p = argparse.ArgumentParser(description="DPS Experiment: Base vs Fine-Tuned (6 Variants)")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/dps_ft_final")
    p.add_argument("--architect_unet_path", type=str, default="finetune_scribble/output/final/merged")
    p.add_argument("--wandb_project", type=str, default="dps_experiment_final_ft")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")
    p.add_argument("--n_faces", type=int, default=3)
    p.add_argument("--strengths", type=str, default="0.25,0.5,0.75")
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--num_variations", type=int, default=20)
    p.add_argument("--num_conditioned", type=int, default=10)
    p.add_argument("--base_zeta", type=float, default=0.2)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str,
                   default="rough pencil scribble outline, loose sketch, minimal")
    p.add_argument("--negative_prompt", type=str,
                   default="detailed, realistic, photograph, complex, colored, shading")
    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--photo_every", type=int, default=5,
                   help="Generate sprinter photos every N denoising steps")
    p.add_argument("--photos_per_step", type=int, default=3,
                   help="Number of sprinter photos per trajectory step")
    return p.parse_args()


def extract_scribble_hed(pil_image):
    """Extract a simple scribble from a PIL image using HED in scribble mode."""
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_image, scribble=True)


def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)


def pil_to_tensor(pil_img, device):
    return TF.to_tensor(pil_img).unsqueeze(0).to(device).float()


def encode_to_latent(vae, image_tensor):
    """Encode a [0,1] image tensor to latent space."""
    scaled = image_tensor * 2.0 - 1.0
    latent_dist = vae.encode(scaled.to(vae.dtype)).latent_dist
    return latent_dist.sample().float() * vae.config.scaling_factor


def partial_denoise_loop(unet, vae, scheduler, image_processor, latents,
                         timesteps_partial, cfg_encoder_states, added_cond_kwargs,
                         guidance_scale, use_cfg=True,
                         dps_guidance=False, sprinter=None, clip_model=None,
                         clip_processor=None, all_clip_embeddings=None,
                         num_variations=20, base_zeta_prime=0.2, device="cuda",
                         save_trajectory_dir=None, photo_every=5, photos_per_step=3,
                         conditioning_outline=None):
    """Run denoising from a partial timestep schedule.

    If use_cfg=False, does a single unconditional forward pass (no CFG doubling).
    If dps_guidance=True, applies CLIP-MMD DPS correction at each step.

    If save_trajectory_dir is set, saves:
      - pred_x0 image at every step
      - sprinter-conditioned photos every photo_every steps + at the final step
    """
    step_metrics = []
    pred_x0_pils = []
    variation_batch_size = 1
    n_total = len(timesteps_partial)

    if save_trajectory_dir:
        os.makedirs(save_trajectory_dir, exist_ok=True)

    for i, t in enumerate(timesteps_partial):
        if dps_guidance:
            latents_step = latents.detach().requires_grad_(True)
        else:
            latents_step = latents.detach()

        # Noise prediction
        if use_cfg:
            # CFG: concat uncond + cond, double batch
            if dps_guidance:
                noise_pred = predict_noise_cfg(
                    unet, scheduler, latents_step, t,
                    cfg_encoder_states, added_cond_kwargs, guidance_scale,
                )
            else:
                with torch.no_grad():
                    noise_pred = predict_noise_cfg(
                        unet, scheduler, latents_step, t,
                        cfg_encoder_states, added_cond_kwargs, guidance_scale,
                    )
        else:
            # Unconditional: single forward pass, no CFG
            lmi = scheduler.scale_model_input(latents_step, t)
            if dps_guidance:
                noise_pred = unet(
                    lmi, t,
                    encoder_hidden_states=cfg_encoder_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            else:
                with torch.no_grad():
                    noise_pred = unet(
                        lmi, t,
                        encoder_hidden_states=cfg_encoder_states,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

        # pred_x0
        if dps_guidance:
            pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents_step)
        else:
            with torch.no_grad():
                pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents_step)

        # Save pred_x0 as image at every step
        if save_trajectory_dir:
            with torch.no_grad():
                pred_x0_pil = latent_to_pil(pred_x0.detach(), vae, image_processor)
            pred_x0_pil.save(os.path.join(save_trajectory_dir, f"pred_x0_step_{i:02d}.png"))
            pred_x0_pils.append(pred_x0_pil)

            # Generate sprinter photos at key steps
            is_photo_step = (i % photo_every == 0) or (i == n_total - 1)
            if is_photo_step and conditioning_outline is not None:
                step_photo_dir = os.path.join(save_trajectory_dir, f"step_{i:02d}_photos")
                os.makedirs(step_photo_dir, exist_ok=True)
                step_photos = generate_conditioned_photos(
                    sprinter, pred_x0_pil, photos_per_step, batch_size=photos_per_step)
                for p_idx, p_img in enumerate(step_photos):
                    p_img.save(os.path.join(step_photo_dir, f"photo_{p_idx:02d}.png"))
                del step_photos

        correction = None
        if dps_guidance:
            pred_x0_scaled = pred_x0 / vae.config.scaling_factor

            def vae_decode_checkpoint(lat):
                return vae.decode(lat.to(vae.dtype)).sample

            pixel_x0 = torch.utils.checkpoint.checkpoint(
                vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
            pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

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

            print(f"    DPS step {i+1}/{n_total} t={t.item():.0f}  "
                  f"MMD={mmd_loss.item():.6f}  grad={grad_norm:.6f}", flush=True)

            if torch.isnan(grad).any():
                print(f"    WARNING: NaN gradient — skipping correction", flush=True)
                correction = torch.zeros_like(latents_step)
            else:
                correction = -zeta_i * grad

            step_metrics.append({
                'step': i, 'timestep': t.item(),
                'mmd_loss': mmd_loss.item(), 'gradient_norm': grad_norm,
                'zeta': zeta_val,
            })

            del grad, mmd_loss, loss_norm, zeta_i, vl_clip_flat
            del pixel_x0, pixel_x0_norm, pred_x0_scaled
        else:
            if not save_trajectory_dir:
                pass  # no extra logging needed
            else:
                print(f"    Step {i+1}/{n_total} t={t.item():.0f}", flush=True)

        # Scheduler step
        latents = denoise_step(scheduler, noise_pred, t, latents_step, correction=correction)

        del latents_step, noise_pred, pred_x0
        if correction is not None:
            del correction
        gc.collect(); torch.cuda.empty_cache()

    return latents, step_metrics, pred_x0_pils


def generate_conditioned_photos(sprinter, scribble_pil, num_photos, batch_size=2):
    """Generate portrait photos conditioned on a scribble via sprinter."""
    all_images = []
    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)

    with torch.no_grad():
        for start in range(0, num_photos, batch_size):
            bs = min(batch_size, num_photos - start)
            result = sprinter(
                prompt=["a superrealistic professional photograph of a person, studio lighting"] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pil", return_dict=True,
            )
            all_images.extend(result.images)

    sprinter.vae.to(dtype=original_vae_dtype)
    return all_images


def build_composite_image(source_face_pil, scribble_pil, strength_results, strengths):
    """Build a single composite image for one face across all strengths.

    Layout:
      Row 0 (header):  [Source Face] [Scribble]  | strength=s1        | strength=s2        | ...
      Row 1: base_prompt_regular                  | [noised→denoised]  | ...
      Row 1 photos:                               | [2×5 thumbnails]   | ...
      Row 2: base_prompt_dps                      | ...
      ...
      Row 6: ft_noprompt_dps                      | ...
    """
    cell = 256
    photo = 100
    photos_per_row = 5
    label_h = 30
    gap = 4

    n_strengths = len(strengths)
    n_variants = len(VARIANTS)

    header_w = 2 * cell + gap
    strength_col_w = max(2 * cell + gap, photos_per_row * (photo + gap))
    total_w = header_w + gap + n_strengths * (strength_col_w + gap)

    photo_block_h = 2 * (photo + gap)
    # Header row + per-variant (denoised row + photo row)
    row_heights = [label_h + cell]  # header
    for _ in range(n_variants):
        row_heights.append(label_h + cell)          # noised → denoised
        row_heights.append(label_h + photo_block_h)  # photos
    total_h = sum(row_heights) + (len(row_heights)) * gap

    canvas = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    y_offsets = [0]
    for h in row_heights[:-1]:
        y_offsets.append(y_offsets[-1] + h + gap)

    # Row 0: Source face and scribble
    draw.text((gap, y_offsets[0]), "Source Face / Scribble", fill='black', font=font)
    y_img = y_offsets[0] + label_h
    canvas.paste(source_face_pil.resize((cell, cell)), (gap, y_img))
    canvas.paste(scribble_pil.resize((cell, cell)), (gap + cell + gap, y_img))

    # Variant labels and colors
    variant_colors = {
        "base_prompt_regular": "gray",
        "base_prompt_dps": "blue",
        "ft_prompt_regular": "green",
        "ft_prompt_dps": "darkgreen",
        "ft_noprompt_regular": "orange",
        "ft_noprompt_dps": "red",
    }
    variant_display = {
        "base_prompt_regular": "Base+prompt (no DPS)",
        "base_prompt_dps": "Base+prompt (DPS)",
        "ft_prompt_regular": "FT+prompt (no DPS)",
        "ft_prompt_dps": "FT+prompt (DPS)",
        "ft_noprompt_regular": "FT+noprompt (no DPS)",
        "ft_noprompt_dps": "FT+noprompt (DPS)",
    }

    for s_idx, strength in enumerate(strengths):
        x_col = header_w + gap + s_idx * (strength_col_w + gap)
        sr = strength_results[s_idx]

        # Column header
        draw.text((x_col, y_offsets[0]), f"strength={strength}", fill='black', font=font)

        for v_idx, (label, _, _, _) in enumerate(VARIANTS):
            row_denoised = 1 + v_idx * 2
            row_photos = 2 + v_idx * 2

            color = variant_colors[label]
            display = variant_display[label]

            # Denoised row
            draw.text((x_col, y_offsets[row_denoised]), display, fill=color, font=font_small)
            y_d = y_offsets[row_denoised] + label_h
            canvas.paste(sr['noised_pil'].resize((cell, cell)), (x_col, y_d))
            canvas.paste(sr[f'{label}_pil'].resize((cell, cell)), (x_col + cell + gap, y_d))

            # Photos row
            draw.text((x_col, y_offsets[row_photos]), f"{display} → photos", fill=color, font=font_small)
            y_p = y_offsets[row_photos] + label_h
            for p_idx, p_img in enumerate(sr[f'{label}_photos'][:10]):
                pr = p_idx // photos_per_row
                pc = p_idx % photos_per_row
                px = x_col + pc * (photo + gap)
                py = y_p + pr * (photo + gap)
                canvas.paste(p_img.resize((photo, photo)), (px, py))

    return canvas


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strengths = [float(s) for s in args.strengths.split(",")]
    print(f"Device: {device}", flush=True)
    print(f"Strengths: {strengths}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        settings=wandb.Settings(init_timeout=300),
    )

    # ── 1. Load models ─────────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    architect, sprinter = load_models(device, architect_unet_path=args.architect_unet_path)
    clip_model, clip_processor = load_clip_model(device)
    setup_gradient_checkpointing(architect, sprinter)

    # The loaded architect has the FT U-Net; also load base U-Net
    from diffusers import UNet2DConditionModel
    ft_unet = architect.unet
    base_unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=torch.float16,
    ).to(device)
    base_unet.enable_gradient_checkpointing()
    for p in base_unet.parameters():
        p.requires_grad_(False)
    print("Both base and FT U-Nets loaded.", flush=True)

    # ── 2. Generate conditioning face from oval ────────────────────────────────
    print("Generating conditioning face outline...", flush=True)
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    # Generate one face from oval via sprinter
    with torch.no_grad():
        face_imgs, _ = generate_and_store_cs(
            sprinter,
            "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, 1, batch_size=1, cn_scale=args.controlnet_scale,
        )
    conditioning_face = face_imgs[0]
    conditioning_face.save(os.path.join(args.output_dir, "conditioning_face.png"))

    # Extract inverted HED scribble → white bg, black lines
    conditioning_outline = ImageOps.invert(extract_scribble_hed(conditioning_face))
    conditioning_outline.save(os.path.join(args.output_dir, "conditioning_outline.png"))
    print("  Conditioning outline saved.", flush=True)

    # ── 3. Generate target distribution ────────────────────────────────────────
    N_TARGETS = args.n_targets
    n_half = N_TARGETS // 2
    print(f"Generating {N_TARGETS} target images ({n_half} man + {n_half} woman)...", flush=True)

    targets_dir = os.path.join(args.output_dir, "targets")
    os.makedirs(targets_dir, exist_ok=True)

    with torch.no_grad():
        man_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            conditioning_outline, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )
        woman_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a woman, studio lighting",
            conditioning_outline, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )

    all_target_images = man_images + woman_images
    for t_idx, t_img in enumerate(all_target_images):
        t_img.save(os.path.join(targets_dir, f"target_{t_idx:02d}.png"))

    plot_row(man_images, "Target Samples (Man)",
             save_path=os.path.join(args.output_dir, "target_man.png"))
    plot_row(woman_images, "Target Samples (Woman)",
             save_path=os.path.join(args.output_dir, "target_woman.png"))

    # ── 4. Encode targets to CLIP ──────────────────────────────────────────────
    with torch.no_grad():
        man_clip = encode_images_clip(pil_images_to_tensor(man_images, device), clip_model, clip_processor)
        woman_clip = encode_images_clip(pil_images_to_tensor(woman_images, device), clip_model, clip_processor)
    all_clip_embeddings = torch.cat([man_clip, woman_clip], dim=0)
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    del man_images, woman_images, man_clip, woman_clip
    clip_model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    # ── 5. Generate source faces ───────────────────────────────────────────────
    print(f"Generating {args.n_faces} source faces...", flush=True)
    source_faces = []
    with torch.no_grad():
        for face_idx in range(args.n_faces):
            face_imgs, _ = generate_and_store_cs(
                sprinter, "a superrealistic portrait photograph of a man, studio lighting",
                conditioning_outline, 1, batch_size=1, cn_scale=args.controlnet_scale,
            )
            source_faces.append(face_imgs[0])

    # ── 6. Prepare DPS components ──────────────────────────────────────────────
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    print(f"  Architect prompt: '{prompt}'", flush=True)
    height, width = 512, 512
    n_steps = args.n_steps

    # CFG embeddings (for prompted variants)
    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
        )

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device)
    cfg_added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids": add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # Unconditional embeddings (for noprompt variants — single pass, no CFG)
    with torch.no_grad():
        (uncond_embeds, _, uncond_pooled, _) = architect.encode_prompt(
            prompt="", device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    uncond_added_cond_kwargs = {
        "text_embeds": uncond_pooled,
        "time_ids": add_time_ids,
    }

    all_metrics = {}
    composites_dir = os.path.join(args.output_dir, "composites")
    os.makedirs(composites_dir, exist_ok=True)

    # ── 7. Per-face loop ───────────────────────────────────────────────────────
    for face_idx, face_pil in enumerate(source_faces):
        print(f"\n{'='*70}", flush=True)
        print(f"FACE {face_idx+1}/{args.n_faces}", flush=True)
        print(f"{'='*70}", flush=True)

        face_dir = os.path.join(args.output_dir, f"face_{face_idx}")
        os.makedirs(face_dir, exist_ok=True)

        face_pil.save(os.path.join(face_dir, "source.png"))

        # Extract inverted HED scribble from source face
        scribble_pil = ImageOps.invert(extract_scribble_hed(face_pil))
        scribble_tensor = pil_to_tensor(scribble_pil, device)
        scribble_pil.save(os.path.join(face_dir, "scribble.png"))
        print(f"  Scribble extracted (inverted HED)", flush=True)

        # Encode scribble to latent
        with torch.no_grad():
            scribble_latent = encode_to_latent(architect.vae, scribble_tensor)
        print(f"  Scribble latent shape: {scribble_latent.shape}", flush=True)

        strength_results = []

        for s_idx, strength in enumerate(strengths):
            print(f"\n  --- Strength {strength} ---", flush=True)

            # Set up scheduler for this strength
            architect.scheduler.set_timesteps(n_steps, device=device)
            timesteps = architect.scheduler.timesteps
            start_step = int(n_steps * (1 - strength))
            timesteps_partial = timesteps[start_step:]
            print(f"  Start step: {start_step}, denoising {len(timesteps_partial)} steps", flush=True)

            # Add noise at t_start
            noise = torch.randn_like(scribble_latent)
            t_start = timesteps_partial[0]
            noised_latent = architect.scheduler.add_noise(scribble_latent, noise, t_start.unsqueeze(0))
            noised_latent = noised_latent.to(prompt_embeds.dtype)

            with torch.no_grad():
                noised_pil = latent_to_pil(noised_latent, architect.vae, architect.image_processor)

            sr = {'noised_pil': noised_pil}

            # Run each of the 6 variants
            for label, unet_key, use_prompt, dps in VARIANTS:
                print(f"\n  Variant: {label}", flush=True)

                # Select U-Net
                unet = base_unet if unet_key == "base" else ft_unet

                # Select embeddings
                if use_prompt:
                    enc_states = cfg_encoder_states
                    add_cond = cfg_added_cond_kwargs
                    gs = args.guidance_scale
                    use_cfg = True
                else:
                    enc_states = uncond_embeds
                    add_cond = uncond_added_cond_kwargs
                    gs = 0.0
                    use_cfg = False

                # Scheduler copy
                scheduler_copy = copy.deepcopy(architect.scheduler)
                scheduler_copy._step_index = start_step

                if dps:
                    sprinter.vae.to(dtype=torch.float32)

                # Trajectory save directory
                traj_dir = os.path.join(face_dir, f"s{strength}_{label}_trajectory")

                denoised_latent, step_metrics, pred_x0_pils = partial_denoise_loop(
                    unet=unet,
                    vae=architect.vae,
                    scheduler=scheduler_copy,
                    image_processor=architect.image_processor,
                    latents=noised_latent.detach().clone(),
                    timesteps_partial=timesteps_partial,
                    cfg_encoder_states=enc_states,
                    added_cond_kwargs=add_cond,
                    guidance_scale=gs,
                    use_cfg=use_cfg,
                    dps_guidance=dps,
                    sprinter=sprinter,
                    clip_model=clip_model if dps else None,
                    clip_processor=clip_processor if dps else None,
                    all_clip_embeddings=all_clip_embeddings if dps else None,
                    num_variations=args.num_variations,
                    base_zeta_prime=args.base_zeta,
                    device=device,
                    save_trajectory_dir=traj_dir,
                    photo_every=args.photo_every,
                    photos_per_step=args.photos_per_step,
                    conditioning_outline=conditioning_outline,
                )

                with torch.no_grad():
                    denoised_pil = latent_to_pil(denoised_latent, architect.vae, architect.image_processor)
                denoised_pil.save(os.path.join(face_dir, f"s{strength}_{label}.png"))
                sr[f'{label}_pil'] = denoised_pil

                # Log DPS metrics
                if dps and step_metrics:
                    for m in step_metrics:
                        wandb.log({
                            f"face{face_idx}/s{strength}/{label}/mmd": m['mmd_loss'],
                            f"face{face_idx}/s{strength}/{label}/grad_norm": m['gradient_norm'],
                            f"face{face_idx}/s{strength}/{label}/step": m['step'],
                        })
                    all_metrics[f"face{face_idx}_s{strength}_{label}"] = step_metrics

                del scheduler_copy, denoised_latent
                gc.collect(); torch.cuda.empty_cache()

                # Generate conditioned photos
                print(f"  Generating {args.num_conditioned} photos from {label}...", flush=True)
                photos = generate_conditioned_photos(
                    sprinter, denoised_pil, args.num_conditioned)
                sr[f'{label}_photos'] = photos

                # Save photos
                photo_dir = os.path.join(face_dir, f"s{strength}_{label}_photos")
                os.makedirs(photo_dir, exist_ok=True)
                for p_idx, p_img in enumerate(photos):
                    p_img.save(os.path.join(photo_dir, f"photo_{p_idx:02d}.png"))

            strength_results.append(sr)

            del noised_latent, noise
            gc.collect(); torch.cuda.empty_cache()

        # -- Build composite image for this face --
        print(f"  Building composite image...", flush=True)
        composite = build_composite_image(face_pil, scribble_pil, strength_results, strengths)
        composite_path = os.path.join(composites_dir, f"composite_face_{face_idx}.png")
        composite.save(composite_path)
        wandb.log({f"composite_face_{face_idx}": wandb.Image(composite_path)})
        print(f"  Composite saved: {composite_path}", flush=True)

        del strength_results, scribble_tensor, scribble_latent
        gc.collect(); torch.cuda.empty_cache()

    # ── 8. Final summary ───────────────────────────────────────────────────────
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"args": vars(args), "all_metrics": all_metrics}, f, indent=2)

    wandb.finish()
    print(f"\nExperiment complete. Outputs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
