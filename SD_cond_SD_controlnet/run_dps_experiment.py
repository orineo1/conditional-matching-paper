"""
DPS Experiment — Scribble-from-Portrait with Partial Noising.

For each of N source faces:
  1. Generate a male portrait via sprinter (NOT from target set)
  2. Extract outline scribble via Sobel edge detection
  3. Encode scribble to latent space (VAE encode)
  4. For each noise strength:
     - Add noise at corresponding timestep
     - Denoise WITH DPS CLIP-MMD guidance → "guided scribble"
     - Denoise WITHOUT guidance → "regular scribble"
     - Generate conditioned photos from each denoised scribble via sprinter
  5. Produce one composite image per face (all strengths)
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
from PIL import Image, ImageDraw, ImageFont

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


def parse_args():
    p = argparse.ArgumentParser(description="DPS Experiment: Scribble-from-Portrait")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/experiment")
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="combined_conditional_flow")
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
    p.add_argument("--prompt", type=str, default="rough pencil scribble outline, loose sketch, minimal line art",
                   help="Architect prompt for denoising (use '' for unconditional)")
    p.add_argument("--negative_prompt", type=str, default="detailed, realistic, photograph, complex, colored, shading")
    p.add_argument("--n_targets", type=int, default=20,
                   help="Total target images (split evenly male/female)")
    p.add_argument("--edge_method", type=str, default="hed_scribble",
                   choices=["sobel", "hed_scribble"],
                   help="Edge extraction method: sobel (detailed) or hed_scribble (simple)")
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
    scaled = image_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    latent_dist = vae.encode(scaled.to(vae.dtype)).latent_dist
    return latent_dist.sample().float() * vae.config.scaling_factor


def partial_denoise_loop(architect, scheduler, latents, timesteps_partial,
                         cfg_encoder_states, added_cond_kwargs, guidance_scale,
                         dps_guidance=False, sprinter=None, clip_model=None,
                         clip_processor=None, all_clip_embeddings=None,
                         num_variations=20, base_zeta_prime=0.2, device="cuda"):
    """Run denoising from a partial timestep schedule.

    If dps_guidance=True, applies CLIP-MMD DPS correction at each step.
    Returns the final denoised latent and a list of per-step metrics (for DPS).
    """
    step_metrics = []
    variation_batch_size = 1

    for i, t in enumerate(timesteps_partial):
        if dps_guidance:
            latents_step = latents.detach().requires_grad_(True)
        else:
            latents_step = latents.detach()

        # Noise prediction with CFG
        if dps_guidance:
            noise_pred = predict_noise_cfg(
                architect.unet, scheduler, latents_step, t,
                cfg_encoder_states, added_cond_kwargs, guidance_scale,
            )
        else:
            with torch.no_grad():
                noise_pred = predict_noise_cfg(
                    architect.unet, scheduler, latents_step, t,
                    cfg_encoder_states, added_cond_kwargs, guidance_scale,
                )

        # pred_x0 — pure formula, no scheduler side effects
        if dps_guidance:
            pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents_step)
        else:
            with torch.no_grad():
                pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents_step)

        correction = None
        if dps_guidance:
            # Decode pred_x0 to pixel space (keep grad)
            pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

            def vae_decode_checkpoint(lat):
                return architect.vae.decode(lat.to(architect.vae.dtype)).sample

            pixel_x0 = torch.utils.checkpoint.checkpoint(
                vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
            pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

            # CLIP-MMD guidance
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

            step_idx = len(timesteps_partial) - len(timesteps_partial) + i
            print(f"    DPS step {i+1}/{len(timesteps_partial)} t={t.item():.0f}  "
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

        # Scheduler step
        latents = denoise_step(scheduler, noise_pred, t, latents_step, correction=correction)

        del latents_step, noise_pred, pred_x0
        if correction is not None:
            del correction
        gc.collect(); torch.cuda.empty_cache()

    return latents, step_metrics


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
      Row 1 (regular):                           | [noised]->[denoised]| ...
      Row 2 (regular photos):                    | [10 small photos]   | ...
      Row 3 (DPS):                               | [noised]->[denoised]| ...
      Row 4 (DPS photos):                        | [10 small photos]   | ...
    """
    cell = 256       # size of main cells (face, scribble, noised, denoised)
    photo = 100      # size of conditioned photo thumbnails
    photos_per_row = 5  # 10 photos in 2 rows of 5
    label_h = 30     # height for text labels
    gap = 4          # gap between elements

    n_strengths = len(strengths)

    # Column widths: header col = 2*cell + gap, then per-strength = 2*cell + gap
    header_w = 2 * cell + gap
    strength_col_w = max(2 * cell + gap, photos_per_row * (photo + gap))
    total_w = header_w + gap + n_strengths * (strength_col_w + gap)

    # Row heights
    photo_block_h = 2 * (photo + gap)  # 2 rows of 5 photos
    row_heights = [
        label_h + cell,          # row 0: header (label + face/scribble)
        label_h + cell,          # row 1: regular noised->denoised
        label_h + photo_block_h, # row 2: regular photos
        label_h + cell,          # row 3: DPS noised->denoised
        label_h + photo_block_h, # row 4: DPS photos
    ]
    total_h = sum(row_heights) + 5 * gap

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

    # Per-strength columns
    for s_idx, strength in enumerate(strengths):
        x_col = header_w + gap + s_idx * (strength_col_w + gap)
        sr = strength_results[s_idx]

        # Column header
        draw.text((x_col, y_offsets[0]), f"strength={strength}", fill='black', font=font)

        # Row 1: Regular noised -> denoised
        draw.text((x_col, y_offsets[1]), "Regular (no DPS)", fill='gray', font=font_small)
        y1 = y_offsets[1] + label_h
        canvas.paste(sr['noised_pil'].resize((cell, cell)), (x_col, y1))
        canvas.paste(sr['regular_pil'].resize((cell, cell)), (x_col + cell + gap, y1))

        # Row 2: Regular conditioned photos
        draw.text((x_col, y_offsets[2]), "Regular → photos", fill='gray', font=font_small)
        y2 = y_offsets[2] + label_h
        for p_idx, p_img in enumerate(sr['regular_photos'][:10]):
            pr = p_idx // photos_per_row
            pc = p_idx % photos_per_row
            px = x_col + pc * (photo + gap)
            py = y2 + pr * (photo + gap)
            canvas.paste(p_img.resize((photo, photo)), (px, py))

        # Row 3: DPS noised -> denoised
        draw.text((x_col, y_offsets[3]), "DPS guided", fill='blue', font=font_small)
        y3 = y_offsets[3] + label_h
        canvas.paste(sr['noised_pil'].resize((cell, cell)), (x_col, y3))
        canvas.paste(sr['dps_pil'].resize((cell, cell)), (x_col + cell + gap, y3))

        # Row 4: DPS conditioned photos
        draw.text((x_col, y_offsets[4]), "DPS → photos", fill='blue', font=font_small)
        y4 = y_offsets[4] + label_h
        for p_idx, p_img in enumerate(sr['dps_photos'][:10]):
            pr = p_idx // photos_per_row
            pc = p_idx % photos_per_row
            px = x_col + pc * (photo + gap)
            py = y4 + pr * (photo + gap)
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
    )

    # ── 1. Load models ─────────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    architect, sprinter = load_models(device, architect_lora_path=args.lora_path)
    clip_model, clip_processor = load_clip_model(device)
    setup_gradient_checkpointing(architect, sprinter)
    print("Models loaded.", flush=True)

    # ── 2. Base image (oval) for target generation ─────────────────────────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    # ── 3. Generate target distribution ────────────────────────────────────────
    N_TARGETS = args.n_targets
    n_half = N_TARGETS // 2
    print(f"Generating {N_TARGETS} target images ({n_half} man + {n_half} woman)...", flush=True)

    with torch.no_grad():
        man_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )
        woman_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a woman, studio lighting",
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )

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
                sobel_cond_pil, 1, batch_size=1, cn_scale=args.controlnet_scale,
            )
            source_faces.append(face_imgs[0])
            source_faces[-1].save(os.path.join(args.output_dir, f"source_face_{face_idx}.png"))
            print(f"  Source face {face_idx} generated.", flush=True)

    # ── 6. Prepare DPS components ──────────────────────────────────────────────
    prompt = args.prompt if args.prompt else ""
    negative_prompt = args.negative_prompt if args.negative_prompt else ""
    print(f"  Architect prompt: '{prompt}'", flush=True)
    height, width = 512, 512
    n_steps = args.n_steps

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

    all_metrics = {}

    # ── 7. Per-face loop ───────────────────────────────────────────────────────
    for face_idx, face_pil in enumerate(source_faces):
        print(f"\n{'='*70}", flush=True)
        print(f"FACE {face_idx+1}/{args.n_faces}", flush=True)
        print(f"{'='*70}", flush=True)

        # Extract scribble
        if args.edge_method == "hed_scribble":
            scribble_pil = extract_scribble_hed(face_pil)
            scribble_tensor = pil_to_tensor(scribble_pil, device)
        else:
            face_tensor = pil_to_tensor(face_pil, device)
            with torch.no_grad():
                scribble_tensor = sobel_proxy(face_tensor, device)
            scribble_pil = T.ToPILImage()(scribble_tensor.squeeze(0).cpu())
        scribble_pil.save(os.path.join(args.output_dir, f"scribble_{face_idx}.png"))
        print(f"  Scribble extracted via {args.edge_method}", flush=True)

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
            # Cast to UNet dtype (fp16) — VAE encode produces float32
            noised_latent = noised_latent.to(prompt_embeds.dtype)

            # Save noised image for visualization
            with torch.no_grad():
                noised_pil = latent_to_pil(noised_latent, architect.vae, architect.image_processor)

            # -- Regular denoising (no DPS) --
            print(f"  Regular denoising...", flush=True)
            scheduler_regular = copy.deepcopy(architect.scheduler)
            # Set step_index to start_step so scheduler knows where we are
            scheduler_regular._step_index = start_step

            regular_latent, _ = partial_denoise_loop(
                architect, scheduler_regular,
                noised_latent.detach().clone(), timesteps_partial,
                cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                dps_guidance=False, device=device,
            )

            with torch.no_grad():
                regular_pil = latent_to_pil(regular_latent, architect.vae, architect.image_processor)
            regular_pil.save(os.path.join(
                args.output_dir, f"regular_f{face_idx}_s{strength}.png"))

            del scheduler_regular
            gc.collect(); torch.cuda.empty_cache()

            # -- DPS denoising --
            print(f"  DPS denoising...", flush=True)
            scheduler_dps = copy.deepcopy(architect.scheduler)
            scheduler_dps._step_index = start_step

            sprinter.vae.to(dtype=torch.float32)

            dps_latent, dps_metrics = partial_denoise_loop(
                architect, scheduler_dps,
                noised_latent.detach().clone(), timesteps_partial,
                cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                dps_guidance=True, sprinter=sprinter, clip_model=clip_model,
                clip_processor=clip_processor, all_clip_embeddings=all_clip_embeddings,
                num_variations=args.num_variations, base_zeta_prime=args.base_zeta,
                device=device,
            )

            with torch.no_grad():
                dps_pil = latent_to_pil(dps_latent, architect.vae, architect.image_processor)
            dps_pil.save(os.path.join(
                args.output_dir, f"dps_f{face_idx}_s{strength}.png"))

            del scheduler_dps
            gc.collect(); torch.cuda.empty_cache()

            # Log DPS metrics to wandb
            for m in dps_metrics:
                wandb.log({
                    f"face{face_idx}/s{strength}/mmd": m['mmd_loss'],
                    f"face{face_idx}/s{strength}/grad_norm": m['gradient_norm'],
                    f"face{face_idx}/s{strength}/step": m['step'],
                })

            # -- Generate conditioned photos from both scribbles --
            print(f"  Generating {args.num_conditioned} photos from regular scribble...", flush=True)
            regular_photos = generate_conditioned_photos(
                sprinter, regular_pil, args.num_conditioned)

            print(f"  Generating {args.num_conditioned} photos from DPS scribble...", flush=True)
            dps_photos = generate_conditioned_photos(
                sprinter, dps_pil, args.num_conditioned)

            strength_results.append({
                'strength': strength,
                'noised_pil': noised_pil,
                'regular_pil': regular_pil,
                'dps_pil': dps_pil,
                'regular_photos': regular_photos,
                'dps_photos': dps_photos,
                'dps_metrics': dps_metrics,
            })

            all_metrics[f"face{face_idx}_s{strength}"] = dps_metrics

            del noised_latent, regular_latent, dps_latent, noise
            gc.collect(); torch.cuda.empty_cache()

        # -- Build composite image for this face --
        print(f"  Building composite image...", flush=True)
        composite = build_composite_image(face_pil, scribble_pil, strength_results, strengths)
        composite_path = os.path.join(args.output_dir, f"composite_face_{face_idx}.png")
        composite.save(composite_path)
        wandb.log({f"composite_face_{face_idx}": wandb.Image(composite_path)})
        print(f"  Composite saved: {composite_path}", flush=True)

        # Free per-face data
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
