"""
DPS Guidance Sweep — pred_x0 Visualization Across Denoising Steps.

Starts from a noised face scribble (like run_dps_experiment.py), varies DPS
guidance strength (zeta), compares with/without LoRA, and shows pred_x0
decoded to PIL at every denoising step.

Output: single composite image (8 rows × N_steps columns).
  Rows: [No LoRA, ζ=0], [LoRA, ζ=0], [No LoRA, ζ=0.2], [LoRA, ζ=0.2], ...
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
    p = argparse.ArgumentParser(description="DPS Guidance Sweep: pred_x0 at every step")
    p.add_argument("--lora_path", type=str, default="scribble_tune/output/checkpoint-50000")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/sweep_guidance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--zetas", type=str, default="0,0.2,1.0,5.0")
    p.add_argument("--strength", type=float, default=0.5,
                   help="Noise level for partial noising")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="CFG scale (fixed)")
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--prompt", type=str,
                   default="rough pencil scribble outline, loose sketch, minimal line art")
    p.add_argument("--negative_prompt", type=str,
                   default="detailed, realistic, photograph, complex, colored, shading")
    p.add_argument("--n_targets", type=int, default=20,
                   help="Total target images (split evenly male/female)")
    p.add_argument("--num_variations", type=int, default=20)
    p.add_argument("--edge_method", type=str, default="hed_scribble",
                   choices=["sobel", "hed_scribble"])
    p.add_argument("--wandb_project", type=str, default="combined_conditional_flow")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")
    p.add_argument("--sprinter_every", type=int, default=3,
                   help="Generate sprinter photos every N denoising steps (0=disabled)")
    p.add_argument("--sprinter_count", type=int, default=5,
                   help="Number of sprinter photos to generate per snapshot")
    p.add_argument("--lora_only", action="store_true",
                   help="Only run WITH LoRA (skip no-LoRA rows)")
    p.add_argument("--no_lora_only", action="store_true",
                   help="Only run WITHOUT LoRA (skip LoRA rows)")
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


def generate_sprinter_photos(sprinter, scribble_pil, num_photos, batch_size=5):
    """Generate portrait photos from a scribble via sprinter (no grad)."""
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


def partial_denoise_loop_with_snapshots(
    architect, scheduler, latents, timesteps_partial,
    cfg_encoder_states, added_cond_kwargs, guidance_scale,
    dps_guidance=False, sprinter=None, clip_model=None,
    clip_processor=None, all_clip_embeddings=None,
    num_variations=20, base_zeta_prime=0.2, device="cuda",
    sprinter_every=0, sprinter_count=5,
):
    """Run denoising from a partial timestep schedule, capturing pred_x0 PIL at every step.

    If sprinter_every > 0, also generates sprinter photos from pred_x0 every N steps.

    Returns: (final_latents, list_of_pred_x0_pil_images, list_of_step_metrics,
              dict of {step_idx: list_of_sprinter_pil_images})
    """
    step_images = []
    step_metrics = []
    sprinter_snapshots = {}  # step_idx -> list of PIL images
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

        # Snapshot: decode pred_x0 to PIL
        with torch.no_grad():
            pil_img = latent_to_pil(pred_x0.detach(), architect.vae, architect.image_processor)
        step_images.append(pil_img)

        # Sprinter photos from pred_x0 scribble at interval
        if sprinter_every > 0 and sprinter is not None and i % sprinter_every == 0:
            print(f"    Generating {sprinter_count} sprinter photos at step {i+1}...", flush=True)
            sprinter_photos = generate_sprinter_photos(sprinter, pil_img, sprinter_count)
            sprinter_snapshots[i] = sprinter_photos

        correction = None
        if dps_guidance:
            # Decode pred_x0 to pixel space (keep grad)
            pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

            def vae_decode_checkpoint(lat):
                return architect.vae.decode(lat.to(architect.vae.dtype)).sample

            pixel_x0 = torch.utils.checkpoint.checkpoint(
                vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
            pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

            # DEBUG: check for NaN before CLIP-MMD
            print(f"      pred_x0 range: [{pred_x0.min().item():.4f}, {pred_x0.max().item():.4f}] nan={torch.isnan(pred_x0).sum().item()}", flush=True)
            print(f"      pixel_x0_norm range: [{pixel_x0_norm.min().item():.4f}, {pixel_x0_norm.max().item():.4f}] nan={torch.isnan(pixel_x0_norm).sum().item()}", flush=True)
            print(f"      all_clip_embeddings device={all_clip_embeddings.device} nan={torch.isnan(all_clip_embeddings).sum().item()}", flush=True)

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

    return latents, step_images, step_metrics, sprinter_snapshots


def build_composite_per_zeta(zeta, rows_for_zeta, n_steps, sprinter_every,
                             source_face_pil=None, scribble_pil=None, cell_size=128):
    """Build composite for one zeta value.

    rows_for_zeta: list of (label, step_images, sprinter_snapshots) tuples (1 or 2 rows).

    Layout per row:
      - pred_x0 row: label + N step images
      - sprinter row(s): below, showing 5 photos at each snapshot step

    Returns a PIL image.
    """
    label_width = 200
    header_height = 20
    photo_size = cell_size  # same size for sprinter photos
    info_height = cell_size + 10 if source_face_pil else 0

    # Count rows: for each entry, 1 pred_x0 row + 1 sprinter row (if snapshots exist)
    total_row_count = 0
    for _, _, sprinter_snaps in rows_for_zeta:
        total_row_count += 1  # pred_x0 row
        if sprinter_snaps:
            total_row_count += 1  # sprinter photos row

    total_w = label_width + n_steps * cell_size
    total_h = info_height + header_height + total_row_count * cell_size

    composite = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    y_cursor = 0

    # Source face + scribble thumbnails
    if source_face_pil and scribble_pil:
        composite.paste(source_face_pil.resize((cell_size, cell_size), Image.LANCZOS), (4, 4))
        composite.paste(scribble_pil.resize((cell_size, cell_size), Image.LANCZOS),
                        (4 + cell_size + 4, 4))
        draw.text((4, cell_size + 6), "Source / Scribble", fill="gray", font=font_small)
        y_cursor = info_height

    # Step number header
    for col in range(n_steps):
        x = label_width + col * cell_size + cell_size // 2
        draw.text((x, y_cursor + 2), str(col + 1), fill="black", font=font_small, anchor="mt")
    y_cursor += header_height

    # Rows
    for label, step_images, sprinter_snaps in rows_for_zeta:
        # pred_x0 row
        draw.text((4, y_cursor + cell_size // 2), label, fill="black", font=font, anchor="lm")
        for col, img in enumerate(step_images):
            if col >= n_steps:
                break
            thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
            composite.paste(thumb, (label_width + col * cell_size, y_cursor))
        y_cursor += cell_size

        # Sprinter photos row
        if sprinter_snaps:
            sprinter_label = f"  sprinter photos"
            draw.text((4, y_cursor + cell_size // 2), sprinter_label,
                      fill="gray", font=font_small, anchor="lm")

            for step_idx, photos in sorted(sprinter_snaps.items()):
                col = step_idx  # step_idx maps to column
                if col >= n_steps:
                    break
                # Tile photos horizontally within the cell's column span
                n_photos = len(photos)
                tile_w = cell_size // n_photos
                for p_idx, photo in enumerate(photos):
                    thumb = photo.resize((tile_w, cell_size), Image.LANCZOS)
                    x = label_width + col * cell_size + p_idx * tile_w
                    composite.paste(thumb, (x, y_cursor))

                # Draw step number above the photos
                draw.text((label_width + col * cell_size + cell_size // 2, y_cursor + 2),
                          f"s{step_idx+1}", fill="yellow", font=font_small, anchor="mt")

            y_cursor += cell_size

    return composite


def run_sweep_for_zetas(architect, sprinter, clip_model, clip_processor,
                        all_clip_embeddings, zetas, noised_latent,
                        timesteps_partial, start_step, cfg_encoder_states,
                        added_cond_kwargs, args, lora_label, device):
    """Run denoising for each zeta value, return list of (label, step_images) tuples."""
    # DEBUG: verify LoRA state matches label
    from peft import PeftModel
    has_lora = isinstance(architect.unet, PeftModel)
    print(f"  DEBUG: lora_label='{lora_label}', UNet is PeftModel={has_lora}", flush=True)
    rows = []

    for zeta in zetas:
        label = f"{lora_label}, \u03b6={zeta}"
        print(f"\n  --- {label} ---", flush=True)

        scheduler = copy.deepcopy(architect.scheduler)
        scheduler._step_index = start_step

        if zeta == 0:
            # No DPS guidance — still need sprinter for snapshot photos
            _, step_images, _, sprinter_snaps = partial_denoise_loop_with_snapshots(
                architect, scheduler,
                noised_latent.detach().clone(), timesteps_partial,
                cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                dps_guidance=False, sprinter=sprinter, device=device,
                sprinter_every=args.sprinter_every, sprinter_count=args.sprinter_count,
            )
        else:
            # DPS guidance
            sprinter.vae.to(dtype=torch.float32)

            _, step_images, step_metrics, sprinter_snaps = partial_denoise_loop_with_snapshots(
                architect, scheduler,
                noised_latent.detach().clone(), timesteps_partial,
                cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                dps_guidance=True, sprinter=sprinter, clip_model=clip_model,
                clip_processor=clip_processor, all_clip_embeddings=all_clip_embeddings,
                num_variations=args.num_variations, base_zeta_prime=zeta,
                device=device,
                sprinter_every=args.sprinter_every, sprinter_count=args.sprinter_count,
            )

        rows.append((label, step_images, sprinter_snaps))

        del scheduler
        gc.collect(); torch.cuda.empty_cache()

    return rows


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zetas = [float(z) for z in args.zetas.split(",")]
    print(f"Device: {device}", flush=True)
    print(f"Zetas: {zetas}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="sweep-guidance",
        config=vars(args),
    )

    # ── 1. Load models ───────────────────────────────────────────────────────
    initial_lora = args.lora_path if args.lora_only else None
    print(f"Loading models (lora_path={initial_lora})...", flush=True)
    architect, sprinter = load_models(device, architect_lora_path=initial_lora)
    clip_model, clip_processor = load_clip_model(device)
    setup_gradient_checkpointing(architect, sprinter)
    print("Models loaded.", flush=True)

    # ── 2. Base image (oval) for target generation ─────────────────────────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    # ── 3. Generate target distribution ────────────────────────────────────────
    n_half = args.n_targets // 2
    print(f"Generating {args.n_targets} target images ({n_half} man + {n_half} woman)...",
          flush=True)

    with torch.no_grad():
        man_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )
        woman_images, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a woman, studio lighting",
            sobel_cond_pil, n_half, batch_size=2, cn_scale=args.controlnet_scale,
        )

    # ── 4. Encode targets to CLIP ──────────────────────────────────────────────
    with torch.no_grad():
        man_clip = encode_images_clip(pil_images_to_tensor(man_images, device),
                                       clip_model, clip_processor)
        woman_clip = encode_images_clip(pil_images_to_tensor(woman_images, device),
                                         clip_model, clip_processor)
    all_clip_embeddings = torch.cat([man_clip, woman_clip], dim=0)
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape} "
          f"nan={torch.isnan(all_clip_embeddings).sum().item()} "
          f"range=[{all_clip_embeddings.min().item():.4f}, {all_clip_embeddings.max().item():.4f}]",
          flush=True)

    del man_images, woman_images, man_clip, woman_clip
    clip_model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    # ── 5. Generate 1 source face ──────────────────────────────────────────────
    print("Generating source face...", flush=True)
    with torch.no_grad():
        face_imgs, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, 1, batch_size=1, cn_scale=args.controlnet_scale,
        )
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
    print(f"Scribble extracted via {args.edge_method}", flush=True)

    # Encode scribble to latent + add noise
    with torch.no_grad():
        scribble_latent = encode_to_latent(architect.vae, scribble_tensor)

    # ── 6. Prepare partial noising ─────────────────────────────────────────────
    n_steps = args.n_steps
    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps
    start_step = int(n_steps * (1 - args.strength))
    timesteps_partial = timesteps[start_step:]
    print(f"Start step: {start_step}, denoising {len(timesteps_partial)} steps", flush=True)

    noise = torch.randn_like(scribble_latent)
    t_start = timesteps_partial[0]
    noised_latent = architect.scheduler.add_noise(scribble_latent, noise, t_start.unsqueeze(0))

    # ── 7. Prepare prompt embeddings ───────────────────────────────────────────
    height, width = 512, 512
    prompt = args.prompt if args.prompt else ""
    negative_prompt = args.negative_prompt if args.negative_prompt else ""

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

    # Cast noised latent to UNet dtype
    noised_latent = noised_latent.to(prompt_embeds.dtype)

    # ── 8. Sweep zetas ───────────────────────────────────────────────────────
    rows_no_lora = None
    rows_lora = None

    if not args.lora_only:
        print("\n" + "=" * 70, flush=True)
        print("SWEEP: No LoRA", flush=True)
        print("=" * 70, flush=True)

        rows_no_lora = run_sweep_for_zetas(
            architect, sprinter, clip_model, clip_processor,
            all_clip_embeddings, zetas, noised_latent,
            timesteps_partial, start_step, cfg_encoder_states,
            added_cond_kwargs, args, "No LoRA", device,
        )

    if not args.no_lora_only:
        if not args.lora_only:
            # Need to reload architect with LoRA
            print("\nReloading architect with LoRA...", flush=True)
            del architect
            gc.collect(); torch.cuda.empty_cache()

            architect, sprinter_new = load_models(device, architect_lora_path=args.lora_path)
            setup_gradient_checkpointing(architect, sprinter_new)
            del sprinter_new
            gc.collect(); torch.cuda.empty_cache()

            # Re-encode prompts with new architect
            with torch.no_grad():
                (prompt_embeds, negative_prompt_embeds,
                 pooled_prompt_embeds, negative_pooled_prompt_embeds,
                ) = architect.encode_prompt(
                    prompt=prompt, negative_prompt=negative_prompt,
                    device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
                )

            added_cond_kwargs = {
                "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
                "time_ids": add_time_ids.repeat(2, 1),
            }
            cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            architect.scheduler.set_timesteps(n_steps, device=device)

        print("\n" + "=" * 70, flush=True)
        print("SWEEP: With LoRA", flush=True)
        print("=" * 70, flush=True)

        rows_lora = run_sweep_for_zetas(
            architect, sprinter, clip_model, clip_processor,
            all_clip_embeddings, zetas, noised_latent,
            timesteps_partial, start_step, cfg_encoder_states,
            added_cond_kwargs, args, "LoRA", device,
        )

    # ── 9. Build per-zeta composite images ──────────────────────────────────
    print("\nBuilding composite images...", flush=True)
    n_actual_steps = len(timesteps_partial)

    for z_idx, zeta in enumerate(zetas):
        # Collect rows for this zeta value
        rows_for_zeta = []
        if rows_no_lora:
            rows_for_zeta.append(rows_no_lora[z_idx])
        if rows_lora:
            rows_for_zeta.append(rows_lora[z_idx])

        composite = build_composite_per_zeta(
            zeta, rows_for_zeta, n_actual_steps,
            sprinter_every=args.sprinter_every,
            source_face_pil=face_pil, scribble_pil=scribble_pil,
        )

        zeta_str = str(zeta).replace(".", "_")
        out_path = os.path.join(args.output_dir, f"sweep_zeta_{zeta_str}.png")
        composite.save(out_path)
        print(f"Saved: {out_path}", flush=True)
        wandb.log({f"sweep_zeta_{zeta}": wandb.Image(composite,
                   caption=f"DPS Sweep zeta={zeta}")})

    wandb.finish()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
