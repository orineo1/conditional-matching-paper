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


def partial_denoise_loop_with_snapshots(
    architect, scheduler, latents, timesteps_partial,
    cfg_encoder_states, added_cond_kwargs, guidance_scale,
    dps_guidance=False, sprinter=None, clip_model=None,
    clip_processor=None, all_clip_embeddings=None,
    num_variations=20, base_zeta_prime=0.2, device="cuda",
):
    """Run denoising from a partial timestep schedule, capturing pred_x0 PIL at every step.

    Returns: (final_latents, list_of_pred_x0_pil_images, list_of_step_metrics)
    """
    step_images = []
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

        # Snapshot: decode pred_x0 to PIL
        with torch.no_grad():
            pil_img = latent_to_pil(pred_x0.detach(), architect.vae, architect.image_processor)
        step_images.append(pil_img)

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

    return latents, step_images, step_metrics


def build_composite(all_rows, zetas, n_steps, source_face_pil=None,
                    scribble_pil=None, cell_size=128):
    """Build composite image: 8 rows × n_steps columns.

    all_rows: list of (label_str, list_of_pil_images) tuples, length 8.
    """
    label_width = 180
    header_height = 20
    info_height = cell_size + 10 if source_face_pil else 0

    total_w = label_width + n_steps * cell_size
    total_h = info_height + header_height + len(all_rows) * cell_size

    composite = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    y_start = 0

    # Source face + scribble thumbnails
    if source_face_pil and scribble_pil:
        thumb_size = cell_size
        composite.paste(source_face_pil.resize((thumb_size, thumb_size), Image.LANCZOS),
                        (4, 4))
        composite.paste(scribble_pil.resize((thumb_size, thumb_size), Image.LANCZOS),
                        (4 + thumb_size + 4, 4))
        draw.text((4, thumb_size + 6), "Source / Scribble", fill="gray", font=font_small)
        y_start = info_height

    # Step number header
    for col in range(n_steps):
        x = label_width + col * cell_size + cell_size // 2
        draw.text((x, y_start + 2), str(col + 1), fill="black", font=font_small, anchor="mt")

    y_start += header_height

    # Rows
    for row_idx, (label, images) in enumerate(all_rows):
        y_offset = y_start + row_idx * cell_size

        # Row label
        draw.text((4, y_offset + cell_size // 2), label, fill="black", font=font, anchor="lm")

        # Cells
        for col, img in enumerate(images):
            if col >= n_steps:
                break
            thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
            composite.paste(thumb, (label_width + col * cell_size, y_offset))

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
            # No DPS guidance
            _, step_images, _ = partial_denoise_loop_with_snapshots(
                architect, scheduler,
                noised_latent.detach().clone(), timesteps_partial,
                cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                dps_guidance=False, device=device,
            )
        else:
            # DPS guidance
            sprinter.vae.to(dtype=torch.float32)

            _, step_images, step_metrics = partial_denoise_loop_with_snapshots(
                architect, scheduler,
                noised_latent.detach().clone(), timesteps_partial,
                cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                dps_guidance=True, sprinter=sprinter, clip_model=clip_model,
                clip_processor=clip_processor, all_clip_embeddings=all_clip_embeddings,
                num_variations=args.num_variations, base_zeta_prime=zeta,
                device=device,
            )

        rows.append((label, step_images))

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

    # ── 1. Load models (no LoRA first) ─────────────────────────────────────────
    print("Loading models (no LoRA)...", flush=True)
    architect, sprinter = load_models(device, architect_lora_path=None)
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

    # ── 8. Sweep zetas WITHOUT LoRA ────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("SWEEP: No LoRA", flush=True)
    print("=" * 70, flush=True)

    rows_no_lora = run_sweep_for_zetas(
        architect, sprinter, clip_model, clip_processor,
        all_clip_embeddings, zetas, noised_latent,
        timesteps_partial, start_step, cfg_encoder_states,
        added_cond_kwargs, args, "No LoRA", device,
    )

    # ── 9. Reload architect WITH LoRA ──────────────────────────────────────────
    print("\nReloading architect with LoRA...", flush=True)
    del architect
    gc.collect(); torch.cuda.empty_cache()

    # Re-load full models with LoRA
    architect, sprinter_new = load_models(device, architect_lora_path=args.lora_path)
    setup_gradient_checkpointing(architect, sprinter_new)
    # Keep existing sprinter (already loaded, same weights)
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

    # Re-setup scheduler for LoRA architect
    architect.scheduler.set_timesteps(n_steps, device=device)

    # ── 10. Sweep zetas WITH LoRA ──────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("SWEEP: With LoRA", flush=True)
    print("=" * 70, flush=True)

    rows_lora = run_sweep_for_zetas(
        architect, sprinter, clip_model, clip_processor,
        all_clip_embeddings, zetas, noised_latent,
        timesteps_partial, start_step, cfg_encoder_states,
        added_cond_kwargs, args, "LoRA", device,
    )

    # ── 11. Build composite ───────────────────────────────────────────────────
    print("\nBuilding composite image...", flush=True)

    # Interleave: No LoRA ζ=0, LoRA ζ=0, No LoRA ζ=0.2, LoRA ζ=0.2, ...
    all_rows = []
    for z_idx in range(len(zetas)):
        all_rows.append(rows_no_lora[z_idx])
        all_rows.append(rows_lora[z_idx])

    n_actual_steps = len(timesteps_partial)
    composite = build_composite(
        all_rows, zetas, n_actual_steps,
        source_face_pil=face_pil, scribble_pil=scribble_pil,
    )

    out_path = os.path.join(args.output_dir, "sweep_guidance.png")
    composite.save(out_path)
    print(f"Saved composite to {out_path}", flush=True)

    wandb.log({"sweep_guidance": wandb.Image(composite, caption="DPS Guidance Sweep")})
    wandb.finish()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
