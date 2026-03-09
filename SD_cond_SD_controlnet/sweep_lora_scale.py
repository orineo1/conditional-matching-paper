"""
LoRA Scale Sweep — How does LoRA strength affect scribble quality?

For each LoRA scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
  1. Set architect LoRA adapter scaling
  2. Denoise from noised scribble (no DPS, just regular denoising)
  3. Save pred_x0 at every step + sprinter photos every N steps

No DPS/CLIP/targets needed — just architect + sprinter.
"""

import argparse
import copy
import gc
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

from generation import compute_pred_x0_direct, denoise_step, generate_and_store_cs, predict_noise_cfg
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from models import load_models, setup_gradient_checkpointing


def parse_args():
    p = argparse.ArgumentParser(description="LoRA Scale Sweep")
    p.add_argument("--lora_path", type=str, default="scribble_tune/output/checkpoint-50000")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/sweep_lora_scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--lora_scales", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--prompt", type=str,
                   default="rough pencil scribble outline, loose sketch, minimal line art")
    p.add_argument("--negative_prompt", type=str,
                   default="detailed, realistic, photograph, complex, colored, shading")
    p.add_argument("--sprinter_every", type=int, default=3)
    p.add_argument("--sprinter_count", type=int, default=5)
    p.add_argument("--edge_method", type=str, default="hed_scribble",
                   choices=["sobel", "hed_scribble"])
    p.add_argument("--wandb_project", type=str, default="combined_conditional_flow")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")
    return p.parse_args()


def extract_scribble_hed(pil_image):
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_image, scribble=True)


def pil_to_tensor(pil_img, device):
    return TF.to_tensor(pil_img).unsqueeze(0).to(device).float()


def encode_to_latent(vae, image_tensor):
    scaled = image_tensor * 2.0 - 1.0
    latent_dist = vae.encode(scaled.to(vae.dtype)).latent_dist
    return latent_dist.sample().float() * vae.config.scaling_factor


def generate_sprinter_photos(sprinter, scribble_pil, num_photos, batch_size=5):
    all_images = []
    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    with torch.no_grad():
        for start in range(0, num_photos, batch_size):
            bs = min(batch_size, num_photos - start)
            result = sprinter(
                prompt=["a superrealistic professional photograph"] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pil", return_dict=True,
            )
            all_images.extend(result.images)
    sprinter.vae.to(dtype=original_vae_dtype)
    return all_images


def denoise_with_snapshots(architect, scheduler, latents, timesteps_partial,
                           cfg_encoder_states, added_cond_kwargs, guidance_scale,
                           sprinter, sprinter_every, sprinter_count):
    """Simple denoising loop (no DPS), saving pred_x0 + sprinter photos."""
    step_images = []
    sprinter_snapshots = {}

    for i, t in enumerate(timesteps_partial):
        with torch.no_grad():
            noise_pred = predict_noise_cfg(
                architect.unet, scheduler, latents, t,
                cfg_encoder_states, added_cond_kwargs, guidance_scale,
            )
            pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents)
            pil_img = latent_to_pil(pred_x0, architect.vae, architect.image_processor)
            step_images.append(pil_img)

            # Sprinter photos at interval
            if sprinter_every > 0 and i % sprinter_every == 0:
                print(f"      Sprinter photos at step {i+1}...", flush=True)
                sprinter_snapshots[i] = generate_sprinter_photos(
                    sprinter, pil_img, sprinter_count)

            latents = denoise_step(scheduler, noise_pred, t, latents)

    return step_images, sprinter_snapshots


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_scales = [float(s) for s in args.lora_scales.split(",")]
    print(f"Device: {device}", flush=True)
    print(f"LoRA scales: {lora_scales}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name="sweep-lora-scale", config=vars(args),
    )

    # ── Load models WITH LoRA ──────────────────────────────────────────────────
    print("Loading models (with LoRA)...", flush=True)
    architect, sprinter = load_models(device, architect_lora_path=args.lora_path)
    print("Models loaded.", flush=True)

    # Verify LoRA is loaded
    from peft import PeftModel
    assert isinstance(architect.unet, PeftModel), "LoRA not loaded!"
    print(f"LoRA loaded from {args.lora_path}", flush=True)

    # ── Generate source face + scribble ────────────────────────────────────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    print("Generating source face...", flush=True)
    with torch.no_grad():
        face_imgs, _ = generate_and_store_cs(
            sprinter, "a superrealistic portrait photograph of a man, studio lighting",
            sobel_cond_pil, 1, batch_size=1, cn_scale=args.controlnet_scale,
        )
    face_pil = face_imgs[0]
    face_pil.save(os.path.join(args.output_dir, "source_face.png"))

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

    # ── Encode + noise ─────────────────────────────────────────────────────────
    with torch.no_grad():
        scribble_latent = encode_to_latent(architect.vae, scribble_tensor)

    n_steps = args.n_steps
    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps
    start_step = int(n_steps * (1 - args.strength))
    timesteps_partial = timesteps[start_step:]
    print(f"Start step: {start_step}, denoising {len(timesteps_partial)} steps", flush=True)

    noise = torch.randn_like(scribble_latent)
    t_start = timesteps_partial[0]
    noised_latent = architect.scheduler.add_noise(scribble_latent, noise, t_start.unsqueeze(0))

    # ── Prompt embeddings ──────────────────────────────────────────────────────
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
    noised_latent = noised_latent.to(prompt_embeds.dtype)

    # ── Sweep LoRA scales ──────────────────────────────────────────────────────
    for scale in lora_scales:
        print(f"\n{'='*60}", flush=True)
        print(f"LoRA scale = {scale}", flush=True)
        print(f"{'='*60}", flush=True)

        # Set LoRA scaling
        for name, module in architect.unet.named_modules():
            if hasattr(module, 'scaling'):
                for key in module.scaling:
                    module.scaling[key] = scale

        # Fresh scheduler
        scheduler = copy.deepcopy(architect.scheduler)
        scheduler._step_index = start_step

        step_images, sprinter_snaps = denoise_with_snapshots(
            architect, scheduler,
            noised_latent.detach().clone(), timesteps_partial,
            cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
            sprinter, args.sprinter_every, args.sprinter_count,
        )

        # Save individual images
        scale_str = str(scale).replace(".", "_")
        img_dir = os.path.join(args.output_dir, f"scale_{scale_str}")
        os.makedirs(img_dir, exist_ok=True)

        for s_idx, img in enumerate(step_images):
            img.save(os.path.join(img_dir, f"pred_x0_step_{s_idx+1:02d}.png"))

        for s_idx, photos in sorted(sprinter_snaps.items()):
            for p_idx, photo in enumerate(photos):
                photo.save(os.path.join(img_dir, f"sprinter_step_{s_idx+1:02d}_img{p_idx+1}.png"))

        n_sprinter = sum(len(p) for p in sprinter_snaps.values())
        print(f"  Saved {len(step_images)} pred_x0 + {n_sprinter} sprinter images to {img_dir}",
              flush=True)

        # Log final pred_x0 to wandb
        wandb.log({f"final_scale_{scale}": wandb.Image(step_images[-1],
                   caption=f"Final pred_x0, LoRA scale={scale}")})

        del scheduler
        gc.collect(); torch.cuda.empty_cache()

    wandb.finish()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
