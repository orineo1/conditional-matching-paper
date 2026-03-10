"""Compare denoising trajectories using generated scribble faces as starting point.

Workflow per face:
  1. Generate a face scribble using the fine-tuned model (pure generation, 4 steps)
  2. Encode scribble → latent → add noise
  3. Denoise with base SDXL Turbo, capture pred_x0 at every step
  4. Denoise with fine-tuned U-Net (same noise), capture pred_x0 at every step
  5. Build composite grids: rows=[base, fine-tuned], columns=steps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image, ImageDraw
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SD_cond_SD_controlnet"))
from generation import compute_pred_x0_direct

DENOISE_PROMPT = ""
DENOISE_NEGATIVE = ""
GUIDANCE_SCALE = 0.0


def pil_to_tensor_batch(images, device):
    """Convert PIL images to [-1, 1] tensor for VAE."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return torch.stack([transform(img) for img in images]).to(device=device, dtype=torch.float32)


def decode_latents(pipe, latents, vae_dtype):
    """Decode latents to PIL via VAE (float32 for stability)."""
    with torch.no_grad():
        pipe.vae.to(torch.float32)
        decoded = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor).sample
        pipe.vae.to(vae_dtype)
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    images = []
    for i in range(decoded.shape[0]):
        arr = (decoded[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def make_grid(images_2d, col_labels=None, row_labels=None, cell_size=128, padding=2, label_height=20):
    """Build grid from 2D list of PIL images with optional labels."""
    n_rows = len(images_2d)
    n_cols = len(images_2d[0])
    left_margin = 80 if row_labels else 0
    top_margin = label_height if col_labels else 0

    grid_w = left_margin + n_cols * (cell_size + padding) - padding
    grid_h = top_margin + n_rows * (cell_size + padding) - padding
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    if col_labels:
        for j, label in enumerate(col_labels):
            x = left_margin + j * (cell_size + padding) + cell_size // 2
            draw.text((x, 2), label, fill="black", anchor="mt")

    for i, row in enumerate(images_2d):
        y = top_margin + i * (cell_size + padding)
        if row_labels:
            draw.text((4, y + cell_size // 2), row_labels[i], fill="black", anchor="lm")
        for j, img in enumerate(row):
            x = left_margin + j * (cell_size + padding)
            grid.paste(img.resize((cell_size, cell_size)), (x, y))

    return grid


def generate_scribble_faces(pipe, num_faces, gen_prompt, gen_negative, gen_guidance, device):
    """Generate face scribbles using the (fine-tuned) pipeline via pure generation."""
    scribbles = []
    for i in range(num_faces):
        generator = torch.Generator(device=device).manual_seed(i + 100)
        with torch.no_grad():
            img = pipe(
                prompt=gen_prompt,
                negative_prompt=gen_negative,
                num_inference_steps=4,
                guidance_scale=gen_guidance,
                generator=generator,
            ).images[0]
        scribbles.append(img)
        print(f"  Generated scribble {i} (seed={i + 100})", flush=True)
    return scribbles


def denoise_with_trajectory(pipe, scribble_images, device, strength, n_steps, seed=42):
    """Denoise scribble images, capturing pred_x0 at every step.

    Returns: list of lists of PIL images — pred_x0_images[face_idx][step_idx]
    """
    pixel_tensor = pil_to_tensor_batch(scribble_images, device)
    vae_dtype = pipe.vae.dtype

    with torch.no_grad():
        pipe.vae.to(torch.float32)
        latent_dist = pipe.vae.encode(pixel_tensor).latent_dist
        original_latents = latent_dist.sample() * pipe.vae.config.scaling_factor
        pipe.vae.to(vae_dtype)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(original_latents.shape, generator=generator, device=device, dtype=torch.float16)

    pipe.scheduler.set_timesteps(n_steps, device=device)
    all_timesteps = pipe.scheduler.timesteps

    init_timestep = min(int(n_steps * strength), n_steps)
    t_start = max(n_steps - init_timestep, 0)
    denoise_timesteps = all_timesteps[t_start:]
    noise_timestep = denoise_timesteps[0]

    print(f"  strength={strength}: noise at t={noise_timestep.item():.0f}, "
          f"{len(denoise_timesteps)} denoise steps", flush=True)

    noisy_latents = pipe.scheduler.add_noise(
        original_latents.to(torch.float16), noise,
        torch.tensor([noise_timestep], device=device),
    )

    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=DENOISE_PROMPT, negative_prompt=DENOISE_NEGATIVE,
        device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)

    bs = len(scribble_images)
    all_pred_x0_images = [[] for _ in range(bs)]

    latents = noisy_latents.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(denoise_timesteps):
            latent_input = pipe.scheduler.scale_model_input(latents, t)
            latent_input = torch.cat([latent_input, latent_input])
            encoder_hs = torch.cat([
                negative_prompt_embeds.expand(bs, -1, -1),
                prompt_embeds.expand(bs, -1, -1),
            ])
            added_kwargs = {
                "text_embeds": torch.cat([
                    negative_pooled_prompt_embeds.expand(bs, -1),
                    pooled_prompt_embeds.expand(bs, -1),
                ]),
                "time_ids": torch.cat([time_ids.expand(bs, -1)] * 2),
            }
            noise_pred = pipe.unet(
                latent_input, t,
                encoder_hidden_states=encoder_hs,
                added_cond_kwargs=added_kwargs,
            ).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

            pred_x0 = compute_pred_x0_direct(pipe.scheduler, noise_pred, t, latents)
            pred_x0_pils = decode_latents(pipe, pred_x0, vae_dtype)
            for face_i, pil_img in enumerate(pred_x0_pils):
                all_pred_x0_images[face_i].append(pil_img)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if (step_idx + 1) % 5 == 0:
                print(f"    step {step_idx + 1}/{len(denoise_timesteps)}", flush=True)

    return all_pred_x0_images


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory comparison using generated scribble faces")
    parser.add_argument("--checkpoint", type=str, default="finetune_scribble/output/checkpoint-2000/merged")
    parser.add_argument("--gen_prompt", type=str, default="simple hand-drawn scribble of a male face")
    parser.add_argument("--gen_negative", type=str, default="detailed, realistic, photograph, shading")
    parser.add_argument("--gen_guidance", type=float, default=7.5)
    parser.add_argument("--strengths", type=str, default="0.5,0.8")
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--num_faces", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="finetune_scribble/output/trajectories_portrait")
    parser.add_argument("--wandb_project", type=str, default="scribble_finetune")
    parser.add_argument("--wandb_entity", type=str, default="conditional-matching")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    strengths = [float(s) for s in args.strengths.split(",")]
    output_dir = Path(args.output_dir)
    checkpoint_name = Path(args.checkpoint).parent.name

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=f"portrait-trajectory-{checkpoint_name}",
        settings=wandb.Settings(init_timeout=300),
        config={
            "checkpoint": args.checkpoint,
            "gen_prompt": args.gen_prompt,
            "strengths": strengths,
            "n_steps": args.n_steps,
            "num_faces": args.num_faces,
            "source": "generated_scribble",
        },
        job_type="portrait_trajectory",
    )

    # --- Load pipeline and generate starting scribbles with fine-tuned model ---
    print("\nLoading SDXL Turbo...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)
    base_unet = pipe.unet  # keep reference to base

    # Swap to fine-tuned for scribble generation
    ft_unet = UNet2DConditionModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16,
    ).to(device)
    pipe.unet = ft_unet

    print(f"\nGenerating {args.num_faces} scribble faces with fine-tuned model...", flush=True)
    print(f"  prompt: '{args.gen_prompt}'", flush=True)
    scribbles = generate_scribble_faces(
        pipe, args.num_faces, args.gen_prompt, args.gen_negative, args.gen_guidance, device)

    # Save scribbles
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, scribble in enumerate(scribbles):
        scribble.save(output_dir / f"source_scribble_{i}.png")

    # --- Base trajectories ---
    pipe.unet = base_unet
    print("\nRunning base trajectories...", flush=True)

    base_trajectories = {}
    for s in strengths:
        print(f"\nBase trajectory, strength={s}:", flush=True)
        base_trajectories[s] = denoise_with_trajectory(pipe, scribbles, device, s, args.n_steps)

    # --- Fine-tuned trajectories ---
    pipe.unet = ft_unet
    print("\nSwitched to fine-tuned U-Net", flush=True)

    ft_trajectories = {}
    for s in strengths:
        print(f"\nFine-tuned trajectory, strength={s}:", flush=True)
        ft_trajectories[s] = denoise_with_trajectory(pipe, scribbles, device, s, args.n_steps)

    # --- Build composites ---
    print("\nBuilding composites...", flush=True)
    all_wandb_images = []

    for face_i in range(len(scribbles)):
        face_dir = output_dir / f"face_{face_i}"
        face_dir.mkdir(parents=True, exist_ok=True)
        scribbles[face_i].save(face_dir / "scribble.png")

        for s in strengths:
            base_steps = base_trajectories[s][face_i]
            ft_steps = ft_trajectories[s][face_i]
            n_active = len(base_steps)

            for step_j, img in enumerate(base_steps):
                img.save(face_dir / f"base_s{s}_step_{step_j + 1:02d}.png")
            for step_j, img in enumerate(ft_steps):
                img.save(face_dir / f"ft_s{s}_step_{step_j + 1:02d}.png")

            # Composite with scribble as first column
            max_cols = 15
            page_idx = 0
            for page_start in range(0, n_active, max_cols):
                page_end = min(page_start + max_cols, n_active)
                base_page = base_steps[page_start:page_end]
                ft_page = ft_steps[page_start:page_end]

                if page_start == 0:
                    col_labels = ["Input"] + [f"t{j+1}" for j in range(page_start, page_end)]
                    grid = make_grid(
                        [[scribbles[face_i]] + base_page,
                         [scribbles[face_i]] + ft_page],
                        col_labels=col_labels,
                        row_labels=["Base", "FT"],
                    )
                else:
                    col_labels = [f"t{j+1}" for j in range(page_start, page_end)]
                    grid = make_grid(
                        [base_page, ft_page],
                        col_labels=col_labels,
                        row_labels=["Base", "FT"],
                    )

                suffix = f"_p{page_idx}" if n_active > max_cols else ""
                grid_path = face_dir / f"composite_s{s}{suffix}.png"
                grid.save(grid_path)

                caption = f"face={face_i} s={s} steps {page_start+1}-{page_end}"
                all_wandb_images.append(wandb.Image(grid, caption=caption))
                page_idx += 1

            print(f"  face {face_i}, strength {s}: {n_active} steps, "
                  f"{page_idx} page(s)", flush=True)

    wandb.log({"portrait_trajectories": all_wandb_images})
    print(f"\nLogged {len(all_wandb_images)} composite images to wandb", flush=True)

    wandb.finish()
    print(f"Done. Results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
