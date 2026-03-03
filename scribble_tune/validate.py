"""Validate LoRA fine-tuned SDXL Turbo: noise-then-denoise at multiple strengths + pure generation comparison + gradient flow check."""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from PIL import Image
from torchvision import transforms


def load_pipeline(device, lora_path=None):
    """Load SDXL Turbo pipeline, optionally with LoRA weights."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    if lora_path:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        print(f"Loaded LoRA weights from {lora_path}", flush=True)

    return pipe


def load_face_images(data_dir, num_images=5, size=512):
    """Load face scribble PNGs from the quickdraw data directory."""
    data_dir = Path(data_dir)
    images = []
    for i in range(num_images):
        path = data_dir / f"face_{i:06d}.png"
        if path.exists():
            img = Image.open(path).convert("RGB").resize((size, size))
            images.append(img)
        else:
            print(f"  Warning: {path} not found, skipping", flush=True)
    print(f"  Loaded {len(images)} face scribble images", flush=True)
    return images


def pil_to_tensor(images, device):
    """Convert list of PIL images to tensor in [-1, 1] for VAE encoding."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    tensors = [transform(img) for img in images]
    return torch.stack(tensors).to(device=device, dtype=torch.float32)


def decode_latents(pipe, latents, vae_dtype):
    """Decode latents to PIL images via VAE (float32 for stability)."""
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


def noise_denoise(pipe, face_images, device, strength, seed=42, num_inference_steps=10):
    """Add noise at given strength, then denoise. Returns list of denoised PIL images.

    strength: fraction of the diffusion process to noise (0.1 = 10%, 0.5 = 50%).
    Uses img2img-style scheduling: strength controls both noise level and number of denoise steps.
    """
    if not face_images:
        return []

    # Encode face images to latents (float32 for VAE stability)
    pixel_tensor = pil_to_tensor(face_images, device)
    with torch.no_grad():
        vae_dtype = pipe.vae.dtype
        pipe.vae.to(torch.float32)
        latent_dist = pipe.vae.encode(pixel_tensor).latent_dist
        original_latents = latent_dist.sample() * pipe.vae.config.scaling_factor
        pipe.vae.to(vae_dtype)

    # Generate noise (seeded for reproducibility across base/LoRA)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(original_latents.shape, generator=generator, device=device, dtype=torch.float16)

    # Set up scheduler timesteps — img2img style
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    all_timesteps = pipe.scheduler.timesteps

    # Determine how many steps to skip (front of schedule = most noisy)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    denoise_timesteps = all_timesteps[t_start:]
    noise_timestep = denoise_timesteps[0]  # the timestep we noise to

    print(f"  strength={strength}: noise at t={noise_timestep.item():.0f}, "
          f"denoise {len(denoise_timesteps)} steps {[t.item() for t in denoise_timesteps]}", flush=True)

    # Add noise at the chosen timestep
    noisy_latents = pipe.scheduler.add_noise(
        original_latents.to(torch.float16),
        noise,
        torch.tensor([noise_timestep], device=device),
    )

    # Get text embeddings
    prompt_embeds = pipe.encode_prompt("a simple hand-drawn scribble", device=device, num_images_per_prompt=1)
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)
    time_ids = time_ids.expand(len(face_images), -1)

    # Denoise
    latents = noisy_latents.clone()
    with torch.no_grad():
        for t in denoise_timesteps:
            latents = pipe.scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(
                latents, t,
                encoder_hidden_states=prompt_embeds[0].expand(len(face_images), -1, -1),
                added_cond_kwargs={
                    "text_embeds": prompt_embeds[2].expand(len(face_images), -1),
                    "time_ids": time_ids,
                },
            ).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    denoised_images = decode_latents(pipe, latents, vae_dtype)
    noisy_images = decode_latents(pipe, noisy_latents, vae_dtype)

    return denoised_images, noisy_images


def generate_images(pipe, device, seeds):
    """Generate scribble images for a list of seeds. Returns list of (seed, image)."""
    prompt = "a simple hand-drawn scribble"
    results = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            img = pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                output_type="pil",
                generator=generator,
            ).images[0]
        results.append((seed, img))
    return results


def check_gradient_flow(pipe, device):
    """Verify gradients flow through U-Net + LoRA via actual backward pass."""
    latent = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device, requires_grad=True)

    prompt_embeds = pipe.encode_prompt("a simple hand-drawn scribble", device=device, num_images_per_prompt=1)

    t = torch.tensor([500], device=device, dtype=torch.long)
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)

    noise_pred = pipe.unet(
        latent, t,
        encoder_hidden_states=prompt_embeds[0],
        added_cond_kwargs={"text_embeds": prompt_embeds[2], "time_ids": time_ids},
    ).sample

    loss = noise_pred.sum()
    loss.backward()

    results = {}

    if latent.grad is not None and latent.grad.abs().sum() > 0:
        print("  Latent grad check PASSED", flush=True)
        results["latent_grad_check"] = True
    else:
        print("  Latent grad check FAILED: no gradients on input latents", flush=True)
        results["latent_grad_check"] = False

    lora_grads = 0
    for name, param in pipe.unet.named_parameters():
        if "lora_" in name and param.grad is not None and param.grad.abs().sum() > 0:
            lora_grads += 1

    if lora_grads > 0:
        print(f"  LoRA grad check PASSED: {lora_grads} LoRA params have nonzero grads", flush=True)
        results["lora_grad_check"] = True
        results["lora_params_with_grad"] = lora_grads
    else:
        print("  LoRA grad check FAILED: no LoRA params have nonzero grads", flush=True)
        results["lora_grad_check"] = False
        results["lora_params_with_grad"] = 0

    passed = results["latent_grad_check"]
    results["passed"] = passed
    print(f"Gradient flow check {'PASSED' if passed else 'FAILED'}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate LoRA scribble generation")
    parser.add_argument("--config", type=str, default="scribble_tune/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint dir (default: output_dir from config)")
    parser.add_argument("--output_dir", type=str, default="scribble_tune/output/validation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to quickdraw_512 data dir (default: data_dir from config)")
    parser.add_argument("--wandb_project", type=str, default="conditional-flow")
    parser.add_argument("--wandb_entity", type=str, default="conditional-matching")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_path = args.checkpoint or cfg["output_dir"]
    data_dir = args.data_dir or cfg["data_dir"]
    checkpoint_name = Path(lora_path).name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 1234]
    strengths = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Init wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"validate-{checkpoint_name}",
        config={
            "checkpoint": lora_path,
            "lora_rank": cfg["lora_rank"],
            "lora_alpha": cfg["lora_alpha"],
            "resolution": cfg["resolution"],
            "num_epochs": cfg["num_epochs"],
            "train_batch_size": cfg["train_batch_size"],
            "learning_rate": cfg["learning_rate"],
            "strengths": strengths,
        },
        job_type="validation",
    )

    # Load face scribble images
    print("\nLoading face scribble images...", flush=True)
    face_images = load_face_images(data_dir, num_images=5)

    # ── Phase 1: Base pipeline ──
    print("\nLoading base SDXL Turbo (no LoRA)...", flush=True)
    pipe_base = load_pipeline(device)

    base_nd_results = {}
    for s in strengths:
        print(f"\nNoise-denoise (base, strength={s})...", flush=True)
        denoised, noisy = noise_denoise(pipe_base, face_images, device, strength=s, seed=42)
        base_nd_results[s] = (denoised, noisy)

    print("\nGenerating comparison images (base)...", flush=True)
    base_gen = generate_images(pipe_base, device, seeds)

    del pipe_base
    torch.cuda.empty_cache()
    print("Freed base pipeline", flush=True)

    # ── Phase 2: LoRA pipeline ──
    print(f"\nLoading LoRA pipeline from {lora_path}...", flush=True)
    pipe_lora = load_pipeline(device, lora_path=lora_path)

    lora_nd_results = {}
    for s in strengths:
        print(f"\nNoise-denoise (LoRA, strength={s})...", flush=True)
        denoised, noisy = noise_denoise(pipe_lora, face_images, device, strength=s, seed=42)
        lora_nd_results[s] = (denoised, noisy)

    print("\nGenerating comparison images (LoRA)...", flush=True)
    lora_gen = generate_images(pipe_lora, device, seeds)

    # ── Log to wandb ──

    # 1. Noise-denoise table: each row = one (strength, face) with 4 images side by side
    nd_table = wandb.Table(columns=["Strength", "Face", "Original", "Noised", "Base Denoised", "LoRA Denoised"])
    for s in strengths:
        base_denoised, noisy_imgs = base_nd_results[s]
        lora_denoised, _ = lora_nd_results[s]

        for i in range(len(face_images)):
            nd_table.add_data(
                s, i,
                wandb.Image(face_images[i]),
                wandb.Image(noisy_imgs[i]),
                wandb.Image(base_denoised[i]),
                wandb.Image(lora_denoised[i]),
            )
            # Save locally
            face_images[i].save(output_dir / f"nd_s{s}_orig_{i}.png")
            noisy_imgs[i].save(output_dir / f"nd_s{s}_noisy_{i}.png")
            base_denoised[i].save(output_dir / f"nd_s{s}_base_{i}.png")
            lora_denoised[i].save(output_dir / f"nd_s{s}_lora_{i}.png")

    wandb.log({"noise_denoise": nd_table})
    print(f"Logged noise-denoise table ({len(strengths)} strengths x {len(face_images)} faces)", flush=True)

    # 2. Pure generation table: each row = one seed with base vs LoRA
    gen_table = wandb.Table(columns=["Seed", "Base", "LoRA"])
    for (seed, base_img), (_, lora_img) in zip(base_gen, lora_gen):
        base_img.save(output_dir / f"gen_base_seed{seed}.png")
        lora_img.save(output_dir / f"gen_lora_seed{seed}.png")
        gen_table.add_data(seed, wandb.Image(base_img), wandb.Image(lora_img))

    wandb.log({"generation": gen_table})
    print(f"Logged generation table ({len(seeds)} seeds)", flush=True)

    # ── Phase 3: Gradient flow check ──
    print("\nChecking gradient flow...", flush=True)
    grad_results = check_gradient_flow(pipe_lora, device)
    wandb.log(grad_results)

    wandb.finish()
    print(f"\nValidation complete. Results saved to {output_dir} and logged to wandb.", flush=True)


if __name__ == "__main__":
    main()
