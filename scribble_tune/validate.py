"""Validate LoRA fine-tuned SDXL Turbo: noise-then-denoise comparison + pure generation comparison + gradient flow check."""

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


def noise_denoise_comparison(pipe, face_images, device, noise_strength=500, seed=42):
    """Noise face scribbles to a midpoint, then denoise. Returns (original, noised_viz, denoised) triplets."""
    if not face_images:
        print("  No face images to process", flush=True)
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

    # Add noise at midpoint timestep
    noisy_latents = pipe.scheduler.add_noise(
        original_latents.to(torch.float16),
        noise,
        torch.tensor([noise_strength], device=device),
    )

    # Get text embeddings for "a simple hand-drawn scribble"
    prompt_embeds = pipe.encode_prompt("a simple hand-drawn scribble", device=device, num_images_per_prompt=1)
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)
    time_ids = time_ids.expand(len(face_images), -1)

    # Set up scheduler and find timesteps >= noise_strength
    pipe.scheduler.set_timesteps(4, device=device)
    all_timesteps = pipe.scheduler.timesteps
    filtered_timesteps = [t for t in all_timesteps if t >= noise_strength]
    print(f"  Scheduler timesteps: {all_timesteps.tolist()}, using: {[t.item() for t in filtered_timesteps]}", flush=True)

    # Denoise from midpoint
    latents = noisy_latents.clone()
    with torch.no_grad():
        for t in filtered_timesteps:
            noise_pred = pipe.unet(
                latents, t,
                encoder_hidden_states=prompt_embeds[0].expand(len(face_images), -1, -1),
                added_cond_kwargs={
                    "text_embeds": prompt_embeds[2].expand(len(face_images), -1),
                    "time_ids": time_ids,
                },
            ).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode denoised latents back to images
    with torch.no_grad():
        pipe.vae.to(torch.float32)
        decoded = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor).sample
        pipe.vae.to(vae_dtype)

    # Convert to PIL
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    denoised_images = []
    for i in range(decoded.shape[0]):
        arr = (decoded[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        denoised_images.append(Image.fromarray(arr))

    # Also decode noisy latents for visualization
    with torch.no_grad():
        pipe.vae.to(torch.float32)
        noisy_decoded = pipe.vae.decode(noisy_latents.to(torch.float32) / pipe.vae.config.scaling_factor).sample
        pipe.vae.to(vae_dtype)

    noisy_decoded = (noisy_decoded / 2 + 0.5).clamp(0, 1)
    noisy_images = []
    for i in range(noisy_decoded.shape[0]):
        arr = (noisy_decoded[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        noisy_images.append(Image.fromarray(arr))

    return list(zip(face_images, noisy_images, denoised_images))


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
    parser.add_argument("--noise_strength", type=int, default=500,
                        help="Timestep for noise injection (default: 500, ~50%% corruption)")
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
            "noise_strength": args.noise_strength,
        },
        job_type="validation",
    )

    # Load face scribble images
    print("\nLoading face scribble images...", flush=True)
    face_images = load_face_images(data_dir, num_images=5)

    # ── Phase 1: Base pipeline ──
    print("\nLoading base SDXL Turbo (no LoRA)...", flush=True)
    pipe_base = load_pipeline(device)

    print("\nNoise-denoise comparison (base)...", flush=True)
    base_triplets = noise_denoise_comparison(pipe_base, face_images, device,
                                              noise_strength=args.noise_strength, seed=42)

    print("\nGenerating comparison images (base)...", flush=True)
    base_gen = generate_images(pipe_base, device, seeds)

    del pipe_base
    torch.cuda.empty_cache()
    print("Freed base pipeline", flush=True)

    # ── Phase 2: LoRA pipeline ──
    print(f"\nLoading LoRA pipeline from {lora_path}...", flush=True)
    pipe_lora = load_pipeline(device, lora_path=lora_path)

    print("\nNoise-denoise comparison (LoRA)...", flush=True)
    lora_triplets = noise_denoise_comparison(pipe_lora, face_images, device,
                                              noise_strength=args.noise_strength, seed=42)

    print("\nGenerating comparison images (LoRA)...", flush=True)
    lora_gen = generate_images(pipe_lora, device, seeds)

    # ── Log to wandb ──

    # 1. Noise-denoise: original → base-denoised → LoRA-denoised
    if base_triplets and lora_triplets:
        nd_images = []
        for i, ((orig, noised, base_dn), (_, _, lora_dn)) in enumerate(zip(base_triplets, lora_triplets)):
            orig.save(output_dir / f"nd_original_{i}.png")
            noised.save(output_dir / f"nd_noised_{i}.png")
            base_dn.save(output_dir / f"nd_base_denoised_{i}.png")
            lora_dn.save(output_dir / f"nd_lora_denoised_{i}.png")

            nd_images.append(wandb.Image(orig, caption=f"Original #{i}"))
            nd_images.append(wandb.Image(noised, caption=f"Noised (t={args.noise_strength}) #{i}"))
            nd_images.append(wandb.Image(base_dn, caption=f"Base denoised #{i}"))
            nd_images.append(wandb.Image(lora_dn, caption=f"LoRA denoised #{i}"))

        wandb.log({"noise_denoise": nd_images})
        print(f"Logged {len(nd_images)} noise-denoise images to wandb", flush=True)

    # 2. Pure generation: base vs LoRA
    gen_images = []
    for (seed, base_img), (_, lora_img) in zip(base_gen, lora_gen):
        base_img.save(output_dir / f"gen_base_seed{seed}.png")
        lora_img.save(output_dir / f"gen_lora_seed{seed}.png")

        gen_images.append(wandb.Image(base_img, caption=f"Base (seed {seed})"))
        gen_images.append(wandb.Image(lora_img, caption=f"LoRA (seed {seed})"))

    wandb.log({"generation": gen_images})
    print(f"Logged {len(gen_images)} generation images to wandb", flush=True)

    # ── Phase 3: Gradient flow check ──
    print("\nChecking gradient flow...", flush=True)
    grad_results = check_gradient_flow(pipe_lora, device)
    wandb.log(grad_results)

    wandb.finish()
    print(f"\nValidation complete. Results saved to {output_dir} and logged to wandb.", flush=True)


if __name__ == "__main__":
    main()
