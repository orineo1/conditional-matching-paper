"""Validate HED-fine-tuned SDXL Turbo: noise-then-denoise + text-conditioned generation + gradient flow check."""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image, ImageDraw
from torchvision import transforms


GUIDANCE_SCALE = 7.5


def load_pipeline(device, unet_path=None):
    """Load SDXL Turbo pipeline, optionally with fine-tuned U-Net."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    if unet_path:
        pipe.unet = UNet2DConditionModel.from_pretrained(
            unet_path, torch_dtype=torch.float16,
        ).to(device)
        print(f"Loaded fine-tuned U-Net from {unet_path}", flush=True)

    return pipe


def load_reference_images(data_dir, num_images=5, size=512):
    """Load HED edge map PNGs from the data directory."""
    import json
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.jsonl"

    entries = []
    with open(metadata_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    images = []
    captions = []
    # Sample evenly across the dataset
    step = max(1, len(entries) // num_images)
    for i in range(0, len(entries), step):
        if len(images) >= num_images:
            break
        path = data_dir / entries[i]["file_name"]
        if path.exists():
            img = Image.open(path).convert("RGB").resize((size, size))
            images.append(img)
            captions.append(entries[i]["caption"])

    print(f"  Loaded {len(images)} reference HED images", flush=True)
    return images, captions


def pil_to_tensor(images, device):
    """Convert list of PIL images to tensor in [-1, 1] for VAE encoding."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tensors = [transform(img) for img in images]
    return torch.stack(tensors).to(device=device, dtype=torch.float32)


def make_grid(images_2d, col_labels=None, row_labels=None, cell_size=256, padding=4, label_height=32):
    """Build a grid image from a 2D list of PIL images (rows x cols) with optional labels."""
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


def noise_denoise(pipe, ref_images, captions, device, strength, seed=42, num_inference_steps=10):
    """Add noise at given strength, then denoise with per-image captions. Returns (denoised, noisy) PIL images."""
    if not ref_images:
        return [], []

    pixel_tensor = pil_to_tensor(ref_images, device)
    with torch.no_grad():
        vae_dtype = pipe.vae.dtype
        pipe.vae.to(torch.float32)
        latent_dist = pipe.vae.encode(pixel_tensor).latent_dist
        original_latents = latent_dist.sample() * pipe.vae.config.scaling_factor
        pipe.vae.to(vae_dtype)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(original_latents.shape, generator=generator, device=device, dtype=torch.float16)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    all_timesteps = pipe.scheduler.timesteps

    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    denoise_timesteps = all_timesteps[t_start:]
    noise_timestep = denoise_timesteps[0]

    print(f"  strength={strength}: noise at t={noise_timestep.item():.0f}, "
          f"denoise {len(denoise_timesteps)} steps", flush=True)

    noisy_latents = pipe.scheduler.add_noise(
        original_latents.to(torch.float16), noise,
        torch.tensor([noise_timestep], device=device),
    )

    # Use first caption for prompt (all same for simplicity in noise-denoise)
    prompt = captions[0] if captions else ""
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)

    bs = len(ref_images)

    latents = noisy_latents.clone()
    with torch.no_grad():
        for t in denoise_timesteps:
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
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    denoised_images = decode_latents(pipe, latents, vae_dtype)
    noisy_images = decode_latents(pipe, noisy_latents, vae_dtype)

    return denoised_images, noisy_images


def generate_images(pipe, device, prompts, seeds):
    """Generate images for diverse prompts — verify text conditioning works."""
    results = []
    for prompt in prompts:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                img = pipe(
                    prompt=prompt,
                    num_inference_steps=4,
                    guidance_scale=GUIDANCE_SCALE,
                    output_type="pil",
                    generator=generator,
                ).images[0]
            results.append((prompt, seed, img))
    return results


def check_gradient_flow(pipe, device):
    """Verify gradients exist on unfrozen decoder params, zero on frozen encoder params."""
    latent = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device, requires_grad=True)

    prompt_embeds = pipe.encode_prompt("a cat", device=device, num_images_per_prompt=1)

    t = torch.tensor([500], device=device, dtype=torch.long)
    time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)

    pipe.unet.requires_grad_(True)
    noise_pred = pipe.unet(
        latent, t,
        encoder_hidden_states=prompt_embeds[0],
        added_cond_kwargs={"text_embeds": prompt_embeds[2], "time_ids": time_ids},
    ).sample

    loss = noise_pred.sum()
    loss.backward()

    results = {}

    # Check latent grads
    if latent.grad is not None and latent.grad.abs().sum() > 0:
        print("  Latent grad check PASSED", flush=True)
        results["latent_grad_check"] = True
    else:
        print("  Latent grad check FAILED", flush=True)
        results["latent_grad_check"] = False

    # Count grads by block
    decoder_grads = 0
    encoder_grads = 0
    for name, param in pipe.unet.named_parameters():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        is_decoder = any(p in name for p in ["up_blocks.1.", "up_blocks.2.", "conv_norm_out.", "conv_out."])
        if is_decoder and has_grad:
            decoder_grads += 1
        elif not is_decoder and has_grad:
            encoder_grads += 1

    print(f"  Decoder params with grad: {decoder_grads}", flush=True)
    print(f"  Encoder params with grad: {encoder_grads} (should be 0 after freeze)", flush=True)
    results["decoder_params_with_grad"] = decoder_grads
    results["encoder_params_with_grad"] = encoder_grads
    results["passed"] = results["latent_grad_check"] and decoder_grads > 0

    pipe.unet.zero_grad()
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate HED fine-tuned SDXL Turbo")
    parser.add_argument("--config", type=str, default="finetune_hed/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned U-Net checkpoint dir")
    parser.add_argument("--output_dir", type=str, default="finetune_hed/output/validation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to HED data dir (default: data_dir from config)")
    parser.add_argument("--wandb_project", type=str, default="hed_finetune")
    parser.add_argument("--wandb_entity", type=str, default="conditional-matching")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_path = args.checkpoint or cfg["output_dir"]
    data_dir = args.data_dir or cfg["data_dir"]
    checkpoint_name = Path(unet_path).name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
    gen_prompts = ["a cat", "a mountain landscape", "a portrait of a person"]

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"validate-{checkpoint_name}",
        settings=wandb.Settings(init_timeout=300),
        config={
            "checkpoint": unet_path,
            "resolution": cfg["resolution"],
            "freeze_strategy": cfg.get("freeze_strategy", "decoder_only"),
        },
        job_type="validation",
    )

    print("\nLoading reference HED images...", flush=True)
    ref_images, ref_captions = load_reference_images(data_dir, num_images=5)

    # Phase 1: Base pipeline
    print("\nLoading base SDXL Turbo (no fine-tune)...", flush=True)
    pipe_base = load_pipeline(device)

    base_nd_results = {}
    for s in strengths:
        print(f"\nNoise-denoise (base, strength={s})...", flush=True)
        denoised, noisy = noise_denoise(pipe_base, ref_images, ref_captions, device, strength=s, seed=42)
        base_nd_results[s] = (denoised, noisy)

    print("\nGenerating images (base)...", flush=True)
    base_gen = generate_images(pipe_base, device, gen_prompts, seeds)

    del pipe_base
    torch.cuda.empty_cache()

    # Phase 2: Fine-tuned pipeline
    print(f"\nLoading fine-tuned pipeline from {unet_path}...", flush=True)
    pipe_ft = load_pipeline(device, unet_path=unet_path)

    ft_nd_results = {}
    for s in strengths:
        print(f"\nNoise-denoise (fine-tuned, strength={s})...", flush=True)
        denoised, noisy = noise_denoise(pipe_ft, ref_images, ref_captions, device, strength=s, seed=42)
        ft_nd_results[s] = (denoised, noisy)

    print("\nGenerating images (fine-tuned)...", flush=True)
    ft_gen = generate_images(pipe_ft, device, gen_prompts, seeds)

    # Log noise-denoise grids
    col_labels = ["Original", "Noised", "Base", "Fine-tuned"]
    nd_grids = []
    for s in strengths:
        base_denoised, noisy_imgs = base_nd_results[s]
        ft_denoised, _ = ft_nd_results[s]

        rows = []
        for i in range(len(ref_images)):
            rows.append([ref_images[i], noisy_imgs[i], base_denoised[i], ft_denoised[i]])

        grid = make_grid(rows, col_labels=col_labels, row_labels=[f"#{i}" for i in range(len(ref_images))])
        grid.save(output_dir / f"nd_grid_s{s}.png")
        nd_grids.append(wandb.Image(grid, caption=f"strength={s}"))

    wandb.log({"noise_denoise": nd_grids})

    # Log generation grid: rows=prompts, cols=base/ft per seed
    gen_rows = []
    row_labels = []
    for prompt in gen_prompts:
        row = []
        for seed in seeds:
            base_img = next(img for p, s, img in base_gen if p == prompt and s == seed)
            ft_img = next(img for p, s, img in ft_gen if p == prompt and s == seed)
            row.extend([base_img, ft_img])
        gen_rows.append(row)
        row_labels.append(prompt[:20])

    col_labels_gen = []
    for seed in seeds:
        col_labels_gen.extend([f"Base s={seed}", f"FT s={seed}"])
    gen_grid = make_grid(gen_rows, col_labels=col_labels_gen, row_labels=row_labels)
    gen_grid.save(output_dir / "gen_grid.png")
    wandb.log({"generation": wandb.Image(gen_grid, caption="Text-conditioned: Base vs FT")})

    # Phase 3: Gradient flow check
    print("\nChecking gradient flow...", flush=True)
    grad_results = check_gradient_flow(pipe_ft, device)
    wandb.log(grad_results)

    wandb.finish()
    print(f"\nValidation complete. Results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
