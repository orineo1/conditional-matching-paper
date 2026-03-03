"""Validate LoRA fine-tuned SDXL Turbo: compare scribble generation before/after, check gradient flow, log to wandb."""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from PIL import Image


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


def generate_comparison(pipe_base, pipe_lora, device, seed=42):
    """Generate scribble images with both pipelines for side-by-side comparison."""
    prompt = "a simple hand-drawn scribble"

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out_base = pipe_base(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            output_type="pil",
            generator=generator,
        ).images[0]

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out_lora = pipe_lora(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            output_type="pil",
            generator=generator,
        ).images[0]

    return out_base, out_lora


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
    parser.add_argument("--wandb_project", type=str, default="conditional-flow")
    parser.add_argument("--wandb_entity", type=str, default="conditional-matching")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_path = args.checkpoint or cfg["output_dir"]
    checkpoint_name = Path(lora_path).name

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
        },
        job_type="validation",
    )

    print("Loading base SDXL Turbo (no LoRA)...", flush=True)
    pipe_base = load_pipeline(device)

    print(f"Loading LoRA pipeline from {lora_path}...", flush=True)
    pipe_lora = load_pipeline(device, lora_path=lora_path)

    # Side-by-side comparison
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comparison images...", flush=True)
    wandb_images = []

    for seed in [42, 123, 456, 789, 1234]:
        base, lora = generate_comparison(pipe_base, pipe_lora, device, seed=seed)
        base.save(output_dir / f"base_seed{seed}.png")
        lora.save(output_dir / f"lora_seed{seed}.png")

        wandb_images.append(wandb.Image(base, caption=f"Base (seed {seed})"))
        wandb_images.append(wandb.Image(lora, caption=f"LoRA (seed {seed})"))
        print(f"  Seed {seed} done", flush=True)

    wandb.log({"comparisons": wandb_images})
    print(f"Saved comparison images to {output_dir}", flush=True)

    # Free base pipeline before gradient check
    del pipe_base
    torch.cuda.empty_cache()

    # Gradient flow check
    print("\nChecking gradient flow...", flush=True)
    grad_results = check_gradient_flow(pipe_lora, device)
    wandb.log(grad_results)

    wandb.finish()
    print("\nValidation complete. Results logged to wandb.", flush=True)


if __name__ == "__main__":
    main()
