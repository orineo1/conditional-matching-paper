"""Validate LoRA fine-tuned sprinter: compare before/after, check gradient flow."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image


def load_sprinter(device, lora_path=None):
    """Load sprinter pipeline, optionally with LoRA weights."""
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16
    ).to(device)

    sprinter = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    if lora_path:
        sprinter.load_lora_weights(lora_path)
        print(f"Loaded LoRA weights from {lora_path}")

    return sprinter


def generate_comparison(sprinter_base, sprinter_lora, condition_image, device, seed=42):
    """Generate images with both sprinters for side-by-side comparison."""
    generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = dict(
        prompt="a simple hand-drawn scribble",
        image=condition_image,
        num_inference_steps=2,
        guidance_scale=0.0,
        controlnet_conditioning_scale=0.5,
        output_type="pil",
    )

    with torch.no_grad():
        out_base = sprinter_base(**kwargs, generator=generator).images[0]

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out_lora = sprinter_lora(**kwargs, generator=generator).images[0]

    return out_base, out_lora


def check_gradient_flow(sprinter, condition_image, device):
    """Verify gradients flow through U-Net + LoRA via actual backward pass."""
    # Prepare condition image as tensor
    cond_np = np.array(condition_image.resize((512, 512))).astype(np.float32) / 255.0
    cond_tensor = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Create a latent that requires grad (simulates DPS loop)
    latent = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device, requires_grad=True)

    # Encode prompt
    prompt_embeds = sprinter.encode_prompt("a simple hand-drawn scribble", device=device, num_images_per_prompt=1)

    # Manual forward through ControlNet + UNet
    t = torch.tensor([500], device=device, dtype=torch.long)

    down_block_res, mid_block_res = sprinter.controlnet(
        latent, t,
        encoder_hidden_states=prompt_embeds[0],
        controlnet_cond=cond_tensor,
        conditioning_scale=0.5,
        added_cond_kwargs={"text_embeds": prompt_embeds[2], "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)},
        return_dict=False,
    )

    noise_pred = sprinter.unet(
        latent, t,
        encoder_hidden_states=prompt_embeds[0],
        down_block_additional_residuals=down_block_res,
        mid_block_additional_residual=mid_block_res,
        added_cond_kwargs={"text_embeds": prompt_embeds[2], "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float16)},
    ).sample

    # Compute scalar loss and backward
    loss = noise_pred.sum()
    loss.backward()

    passed = True

    # Check that the input latent received gradients
    if latent.grad is not None and latent.grad.abs().sum() > 0:
        print("  Latent grad check PASSED")
    else:
        print("  Latent grad check FAILED: no gradients on input latents")
        passed = False

    # Check that LoRA parameters specifically received gradients
    lora_grads = 0
    for name, param in sprinter.unet.named_parameters():
        if "lora_" in name and param.grad is not None and param.grad.abs().sum() > 0:
            lora_grads += 1
    if lora_grads > 0:
        print(f"  LoRA grad check PASSED: {lora_grads} LoRA params have nonzero grads")
    else:
        print("  LoRA grad check FAILED: no LoRA params have nonzero grads")
        passed = False

    if passed:
        print("Gradient flow check PASSED")
    else:
        print("Gradient flow check FAILED")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate LoRA sprinter")
    parser.add_argument("--config", type=str, default="scribble_tune/config.yaml")
    parser.add_argument("--condition_image", type=str, required=True,
                        help="Path to a scribble/edge image for conditioning")
    parser.add_argument("--output_dir", type=str, default="scribble_tune/output/validation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_path = cfg["output_dir"]

    condition_image = Image.open(args.condition_image).convert("RGB")

    print("Loading base sprinter (no LoRA)...")
    sprinter_base = load_sprinter(device)

    print("Loading LoRA sprinter...")
    sprinter_lora = load_sprinter(device, lora_path=lora_path)

    # Side-by-side comparison
    print("\nGenerating comparison images...")
    out_base, out_lora = generate_comparison(
        sprinter_base, sprinter_lora, condition_image, device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_base.save(output_dir / "base_output.png")
    out_lora.save(output_dir / "lora_output.png")
    condition_image.save(output_dir / "condition.png")
    print(f"Saved comparison images to {output_dir}")

    # Multi-seed comparison
    print("\nGenerating multi-seed comparison...")
    for seed in [42, 123, 456]:
        base, lora = generate_comparison(
            sprinter_base, sprinter_lora, condition_image, device, seed=seed
        )
        base.save(output_dir / f"base_seed{seed}.png")
        lora.save(output_dir / f"lora_seed{seed}.png")

    # Gradient flow check (with LoRA sprinter only)
    print("\nChecking gradient flow...")
    check_gradient_flow(sprinter_lora, condition_image, device)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
