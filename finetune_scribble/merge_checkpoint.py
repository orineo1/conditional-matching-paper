"""Merge FSDP sharded checkpoint into a single UNet2DConditionModel for inference."""

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import UNet2DConditionModel


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP sharded checkpoint")
    parser.add_argument("checkpoint_dir", type=str, help="Path to FSDP sharded checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for merged U-Net (default: checkpoint_dir/merged)")
    parser.add_argument("--model_name", type=str, default="stabilityai/sdxl-turbo")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "merged"

    print(f"Loading base U-Net from {args.model_name}...")
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name, subfolder="unet", torch_dtype=torch.float16,
    )

    # Use accelerator to load FSDP sharded state
    accelerator = Accelerator()
    unet = accelerator.prepare(unet)
    accelerator.load_state(str(checkpoint_dir))

    # Save as standard pretrained format
    unwrapped = accelerator.unwrap_model(unet)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped.save_pretrained(output_dir)
    print(f"Saved merged U-Net to {output_dir}")


if __name__ == "__main__":
    main()
