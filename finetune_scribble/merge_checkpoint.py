"""Merge FSDP-flattened checkpoint into a standard UNet2DConditionModel for inference.

FSDP flattens each parameter to 1D when saving. This script loads the flat
state dict, reshapes each tensor to match the base model's expected shapes,
and saves a clean checkpoint that UNet2DConditionModel.from_pretrained() can load.
"""

import argparse
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP-flattened checkpoint")
    parser.add_argument("checkpoint_dir", type=str, help="Path to FSDP checkpoint with flat safetensors")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for merged U-Net (default: checkpoint_dir/merged)")
    parser.add_argument("--model_name", type=str, default="stabilityai/sdxl-turbo")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "merged"

    # 1. Load base U-Net to get expected shapes
    print(f"Loading base U-Net from {args.model_name} for shape reference...")
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name, subfolder="unet", torch_dtype=torch.float16,
    )
    expected_shapes = {name: param.shape for name, param in unet.state_dict().items()}
    print(f"  {len(expected_shapes)} parameters in base model")

    # 2. Load flat state dict from safetensors shards
    print("Loading FSDP-flattened checkpoint...")
    flat_state_dict = {}
    shard_files = sorted(checkpoint_dir.glob("diffusion_pytorch_model*.safetensors"))
    shard_files = [f for f in shard_files if "index" not in f.name]
    for shard_file in shard_files:
        print(f"  Loading {shard_file.name}...")
        shard = load_file(shard_file)
        flat_state_dict.update(shard)
    print(f"  {len(flat_state_dict)} tensors loaded")

    # 3. Reshape flat tensors to match expected shapes
    print("Reshaping flat tensors...")
    reshaped = {}
    mismatches = []
    for name, expected_shape in expected_shapes.items():
        if name not in flat_state_dict:
            print(f"  WARNING: {name} missing from checkpoint")
            continue

        flat_tensor = flat_state_dict[name]
        if flat_tensor.shape == expected_shape:
            reshaped[name] = flat_tensor
        elif flat_tensor.numel() == torch.Size(expected_shape).numel():
            reshaped[name] = flat_tensor.reshape(expected_shape)
        else:
            mismatches.append((name, flat_tensor.shape, expected_shape))

    if mismatches:
        print(f"\n  ERROR: {len(mismatches)} parameters have incompatible sizes:")
        for name, got, expected in mismatches[:10]:
            print(f"    {name}: got {got} ({got.numel()}), expected {expected} ({torch.Size(expected).numel()})")
        return

    extra = set(flat_state_dict.keys()) - set(expected_shapes.keys())
    if extra:
        print(f"  Note: {len(extra)} extra keys in checkpoint (ignored): {list(extra)[:5]}")

    print(f"  Reshaped {len(reshaped)}/{len(expected_shapes)} parameters")

    # 4. Load reshaped state dict and save
    unet.load_state_dict(reshaped, strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(output_dir)
    print(f"\nSaved merged U-Net to {output_dir}")
    print("You can now load it with: UNet2DConditionModel.from_pretrained(output_dir)")


if __name__ == "__main__":
    main()
