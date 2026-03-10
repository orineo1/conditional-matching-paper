"""Compare random generation: base vs fine-tuned U-Net with a given prompt."""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image, ImageDraw


def make_grid(images_2d, col_labels=None, row_labels=None, cell_size=256, padding=4, label_height=32):
    n_rows = len(images_2d)
    n_cols = len(images_2d[0])
    left_margin = 100 if row_labels else 0
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


def generate(pipe, prompt, negative_prompt, guidance_scale, seeds, device):
    results = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            img = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=4,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        results.append(img)
        print(f"  seed {seed} done", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="finetune_scribble/output/checkpoint-1000/merged")
    parser.add_argument("--prompt", type=str, default="male face")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="finetune_scribble/output/compare_gen")
    parser.add_argument("--wandb_project", type=str, default="scribble_finetune")
    parser.add_argument("--wandb_entity", type=str, default="conditional-matching")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(args.checkpoint).parent.name

    seeds = list(range(args.num_seeds))

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=f"compare-gen-{checkpoint_name}",
        settings=wandb.Settings(init_timeout=300),
        config=vars(args), job_type="compare_generation",
    )

    # Base
    print("Loading base SDXL Turbo...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    print(f"Generating {args.num_seeds} base images...", flush=True)
    base_imgs = generate(pipe, args.prompt, args.negative_prompt, args.guidance_scale, seeds, device)

    # Swap to fine-tuned
    print(f"Loading fine-tuned U-Net from {args.checkpoint}...", flush=True)
    pipe.unet = UNet2DConditionModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16,
    ).to(device)

    print(f"Generating {args.num_seeds} fine-tuned images...", flush=True)
    ft_imgs = generate(pipe, args.prompt, args.negative_prompt, args.guidance_scale, seeds, device)

    # Save and build grid
    for i, seed in enumerate(seeds):
        base_imgs[i].save(output_dir / f"base_seed{seed}.png")
        ft_imgs[i].save(output_dir / f"ft_seed{seed}.png")

    grid = make_grid(
        [base_imgs, ft_imgs],
        col_labels=[f"s={s}" for s in seeds],
        row_labels=["Base", "FT"],
    )
    grid.save(output_dir / "comparison_grid.png")
    wandb.log({"generation_comparison": wandb.Image(grid, caption=f'prompt="{args.prompt}"')})

    wandb.finish()
    print(f"Done. Grid saved to {output_dir / 'comparison_grid.png'}", flush=True)


if __name__ == "__main__":
    main()
