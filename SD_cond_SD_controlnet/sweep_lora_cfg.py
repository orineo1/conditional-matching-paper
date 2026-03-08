"""
LoRA + CFG Sweep — Scribble Visualization Across Denoising Steps.

Generates a composite image: 8 rows × N columns (steps).
Rows alternate No-LoRA / LoRA for each CFG scale.
Each cell = pred_x0 decoded to PIL at that denoising step.
Same seed/noise for all rows.
"""

import argparse
import copy
import gc
import os
import sys

import matplotlib
matplotlib.use("Agg")
import torch
from PIL import Image, ImageDraw, ImageFont

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from generation import compute_pred_x0_direct, denoise_step, predict_noise_cfg
from image_utils import latent_to_pil


def parse_args():
    p = argparse.ArgumentParser(description="LoRA + CFG sweep: pred_x0 at every step")
    p.add_argument("--lora_path", type=str, default="scribble_tune/output/checkpoint-50000")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/sweep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--guidance_scales", type=str, default="3.0,5.0,7.5,12.0")
    p.add_argument("--prompt", type=str,
                   default="rough pencil scribble outline, loose sketch, minimal line art")
    p.add_argument("--negative_prompt", type=str,
                   default="detailed, realistic, photograph, complex, colored, shading")
    p.add_argument("--wandb_project", type=str, default="combined_conditional_flow")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")
    return p.parse_args()


def load_architect(device, lora_path=None):
    """Load architect pipeline (SDXL Turbo), optionally with LoRA."""
    from diffusers import StableDiffusionXLPipeline
    architect = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    if lora_path:
        from peft import PeftModel
        architect.unet = PeftModel.from_pretrained(architect.unet, lora_path)
        print(f"Loaded LoRA from {lora_path}")
    architect.vae.to(dtype=torch.float32)
    return architect


def run_sweep(architect, cfg_scales, n_steps, initial_noise, prompt, negative_prompt, device):
    """Run denoising for each CFG scale, collect pred_x0 images at every step.

    Returns: list of (cfg_value, list_of_pil_images) tuples.
    """
    height, width = 512, 512

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

    rows = []
    for gs in cfg_scales:
        print(f"  CFG={gs} ...", end="", flush=True)
        scheduler = copy.deepcopy(architect.scheduler)
        scheduler.set_timesteps(n_steps, device=device)
        latents = initial_noise.clone()
        step_images = []

        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                noise_pred = predict_noise_cfg(
                    architect.unet, scheduler, latents, t,
                    cfg_encoder_states, added_cond_kwargs, gs,
                )
                pred_x0 = compute_pred_x0_direct(scheduler, noise_pred, t, latents)
                pil_img = latent_to_pil(pred_x0, architect.vae, architect.image_processor)
                step_images.append(pil_img)
                latents = denoise_step(scheduler, noise_pred, t, latents)

        rows.append((gs, step_images))
        print(f" done ({len(step_images)} steps)", flush=True)

    return rows


def build_composite(rows_no_lora, rows_lora, cell_size=128):
    """Build composite image: interleaved No-LoRA / LoRA rows per CFG scale."""
    n_cols = len(rows_no_lora[0][1])
    n_rows = len(rows_no_lora) * 2  # 2 per CFG (no-lora, lora)
    label_width = 160

    total_w = label_width + n_cols * cell_size
    total_h = 20 + n_rows * cell_size  # 20px header for step numbers

    composite = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Step numbers header
    for col in range(n_cols):
        x = label_width + col * cell_size + cell_size // 2
        draw.text((x, 2), str(col + 1), fill="black", font=font, anchor="mt")

    # Rows
    for cfg_idx, (cfg_val, _) in enumerate(rows_no_lora):
        for lora_idx, (label_prefix, rows_data) in enumerate([
            ("No LoRA", rows_no_lora),
            ("LoRA", rows_lora),
        ]):
            row = cfg_idx * 2 + lora_idx
            y_offset = 20 + row * cell_size
            label = f"{label_prefix}, CFG={cfg_val}"
            draw.text((4, y_offset + cell_size // 2), label, fill="black", font=font, anchor="lm")

            _, images = rows_data[cfg_idx]
            for col, img in enumerate(images):
                thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
                composite.paste(thumb, (label_width + col * cell_size, y_offset))

    return composite


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_scales = [float(x) for x in args.guidance_scales.split(",")]

    # wandb init
    import wandb
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name="sweep-lora-cfg",
        config=vars(args),
    )

    # Generate initial noise (same for all rows)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(1, 4, 64, 64, generator=generator, device=device, dtype=torch.float16)

    # ── Run WITHOUT LoRA ──────────────────────────────────────────────────────
    print("Loading architect (no LoRA)...", flush=True)
    architect = load_architect(device, lora_path=None)
    print("Running sweep (no LoRA)...", flush=True)
    rows_no_lora = run_sweep(
        architect, cfg_scales, args.n_steps, initial_noise,
        args.prompt, args.negative_prompt, device,
    )

    del architect
    gc.collect()
    torch.cuda.empty_cache()

    # ── Run WITH LoRA ─────────────────────────────────────────────────────────
    print("Loading architect (with LoRA)...", flush=True)
    architect = load_architect(device, lora_path=args.lora_path)
    print("Running sweep (with LoRA)...", flush=True)
    rows_lora = run_sweep(
        architect, cfg_scales, args.n_steps, initial_noise,
        args.prompt, args.negative_prompt, device,
    )

    del architect
    gc.collect()
    torch.cuda.empty_cache()

    # ── Build composite ───────────────────────────────────────────────────────
    print("Building composite image...", flush=True)
    composite = build_composite(rows_no_lora, rows_lora)
    out_path = os.path.join(args.output_dir, "sweep_lora_cfg.png")
    composite.save(out_path)
    print(f"Saved composite to {out_path}", flush=True)

    wandb.log({"sweep_composite": wandb.Image(composite, caption="LoRA + CFG Sweep")})
    wandb.finish()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
