#!/usr/bin/env python
"""
time_dps_step.py — measure the wall-clock time of ONE MLGD-F guidance step:
Architect U-Net forward -> pred_x0 -> VAE decode -> run_dps_step_clip
(sample --num_variations candidate points via Sprinter, score/select via
--backsel_rule, backprop the loss to get the gradient). No file saving, no
wandb, no visualization, no outer diffusion loop over many steps -- just the
per-step cost that actually scales with --num_variations/--backsel_k, timed
directly, plus a naive n_steps x per_step_time projection.

Usage:
    python time_dps_step.py --num_variations 100 --backsel_k 50 --backsel_rule witness --n_steps 250
    python time_dps_step.py --num_variations 6   --backsel_k 20 --backsel_rule uniform --n_steps 30
"""
import os
import sys
import time
import argparse

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_HERE), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_variations", type=int, default=6,
                   help="Fresh Sprinter samples drawn per step.")
    p.add_argument("--backsel_k", type=int, default=None,
                   help="Of num_variations, how many to backprop through (None = all).")
    p.add_argument("--backsel_rule", type=str, default="uniform", choices=["uniform", "witness"])
    p.add_argument("--witness_floor", type=float, default=0.3)
    p.add_argument("--witness_temperature", type=float, default=1.0)
    p.add_argument("--witness_bandwidth_scale", type=float, default=1.0)
    p.add_argument("--witness_kernel_alpha", type=float, default=1.0)
    p.add_argument("--witness_replacement", action="store_true")
    p.add_argument("--base_zeta", type=float, default=5.0)
    p.add_argument("--loss_fn", type=str, default="mmd", choices=["mmd", "swd"])
    p.add_argument("--loss_scale", type=float, default=1.0)
    p.add_argument("--variation_batch_size", type=int, default=1)
    p.add_argument("--variation_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--n_target", type=int, default=250,
                   help="Size of the (random, content-irrelevant) target CLIP set -- "
                        "only affects the loss kernel's matrix size, not what generates "
                        "the timed cost (Sprinter + VAE + CLIP + backward).")
    p.add_argument("--n_warmup", type=int, default=1, help="Untimed calls first (CUDA/cuDNN warmup).")
    p.add_argument("--n_repeats", type=int, default=5, help="Timed calls to average over.")
    p.add_argument("--n_steps", type=int, default=None,
                   help="If given, print an approximate total-run-time projection "
                        "(n_steps x measured per-step time).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] No CUDA device found -- timings will not be representative of a real run.")
    torch.manual_seed(args.seed)

    from models import load_models, setup_gradient_checkpointing
    from clip_utils import load_clip_model
    from generation import predict_noise_cfg, compute_pred_x0_direct, run_dps_step_clip
    from metrics import compute_mmd, compute_swd

    loss_fn = {"mmd": compute_mmd, "swd": compute_swd}[args.loss_fn]

    print("Loading models...", flush=True)
    architect, sprinter = load_models(device)
    clip_model, clip_processor = load_clip_model(device)
    setup_gradient_checkpointing(architect, sprinter)
    sprinter.vae.to(dtype=torch.float32)
    print("Models loaded.", flush=True)

    # ── One-time setup identical for every repeat (not timed): CFG text state,
    # a single denoising timestep, and a random starting latent. Content is
    # irrelevant for timing -- only shapes/costs matter.
    height, width = 512, 512
    with torch.no_grad():
        (
            prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=args.variation_prompt, negative_prompt="",
            device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
        )
    architect.scheduler.set_timesteps(50, device=device)
    t = architect.scheduler.timesteps[25]  # an arbitrary mid-schedule step

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids": add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    latents_shape = (1, architect.unet.config.in_channels, height // 8, width // 8)
    all_clip_embeddings = torch.randn(args.n_target, 768, device=device)

    def one_step():
        latents = torch.randn(latents_shape, device=device, dtype=torch.float16)
        latents_step = latents.detach().requires_grad_(True)

        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, gs=0.0,
        )
        pred_x0 = compute_pred_x0_direct(architect.scheduler, noise_pred, t, latents_step)
        pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

        def vae_decode_checkpoint(lat):
            return architect.vae.decode(lat.to(architect.vae.dtype)).sample

        pixel_x0 = torch.utils.checkpoint.checkpoint(
            vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False
        )
        pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

        grad, loss_scaled, zeta_i, loss_norm, _ = run_dps_step_clip(
            latents=latents,
            latents_step=latents_step,
            noise_pred=noise_pred,
            pixel_x0_norm=pixel_x0_norm,
            sprinter=sprinter,
            all_clip_embeddings=all_clip_embeddings,
            num_variations=args.num_variations,
            variation_batch_size=args.variation_batch_size,
            base_zeta_prime=args.base_zeta,
            clip_model=clip_model,
            clip_processor=clip_processor,
            vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=args.variation_prompt,
            loss_fn=loss_fn,
            loss_scale=args.loss_scale,
            backsel_k=args.backsel_k,
            backsel_rule=args.backsel_rule,
            backsel_generator=None,
            witness_floor=args.witness_floor,
            witness_temperature=args.witness_temperature,
            witness_replacement=args.witness_replacement,
            witness_bandwidth_scale=args.witness_bandwidth_scale,
            witness_kernel_alpha=args.witness_kernel_alpha,
        )
        return grad

    print(f"\nConfig: num_variations={args.num_variations} backsel_k={args.backsel_k} "
          f"backsel_rule={args.backsel_rule} loss_fn={args.loss_fn}", flush=True)

    print(f"Warming up ({args.n_warmup} call(s))...", flush=True)
    for _ in range(args.n_warmup):
        one_step()
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"Timing ({args.n_repeats} call(s))...", flush=True)
    times = []
    for i in range(args.n_repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"  [{i + 1}/{args.n_repeats}] {dt:.3f}s", flush=True)

    mean_t = sum(times) / len(times)
    std_t = (sum((x - mean_t) ** 2 for x in times) / len(times)) ** 0.5
    print(f"\nPer-step time: {mean_t:.3f}s +/- {std_t:.3f}s  (over {args.n_repeats} repeats)")

    if args.n_steps is not None:
        total_s = mean_t * args.n_steps
        print(f"Projected total for --n_steps {args.n_steps}: "
              f"{total_s:.1f}s  ({total_s / 60:.1f} min, {total_s / 3600:.2f} h)")


if __name__ == "__main__":
    main()
