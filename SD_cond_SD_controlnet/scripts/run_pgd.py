"""
run_pgd.py — PGD pipeline entry point, the projected-gradient-descent
competitor to MLGD-F (run_mlgd_f.py).

Follows Shah & Hegde-style PGD for generative priors: alternate an
unconstrained gradient step in ambient (pixel) space with a projection back
onto the generator's range, found by Adam-searching the generator's own
input space (exactly the paper's P_G(w) = G(argmin_z ||w - G(z)||)).

Mapped onto this codebase:
    - "Ambient space"   = pixel-space scribble (Architect VAE decode output).
    - "Generator G"     = the Architect's VAE decoder ONLY (no UNet/diffusion
                          trajectory — matches the paper's own experiments,
                          which use a plain VAE/GAN decoder as G).
    - "Measurement op"  = Sprinter (+ ControlNet) + CLIP, exactly as in
                          MLGD-F: turns a scribble into CLIP embeddings of
                          conditioned portraits, which is what the L2/MMD
                          loss is computed on.

One round:
    1. Optimization step (opt_steps iters): w = decode(x); repeatedly
       w <- w - opt_lr * grad_w L(w), L in {l2, mmd} on Sprinter+CLIP samples
       conditioned on w. Ambient/pixel-space, unconstrained.
    2. Projection step (proj_adam_steps Adam iters): z* = argmin_z
       ||w - decode(z)||_2, initialised at the current latent x.
    3. x <- z*.

Total rounds are picked to match a target wall-clock budget (the
corresponding MLGD-F run's runtime), the same pattern eval_baselines.py uses
for its SDEdit search budget: measure round 1's time, then compute n_rounds.

Usage:
    python scripts/run_pgd.py --output_dir output/pgd_run --target_minutes 45 \\
        --target_prompts "Woman:a superrealistic portrait photograph of a woman, studio lighting:100"
"""

import argparse
import gc
import json
import os
import sys
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torchvision.transforms.functional as TF

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from clip_utils import encode_images_clip, load_clip_model
from image_utils import build_base_image, sobel_proxy
from metrics import compute_l2, compute_mmd, evaluate_distribution_mmd
from models import load_models, setup_gradient_checkpointing
from visualization import plot_row, visualize_round

from run_mlgd_f import build_targets_age, build_targets_gender, save_image_list_npy

_MAN   = "a superrealistic portrait photograph of a man, studio lighting"
_WOMAN = "a superrealistic portrait photograph of a woman, studio lighting"

# MLGD-F hardcodes controlnet_conditioning_scale=0.8 for every loop-time Sprinter
# call -- generation.run_dps_step_clip (the actual guidance loss), metrics.
# evaluate_distribution_mmd, and visualization.visualize_step's "Cond" images --
# independent of its --controlnet_scale flag (0.5), which is only used to build
# the target distribution. PGD's optimization-step loss and per-round preview
# photos are the analogues of those loop-time calls, so they use this same 0.8
# rather than --controlnet_scale, to keep the two methods' Sprinter behavior
# identical wherever it isn't the thing being compared.
_LOOP_CONTROLNET_SCALE = 0.8

# Same source (man) scribble, same target-distribution presets used by the
# corresponding MLGD-F / eval_baselines.py experiments, so results are
# directly comparable. Per-experiment default seed matches
# eval_baselines.py's EXPERIMENT_CONFIGS. --seed/--controlnet_scale/etc. on
# the CLI still override a preset's value when explicitly passed.
EXPERIMENT_PRESETS = {
    "GenderTarget100": dict(
        mode="gender", seed=1, controlnet_scale=0.5,
        target_prompts=[f"Woman:{_WOMAN}:100"],
    ),
    "GenderTarget1": dict(
        mode="gender", seed=1, controlnet_scale=0.5,
        target_prompts=[f"Woman:{_WOMAN}:1"],
    ),
    "SkewedTarget": dict(  # 25% male / 75% female, matches experiments/SkewedTarget
        mode="gender", seed=5, controlnet_scale=0.5,
        target_prompts=[f"Man:{_MAN}:25", f"Woman:{_WOMAN}:75"],
    ),
    "BalancedTarget": dict(  # 50% male / 50% female, matches experiments/BalancedTarget
        mode="gender", seed=5, controlnet_scale=0.5,
        target_prompts=[f"Man:{_MAN}:50", f"Woman:{_WOMAN}:50"],
    ),
    "GenderInterpolation": dict(  # 4-class, matches experiments/GenderInterpolation
        mode="gender", seed=5, controlnet_scale=0.5,
        target_prompts=[
            "Woman:superrealistic portrait photograph of a woman, extremely feminine features, studio lighting:25",
            "Woman w/ masc features:a superrealistic portrait photograph of a woman with masculine features, heavy brow ridge, studio lighting:25",
            "Man w/ fem features:a superrealistic portrait photograph of a man with extremely feminine features, soft delicate face, high cheekbones, studio lighting:25",
            f"Man:{_MAN}:25",
        ],
    ),
    "AgeInterpolation": dict(  # matches experiments/AgeInterpolation
        mode="age", seed=42, controlnet_scale=0.5,
        age_min=40, age_max=79, age_step=1, age_gender="man",
    ),
}


def apply_experiment_preset(args):
    """Fill in mode/target_prompts/controlnet_scale/age-range from
    EXPERIMENT_PRESETS[args.experiment]. --target_prompts and --seed are only
    filled in when not explicitly passed (both default to None); the other
    preset fields (controlnet_scale, age range) are authoritative for a named
    experiment, so they're applied whenever --experiment is given -- pass no
    --experiment and set flags manually for full manual control instead."""
    if not args.experiment:
        return args
    preset = EXPERIMENT_PRESETS[args.experiment]
    args.mode = preset["mode"]
    if args.target_prompts is None and "target_prompts" in preset:
        args.target_prompts = preset["target_prompts"]
    if args.seed is None:
        args.seed = preset.get("seed")
    for key in ("controlnet_scale", "age_min", "age_max", "age_step", "age_gender"):
        if key in preset:
            setattr(args, key, preset[key])
    return args


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PGD Pipeline (MLGD-F competitor)")

    p.add_argument("--output_dir",    type=str, default="output/pgd_run")
    p.add_argument("--wandb_project", type=str, default="PGD-EXP")
    p.add_argument("--wandb_entity",  type=str, default="")
    p.add_argument("--experiment",    type=str, default=None,
                   choices=list(EXPERIMENT_PRESETS.keys()),
                   help="Optional preset name -- fills in mode/target_prompts/"
                        "seed/age range from EXPERIMENT_PRESETS, matching the "
                        "corresponding MLGD-F/eval_baselines.py experiment. "
                        "Explicit --target_prompts/--seed/etc. still win.")

    # PGD schedule
    p.add_argument("--target_minutes",   type=float, required=True,
                   help="Wall-clock budget in minutes -- set to the matching "
                        "MLGD-F run's measured runtime so both methods get "
                        "the same time budget. n_rounds is computed from "
                        "this by timing round 1, same pattern as "
                        "eval_baselines.py's SDEdit search budget.")
    p.add_argument("--opt_steps",        type=int,   default=3,
                   help="Ambient (pixel-space) gradient steps per round")
    p.add_argument("--opt_lr",           type=float, default=0.05,
                   help="Step size nu for the ambient gradient step")
    p.add_argument("--proj_adam_steps",  type=int,   default=200,
                   help="Adam iterations for the projection's inner "
                        "argmin_z ||w - decode(z)|| search (200 = paper default)")
    p.add_argument("--proj_lr",          type=float, default=0.03,
                   help="Adam learning rate for the projection search")

    # Logging
    p.add_argument("--n_photos_per_round", type=int, default=5,
                   help="Sprinter photos conditioned on the current scribble "
                        "to log to wandb each logged round")
    p.add_argument("--log_image_every",  type=int, default=1,
                   help="Log the scribble + conditioned photos every N rounds")
    p.add_argument("--n_eval",           type=int, default=10,
                   help="Sprinter samples for the quick init/per-round MMD check")
    p.add_argument("--n_eval_final",     type=int, default=250,
                   help="Sprinter samples for the final, higher-fidelity MMD")

    # Loss / guidance
    p.add_argument("--loss_fn",          type=str,   default="mmd",
                   choices=["l2", "mmd"])
    p.add_argument("--bandwidth_scale",  type=float, default=1.0)
    p.add_argument("--kernel_alpha",     type=float, default=1.0)
    p.add_argument("--num_variations",   type=int,   default=6,
                   help="Sprinter samples per loss evaluation (matches "
                        "MLGD-F's --num_variations for a fair comparison)")
    p.add_argument("--variation_batch_size", type=int, default=1)
    p.add_argument("--controlnet_scale", type=float, default=0.5)

    # Prompts
    p.add_argument("--sprinter_variation_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--sprinter_eval_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument(
        "--target_prompts", type=str, nargs="+", default=None,
        metavar="NAME:PROMPT:N",
        help="Same format as run_mlgd_f.py: 'name:prompt:n' triples.",
    )

    # Models
    p.add_argument("--controlnet_model_id", type=str,
                   default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--sprinter_model_id",   type=str,
                   default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id",  type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")

    p.add_argument("--seed", type=int, default=None)

    # Mode (reuses run_mlgd_f.py's target builders unchanged)
    p.add_argument("--mode", type=str, default="gender", choices=["gender", "age"])
    p.add_argument("--age_min",    type=int, default=10)
    p.add_argument("--age_max",    type=int, default=80)
    p.add_argument("--age_step",   type=int, default=1)
    p.add_argument("--n_per_age",  type=int, default=0)
    p.add_argument("--age_gender", type=str, default="man")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Pixel-space VAE helpers (Architect VAE only -- the PGD "generator" G)
# ---------------------------------------------------------------------------

def encode_01(pixel_01, vae):
    """[1,3,H,W] pixels in [0,1] -> Architect latent, same convention run_mlgd_f.py uses."""
    signed = (pixel_01.to(vae.dtype) * 2.0) - 1.0
    latent = vae.encode(signed).latent_dist.mean
    return latent * vae.config.scaling_factor


def decode_01(latent, vae):
    """Architect latent -> [1,3,H,W] pixels in [0,1], inverse of encode_01."""
    pix = vae.decode((latent / vae.config.scaling_factor).to(vae.dtype)).sample
    return torch.clamp((pix.float() + 1.0) / 2.0, 0.0, 1.0)


def generate_clip_embeddings(pixel_01, sprinter, num_variations, variation_batch_size,
                             variation_prompt, controlnet_scale, clip_model, clip_processor):
    """
    Differentiable: pixel_01 (the scribble) -> num_variations Sprinter portraits
    -> CLIP embeddings. Gradient flows back to pixel_01. Mirrors the Sprinter+VAE
    +CLIP chain in generation.run_dps_step_clip, decoupled from the Architect's
    UNet/pred_x0 (PGD never touches the Architect's diffusion trajectory).
    """
    clip_model.to(pixel_01.device)

    def forward(ctrl):
        var_latents = sprinter(
            prompt=[variation_prompt] * ctrl.shape[0],
            image=ctrl,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="latent",
            return_dict=True,
        ).images
        var_pixels = sprinter.vae.decode(
            (var_latents.float() / sprinter.vae.config.scaling_factor).to(sprinter.vae.dtype)
        ).sample
        var_pixels = torch.clamp((var_pixels.float() + 1.0) / 2.0, 0.0, 1.0)
        with torch.amp.autocast("cuda", enabled=False):
            return encode_images_clip(var_pixels.float(), clip_model, clip_processor)

    rows = []
    for start in range(0, num_variations, variation_batch_size):
        bs = min(variation_batch_size, num_variations - start)
        ctrl_batch = pixel_01[0].unsqueeze(0).repeat(bs, 1, 1, 1)
        chunk = torch.utils.checkpoint.checkpoint(forward, ctrl_batch, use_reentrant=False)
        rows.append(chunk)
    return torch.cat(rows, dim=0)


def conditioned_photos(scribble_pil, sprinter, prompt, controlnet_scale, n):
    """n Sprinter portraits conditioned on scribble_pil, for visualization only
    (no_grad, PIL output) -- e.g. the per-round wandb preview."""
    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    with torch.no_grad():
        photos = sprinter(
            prompt=[prompt] * n,
            image=[scribble_pil] * n,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="pil",
        ).images
    sprinter.vae.to(dtype=original_vae_dtype)
    return photos


# ---------------------------------------------------------------------------
# PGD steps
# ---------------------------------------------------------------------------

def optimization_step(x_latent, architect, sprinter, clip_model, clip_processor,
                      all_clip_embeddings, loss_fn, args):
    """Ambient (pixel-space) gradient descent: decode once, then take
    opt_steps gradient steps directly on the pixel tensor.

    Returns (w, last_loss, last_gen_embs) -- last_gen_embs is the full
    num_variations-sized CLIP embedding batch from the final iteration
    (detached), for reuse in the PCA visualization instead of a fresh,
    differently-sized batch."""
    w = decode_01(x_latent.detach(), architect.vae).detach()
    last_loss, last_gen_embs = None, None
    for _ in range(args.opt_steps):
        w = w.clone().requires_grad_(True)
        gen_embs = generate_clip_embeddings(
            w, sprinter, args.num_variations, args.variation_batch_size,
            args.sprinter_variation_prompt, _LOOP_CONTROLNET_SCALE,
            clip_model, clip_processor,
        )
        loss = loss_fn(gen_embs, all_clip_embeddings)
        grad_w = torch.autograd.grad(loss, w)[0]
        w = (w.detach() - args.opt_lr * grad_w).clamp(0.0, 1.0)
        last_loss, last_gen_embs = loss.detach(), gen_embs.detach()
    return w.detach(), last_loss, last_gen_embs


def projection_step(w_pixels, x_init, architect, args):
    """z* = argmin_z ||w_pixels - decode(z)||_2 via Adam, init z = x_init."""
    z = x_init.detach().clone().float().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.proj_lr)
    for _ in range(args.proj_adam_steps):
        opt.zero_grad()
        decoded = decode_01(z, architect.vae)
        proj_loss = ((decoded - w_pixels) ** 2).mean()
        proj_loss.backward()
        opt.step()
    return z.detach().to(x_init.dtype), proj_loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = apply_experiment_preset(parse_args())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    import wandb

    print("Loading models...", flush=True)
    architect, sprinter = load_models(
        device,
        controlnet_model_id=args.controlnet_model_id,
        sprinter_model_id=args.sprinter_model_id,
        architect_model_id=args.architect_model_id,
    )
    clip_model, clip_processor = load_clip_model(device)
    setup_gradient_checkpointing(architect, sprinter)
    print("Models loaded.", flush=True)

    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil    = TF.to_pil_image(sobel_cond_tensor.squeeze(0).cpu())

    print(f"Mode: {args.mode}", flush=True)
    builder = build_targets_gender if args.mode == "gender" else build_targets_age
    (target_images_per_group, clip_embs_per_group, all_clip_embeddings,
     group_names, group_sizes, group_colors, group_markers,
     pca_fixed, source_image, scribble_pil, N_total, target_groups, pca_path) = builder(
        args, sprinter, clip_model, clip_processor, device, sobel_cond_pil)

    source_image.save(os.path.join(args.output_dir, "source_portrait.png"))
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        config={
            "algorithm":           "PGD",
            "experiment":          args.experiment,
            "n_targets":           N_total,
            "target_groups":       {name: {"prompt": pt, "n": n} for name, pt, n, _, _ in target_groups},
            "loss_fn":             args.loss_fn,
            "num_variations":      args.num_variations,
            "controlnet_scale":    args.controlnet_scale,
            "opt_steps":           args.opt_steps,
            "opt_lr":              args.opt_lr,
            "proj_adam_steps":     args.proj_adam_steps,
            "proj_lr":             args.proj_lr,
            "target_minutes":      args.target_minutes,
            "mode":                args.mode,
            "seed":                args.seed,
        },
    )
    print(f"✅ wandb run: {run.name}", flush=True)
    wandb.log({
        "scribble":        wandb.Image(scribble_pil),
        "source_portrait": wandb.Image(source_image),
        **{f"target_samples/{name}": [wandb.Image(p) for p in imgs]
           for name, imgs in target_images_per_group.items()},
    })

    target_clip_np = all_clip_embeddings.cpu().numpy()

    architect.vae.to(dtype=torch.float32)
    sprinter.vae.to(dtype=torch.float32)
    loss_fn = (partial(compute_mmd, bandwidth_scale=args.bandwidth_scale,
                       kernel_alpha=args.kernel_alpha)
              if args.loss_fn == "mmd" else compute_l2)

    # ── Init: source scribble -> Architect latent, no noising (arbitrary init) ──
    with torch.no_grad():
        scribble_01 = TF.to_tensor(scribble_pil).unsqueeze(0).to(device).float()
        x = encode_01(scribble_01, architect.vae)

    n_eval = args.n_eval
    print("Evaluating initial (pre-PGD) MMD...", flush=True)
    init_mmd, _, _ = evaluate_distribution_mmd(
        x, architect.vae, architect.image_processor, sprinter, clip_model, clip_processor,
        all_clip_embeddings, args.sprinter_eval_prompt, n_eval=n_eval, device=device,
    )
    print(f"  init_mmd={init_mmd:.6f}", flush=True)
    wandb.log({"init_mmd": init_mmd}, commit=False)

    # ── PGD loop: run round 1, time it, then compute n_rounds to hit the budget ──
    rounds_dir = os.path.join(args.output_dir, "rounds")
    os.makedirs(rounds_dir, exist_ok=True)
    round_idx = 0
    n_rounds = None
    pgd_start_time = time.time()

    while n_rounds is None or round_idx < n_rounds:
        t0 = time.time()
        w, opt_loss, opt_gen_embs = optimization_step(
            x, architect, sprinter, clip_model, clip_processor,
            all_clip_embeddings, loss_fn, args,
        )
        x, proj_loss = projection_step(w, x, architect, args)
        round_time = time.time() - t0
        round_idx += 1

        print(f"Round {round_idx}{f'/{n_rounds}' if n_rounds else ''}  "
              f"opt_loss={opt_loss.item():.6f}  proj_l2={proj_loss:.6f}  "
              f"round_time={round_time:.1f}s", flush=True)
        log_data = {"round": round_idx, "opt_loss": opt_loss.item(),
                   "proj_l2": proj_loss, "round_time_sec": round_time}
        wandb.log(log_data, commit=(round_idx % args.log_image_every != 0))

        if round_idx % args.log_image_every == 0:
            with torch.no_grad():
                round_scribble_pil = TF.to_pil_image(decode_01(x, architect.vae).squeeze(0).cpu())
            round_photos = conditioned_photos(
                round_scribble_pil, sprinter, args.sprinter_eval_prompt,
                _LOOP_CONTROLNET_SCALE, args.n_photos_per_round,
            )
            round_scribble_pil.save(
                os.path.join(rounds_dir, f"round_{round_idx:04d}_scribble.png"))

            # PCA reuses the actual num_variations-sized batch the optimization
            # step just computed its loss on -- not a fresh, differently-sized
            # batch -- so "Generated" reflects --num_variations, matching how
            # MLGD-F's visualize_step plots its own full variation_clip_flat
            # (while only ever *showing* num_cond thumbnail photos).
            visualize_round(
                round_idx, round_scribble_pil, round_photos,
                opt_gen_embs.cpu().numpy(),
                target_clip_np, pca_fixed, group_names, group_sizes,
                save_path=os.path.join(rounds_dir, f"round_{round_idx:04d}_viz.png"),
            )

        if n_rounds is None:
            n_rounds = max(1, round(args.target_minutes * 60 / round_time))
            print(f"  [budget] round_time={round_time:.1f}s -> "
                  f"target={args.target_minutes}min -> n_rounds={n_rounds}", flush=True)
            wandb.log({"n_rounds": n_rounds}, commit=False)

        gc.collect(); torch.cuda.empty_cache()

    optimization_time_sec = time.time() - pgd_start_time
    print(f"\n✅ PGD complete! {round_idx} rounds, {optimization_time_sec/60:.1f} min.", flush=True)

    # ── Final evaluation ────────────────────────────────────────────────────
    print("Computing final PGD MMD...", flush=True)
    pgd_mmd, pgd_eval_photos, _ = evaluate_distribution_mmd(
        x, architect.vae, architect.image_processor, sprinter, clip_model, clip_processor,
        all_clip_embeddings, args.sprinter_eval_prompt, n_eval=n_eval, device=device,
    )
    print(f"Init MMD : {init_mmd:.6f}", flush=True)
    print(f"PGD MMD  : {pgd_mmd:.6f}",  flush=True)
    print(f"Delta (↓ better): {init_mmd - pgd_mmd:.6f}", flush=True)

    print(f"Computing final PGD MMD at n_eval={args.n_eval_final} (higher-fidelity)...", flush=True)
    pgd_mmd_final, _, _ = evaluate_distribution_mmd(
        x, architect.vae, architect.image_processor, sprinter, clip_model, clip_processor,
        all_clip_embeddings, args.sprinter_eval_prompt, n_eval=args.n_eval_final, device=device,
    )
    print(f"PGD MMD (n={args.n_eval_final}): {pgd_mmd_final:.6f}", flush=True)

    with torch.no_grad():
        final_pgd_pil = TF.to_pil_image(decode_01(x, architect.vae).squeeze(0).cpu())
    final_pgd_pil.save(os.path.join(args.output_dir, "final_scribble_pgd.png"))

    plot_row(pgd_eval_photos, f"PGD final photos (MMD={pgd_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_pgd.png"))

    photo_dir = os.path.join(args.output_dir, "photos_pgd")
    os.makedirs(photo_dir, exist_ok=True)
    for idx, photo in enumerate(pgd_eval_photos):
        photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))

    wandb.log({
        "final_pgd_mmd":         pgd_mmd,
        "final_pgd_mmd_250":     pgd_mmd_final,
        "final_init_mmd":        init_mmd,
        "mmd_delta":             init_mmd - pgd_mmd,
        "final_scribble_pgd":    wandb.Image(final_pgd_pil),
        "pgd_eval_photos":       [wandb.Image(p) for p in pgd_eval_photos],
    })
    wandb.summary["final_pgd_mmd"]      = pgd_mmd
    wandb.summary["final_pgd_mmd_250"]  = pgd_mmd_final
    wandb.summary["final_init_mmd"]     = init_mmd
    wandb.summary["mmd_delta"]          = init_mmd - pgd_mmd
    wandb.summary["n_rounds"]           = round_idx

    npy_dir = os.path.join(args.output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    save_image_list_npy(pgd_eval_photos, os.path.join(npy_dir, "photos_pgd.npy"))
    for name, imgs in target_images_per_group.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        save_image_list_npy(imgs, os.path.join(npy_dir, f"targets_{safe_name}.npy"))
    save_image_list_npy([source_image],   os.path.join(npy_dir, "source_portrait.npy"))
    save_image_list_npy([scribble_pil],   os.path.join(npy_dir, "scribble.npy"))
    save_image_list_npy([final_pgd_pil],  os.path.join(npy_dir, "final_scribble_pgd.npy"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "args":                  vars(args),
            "n_rounds":              round_idx,
            "final_pgd_mmd":         pgd_mmd,
            "final_pgd_mmd_n_eval_final": pgd_mmd_final,
            "n_eval_final":          args.n_eval_final,
            "final_init_mmd":        init_mmd,
            "mmd_delta":             init_mmd - pgd_mmd,
            "optimization_time_sec": optimization_time_sec,
        }, f, indent=2)
    print(f"✅ metrics.json saved. Optimization time: {optimization_time_sec/60:.1f} min", flush=True)

    wandb.finish()
    print(f"\n✅ All outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
