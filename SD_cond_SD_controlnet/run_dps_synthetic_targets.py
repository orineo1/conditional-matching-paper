"""
DPS CLIP-MMD Pipeline with synthetic target distributions.

Instead of generating target portraits with the sprinter, uses CLIP embeddings
of two anchor images to construct synthetic target distributions:
  - binary: 50× CLIP(A) + 50× CLIP(B) — bimodal target
  - interpolated: N evenly spaced points along the CLIP geodesic from A to B

Sections 3-6 of run_dps.py are replaced; sections 7-12 are identical.
"""

import argparse
import copy
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA

# Ensure SD_cond_SD_controlnet/ is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0_direct,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
    run_dps_step_clip,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd, evaluate_distribution_mmd, compute_swd
from models import load_models, setup_gradient_checkpointing
from visualization import plot_row, visualize_step, compare_scribbles_heatmap

LOSS_FNS = {"mmd": compute_mmd, "swd": compute_swd}


def parse_args():
    p = argparse.ArgumentParser(description="DPS CLIP-MMD Pipeline with synthetic targets")
    p.add_argument("--output_dir", type=str, default="SD_cond_SD_controlnet/output/dps_synthetic")
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--architect_unet_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="measure_MMD_between_uncond_dps")
    p.add_argument("--wandb_entity", type=str, default="conditional-matching")

    # Synthetic targets
    p.add_argument("--anchor_a_path", type=str, default=None,
                   help="Path to anchor image A (required for binary/interpolated/gender_bimodal)")
    p.add_argument("--anchor_b_path", type=str, default=None,
                   help="Path to anchor image B (required for binary/interpolated/gender_bimodal)")
    p.add_argument("--target_mode", type=str, required=True,
                   choices=["binary", "interpolated",
                            "age_continuous", "gender_bimodal", "age_gender_combined"],
                   help=("binary/gender_bimodal: 50/50 copies of A,B; "
                         "interpolated: geodesic A→B; "
                         "age_continuous: real CLIP embeddings of age_* reference images; "
                         "age_gender_combined: age_woman + age_man reference images"))
    p.add_argument("--n_targets", type=int, default=100,
                   help="Total number of synthetic target embeddings")
    p.add_argument("--reference_images_dir", type=str, default="reference_images",
                   help="Base dir for age reference images (contains age_man/ and age_woman/)")
    p.add_argument("--reference_gender", type=str, default="woman",
                   choices=["man", "woman"],
                   help="Gender subdir to use for age_continuous mode")
    p.add_argument("--scribble_path", type=str, required=True,
                   help="Path to input scribble image for DPS")

    # Scheduler / loop
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--start_step", type=int, default=15,
                   help="SDEdit start step — DPS runs from here to n_steps")

    # Guidance
    p.add_argument("--base_zeta", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="CFG scale for architect (0.0 = unconditional)")
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--loss_fn", type=str, default="mmd", choices=["mmd", "swd"])
    p.add_argument("--bandwidth_scale", type=float, default=1.0,
                   help="Scale factor for MMD bandwidth (< 1 = sharper kernel)")
    p.add_argument("--loss_scale", type=float, default=1.0,
                   help="Multiply loss by this factor before grad computation")
    p.add_argument("--kernel_alpha", type=float, default=1.0,
                   help="Generalized RBF exponent. >1 = sharper falloff.")

    # Variations / eval
    p.add_argument("--num_variations", type=int, default=6)
    p.add_argument("--n_eval", type=int, default=10,
                   help="Sprinter photos per MMD evaluation")
    p.add_argument("--eval_interval", type=int, default=0,
                   help="Evaluate intermediate MMD every N steps (0 = auto ~5 checkpoints)")

    # Prompts
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--sprinter_variation_prompt", type=str,
                   default="a superrealistic professional photograph of")
    p.add_argument("--sprinter_eval_prompt", type=str,
                   default="a superrealistic professional photograph of")

    # Models
    p.add_argument("--controlnet_model_id", type=str,
                   default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--sprinter_model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--architect_model_id", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")

    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    steps_dir = os.path.join(args.output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── wandb ──────────────────────────────────────────────────────────────────
    import wandb

    # ── 1. Load models ─────────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    architect, sprinter = load_models(
        device,
        architect_lora_path=args.lora_path,
        architect_unet_path=args.architect_unet_path,
        controlnet_model_id=args.controlnet_model_id,
        sprinter_model_id=args.sprinter_model_id,
        architect_model_id=args.architect_model_id,
    )
    clip_model, clip_processor = load_clip_model(device)
    print("Models loaded.", flush=True)

    # ── 2. Load scribble from file (skip base oval + HED extraction) ──────────
    print(f"Loading scribble from {args.scribble_path}...", flush=True)
    scribble_pil = Image.open(args.scribble_path).convert("RGB").resize((512, 512))
    sobel_cond_pil = scribble_pil  # used as controlnet conditioning
    scribble_pil.save(os.path.join(args.output_dir, "scribble.png"))
    print("Scribble loaded.", flush=True)

    # ── 3-6. Construct target CLIP embeddings ─────────────────────────────────
    anchor_a_pil = anchor_b_pil = None
    e_a = e_b = None
    anchor_sim = None
    target_ages = None       # list[int] or None — set for age-based modes
    target_genders = None    # list[int] 0/1 or None — set for combined mode

    anchor_modes = {"binary", "interpolated", "gender_bimodal"}
    age_modes    = {"age_continuous", "age_gender_combined"}

    if args.target_mode in anchor_modes:
        if args.anchor_a_path is None or args.anchor_b_path is None:
            raise ValueError(f"--anchor_a_path and --anchor_b_path are required "
                             f"for target_mode={args.target_mode}")
        print("Loading anchor images...", flush=True)
        anchor_a_pil = Image.open(args.anchor_a_path).convert("RGB").resize((512, 512))
        anchor_b_pil = Image.open(args.anchor_b_path).convert("RGB").resize((512, 512))
        anchor_a_pil.save(os.path.join(args.output_dir, "anchor_a.png"))
        anchor_b_pil.save(os.path.join(args.output_dir, "anchor_b.png"))

        with torch.no_grad():
            anchor_a_tensor = TF.to_tensor(anchor_a_pil).unsqueeze(0).to(device)
            anchor_b_tensor = TF.to_tensor(anchor_b_pil).unsqueeze(0).to(device)
            e_a = encode_images_clip(anchor_a_tensor, clip_model, clip_processor)
            e_b = encode_images_clip(anchor_b_tensor, clip_model, clip_processor)

        anchor_sim = (e_a @ e_b.T).item()
        print(f"  Anchor A: {args.anchor_a_path}")
        print(f"  Anchor B: {args.anchor_b_path}")
        print(f"  Anchor cosine similarity: {anchor_sim:.4f}")

    def _load_reference_dir(ref_dir):
        """Load all PNG images from ref_dir, return (pil_list, age_list)."""
        import re
        pngs = sorted([f for f in os.listdir(ref_dir) if f.endswith(".png")])
        pil_imgs, ages_list = [], []
        for fname in pngs:
            m = re.match(r"age_(\d+)_", fname)
            if m is None:
                continue
            pil_imgs.append(Image.open(os.path.join(ref_dir, fname)).convert("RGB").resize((512, 512)))
            ages_list.append(int(m.group(1)))
        return pil_imgs, ages_list

    def _encode_pil_list(pil_imgs):
        """Encode a list of PIL images to CLIP embeddings [N, 768]."""
        tensors = torch.cat(
            [TF.to_tensor(img).unsqueeze(0) for img in pil_imgs], dim=0
        ).to(device)
        with torch.no_grad():
            embs = encode_images_clip(tensors, clip_model, clip_processor)
        return embs

    N = args.n_targets

    if args.target_mode in ("binary", "gender_bimodal"):
        n_a = N // 2
        n_b = N - n_a
        all_clip_embeddings = torch.cat([e_a.repeat(n_a, 1), e_b.repeat(n_b, 1)], dim=0)
        print(f"  {args.target_mode} target: {n_a}× A + {n_b}× B = {N} embeddings")

    elif args.target_mode == "interpolated":
        alphas = torch.linspace(0.0, 1.0, N, device=device)
        interp = alphas.unsqueeze(1) * e_a + (1.0 - alphas.unsqueeze(1)) * e_b
        all_clip_embeddings = interp / interp.norm(dim=-1, keepdim=True)
        print(f"  Interpolated target: {N} points along A→B geodesic")

    elif args.target_mode == "age_continuous":
        ref_dir = os.path.join(args.reference_images_dir, f"age_{args.reference_gender}")
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(f"Reference dir not found: {ref_dir}")
        pil_imgs, target_ages = _load_reference_dir(ref_dir)
        if len(pil_imgs) == 0:
            raise RuntimeError(f"No age_*.png images found in {ref_dir}")
        print(f"  age_continuous: loaded {len(pil_imgs)} images from {ref_dir}", flush=True)
        all_clip_embeddings = _encode_pil_list(pil_imgs)
        N = all_clip_embeddings.shape[0]
        print(f"  Age range: [{min(target_ages)}, {max(target_ages)}]  mean={np.mean(target_ages):.1f}")

    elif args.target_mode == "age_gender_combined":
        half = N // 2
        for gender_tag in ("woman", "man"):
            ref_dir = os.path.join(args.reference_images_dir, f"age_{gender_tag}")
            if not os.path.isdir(ref_dir):
                raise FileNotFoundError(f"Reference dir not found: {ref_dir}")

        ref_dir_w = os.path.join(args.reference_images_dir, "age_woman")
        ref_dir_m = os.path.join(args.reference_images_dir, "age_man")
        pil_w, ages_w = _load_reference_dir(ref_dir_w)
        pil_m, ages_m = _load_reference_dir(ref_dir_m)

        # Sample half from each; take first `half` after shuffle for reproducibility
        rng_np = np.random.default_rng(42)
        idx_w = rng_np.choice(len(pil_w), size=min(half, len(pil_w)), replace=False)
        idx_m = rng_np.choice(len(pil_m), size=min(half, len(pil_m)), replace=False)
        pil_w = [pil_w[i] for i in idx_w]
        pil_m = [pil_m[i] for i in idx_m]
        ages_w = [ages_w[i] for i in idx_w]
        ages_m = [ages_m[i] for i in idx_m]

        emb_w = _encode_pil_list(pil_w)
        emb_m = _encode_pil_list(pil_m)
        all_clip_embeddings = torch.cat([emb_w, emb_m], dim=0)
        target_ages    = ages_w + ages_m
        target_genders = [0] * len(ages_w) + [1] * len(ages_m)  # 0=woman, 1=man
        N = all_clip_embeddings.shape[0]
        print(f"  age_gender_combined: {len(ages_w)} women + {len(ages_m)} men = {N} embeddings")

    print(f"  Target embeddings shape: {all_clip_embeddings.shape}")
    norms = all_clip_embeddings.norm(dim=-1)
    print(f"  Norms min/max: {norms.min():.4f} / {norms.max():.4f}")

    # Target CLIP PCA
    _pca = PCA(n_components=2)
    _coords = _pca.fit_transform(all_clip_embeddings.cpu().numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    if args.target_mode in ("binary", "gender_bimodal"):
        n_a_plot = N // 2
        ax.scatter(_coords[:n_a_plot, 0], _coords[:n_a_plot, 1],
                   c='dodgerblue', label=f'Anchor A ({n_a_plot})', alpha=0.7)
        ax.scatter(_coords[n_a_plot:, 0], _coords[n_a_plot:, 1],
                   c='crimson', label=f'Anchor B ({N - n_a_plot})', alpha=0.7)
        ax.legend()
    elif args.target_mode == "interpolated":
        sc = ax.scatter(_coords[:, 0], _coords[:, 1],
                        c=np.linspace(0, 1, N), cmap='coolwarm', alpha=0.7)
        plt.colorbar(sc, ax=ax, label='α (A→B)')
    elif args.target_mode == "age_continuous":
        sc = ax.scatter(_coords[:, 0], _coords[:, 1],
                        c=target_ages, cmap='plasma', alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Age')
    elif args.target_mode == "age_gender_combined":
        n_w = sum(1 for g in target_genders if g == 0)
        sc_w = ax.scatter(_coords[:n_w, 0], _coords[:n_w, 1],
                          c=target_ages[:n_w], cmap='Blues', alpha=0.7,
                          marker='o', label='Woman', vmin=20, vmax=80)
        sc_m = ax.scatter(_coords[n_w:, 0], _coords[n_w:, 1],
                          c=target_ages[n_w:], cmap='Reds', alpha=0.7,
                          marker='^', label='Man', vmin=20, vmax=80)
        plt.colorbar(sc_m, ax=ax, label='Age')
        ax.legend()
    ax.set_title(f"PCA of Target CLIP Embeddings ({args.target_mode})")
    ax.grid(True, alpha=0.3)
    pca_path = os.path.join(args.output_dir, "target_clip_pca.png")
    fig.savefig(pca_path, dpi=100, bbox_inches='tight'); plt.close(fig)
    pca_fixed = _pca
    del _coords

    # ── 7. wandb init ─────────────────────────────────────────────────────────
    eval_interval = args.eval_interval if args.eval_interval > 0 else max(1, (args.n_steps - args.start_step) // 5)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "target_mode":                  args.target_mode,
            "anchor_a_path":                args.anchor_a_path,
            "anchor_b_path":                args.anchor_b_path,
            "anchor_cosine_sim":            anchor_sim,
            "reference_images_dir":         args.reference_images_dir,
            "reference_gender":             args.reference_gender,
            "scribble_path":                args.scribble_path,
            "prompt":                       args.prompt,
            "negative_prompt":              args.negative_prompt,
            "n_targets":                    N,
            "n_steps":                      args.n_steps,
            "start_step":                   args.start_step,
            "strength":                     1 - args.start_step / args.n_steps,
            "steps_run":                    args.n_steps - args.start_step,
            "scheduler_type":               type(architect.scheduler).__name__,
            "num_variations":               args.num_variations,
            "base_zeta":                    args.base_zeta,
            "guidance_scale":               args.guidance_scale,
            "controlnet_scale":             args.controlnet_scale,
            "n_eval":                       args.n_eval,
            "eval_interval":                eval_interval,
            "lora_path":                    args.lora_path,
            "architect_unet_path":          args.architect_unet_path,
            "architect_model":              args.architect_model_id,
            "sprinter_model":               args.sprinter_model_id,
            "sprinter_variation_prompt":    args.sprinter_variation_prompt,
            "sprinter_eval_prompt":         args.sprinter_eval_prompt,
            "loss_fn":                      args.loss_fn,
            "loss_scale":                   args.loss_scale,
            "bandwidth_scale":              args.bandwidth_scale,
            "kernel_alpha":                 args.kernel_alpha,
        },
    )
    print(f"wandb run: {run.name}", flush=True)

    # Log input images
    wandb_log = {
        "scribble":        wandb.Image(scribble_pil),
        "target_clip_pca": wandb.Image(pca_path),
    }
    if anchor_a_pil is not None:
        wandb_log["anchor_a"] = wandb.Image(anchor_a_pil)
    if anchor_b_pil is not None:
        wandb_log["anchor_b"] = wandb.Image(anchor_b_pil)
    wandb.log(wandb_log)
    print("Input images logged to wandb.", flush=True)

    # ── 8. Prepare DPS ─────────────────────────────────────────────────────────
    height, width = 512, 512
    n_steps = args.n_steps
    start_step = args.start_step
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    sprinter.vae.to(dtype=torch.float32)
    setup_gradient_checkpointing(architect, sprinter)

    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = architect.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
        )

    architect.scheduler.set_timesteps(n_steps, device=device)
    timesteps = architect.scheduler.timesteps

    scheduler_regular = copy.deepcopy(architect.scheduler)

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids":    add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # ── SDEdit-style init: encode scribble → latent, noise to start_step ──
    with torch.no_grad():
        scribble_tensor = TF.to_tensor(scribble_pil).unsqueeze(0).to(device).to(torch.float32)
        scribble_tensor = (scribble_tensor * 2.0) - 1.0  # [0,1] → [-1,1]
        scribble_latent = architect.vae.encode(scribble_tensor).latent_dist.mean
        scribble_latent = scribble_latent * architect.vae.config.scaling_factor

    t_start        = timesteps[start_step]
    alphas_cumprod = architect.scheduler.alphas_cumprod.to(device)
    alpha          = alphas_cumprod[t_start.long()].to(torch.float32)
    noise          = torch.randn_like(scribble_latent)
    latents        = ((alpha ** 0.5) * scribble_latent + ((1 - alpha) ** 0.5) * noise).to(torch.float16)
    latents_regular = latents.detach().clone()

    timesteps_to_run = timesteps[start_step:]

    print(f"Ready. Starting from step {start_step}/{n_steps}  (t={t_start.item():.0f})", flush=True)
    print(f"   Running {len(timesteps_to_run)} DPS steps...", flush=True)

    step_gradients = []
    step_vis_data  = []
    target_clip_np = all_clip_embeddings.cpu().numpy()
    from functools import partial
    if args.loss_fn == "mmd":
        loss_fn = partial(compute_mmd, bandwidth_scale=args.bandwidth_scale, kernel_alpha=args.kernel_alpha)
    else:
        loss_fn = LOSS_FNS[args.loss_fn]

    # ── Baseline visualization (step 0, before any DPS correction) ────────────
    with torch.no_grad():
        baseline_noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents.detach(), timesteps_to_run[0],
            cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        baseline_pred_x0 = compute_pred_x0_direct(
            architect.scheduler, baseline_noise_pred, timesteps_to_run[0], latents.detach()
        )

        baseline_px = architect.vae.decode(
            (baseline_pred_x0 / architect.vae.config.scaling_factor).to(architect.vae.dtype)
        ).sample
        baseline_px_norm = torch.clamp((baseline_px + 1.0) / 2.0, 0.0, 1.0)

        sprinter.vae.to(dtype=torch.float16)
        baseline_var_images = [
            sprinter(
                prompt=args.sprinter_variation_prompt,
                image=baseline_px_norm,
                num_inference_steps=2, guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale, output_type="pil",
            ).images[0]
            for _ in range(args.n_eval)
        ]
        sprinter.vae.to(dtype=torch.float32)

        var_tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in baseline_var_images], dim=0).to(device)
        clip_model.to(device)
        baseline_clip_flat = encode_images_clip(var_tensors, clip_model, clip_processor).cpu().numpy()
        clip_model.to("cpu")

        sd_baseline = {
            "step": 0,
            "timestep": timesteps_to_run[0].item(),
            "mmd_loss": 0.0,
            "zeta_i": 0.0,
            "latents_step_cpu": latents.detach().cpu(),
            "latents_step_regular_cpu": latents.detach().cpu(),
            "pred_x0_cpu": baseline_pred_x0.detach().cpu(),
            "pred_x0_regular_cpu": baseline_pred_x0.detach().cpu(),
            "variation_clip_flat": baseline_clip_flat,
        }

    baseline_save_path = os.path.join(steps_dir, "step_baseline.png")
    visualize_step(sd_baseline, architect, sprinter, target_clip_np,
                   num_cond=4, save_path=baseline_save_path, pca_fixed=pca_fixed)
    print("Baseline visualization saved.", flush=True)

    # ── 9. DPS loop ────────────────────────────────────────────────────────────
    for i, t in enumerate(timesteps_to_run):
        print(f"\n{'='*60}", flush=True)
        print(f"Step {i+1}/{len(timesteps_to_run)}  (t={t})", flush=True)
        print(f"{'='*60}", flush=True)

        latents_step         = latents.detach().requires_grad_(True)
        latents_step_regular = latents_regular.detach()

        # Noise prediction
        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
        )
        with torch.no_grad():
            noise_pred_regular = predict_noise_cfg(
                architect.unet, scheduler_regular,
                latents_step_regular, t, cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
            )

        # pred_x0
        pred_x0 = compute_pred_x0_direct(architect.scheduler, noise_pred, t, latents_step)
        with torch.no_grad():
            pred_x0_regular = compute_pred_x0_direct(
                scheduler_regular, noise_pred_regular, t, latents_step_regular)

        # Decode pred_x0 → pixel space (keep grad)
        pred_x0_scaled = pred_x0 / architect.vae.config.scaling_factor

        def vae_decode_checkpoint(lat):
            return architect.vae.decode(lat.to(architect.vae.dtype)).sample

        pixel_x0 = torch.utils.checkpoint.checkpoint(
            vae_decode_checkpoint, pred_x0_scaled, use_reentrant=False)
        pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)

        # CLIP-MMD + gradient
        grad, mmd_loss, zeta_i, loss_norm, vl_clip_flat = run_dps_step_clip(
            latents=latents,
            latents_step=latents_step,
            noise_pred=noise_pred,
            pixel_x0_norm=pixel_x0_norm,
            sprinter=sprinter,
            all_clip_embeddings=all_clip_embeddings,
            num_variations=args.num_variations,
            variation_batch_size=1,
            base_zeta_prime=args.base_zeta,
            clip_model=clip_model,
            clip_processor=clip_processor,
            vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=args.sprinter_variation_prompt,
            loss_fn=loss_fn,
            loss_scale=args.loss_scale,
        )

        grad_norm = grad.norm().item()
        zeta_val  = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i
        print(f"  MMD={mmd_loss.item():.6f}  zeta_i={zeta_val:.4f}  ||grad||={grad_norm:.6f}", flush=True)

        if torch.isnan(grad).any():
            print(f"  NaN in gradient at step {i} — skipping correction", flush=True)
            correction = torch.zeros_like(latents_step)
        else:
            correction = -zeta_i * grad

        step_gradients.append({
            "step":            i+1,
            "timestep":        t.item(),
            "gradient_norm":   grad_norm,
            "mmd_loss":        mmd_loss.item(),
            "zeta_i":          zeta_val,
            "loss_norm":       loss_norm.item(),
            "correction_norm": zeta_val * grad_norm,
        })

        wandb_log = {
            "step":            i+1,
            "mmd_loss":        mmd_loss.item(),
            "gradient_norm":   grad_norm,
            "zeta":            zeta_val,
            "correction_norm": zeta_val * grad_norm,
        }

        # Intermediate MMD evaluation
        if i % eval_interval == 0:
            unguided_mmd, _, _ = evaluate_distribution_mmd(
                pred_x0_regular.detach(), architect.vae, architect.image_processor,
                sprinter, clip_model, clip_processor,
                all_clip_embeddings, args.sprinter_eval_prompt,
                n_eval=args.n_eval, device=device,
            )
            wandb_log["intermediate/unguided_cond_mmd"] = unguided_mmd
            wandb_log["intermediate/guided_cond_mmd"]   = mmd_loss.item()
            wandb_log["intermediate/cond_mmd_delta"]    = mmd_loss.item() - unguided_mmd
            print(f"  [eval] guided={mmd_loss.item():.6f}  "
                  f"unguided={unguided_mmd:.6f}  "
                  f"delta={mmd_loss.item()-unguided_mmd:.6f}", flush=True)

        wandb.log(wandb_log, commit=False)

        # Store step data for visualization
        with torch.no_grad():
            sd = {
                "step":                     i+1,
                "timestep":                 t.item(),
                "mmd_loss":                 mmd_loss.item(),
                "zeta_i":                   zeta_val,
                "latents_step_cpu":         latents_step.detach().cpu(),
                "latents_step_regular_cpu": latents_step_regular.detach().cpu(),
                "pred_x0_cpu":              pred_x0.detach().cpu(),
                "pred_x0_regular_cpu":      pred_x0_regular.detach().cpu(),
                "variation_clip_flat":      vl_clip_flat,
            }
            step_vis_data.append(sd)

        step_save_path = os.path.join(steps_dir, f"step_{i:03d}.png")
        visualize_step(sd, architect, sprinter, target_clip_np,
                       num_cond=5, save_path=step_save_path, pca_fixed=pca_fixed)

        # Scheduler step
        latents = denoise_step(architect.scheduler, noise_pred, t, latents_step,
                               correction=correction)
        with torch.no_grad():
            latents_regular = denoise_step(scheduler_regular, noise_pred_regular,
                                           t, latents_step_regular)

        # Cleanup
        del grad, mmd_loss, loss_norm, zeta_i, correction
        del pixel_x0, pixel_x0_norm, pred_x0, pred_x0_regular
        del latents_step_regular, noise_pred_regular
        gc.collect(); torch.cuda.empty_cache()

    del latents_step, noise_pred
    torch.cuda.empty_cache()
    print(f"\nCLIP-MMD DPS Complete! {len(step_vis_data)} steps stored.", flush=True)

    # ── 10. Final MMD evaluation ───────────────────────────────────────────────
    print("Computing final MMD (regular)...", flush=True)
    regular_mmd, regular_eval_photos, _ = evaluate_distribution_mmd(
        latents_regular, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=args.n_eval, device=device,
    )

    print("Computing final MMD (DPS)...", flush=True)
    dps_mmd, dps_eval_photos, _ = evaluate_distribution_mmd(
        latents, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=args.sprinter_eval_prompt,
        n_eval=args.n_eval, device=device,
    )

    print(f"Regular MMD : {regular_mmd:.6f}", flush=True)
    print(f"DPS MMD     : {dps_mmd:.6f}", flush=True)
    print(f"Delta (positive = DPS better): {regular_mmd - dps_mmd:.6f}", flush=True)

    # ── 11. Final visualizations ───────────────────────────────────────────────
    with torch.no_grad():
        final_dps_pil     = latent_to_pil(latents,         architect.vae, architect.image_processor)
        final_regular_pil = latent_to_pil(latents_regular, architect.vae, architect.image_processor)

    final_dps_pil.save(os.path.join(args.output_dir, "final_scribble_dps.png"))
    final_regular_pil.save(os.path.join(args.output_dir, "final_scribble_regular.png"))
    heatmap_path = os.path.join(args.output_dir, "scribble_heatmap.png")
    compare_scribbles_heatmap(final_dps_pil, final_regular_pil, save_path=heatmap_path)
    print("Scribble heatmap saved.", flush=True)

    plot_row(regular_eval_photos, f"Regular final photos  (MMD={regular_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_regular.png"))
    plot_row(dps_eval_photos, f"DPS final photos      (MMD={dps_mmd:.4f})",
             save_path=os.path.join(args.output_dir, "final_photos_dps.png"))

    # Save individual photos for downstream evaluation
    for folder, photos in [("photos_regular", regular_eval_photos),
                           ("photos_dps", dps_eval_photos)]:
        photo_dir = os.path.join(args.output_dir, folder)
        os.makedirs(photo_dir, exist_ok=True)
        for idx, photo in enumerate(photos):
            photo.save(os.path.join(photo_dir, f"photo_{idx:03d}.png"))

    # Training curves
    steps_list = [d["step"]          for d in step_gradients]
    mmd_vals   = [d["mmd_loss"]      for d in step_gradients]
    grad_norms = [d["gradient_norm"] for d in step_gradients]
    zetas      = [d["zeta_i"]        for d in step_gradients]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"CLIP-MMD DPS Training Curves ({args.target_mode} target)", fontsize=14, fontweight="bold")
    axes[0].plot(steps_list, mmd_vals,   color="royalblue"); axes[0].set_title("MMD Loss");      axes[0].set_xlabel("Step"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps_list, grad_norms, color="crimson");   axes[1].set_title("Gradient Norm"); axes[1].set_xlabel("Step"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps_list, zetas,      color="seagreen");  axes[2].set_title("Zeta"); axes[2].set_xlabel("Step"); axes[2].grid(True, alpha=0.3)
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=100, bbox_inches="tight"); plt.close(fig)

    # Final CLIP PCA comparison
    def pil_list_to_clip(pil_list):
        tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
        tensor = torch.cat(tensors, dim=0).to(device)
        clip_model.to(device)
        with torch.no_grad():
            embs = encode_images_clip(tensor, clip_model, clip_processor)
        clip_model.to("cpu")
        return embs.cpu().numpy()

    regular_eval_clip = pil_list_to_clip(regular_eval_photos)
    dps_eval_clip     = pil_list_to_clip(dps_eval_photos)
    target_clip_np_   = all_clip_embeddings.cpu().numpy()

    combined = np.vstack([target_clip_np_, regular_eval_clip, dps_eval_clip])
    coords = pca_fixed.transform(combined)

    n_target  = target_clip_np_.shape[0]
    n_ev      = regular_eval_clip.shape[0]
    target_c  = coords[:n_target]
    regular_c = coords[n_target : n_target + n_ev]
    dps_c     = coords[n_target + n_ev :]

    fig, ax = plt.subplots(figsize=(9, 7))
    if args.target_mode in ("binary", "gender_bimodal"):
        n_a_plot = n_target // 2
        ax.scatter(target_c[:n_a_plot, 0], target_c[:n_a_plot, 1],
                   c="royalblue", alpha=0.6, s=60, label=f"Target A ({n_a_plot})")
        ax.scatter(target_c[n_a_plot:, 0], target_c[n_a_plot:, 1],
                   c="crimson", alpha=0.6, s=60, label=f"Target B ({n_target - n_a_plot})")
    elif args.target_mode == "interpolated":
        sc = ax.scatter(target_c[:, 0], target_c[:, 1],
                        c=np.linspace(0, 1, n_target), cmap='coolwarm', alpha=0.6, s=60)
        plt.colorbar(sc, ax=ax, label='α (A→B)')
    elif args.target_mode == "age_continuous":
        sc = ax.scatter(target_c[:, 0], target_c[:, 1],
                        c=target_ages, cmap='plasma', alpha=0.6, s=60)
        plt.colorbar(sc, ax=ax, label='Age')
    elif args.target_mode == "age_gender_combined":
        n_w = sum(1 for g in target_genders if g == 0)
        ax.scatter(target_c[:n_w, 0], target_c[:n_w, 1],
                   c=target_ages[:n_w], cmap='Blues', alpha=0.6, s=60,
                   marker='o', label='Woman', vmin=20, vmax=80)
        sc_m = ax.scatter(target_c[n_w:, 0], target_c[n_w:, 1],
                          c=target_ages[n_w:], cmap='Reds', alpha=0.6, s=60,
                          marker='^', label='Man', vmin=20, vmax=80)
        plt.colorbar(sc_m, ax=ax, label='Age')
    ax.scatter(regular_c[:, 0], regular_c[:, 1],
               c="orange", alpha=0.8, s=80, marker="s",
               label=f"Unguided eval (MMD={regular_mmd:.4f})")
    ax.scatter(dps_c[:, 0], dps_c[:, 1],
               c="limegreen", alpha=0.8, s=80, marker="x",
               label=f"DPS eval     (MMD={dps_mmd:.4f})")
    ax.set_title(f"Final CLIP PCA — {args.target_mode} target vs Unguided vs DPS\n"
                 f"Var explained: {pca_fixed.explained_variance_ratio_.sum():.1%}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_final_path = os.path.join(args.output_dir, "final_pca_comparison.png")
    fig.savefig(pca_final_path, dpi=120, bbox_inches="tight"); plt.close(fig)

    # ── 12. wandb final logs ───────────────────────────────────────────────────
    wandb.log({
        "final_dps_mmd":            dps_mmd,
        "final_regular_mmd":        regular_mmd,
        "mmd_delta":                regular_mmd - dps_mmd,
        "mmd_relative_improvement": (regular_mmd - dps_mmd) / (regular_mmd + 1e-8),
        "final_scribble_dps":       wandb.Image(final_dps_pil),
        "final_scribble_regular":   wandb.Image(final_regular_pil),
        "dps_eval_photos":          [wandb.Image(p) for p in dps_eval_photos],
        "regular_eval_photos":      [wandb.Image(p) for p in regular_eval_photos],
        "final_pca_comparison":     wandb.Image(pca_final_path),
        "training_curves":          wandb.Image(curves_path),
        "scribble_heatmap":         wandb.Image(heatmap_path),
    })

    wandb.summary["final_dps_mmd"]     = dps_mmd
    wandb.summary["final_regular_mmd"] = regular_mmd
    wandb.summary["mmd_delta"]         = regular_mmd - dps_mmd
    wandb.summary["final_grad_norm"]   = step_gradients[-1]["gradient_norm"]

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "args": vars(args),
            "steps": step_gradients,
            "final_dps_mmd": dps_mmd,
            "final_regular_mmd": regular_mmd,
            "mmd_delta": regular_mmd - dps_mmd,
        }, f, indent=2)

    wandb.finish()
    print(f"\nAll outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
