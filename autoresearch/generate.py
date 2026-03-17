"""
Autonomous Research Orchestrator — DPS CLIP-MMD Pipeline.

This is the ONLY file the autoresearch agent modifies.
All knobs are at the top. Imports from SD_cond_SD_controlnet/ modules.

Phases:
  1. Startup (not timed): load models, generate targets, encode CLIP
  2. DPS loop (timed): denoise with CLIP-MMD gradient corrections
  3. Evaluation (not timed): final MMD for guided + unguided
  4. Output: wandb + standardized stdout block
"""

import copy
import gc
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════════════════════════
# KNOBS — everything the autoresearch agent may tune
# ═══════════════════════════════════════════════════════════════════════════════

# -- Scheduler / denoising --
N_STEPS = 250                    # total scheduler timesteps
START_STEP = 125                 # SDEdit start index (strength = 1 - start/n)
SCHEDULER_TYPE = "ddim"          # "ddim" | "euler_discrete"
NOISE_SCHEDULE = "karras"        # "linear" (default DDIM spacing) | "karras" (Karras eq 5)
KARRAS_RHO = 7.0                 # Karras schedule shape: higher = more steps near end (low noise)

# -- DPS guidance --
BASE_ZETA = 5.0                  # adaptive ζ = BASE_ZETA / ||MMD_loss||
GUIDANCE_SCALE = 0.0             # CFG scale for architect (0 = unconditional)
GRADIENT_MOMENTUM = 0.0          # EMA coefficient for gradient accumulation (0 = off)

# -- Sprinter / variations --
NUM_VARIATIONS = 100             # variations per DPS step
CONTROLNET_SCALE = 0.5           # ControlNet conditioning scale
SPRINTER_INFERENCE_STEPS = 2     # sprinter denoising steps

# -- Targets / evaluation --
N_TARGETS = 100                  # target distribution size (half man, half woman)
N_EVAL = 100                     # sprinter photos for final MMD eval

# -- Models --
ARCHITECT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SPRINTER_MODEL_ID = "stabilityai/sdxl-turbo"
CONTROLNET_MODEL_ID = "xinsir/controlnet-scribble-sdxl-1.0"
ARCHITECT_UNET_PATH = None       # None = base, or path to fine-tuned merged U-Net

# -- Prompts --
PROMPT = ""                      # architect prompt (empty = unconditional)
NEGATIVE_PROMPT = ""
VARIATION_PROMPT = "a superrealistic professional photograph of"
TARGET_MAN_PROMPT = "a superrealistic portrait photograph of a man, studio lighting"
TARGET_WOMAN_PROMPT = "a superrealistic portrait photograph of a woman, studio lighting"
EVAL_PROMPT = "a superrealistic professional photograph of"

# -- Misc --
SEED = 1
OUTPUT_DIR = "autoresearch_output"
FAIRFACE_WEIGHTS = "/sci/labs/orzuk/shaulytolk/models/fairface/res34_fair_align_multi_7_20190809.pt"

# -- wandb --
WANDB_PROJECT = "autoresearch"
WANDB_ENTITY = "conditional-matching"

# ═══════════════════════════════════════════════════════════════════════════════
# ENV VAR OVERRIDES — allows running different configs in parallel
# ═══════════════════════════════════════════════════════════════════════════════
BASE_ZETA = float(os.environ.get("AR_BASE_ZETA", BASE_ZETA))
NOISE_SCHEDULE = os.environ.get("AR_NOISE_SCHEDULE", NOISE_SCHEDULE)
KARRAS_RHO = float(os.environ.get("AR_KARRAS_RHO", KARRAS_RHO))
OUTPUT_DIR = os.environ.get("AR_OUTPUT_DIR", OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# END KNOBS
# ═══════════════════════════════════════════════════════════════════════════════

# Add SD_cond_SD_controlnet to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sd_dir = os.path.join(script_dir, "SD_cond_SD_controlnet")
if sd_dir not in sys.path:
    sys.path.insert(0, sd_dir)

from clip_utils import encode_images_clip, load_clip_model
from generation import (
    compute_pred_x0_direct,
    denoise_step,
    generate_and_store_cs,
    predict_noise_cfg,
    run_dps_step_clip,
)
from image_utils import build_base_image, latent_to_pil, sobel_proxy
from metrics import compute_mmd, evaluate_distribution_mmd
from models import load_models, setup_gradient_checkpointing

# Add scripts/ to path for gender classifier
scripts_dir = os.path.join(script_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)


def compute_karras_timesteps(scheduler, n_steps, rho, device):
    """Compute custom timesteps using Karras noise schedule (EDM eq 5).

    σ_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

    Then map each σ to the nearest DDIM timestep via alphas_cumprod.
    Higher rho = more steps concentrated near low noise (later denoising).
    """
    alphas_cumprod = scheduler.alphas_cumprod
    # Convert alpha schedule to sigma: σ = sqrt((1-α)/α)
    sigmas_full = ((1 - alphas_cumprod) / alphas_cumprod).sqrt()
    sigma_min = sigmas_full[-1].item()  # smallest sigma (cleanest)
    sigma_max = sigmas_full[0].item()   # largest sigma (noisiest)

    # Karras eq 5: inverse CDF spacing
    ramp = torch.linspace(0, 1, n_steps)
    karras_sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Map each Karras sigma to nearest scheduler timestep
    timestep_indices = []
    for sigma in karras_sigmas:
        # Find nearest sigma in the full schedule
        idx = (sigmas_full - sigma).abs().argmin().item()
        timestep_indices.append(idx)

    # Deduplicate while preserving order (keep first occurrence)
    seen = set()
    unique_indices = []
    for idx in timestep_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    custom_timesteps = torch.tensor(unique_indices, device=device, dtype=torch.long)
    print(f"Karras schedule (rho={rho}): {len(custom_timesteps)} unique timesteps "
          f"from t={custom_timesteps[0].item()} to t={custom_timesteps[-1].item()}", flush=True)
    return custom_timesteps


def pil_images_to_tensor(pil_list, device):
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_list]
    return torch.cat(tensors, dim=0).to(device)


def extract_scribble_hed(pil_image):
    from controlnet_aux import HEDdetector
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_image, scribble=True)


from fairface.gender_classifier import evaluate_gender_balance  # noqa: E402


def main():
    time_budget = int(os.environ.get("TIME_BUDGET_SECONDS", 600))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Time budget: {time_budget}s", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    # ── 1. Load models (startup, not timed) ───────────────────────────────────
    print("Loading models...", flush=True)
    architect, sprinter = load_models(
        device,
        architect_unet_path=ARCHITECT_UNET_PATH,
        controlnet_model_id=CONTROLNET_MODEL_ID,
        sprinter_model_id=SPRINTER_MODEL_ID,
        architect_model_id=ARCHITECT_MODEL_ID,
    )
    clip_model, clip_processor = load_clip_model(device)
    print("Models loaded.", flush=True)

    # ── 2. Base image + target generation (startup) ───────────────────────────
    base_image_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_cond_tensor = sobel_proxy(base_tensor, device)
        sobel_cond_pil = T.ToPILImage()(sobel_cond_tensor.squeeze(0).cpu())

    n_half = N_TARGETS // 2
    print(f"Generating {N_TARGETS} target images...", flush=True)
    with torch.no_grad():
        man_images_init, _ = generate_and_store_cs(
            sprinter, TARGET_MAN_PROMPT, sobel_cond_pil, n_half,
            batch_size=2, cn_scale=CONTROLNET_SCALE,
        )
        woman_images_init, _ = generate_and_store_cs(
            sprinter, TARGET_WOMAN_PROMPT, sobel_cond_pil, n_half,
            batch_size=2, cn_scale=CONTROLNET_SCALE,
        )

    # ── 3. Extract HED scribble ───────────────────────────────────────────────
    print("Extracting HED scribble...", flush=True)
    source_image = man_images_init[min(2, len(man_images_init) - 1)]
    scribble_pil = extract_scribble_hed(source_image)
    sobel_cond_pil = scribble_pil

    # ── 4. Regenerate targets conditioned on HED scribble ─────────────────────
    print(f"Regenerating targets on HED scribble...", flush=True)
    with torch.no_grad():
        man_images, _ = generate_and_store_cs(
            sprinter, TARGET_MAN_PROMPT, sobel_cond_pil, n_half,
            batch_size=2, cn_scale=CONTROLNET_SCALE,
        )
        woman_images, _ = generate_and_store_cs(
            sprinter, TARGET_WOMAN_PROMPT, sobel_cond_pil, n_half,
            batch_size=2, cn_scale=CONTROLNET_SCALE,
        )

    # ── 5. Encode targets to CLIP ─────────────────────────────────────────────
    print("Encoding targets to CLIP...", flush=True)
    with torch.no_grad():
        man_clip_embs = encode_images_clip(
            pil_images_to_tensor(man_images, device), clip_model, clip_processor)
        woman_clip_embs = encode_images_clip(
            pil_images_to_tensor(woman_images, device), clip_model, clip_processor)
    all_clip_embeddings = torch.cat([man_clip_embs, woman_clip_embs], dim=0)
    print(f"Target CLIP embeddings: {all_clip_embeddings.shape}", flush=True)

    # ── 6. wandb init ─────────────────────────────────────────────────────────
    import wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "n_steps": N_STEPS,
            "start_step": START_STEP,
            "scheduler_type": SCHEDULER_TYPE,
            "noise_schedule": NOISE_SCHEDULE,
            "karras_rho": KARRAS_RHO,
            "base_zeta": BASE_ZETA,
            "guidance_scale": GUIDANCE_SCALE,
            "gradient_momentum": GRADIENT_MOMENTUM,
            "num_variations": NUM_VARIATIONS,
            "controlnet_scale": CONTROLNET_SCALE,
            "sprinter_inference_steps": SPRINTER_INFERENCE_STEPS,
            "n_targets": N_TARGETS,
            "n_eval": N_EVAL,
            "architect_model": ARCHITECT_MODEL_ID,
            "architect_unet_path": ARCHITECT_UNET_PATH,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "variation_prompt": VARIATION_PROMPT,
            "seed": SEED,
        },
    )
    print(f"wandb run: {run.name}", flush=True)

    # ── 7. Prepare DPS ────────────────────────────────────────────────────────
    height, width = 512, 512
    sprinter.vae.to(dtype=torch.float32)
    setup_gradient_checkpointing(architect, sprinter)

    # Configure scheduler
    if SCHEDULER_TYPE == "euler_discrete":
        from diffusers import EulerDiscreteScheduler
        architect.scheduler = EulerDiscreteScheduler.from_config(architect.scheduler.config)
    # else: keep DDIMScheduler (set by load_models)

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            architect.encode_prompt(
                prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
                device=device, do_classifier_free_guidance=True, num_images_per_prompt=1,
            )
        )

    architect.scheduler.set_timesteps(N_STEPS, device=device)

    if NOISE_SCHEDULE == "karras":
        # Compute Karras-spaced timesteps, then override the scheduler's timesteps
        karras_ts = compute_karras_timesteps(architect.scheduler, N_STEPS, KARRAS_RHO, device)
        architect.scheduler.timesteps = karras_ts
        timesteps = karras_ts
    else:
        timesteps = architect.scheduler.timesteps

    scheduler_regular = copy.deepcopy(architect.scheduler)

    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=device)
    added_cond_kwargs = {
        "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
        "time_ids": add_time_ids.repeat(2, 1),
    }
    cfg_encoder_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # SDEdit init: encode scribble → latent, noise to start_step
    with torch.no_grad():
        scribble_tensor = TF.to_tensor(scribble_pil).unsqueeze(0).to(device).to(torch.float32)
        scribble_tensor = (scribble_tensor * 2.0) - 1.0
        scribble_latent = architect.vae.encode(scribble_tensor).latent_dist.mean
        scribble_latent = scribble_latent * architect.vae.config.scaling_factor

    t_start = timesteps[START_STEP]
    alphas_cumprod = architect.scheduler.alphas_cumprod.to(device)
    alpha = alphas_cumprod[t_start.long()].to(torch.float32)
    noise = torch.randn_like(scribble_latent)
    latents = ((alpha ** 0.5) * scribble_latent + ((1 - alpha) ** 0.5) * noise).to(torch.float16)
    latents_regular = latents.detach().clone()

    timesteps_to_run = timesteps[START_STEP:]
    total_params = sum(p.numel() for p in architect.unet.parameters())

    print(f"Starting DPS from step {START_STEP}/{N_STEPS} (t={t_start.item():.0f})", flush=True)
    print(f"Running {len(timesteps_to_run)} steps, {NUM_VARIATIONS} variations each", flush=True)

    # ── 8. DPS loop (TIMED) ───────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    dps_start_time = time.time()
    step_data = []
    grad_momentum = None

    for i, t in enumerate(timesteps_to_run):
        elapsed = time.time() - dps_start_time
        if elapsed > time_budget:
            print(f"Time budget exceeded at step {i} ({elapsed:.1f}s). Stopping.", flush=True)
            break

        print(f"\nStep {i+1}/{len(timesteps_to_run)} (t={t}) [{elapsed:.0f}s]", flush=True)

        latents_step = latents.detach().requires_grad_(True)
        latents_step_regular = latents_regular.detach()

        # Noise prediction
        noise_pred = predict_noise_cfg(
            architect.unet, architect.scheduler,
            latents_step, t, cfg_encoder_states, added_cond_kwargs, GUIDANCE_SCALE,
        )
        with torch.no_grad():
            noise_pred_regular = predict_noise_cfg(
                architect.unet, scheduler_regular,
                latents_step_regular, t, cfg_encoder_states, added_cond_kwargs, GUIDANCE_SCALE,
            )

        # pred_x0
        pred_x0 = compute_pred_x0_direct(architect.scheduler, noise_pred, t, latents_step)
        with torch.no_grad():
            pred_x0_regular = compute_pred_x0_direct(
                scheduler_regular, noise_pred_regular, t, latents_step_regular)

        # Decode pred_x0 → pixel
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
            num_variations=NUM_VARIATIONS,
            variation_batch_size=1,
            base_zeta_prime=BASE_ZETA,
            clip_model=clip_model,
            clip_processor=clip_processor,
            vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=VARIATION_PROMPT,
        )

        grad_norm = grad.norm().item()
        zeta_val = zeta_i.item() if isinstance(zeta_i, torch.Tensor) else zeta_i

        # Gradient momentum
        if GRADIENT_MOMENTUM > 0 and grad_momentum is not None:
            grad = GRADIENT_MOMENTUM * grad_momentum + (1 - GRADIENT_MOMENTUM) * grad
        if GRADIENT_MOMENTUM > 0:
            grad_momentum = grad.detach().clone()

        if torch.isnan(grad).any():
            print(f"  NaN gradient — skipping correction", flush=True)
            correction = torch.zeros_like(latents_step)
        else:
            correction = -zeta_i * grad

        correction_norm = correction.norm().item()
        print(f"  MMD={mmd_loss.item():.6f} zeta={zeta_val:.4f} "
              f"grad={grad_norm:.6f} corr={correction_norm:.6f}", flush=True)

        step_data.append({
            "step": i, "timestep": t.item(),
            "mmd_loss": mmd_loss.item(), "gradient_norm": grad_norm,
            "zeta": zeta_val, "correction_norm": correction_norm,
        })

        wandb.log({
            "step": i, "mmd_loss": mmd_loss.item(),
            "gradient_norm": grad_norm, "zeta": zeta_val,
            "correction_norm": correction_norm,
        })

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
        gc.collect()
        torch.cuda.empty_cache()

    del latents_step, noise_pred
    torch.cuda.empty_cache()

    training_seconds = time.time() - dps_start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    print(f"\nDPS loop complete. {len(step_data)} steps in {training_seconds:.1f}s", flush=True)

    # ── 9. Final evaluation (not timed) ───────────────────────────────────────
    print("Computing final MMD (unguided)...", flush=True)
    regular_mmd, regular_photos, _ = evaluate_distribution_mmd(
        latents_regular, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=EVAL_PROMPT,
        n_eval=N_EVAL, device=device,
    )

    print("Computing final MMD (guided)...", flush=True)
    dps_mmd, dps_photos, _ = evaluate_distribution_mmd(
        latents, architect.vae, architect.image_processor,
        sprinter, clip_model, clip_processor,
        all_clip_embeddings, eval_prompt=EVAL_PROMPT,
        n_eval=N_EVAL, device=device,
    )

    mmd_delta = regular_mmd - dps_mmd
    mmd_relative_improvement = mmd_delta / (regular_mmd + 1e-8)

    print(f"\nUnguided MMD: {regular_mmd:.6f}", flush=True)
    print(f"Guided MMD:   {dps_mmd:.6f}", flush=True)
    print(f"Delta:        {mmd_delta:.6f} (positive = DPS helped)", flush=True)
    print(f"Relative:     {mmd_relative_improvement:.6f}", flush=True)

    # ── 10. Gender balance evaluation ─────────────────────────────────────────
    print("Evaluating gender balance...", flush=True)
    gender_unguided = evaluate_gender_balance(
        regular_photos, device, save_dir=os.path.join(OUTPUT_DIR, "eval_photos_unguided"),
        weights_path=FAIRFACE_WEIGHTS)
    gender_guided = evaluate_gender_balance(
        dps_photos, device, save_dir=os.path.join(OUTPUT_DIR, "eval_photos_guided"),
        weights_path=FAIRFACE_WEIGHTS)
    print(f"Unguided gender: {gender_unguided['n_male']}M/{gender_unguided['n_female']}F "
          f"L={gender_unguided['gender_L']}", flush=True)
    print(f"Guided gender:   {gender_guided['n_male']}M/{gender_guided['n_female']}F "
          f"L={gender_guided['gender_L']}", flush=True)

    # ── 10b. Save images ──────────────────────────────────────────────────────
    print("Saving images...", flush=True)
    with torch.no_grad():
        final_dps_pil = latent_to_pil(latents, architect.vae, architect.image_processor)
        final_regular_pil = latent_to_pil(latents_regular, architect.vae, architect.image_processor)

    final_dps_pil.save(os.path.join(OUTPUT_DIR, "final_scribble_guided.png"))
    final_regular_pil.save(os.path.join(OUTPUT_DIR, "final_scribble_unguided.png"))
    scribble_pil.save(os.path.join(OUTPUT_DIR, "input_scribble.png"))

    # Save eval photos
    eval_guided_dir = os.path.join(OUTPUT_DIR, "eval_photos_guided")
    eval_unguided_dir = os.path.join(OUTPUT_DIR, "eval_photos_unguided")
    os.makedirs(eval_guided_dir, exist_ok=True)
    os.makedirs(eval_unguided_dir, exist_ok=True)
    for idx, img in enumerate(dps_photos):
        img.save(os.path.join(eval_guided_dir, f"{idx:03d}.png"))
    for idx, img in enumerate(regular_photos):
        img.save(os.path.join(eval_unguided_dir, f"{idx:03d}.png"))
    print(f"Saved {len(dps_photos)} guided + {len(regular_photos)} unguided eval photos", flush=True)

    # ── 11. wandb final ───────────────────────────────────────────────────────
    wandb.log({
        "final_mmd_unguided": regular_mmd,
        "final_mmd_guided": dps_mmd,
        "mmd_delta": mmd_delta,
        "mmd_relative_improvement": mmd_relative_improvement,
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "total_params": total_params,
        "steps_completed": len(step_data),
        "gender_L_unguided": gender_unguided["gender_L"],
        "gender_L_guided": gender_guided["gender_L"],
        "gender_n_male_unguided": gender_unguided["n_male"],
        "gender_n_female_unguided": gender_unguided["n_female"],
        "gender_n_male_guided": gender_guided["n_male"],
        "gender_n_female_guided": gender_guided["n_female"],
        "gender_conf_unguided": gender_unguided["gender_conf_mean"],
        "gender_conf_guided": gender_guided["gender_conf_mean"],
    })
    wandb.summary["mmd_relative_improvement"] = mmd_relative_improvement
    wandb.summary["mmd_unguided"] = regular_mmd
    wandb.summary["mmd_guided"] = dps_mmd
    wandb.summary["gender_L_guided"] = gender_guided["gender_L"]
    wandb.summary["gender_L_unguided"] = gender_unguided["gender_L"]
    wandb.finish()

    # ── 12. Standardized output block ─────────────────────────────────────────
    print(f"\n---", flush=True)
    print(f"mmd_relative_improvement: {mmd_relative_improvement:.6f}", flush=True)
    print(f"mmd_guided: {dps_mmd:.6f}", flush=True)
    print(f"mmd_unguided: {regular_mmd:.6f}", flush=True)
    print(f"gender_L_guided: {gender_guided['gender_L']}", flush=True)
    print(f"gender_L_unguided: {gender_unguided['gender_L']}", flush=True)
    print(f"gender_guided: {gender_guided['n_male']}M/{gender_guided['n_female']}F", flush=True)
    print(f"gender_unguided: {gender_unguided['n_male']}M/{gender_unguided['n_female']}F", flush=True)
    print(f"training_seconds: {training_seconds:.1f}", flush=True)
    print(f"peak_vram_mb: {peak_vram_mb:.0f}", flush=True)
    print(f"total_params: {total_params}", flush=True)
    print(f"---", flush=True)


if __name__ == "__main__":
    main()
