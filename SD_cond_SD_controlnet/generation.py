import torch
import numpy as np
import gc
from metrics import compute_clip_mmd, encode_images_clip


def generate_and_store(pipe, prompt, sobel_cond_pil, num_samples, batch_size=2):
    """Unchanged — still used for generating target images."""
    original_vae_dtype = pipe.vae.dtype
    pipe.vae.to(dtype=torch.float16)
    all_images, all_lats = [], []

    def latents_callback(p, step_index, timestep, cb_kwargs):
        if step_index == p.num_timesteps - 1:
            p._current_latents = cb_kwargs["latents"].detach().cpu().numpy()
        return cb_kwargs

    for i in range(0, num_samples, batch_size):
        curr = min(batch_size, num_samples - i)
        result = pipe(
            prompt=[prompt] * curr, image=[sobel_cond_pil] * curr,
            num_inference_steps=2, guidance_scale=0.0,
            controlnet_conditioning_scale=1.0,
            callback_on_step_end=latents_callback,
        )
        all_images.extend(result.images)
        all_lats.append(pipe._current_latents.reshape(curr, -1))
        print(f"  Progress: {len(all_images)}/{num_samples}", end="\r")

    print()
    pipe.vae.to(dtype=original_vae_dtype)
    return all_images, np.vstack(all_lats)


def predict_noise_cfg(unet, scheduler, latents_in, t, encoder_states, added_cond, gs):
    lmi = scheduler.scale_model_input(torch.cat([latents_in] * 2), t)
    np_out = unet(lmi, t, encoder_hidden_states=encoder_states,
                  added_cond_kwargs=added_cond, return_dict=False)[0]
    np_u, np_t = np_out.chunk(2)
    return np_u + gs * (np_t - np_u)


def compute_pred_x0(scheduler, noise_pred, t, latents_in):
    so = scheduler.step(noise_pred, t, latents_in, return_dict=True)
    if hasattr(so, 'pred_original_sample') and so.pred_original_sample is not None:
        return so.pred_original_sample
    alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    return (latents_in - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5


def denoise_step(scheduler, noise_pred, t, latents_in, correction=None):
    x_t_minus_1 = scheduler.step(noise_pred, t, latents_in, return_dict=True).prev_sample
    if correction is not None:
        x_t_minus_1 = x_t_minus_1 + correction
    return x_t_minus_1.detach()


def run_dps_step(latents, latents_step, noise_pred, pixel_x0_norm,
                 sprinter, clip_model, target_clip_embs,
                 num_variations, variation_batch_size, base_zeta_prime):
    """
    DPS guidance step using MMD in CLIP embedding space (Option B).

    Gradient path:
        latents_step → pixel_x0_norm → sprinter (checkpointed)
        → variation_pixels (float32) → CLIP encoder → CLIP embs
        → CLIP-MMD loss → grad w.r.t latents_step

    target_clip_embs: precomputed + detached CLIP embs of target images (M, D)
    """
    variation_pixels_list = []
    device = pixel_x0_norm.device
    print(f"  Generating {num_variations} variations...")

    for start_idx in range(0, num_variations, variation_batch_size):
        bs = min(variation_batch_size, num_variations - start_idx)
        ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)

        def sprinter_forward(ctrl):
            # output_type="pt" → returns pixel tensor (B, 3, H, W) in [0,1]
            # grad flows through this call back to ctrl → pixel_x0_norm → latents_step
            return sprinter(
                prompt=["a superrealistic professional photograph of"] * ctrl.shape[0],
                image=ctrl,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pt",  # ← pixel tensors, not latents
                return_dict=True,
            ).images  # (B, 3, H, W) float in [0, 1]

        with torch.cuda.amp.autocast():
            pix = torch.utils.checkpoint.checkpoint(
                sprinter_forward, ctrl_batch, use_reentrant=False
            )
        # Cast to float32 before CLIP — CLIP encoder expects float32
        variation_pixels_list.append(pix.float())

    variation_pixels = torch.cat(variation_pixels_list, dim=0)  # (N, 3, H, W)
    torch.cuda.empty_cache()

    # ── CLIP encode variations — grad flows through here ─────────────────────
    # DO NOT wrap in no_grad — this is Option B, graph must stay live
    gen_clip_embs = encode_images_clip(variation_pixels, clip_model)  # (N, D)

    # ── MMD in CLIP space ─────────────────────────────────────────────────────
    # target_clip_embs is detached (precomputed) — grad only through gen_clip_embs
    mmd_squared = compute_clip_mmd(gen_clip_embs, target_clip_embs)
    mmd_loss = torch.sqrt(mmd_squared + 1e-8)
    loss_norm = mmd_loss.detach()
    zeta_i = base_zeta_prime / loss_norm

    # ── Gradient w.r.t latents_step ──────────────────────────────────────────
    grad = torch.autograd.grad(
        mmd_loss, latents_step,
        retain_graph=False, create_graph=False
    )[0]

    # Store CLIP embs (not pixels) for PCA visualization
    vl_flat = gen_clip_embs.detach().cpu().numpy()  # (N, 512)

    del variation_pixels_list, variation_pixels, gen_clip_embs
    return grad, mmd_loss, zeta_i, loss_norm, vl_flat