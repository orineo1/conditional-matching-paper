import torch
import numpy as np
import gc
from metrics import compute_mmd


def generate_and_store(pipe, prompt, sobel_cond_pil, num_samples, batch_size=2):
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


def generate_and_store_cs(pipe, prompt, cond_pil, num_samples, batch_size=2, cn_scale=0.5):
    """generate_and_store but with configurable controlnet_conditioning_scale."""
    original_vae_dtype = pipe.vae.dtype
    pipe.vae.to(dtype=torch.float16)
    all_images, all_lats = [], []

    def latents_callback(p, step_index, timestep, cb_kwargs):
        if step_index == p.num_timesteps - 1:
            p._current_latents = cb_kwargs["latents"].detach().cpu().numpy()
        return cb_kwargs

    for i in range(0, num_samples, batch_size):
        curr   = min(batch_size, num_samples - i)
        result = pipe(
            prompt=[prompt] * curr,
            image=[cond_pil] * curr,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=cn_scale,
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
    return (latents_in - (1 - alpha)**0.5 * noise_pred) / alpha**0.5


def compute_pred_x0_direct(scheduler, noise_pred, t, latents_in):
    """Compute pred_x0 without calling scheduler.step() (avoids step_index side effects).

    Uses the diffusion formula directly:
        x0 = (x_t - sqrt(1-alpha) * noise_pred) / sqrt(alpha)
    """
    if hasattr(scheduler, 'alphas_cumprod'):
        alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    else:
        # Euler-type schedulers: compute alpha from sigma
        sigma = scheduler.sigmas[scheduler.step_index]
        alpha = (1 / (sigma**2 + 1)).to(latents_in.device)
    return (latents_in - (1 - alpha)**0.5 * noise_pred) / alpha**0.5


def denoise_step(scheduler, noise_pred, t, latents_in, correction=None):
    x_t_minus_1 = scheduler.step(noise_pred, t, latents_in, return_dict=True).prev_sample
    if correction is not None:
        x_t_minus_1 = x_t_minus_1 + correction
    return x_t_minus_1.detach()


def run_dps_step(latents, latents_step, noise_pred, pixel_x0_norm,
                 sprinter, all_latents, num_variations, variation_batch_size, base_zeta_prime):
    variation_latents_list = []
    print(f"  Generating {num_variations} variations...")

    for start_idx in range(0, num_variations, variation_batch_size):
        end_idx = min(start_idx + variation_batch_size, num_variations)
        bs = end_idx - start_idx
        ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)

        def sprinter_forward(ctrl):
            return sprinter(
                prompt=["a superrealistic professional photograph of"] * ctrl.shape[0],
                image=ctrl, num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8, output_type="latent", return_dict=True,
            ).images

        with torch.cuda.amp.autocast():
            vl = torch.utils.checkpoint.checkpoint(sprinter_forward, ctrl_batch, use_reentrant=False)
        variation_latents_list.append(vl)

    variation_latents = torch.cat(variation_latents_list, dim=0)
    torch.cuda.empty_cache()
    mmd_squared = compute_mmd(variation_latents, torch.tensor(all_latents, device=variation_latents.device))
    mmd_loss = torch.sqrt(mmd_squared.abs() + 1e-8)
    loss_norm = mmd_loss.detach()
    zeta_i = base_zeta_prime / loss_norm

    grad = torch.autograd.grad(mmd_loss, latents_step, retain_graph=False, create_graph=False)[0]
    vl_flat = variation_latents.detach().cpu().numpy().reshape(variation_latents.shape[0], -1)
    del variation_latents_list, variation_latents

    return grad, mmd_loss, zeta_i, loss_norm, vl_flat


def run_dps_step_clip(latents, latents_step, noise_pred, pixel_x0_norm,
                      sprinter, all_clip_embeddings, num_variations,
                      variation_batch_size, base_zeta_prime,
                      clip_model, clip_processor, vae, vae_scaling_factor):
    """
    DPS step using CLIP embeddings + MMD guidance.
    current pred_x0 is encoded through CLIP (grad flows),
    variations are encoded detached and concatenated with the pre-computed
    target embeddings to form the reference distribution.
    """
    from clip_utils import encode_images_clip

    # ── A. Encode current pred_x0 through CLIP — carries the gradient ─────────
    current_clip_emb = encode_images_clip(pixel_x0_norm, clip_model, clip_processor)
    # shape: [1, 768]

    # ── B. Generate variations — fully detached reference distribution ─────────
    variation_clip_list = []
    with torch.no_grad():
        for start_idx in range(0, num_variations, variation_batch_size):
            end_idx    = min(start_idx + variation_batch_size, num_variations)
            bs         = end_idx - start_idx
            ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1).detach()

            var_latents = sprinter(
                prompt=["a superrealistic professional photograph of"] * bs,
                image=ctrl_batch,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="latent", return_dict=True,
            ).images

            var_pixels = vae.decode(
                (var_latents.float() / vae_scaling_factor).to(vae.dtype)
            ).sample
            var_pixels = torch.clamp((var_pixels.float() + 1.0) / 2.0, 0.0, 1.0)

            var_clip = encode_images_clip(var_pixels, clip_model, clip_processor)
            variation_clip_list.append(var_clip)

    variation_clip_embs = torch.cat(variation_clip_list, dim=0).detach()  # [num_variations, 768]
    torch.cuda.empty_cache()

    # ── C. Build full reference set: variations + pre-computed target ──────────
    full_target = torch.cat([variation_clip_embs, all_clip_embeddings], dim=0).detach()

    # ── D. MMD + gradient ─────────────────────────────────────────────────────
    mmd_squared = compute_mmd(current_clip_emb, full_target)
    # Use abs() so sqrt is safe for slightly negative unbiased estimates
    mmd_loss    = torch.sqrt(mmd_squared.abs() + 1e-8)
    loss_norm   = mmd_loss.detach()
    zeta_i      = base_zeta_prime / loss_norm

    # ── E. Gradient ───────────────────────────────────────────────────────────
    grad = torch.autograd.grad(
        mmd_loss, latents_step, retain_graph=False, create_graph=False
    )[0]

    vl_clip_flat = variation_clip_embs.cpu().numpy()
    del variation_clip_list, variation_clip_embs

    return grad, mmd_loss, zeta_i, loss_norm, vl_clip_flat