"""
generation.py — Noise prediction, pred_x0 computation, scheduler steps,
and DPS gradient steps for the MLGD-F pipeline.

Functions:
    generate_and_store_cs    Sprinter batch generation with latent capture.
    predict_noise_cfg        CFG noise prediction from the Architect U-Net.
    compute_pred_x0_direct   Clean-image estimate without scheduler side-effects.
    denoise_step             One DDIM step, optionally with a DPS correction.
    run_dps_step             Latent-space MMD guidance (original formulation).
    run_dps_step_clip        CLIP-space MMD/SWD guidance (MLGD-F formulation).
"""

import gc

import numpy as np
import torch

from metrics import compute_mmd, compute_witness_scores


def generate_and_store(pipe, prompt, sobel_cond_pil, num_samples, batch_size=2):
    """Generate images from the Sprinter and capture final latents."""
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
            prompt=[prompt] * curr,
            image=[sobel_cond_pil] * curr,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=1.0,
            callback_on_step_end=latents_callback,
        )
        all_images.extend(result.images)
        all_lats.append(pipe._current_latents.reshape(curr, -1))
        print(f"  Progress: {len(all_images)}/{num_samples}", end="\r")

    print()
    pipe.vae.to(dtype=original_vae_dtype)
    return all_images, np.vstack(all_lats)


def generate_and_store_cs(
    pipe, prompt, cond_pil, num_samples, batch_size=2, cn_scale=0.5
):
    """generate_and_store with configurable controlnet_conditioning_scale."""
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
    """
    Classifier-free guidance noise prediction.

    Concatenates unconditional + conditional latents, runs one U-Net forward
    pass, and blends the two outputs with CFG scale `gs`.
    """
    lmi = scheduler.scale_model_input(torch.cat([latents_in] * 2), t)
    np_out = unet(
        lmi, t,
        encoder_hidden_states=encoder_states,
        added_cond_kwargs=added_cond,
        return_dict=False,
    )[0]
    np_u, np_t = np_out.chunk(2)
    return np_u + gs * (np_t - np_u)


def compute_pred_x0(scheduler, noise_pred, t, latents_in):
    """
    Compute predicted clean image via scheduler.step().

    NOTE: advances the scheduler's internal step_index as a side-effect.
    Prefer compute_pred_x0_direct when calling multiple times per iteration.
    """
    so = scheduler.step(noise_pred, t, latents_in, return_dict=True)
    if hasattr(so, "pred_original_sample") and so.pred_original_sample is not None:
        return so.pred_original_sample
    alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    return (latents_in - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5


def compute_pred_x0_direct(scheduler, noise_pred, t, latents_in):
    """
    Compute predicted clean image using the diffusion formula directly.

    Does NOT call scheduler.step() — the internal step_index is not advanced.
    Safe to call multiple times per iteration.

        x0 = (x_t - sqrt(1 - alpha) * noise_pred) / sqrt(alpha)
    """
    if hasattr(scheduler, "alphas_cumprod"):
        alpha = scheduler.alphas_cumprod[t.long().cpu()].to(latents_in.device)
    else:
        # Euler-type schedulers: derive alpha from sigma
        sigma = scheduler.sigmas[scheduler.step_index]
        alpha = (1 / (sigma ** 2 + 1)).to(latents_in.device)
    return (latents_in - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5


def denoise_step(scheduler, noise_pred, t, latents_in, correction=None):
    """
    Run one DDIM denoising step, optionally adding a DPS correction.

    The correction is added after the scheduler step (standard DPS formulation).
    Returns a detached tensor.
    """
    x_t_minus_1 = scheduler.step(
        noise_pred, t, latents_in, return_dict=True
    ).prev_sample
    if correction is not None:
        x_t_minus_1 = x_t_minus_1 + correction
    return x_t_minus_1.detach()


def run_dps_step(
    latents,
    latents_step,
    noise_pred,
    pixel_x0_norm,
    sprinter,
    all_latents,
    num_variations,
    variation_batch_size,
    base_zeta_prime,
):
    """
    Latent-space MMD DPS step (original formulation).

    Gradient flows through num_variations Sprinter passes in latent space.
    Targets (all_latents) are fully detached.

    Returns:
        (grad, mmd_loss, zeta_i, loss_norm, vl_flat)
    """
    variation_latents_list = []
    print(f"  Generating {num_variations} variations...")

    for start_idx in range(0, num_variations, variation_batch_size):
        end_idx = min(start_idx + variation_batch_size, num_variations)
        bs = end_idx - start_idx
        ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)

        def sprinter_forward(ctrl):
            return sprinter(
                prompt=["a superrealistic professional photograph of"] * ctrl.shape[0],
                image=ctrl,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="latent",
                return_dict=True,
            ).images

        with torch.cuda.amp.autocast():
            vl = torch.utils.checkpoint.checkpoint(
                sprinter_forward, ctrl_batch, use_reentrant=False
            )
        variation_latents_list.append(vl)

    variation_latents = torch.cat(variation_latents_list, dim=0)
    torch.cuda.empty_cache()

    mmd_squared = compute_mmd(
        variation_latents,
        torch.tensor(all_latents, device=variation_latents.device),
    )
    mmd_loss = torch.sqrt(mmd_squared.abs() + 1e-8)
    loss_norm = mmd_loss.detach()
    zeta_i = base_zeta_prime / loss_norm

    grad = torch.autograd.grad(
        mmd_loss, latents_step, retain_graph=False, create_graph=False
    )[0]
    vl_flat = (
        variation_latents.detach().cpu().numpy()
        .reshape(variation_latents.shape[0], -1)
    )
    del variation_latents_list, variation_latents

    return grad, mmd_loss, zeta_i, loss_norm, vl_flat


def _witness_select_indices(scores, k, witness_floor, witness_temperature, replacement, generator):
    """
    Defensive-mixture importance selection from witness scores:
    p_i = floor/n + (1-floor)*|w_i|^(1/T)/sum(|w|^(1/T)), then draw k indices from p
    (with/without replacement).

    Returns (grad_counts, probs): grad_counts maps index -> draw count (dict, only
    entries with count > 0); probs is the [n] float tensor of selection
    probabilities actually used (for diagnostics/logging).
    """
    n = scores.shape[0]
    probs = scores.abs().double()
    if witness_temperature != 1.0:
        probs = probs.clamp_min(1e-12) ** (1.0 / witness_temperature)
    probs = (1.0 - witness_floor) * probs / probs.sum().clamp_min(1e-12) + witness_floor / n
    probs = probs / probs.sum()
    # multinomial + a CPU generator require CPU probs.
    idx_t = torch.multinomial(probs.cpu(), k, replacement=replacement, generator=generator)
    # Without replacement: distinct indices, count 1 each. With replacement: an
    # index drawn c times gets count c and is included c times downstream.
    unique_idx, counts = torch.unique(idx_t, return_counts=True)
    grad_counts = dict(zip(unique_idx.tolist(), counts.tolist()))
    return grad_counts, probs


def _log_witness_selection(scores, probs, grad_counts, replacement):
    """Print scores/probs/selected indices for one witness-selection step, so a
    run's log shows whether selection is collapsing onto the same few indices."""
    scores_np = scores.cpu().numpy()
    probs_np = probs.cpu().numpy()
    selected_sorted = sorted(grad_counts)
    dup_counts = {i: c for i, c in grad_counts.items() if c > 1}
    print(f"      [witness] scores={np.round(scores_np, 4).tolist()}", flush=True)
    print(f"      [witness] probs ={np.round(probs_np, 4).tolist()}", flush=True)
    print(
        f"      [witness] selected_idx={selected_sorted}  "
        f"selected_scores={np.round(scores_np[selected_sorted], 4).tolist()}  "
        f"score_range=[{scores_np.min():.4f}, {scores_np.max():.4f}]  "
        f"replacement={replacement}  duplicate_draws={dup_counts or 'none'}",
        flush=True,
    )


def _expand_rows_by_count(rows, counts):
    """Reassemble `rows` per a {index: count} dict: a row with count > 0 is
    repeated that many times (autograd sums its gradient across the reuses); a
    row with no entry is individually detached once."""
    out = []
    for i, row in enumerate(rows):
        c = counts.get(i, 0)
        out.extend([row] * c if c > 0 else [row.detach()])
    return out


def run_dps_step_clip(
    latents,
    latents_step,
    noise_pred,
    pixel_x0_norm,
    sprinter,
    all_clip_embeddings,
    num_variations,
    variation_batch_size,
    base_zeta_prime,
    clip_model,
    clip_processor,
    vae,
    vae_scaling_factor,
    variation_prompt,
    loss_fn=compute_mmd,
    loss_scale=1.0,
    backsel_k=None,
    backsel_rule="uniform",
    backsel_generator=None,
    witness_floor=0.3,
    witness_temperature=1.0,
    witness_bandwidth_scale=1.0,
    witness_kernel_alpha=1.0,
    witness_replacement=False,
):
    """
    CLIP-space MMD/SWD DPS step — core of the MLGD-F algorithm.

    Gradient flows through the freshly-generated Sprinter passes:
        latents_step -> UNet -> pred_x0 -> VAE decode -> pixels
        -> CLIP encode -> loss (MMD or SWD) vs target embeddings

    Targets (all_clip_embeddings) are fully detached.
    Uses torch.utils.checkpoint for memory efficiency (batch_size=1).

    Args:
        loss_fn:              callable with signature loss_fn(generated, targets) -> scalar.
        loss_scale:           multiply loss before grad to amplify weak gradients.
        backsel_k:             of the freshly-generated variations, how many to backprop
                               through (None = all). The rest are generated under
                               torch.no_grad() -- still counted in the loss, no gradient.
                               All variations are i.i.d., so only the count matters.
        backsel_rule:          'uniform' (default) -- first n_grad draws are differentiable,
                               no extra cost. 'witness' -- generate all n_new via checkpoint
                               (same forward cost as no_grad), score with the MMD witness
                               function, detach all but the n_grad highest-|score| rows
                               individually before concatenating. Detaching prunes that row's
                               checkpoint node, so only the kept rows get recomputed+backprop'd
                               at backward time -- and since it's the same node (not a fresh
                               redraw), checkpoint's RNG preservation reproduces the exact
                               sample that was scored, for a lower-variance gradient at no
                               extra Sprinter cost.
        backsel_generator:     optional torch.Generator for reproducible witness sampling.
        witness_floor:         defensive mixture p_i = floor/n + (1-floor)*|w_i|^(1/T)/sum(|w|^(1/T))
                               instead of pure |w|-proportional -- bounds how much weight any
                               one outlier can dominate. Default 0.3 (recommended 0.3-0.5);
                               0 = pure importance sampling, 1 = uniform.
        witness_temperature:   T above. Default 1.0 (plain |w|); T>1 flattens the selection
                               distribution toward uniform, T<1 sharpens it toward the
                               top-|score| rows.
        witness_bandwidth_scale, witness_kernel_alpha:
                               RBF kernel params for the witness score (independent of
                               loss_fn's own kernel; only used when backsel_rule='witness').
        witness_replacement:   False (default) -- draw n_grad distinct indices, avoiding
                               repeat picks. True -- draw with replacement (standard
                               importance-resampling); a row drawn c times is included c
                               times (same checkpoint node, recomputed once, autograd sums
                               its gradient across uses), so it can carry >1x weight but is
                               more prone to concentrating on a few samples.

    Returns:
        (grad, loss_scaled, zeta_i, loss_norm, vl_clip_flat)
    """
    from clip_utils import encode_images_clip

    device = pixel_x0_norm.device
    clip_model.to(device)

    n_new = num_variations

    n_grad = n_new if backsel_k is None else min(backsel_k, n_new)

    def sprinter_vae_clip_forward(ctrl):
        var_latents = sprinter(
            prompt=[variation_prompt] * ctrl.shape[0],
            image=ctrl,
            num_inference_steps=2,
            guidance_scale=0.0,
            controlnet_conditioning_scale=0.8,
            output_type="latent",
            return_dict=True,
        ).images
        var_pixels = vae.decode(
            (var_latents.float() / vae_scaling_factor).to(vae.dtype)
        ).sample
        var_pixels = torch.clamp((var_pixels.float() + 1.0) / 2.0, 0.0, 1.0)
        # Disable autocast around CLIP ViT to prevent fp16/fp32 mismatches
        with torch.amp.autocast("cuda", enabled=False):
            return encode_images_clip(var_pixels.float(), clip_model, clip_processor)

    def checkpointed_forward(count):
        """
        Differentiable forward for `count` fresh slots, checkpointed, one row-tensor
        per slot. Cost of *this* call is ~identical to a no_grad forward — checkpoint
        doesn't store activations either way; the recompute-and-backward cost of a
        row only actually happens later, at backward time, and only for rows that
        are still attached to the loss graph then (see the 'witness' branch below).
        """
        rows = []
        for start in range(0, count, variation_batch_size):
            bs = min(variation_batch_size, count - start)
            ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)
            chunk = torch.utils.checkpoint.checkpoint(
                sprinter_vae_clip_forward, ctrl_batch, use_reentrant=False
            )
            rows.extend(chunk[j:j + 1] for j in range(bs))
        return rows

    variation_clip_list = []

    if backsel_rule == "witness" and 0 < n_grad < n_new:
        # Generate all candidates once via checkpoint (same cost as no_grad), score
        # them for real, then detach all but the top-n_grad |score| rows individually
        # before concatenating -- detaching prunes that row's checkpoint node, so
        # autograd.grad() below only recomputes+backprops the kept rows, and because
        # it's the same node (not a fresh redraw), it reproduces the exact sample
        # that was scored.
        all_rows = checkpointed_forward(n_new)
        all_embs = torch.cat(all_rows, dim=0)

        with torch.no_grad():
            scores, _ = compute_witness_scores(
                all_embs.detach(), all_clip_embeddings,
                bandwidth_scale=witness_bandwidth_scale, kernel_alpha=witness_kernel_alpha,
            )
            grad_counts, probs = _witness_select_indices(
                scores, n_grad, witness_floor, witness_temperature,
                witness_replacement, backsel_generator,
            )
            _log_witness_selection(scores, probs, grad_counts, witness_replacement)

        variation_clip_list.extend(_expand_rows_by_count(all_rows, grad_counts))
        # Number of distinct candidates left fully undifferentiated -- with
        # replacement this can exceed n_new - n_grad, since duplicate draws leave
        # more candidates untouched than a without-replacement draw of the same size.
        n_nograd = n_new - len(grad_counts)
    else:
        # Differentiable slots: checkpointed forward, gradient flows to latents_step.
        variation_clip_list.extend(checkpointed_forward(n_grad))

        # Selected-out slots: plain no_grad forward — cheaper (no checkpoint
        # recompute, no autograd graph), still included in the loss for full-N stats.
        n_nograd = n_new - n_grad
        if n_nograd > 0:
            with torch.no_grad():
                for start_idx in range(0, n_nograd, variation_batch_size):
                    end_idx = min(start_idx + variation_batch_size, n_nograd)
                    bs = end_idx - start_idx
                    ctrl_batch = pixel_x0_norm[0].unsqueeze(0).repeat(bs, 1, 1, 1)
                    variation_clip_list.append(sprinter_vae_clip_forward(ctrl_batch))

    variation_clip_embs = torch.cat(variation_clip_list, dim=0)
    torch.cuda.empty_cache()

    print(
        f"      var_clip_embs: shape={variation_clip_embs.shape} "
        f"(n_new={n_new}, n_grad={n_grad}, n_nograd={n_nograd}) "
        f"nan={torch.isnan(variation_clip_embs).sum().item()} "
        f"range=[{variation_clip_embs.min().item():.4f}, "
        f"{variation_clip_embs.max().item():.4f}] "
        f"grad_fn={variation_clip_embs.grad_fn}",
        flush=True,
    )

    loss_value = loss_fn(variation_clip_embs, all_clip_embeddings.detach())
    loss_scaled = loss_value * loss_scale
    loss_norm = loss_scaled.detach()
    zeta_i = base_zeta_prime / loss_norm

    grad = torch.autograd.grad(
        loss_scaled, latents_step, retain_graph=False, create_graph=False
    )[0]

    vl_clip_flat = variation_clip_embs.detach().cpu().numpy()
    del variation_clip_list, variation_clip_embs

    return grad, loss_scaled, zeta_i, loss_norm, vl_clip_flat
