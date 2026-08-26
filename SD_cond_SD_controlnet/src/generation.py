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
    prev_variation_clip=None,
    reuse_frac=0.0,
    backsel_k=None,
    backsel_rule="uniform",
    backsel_generator=None,
    witness_floor=0.3,
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
        prev_variation_clip:  detached CLIP embeddings freshly generated at the *previous*
                               step (one step old), or None on the first step / when unused.
        reuse_frac:           fraction of num_variations reused from prev_variation_clip
                               instead of generated fresh here (0.0 = original behavior).
        backsel_k:            of the freshly-generated variations, how many to actually
                               backprop through (None = all, i.e. original behavior). The
                               rest are generated under torch.no_grad() — cheaper, and still
                               counted in the loss for a full-N statistic, but contribute no
                               gradient. Since every variation is an i.i.d. Sprinter draw
                               conditioned on the same pixel_x0_norm, there is no fixed pool
                               to pick "which" k from — only the count matters, so this is
                               equivalent to a uniform-random subsample of an implicit N-pool.
                               Lets `num_variations` grow for loss fidelity without growing
                               backward-pass cost.
        backsel_rule:          'uniform' (default) — first n_grad fresh draws are
                               differentiable, no extra cost.
                               'witness' — generate all n_new candidates through
                               checkpoint (their forward cost is ~identical to
                               no_grad — checkpoint doesn't store activations
                               either way), score them for real with the MMD
                               witness function (see metrics.compute_witness_scores),
                               then keep only the n_grad highest-|score| rows
                               attached and .detach() the rest individually before
                               concatenating. Detaching a row structurally removes
                               its checkpoint node from the graph, so it's never
                               recomputed or backpropped — total cost is the same
                               n_new + n_grad forward-equivalents as if you'd
                               written two separate loops, but because it's the
                               *same* checkpoint node that's kept (not a fresh
                               regeneration), checkpoint's RNG-state preservation
                               reproduces the exact sample that was scored when it
                               recomputes for backward, giving a lower-variance
                               gradient for the same k with no extra Sprinter calls
                               versus doing it "the naive way".
        backsel_generator:     optional torch.Generator for reproducible witness
                               sampling.
        witness_floor:         defensive mixture: p_i = witness_floor/n + (1 - witness_floor)
                               * |w_i| / sum(|w|), rather than sampling purely
                               proportional to |w|. Bounds the worst-case weight so
                               low-|score| samples still have a nonzero chance of
                               being selected, instead of one or two outliers
                               dominating every step. Combined with the
                               replacement=False draw below (never pick the same
                               index twice within one step's n_grad), this is the
                               standard fix for importance sampling collapsing onto
                               a handful of samples. Default 0.3 (recommended range
                               0.3-0.5); 0 = pure |w|-proportional, 1 = uniform.
        witness_bandwidth_scale, witness_kernel_alpha:
                               RBF kernel params for the witness score (independent of
                               loss_fn's own kernel settings; only used when
                               backsel_rule='witness').
        witness_replacement:   False (default) — draw n_grad *distinct* indices;
                               removes the "same outlier picked repeatedly" failure
                               mode entirely, at the cost of needing n_grad <= n_new
                               (already true here) and being unable to concentrate
                               more than 1x weight on any single sample. True —
                               draw n_grad indices with replacement (standard
                               importance-resampling); an index drawn c times is
                               included c times in the differentiable batch (the
                               *same* checkpoint node reused c times — autograd
                               correctly sums its gradient across all c uses, and it
                               is still only recomputed once), so a high-|witness|
                               sample can end up carrying more than 1x weight. This
                               is the textbook with-replacement importance-sampling
                               estimator, but is more prone to concentrating on a
                               handful of samples than the without-replacement
                               version — watch the "duplicate draws" line in the
                               printed diagnostics below.

    Returns:
        (grad, loss_scaled, zeta_i, loss_norm, vl_clip_flat, new_variation_clip)
        new_variation_clip is this step's freshly-generated embeddings (detached),
        to pass back in as prev_variation_clip on the next call.
    """
    from clip_utils import encode_images_clip

    device = pixel_x0_norm.device
    clip_model.to(device)

    n_reuse = min(int(round(reuse_frac * num_variations)), prev_variation_clip.shape[0]) \
        if prev_variation_clip is not None else 0
    n_new = num_variations - n_reuse

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
        # Generate every candidate ONCE via checkpoint (same forward cost as
        # no_grad). Score them for real — their actual realized values, not a
        # proxy — pick which n_grad to keep differentiable, and detach the rest
        # *individually*, before concatenating. Detaching a specific row's tensor
        # structurally removes that row's checkpoint node from the backward graph,
        # so autograd.grad() below recomputes+backprops only through the kept
        # n_grad rows (checkpoint's RNG-state preservation reproduces them
        # bit-identically — this differentiates through the literal sample that
        # was scored, not a fresh independent redraw at the same slot).
        all_rows = checkpointed_forward(n_new)
        all_embs = torch.cat(all_rows, dim=0)

        with torch.no_grad():
            scores, _ = compute_witness_scores(
                all_embs.detach(), all_clip_embeddings,
                bandwidth_scale=witness_bandwidth_scale, kernel_alpha=witness_kernel_alpha,
            )
            # Defensive mixture: blend the |score|-proportional distribution with
            # witness_floor mass of uniform (recommended ~0.3-0.5). Pure |w|-
            # proportional sampling can put almost all the mass on one or two
            # outliers step after step, starving everything else; the uniform floor
            # bounds how skewed p can get.
            probs = scores.abs().double()
            probs = (1.0 - witness_floor) * probs / probs.sum().clamp_min(1e-12) + witness_floor / n_new
            probs = probs / probs.sum()
            # multinomial + a CPU generator require CPU probs.
            grad_idx_t = torch.multinomial(
                probs.cpu(), n_grad, replacement=witness_replacement, generator=backsel_generator
            )
            # Without replacement: every draw is a distinct index, count 1 each --
            # this is what removes the "same outlier picked repeatedly" failure
            # mode entirely (the standard fix when k isn't tiny relative to n).
            # With replacement: draws can repeat: an index drawn c times gets
            # count c, and is included c times in the differentiable batch below
            # (the textbook with-replacement importance-resampling estimator).
            unique_idx, counts = torch.unique(grad_idx_t, return_counts=True)
            grad_counts = dict(zip(unique_idx.tolist(), counts.tolist()))

            # Diagnostic: print scores/probs/selection every step so you can check
            # across a run's log whether selection is collapsing onto the same few
            # indices.
            scores_np = scores.cpu().numpy()
            probs_np = probs.cpu().numpy()
            selected_sorted = sorted(grad_counts)
            dup_counts = {i: c for i, c in grad_counts.items() if c > 1}
            print(
                f"      [witness] scores={np.round(scores_np, 4).tolist()}",
                flush=True,
            )
            print(
                f"      [witness] probs ={np.round(probs_np, 4).tolist()}",
                flush=True,
            )
            print(
                f"      [witness] selected_idx={selected_sorted}  "
                f"selected_scores={np.round(scores_np[selected_sorted], 4).tolist()}  "
                f"score_range=[{scores_np.min():.4f}, {scores_np.max():.4f}]  "
                f"replacement={witness_replacement}  duplicate_draws={dup_counts or 'none'}",
                flush=True,
            )

        rows_out = []
        for i, row in enumerate(all_rows):
            c = grad_counts.get(i, 0)
            rows_out.extend([row] * c if c > 0 else [row.detach()])
        variation_clip_list.extend(rows_out)
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

    new_variation_clip = torch.cat(variation_clip_list, dim=0) if variation_clip_list else None
    # Snapshot for the next step's reuse buffer immediately, before variation_clip_embs
    # is handed to loss_fn/autograd — independent of anything that happens downstream.
    new_variation_clip_detached = new_variation_clip.detach().clone() if new_variation_clip is not None else None
    torch.cuda.empty_cache()

    if n_reuse > 0:
        reused = prev_variation_clip[:n_reuse]
        variation_clip_embs = torch.cat([reused, new_variation_clip], dim=0) \
            if new_variation_clip is not None else reused
    else:
        variation_clip_embs = new_variation_clip

    print(
        f"      var_clip_embs: shape={variation_clip_embs.shape} "
        f"(n_new={n_new}, n_reuse={n_reuse}, n_grad={n_grad}, n_nograd={n_nograd}) "
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

    return grad, loss_scaled, zeta_i, loss_norm, vl_clip_flat, new_variation_clip_detached
