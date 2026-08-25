"""
metrics.py — Loss functions and distribution evaluation for MLGD-F.

Functions:
    compute_mmd               Unbiased MMD with generalised RBF kernel.
    compute_swd                Sliced Wasserstein Distance (adaptive projections).
    compute_witness_scores    Per-sample MMD witness function (for importance backsel).
    evaluate_distribution_mmd Decode latent -> scribble -> photos -> CLIP -> MMD.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF


def rbf_kernel(a, b, bw, alpha):
    """Generalised RBF kernel: exp(-(||a-b||^2 / 2bw^2)^alpha)."""
    a_sq = (a ** 2).sum(dim=1, keepdim=True)
    b_sq = (b ** 2).sum(dim=1, keepdim=True)
    dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
    return torch.exp(-(dist_sq / (2 * bw ** 2)) ** alpha)


def estimate_bandwidth(x, y, bandwidth_scale=1.0):
    """Median-heuristic RBF bandwidth between x and y (detached, no_grad)."""
    dev = x.device
    ss = min(1000, x.shape[0], y.shape[0])
    with torch.no_grad():
        x_sq = (x[:ss].detach() ** 2).sum(dim=1, keepdim=True)
        y_sq = (y[:ss].detach() ** 2).sum(dim=1, keepdim=True)
        dists = x_sq + y_sq.T - 2 * torch.mm(x[:ss].detach(), y[:ss].detach().T)
        dists = dists[dists > 0]
        bandwidth = (
            torch.sqrt(torch.median(dists) / 2)
            if len(dists) > 0
            else torch.tensor(1.0, device=dev)
        )
    return bandwidth.detach() * bandwidth_scale


def compute_witness_scores(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    """
    Per-sample MMD witness function w(x_l) = mean_i k(x_l, x_i) - mean_j k(x_l, y_j).

    Positive and large where x has "too much mass" relative to y — i.e. the samples
    whose removal/change would most reduce the MMD. |w| is the natural importance
    score for subsampling which x's to backprop through (see backsel_rule="witness"
    in generation.run_dps_step_clip): cheap (no_grad, row-means of kernel matrices
    already used by compute_mmd), and doesn't require gradients through the network
    that produced x.

    Args:
        x:         [n, d] candidate samples (detached; no_grad here regardless).
        y:         [m, d] target samples (detached).
        bandwidth: kernel bandwidth; estimated via median heuristic if None.

    Returns:
        (scores, bandwidth) — scores: [n] tensor of w(x_l); bandwidth: the (possibly
        estimated) bandwidth used, so callers can reuse it for the actual loss.
    """
    with torch.no_grad():
        x = x.float().detach()
        y = y.float().detach()
        if bandwidth is None:
            bandwidth = estimate_bandwidth(x, y, bandwidth_scale)
        K_xx = rbf_kernel(x, x, bandwidth, kernel_alpha)
        K_xy = rbf_kernel(x, y, bandwidth, kernel_alpha)
        scores = K_xx.mean(dim=1) - K_xy.mean(dim=1)
    return scores, bandwidth


def compute_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    """
    Unbiased Maximum Mean Discrepancy with a generalised RBF kernel.

    Args:
        x:               [n, d] generated embeddings (grad flows through).
        y:               [m, d] target embeddings (detached).
        bandwidth:       kernel bandwidth; estimated via median heuristic if None.
        bandwidth_scale: multiplicative scale applied after median estimation.
        kernel_alpha:    RBF exponent. 1 = standard Gaussian.
                         >1 = flatter centre, sharper falloff.

    Returns:
        Scalar MMD estimate (sqrt of unbiased MMD²).
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)

    n, m = x.shape[0], y.shape[0]

    if bandwidth is None:
        bandwidth = estimate_bandwidth(x.detach(), y, bandwidth_scale)

    K_xx = rbf_kernel(x, x, bandwidth, kernel_alpha)
    K_yy = rbf_kernel(y, y, bandwidth, kernel_alpha)
    K_xy = rbf_kernel(x, y, bandwidth, kernel_alpha)

    # Skip K_xx diagonal term when n=1 to avoid 0/0
    xx_term = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) if n > 1 else 0.0
    yy_term = (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) if m > 1 else 0.0
    xy_term = 2 * K_xy.sum() / (n * m)

    mmd_sq = xx_term - xy_term + yy_term
    # abs() before sqrt handles slightly-negative unbiased estimates
    return torch.sqrt(mmd_sq.abs() + 1e-8)


def compute_swd(
    x,
    y,
    n_projections=None,
    tol=1e-3,
    min_projections=10,
    step=10,
    max_projections=500,
):
    """
    Sliced Wasserstein Distance between two sets of embeddings.

    Args:
        x:             [n, d] generated embeddings (grad flows through).
        y:             [m, d] target embeddings (detached).
        n_projections: if given, uses exactly that many projections.
                       Otherwise grows adaptively until convergence.
        tol:           convergence threshold |SWD(n+step) - SWD(n)| < tol.
        min_projections: starting number for adaptive mode.
        step:          projections added per iteration in adaptive mode.
        max_projections: hard cap for adaptive mode.

    Returns:
        Scalar SWD estimate.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)

    d = x.shape[1]

    def _swd_fixed(n_proj):
        projections = torch.randn(n_proj, d, device=dev)
        projections = projections / projections.norm(dim=1, keepdim=True)

        x_proj = projections @ x.T  # [n_proj, n]
        y_proj = projections @ y.T  # [n_proj, m]

        x_sorted = x_proj.sort(dim=1).values
        y_sorted = y_proj.sort(dim=1).values

        if x_sorted.shape[1] != y_sorted.shape[1]:
            y_sorted = torch.nn.functional.interpolate(
                y_sorted.unsqueeze(0),
                size=x_sorted.shape[1],
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        return (x_sorted - y_sorted).abs().mean()

    if n_projections is not None:
        return _swd_fixed(n_projections)

    # Adaptive: grow until converged
    n = min_projections
    prev_swd = _swd_fixed(n)
    while n + step <= max_projections:
        n += step
        curr_swd = _swd_fixed(n)
        delta = (curr_swd - prev_swd).abs()
        print(
            f"      [SWD] n_proj={n}  swd={curr_swd.item():.6f}  "
            f"delta={delta.item():.6f}",
            flush=True,
        )
        if delta.item() < tol:
            print(f"      [SWD] Converged at n_projections={n}", flush=True)
            return curr_swd
        prev_swd = curr_swd

    print(
        f"      [SWD] Reached max_projections={max_projections} without convergence",
        flush=True,
    )
    return prev_swd


def evaluate_distribution_mmd(
    latent,
    architect_vae,
    architect_image_processor,
    sprinter,
    clip_model,
    clip_processor,
    all_clip_embeddings,
    eval_prompt,
    n_eval=10,
    device="cuda",
):
    """
    Full evaluation: latent -> scribble PIL -> sprinter photos -> CLIP -> MMD.

    Args:
        latent:                  Architect latent tensor [1, C, H, W].
        architect_vae:           VAE from the Architect pipeline.
        architect_image_processor: image processor from the Architect pipeline.
        sprinter:                Sprinter pipeline.
        clip_model:              Frozen CLIP model.
        clip_processor:          CLIP processor.
        all_clip_embeddings:     [N, 768] target CLIP embeddings (detached).
        eval_prompt:             Text prompt for Sprinter generation.
        n_eval:                  Number of Sprinter photos to generate.
        device:                  torch device string.

    Returns:
        (mmd_scalar, eval_photos_list, clip_embs)
    """
    from clip_utils import encode_images_clip
    from image_utils import latent_to_pil

    with torch.no_grad():
        scribble_pil = latent_to_pil(latent, architect_vae, architect_image_processor)

    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    eval_photos = []
    with torch.no_grad():
        for start in range(0, n_eval, 2):
            bs = min(2, n_eval - start)
            result = sprinter(
                prompt=[eval_prompt] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pil",
                return_dict=True,
            )
            eval_photos.extend(result.images)
    sprinter.vae.to(dtype=original_vae_dtype)

    tensors = [TF.to_tensor(img).unsqueeze(0) for img in eval_photos]
    photo_tensor = torch.cat(tensors, dim=0).to(device)
    clip_model.to(device)
    with torch.no_grad():
        clip_embs = encode_images_clip(photo_tensor, clip_model, clip_processor)
    clip_model.to("cpu")

    mmd = compute_mmd(clip_embs, all_clip_embeddings).item()

    return mmd, eval_photos, clip_embs
