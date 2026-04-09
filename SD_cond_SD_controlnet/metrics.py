import torch
import numpy as np
import torchvision.transforms.functional as TF


def compute_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    # kernel_alpha: exponent on the RBF argument. alpha=1 → standard RBF.
    # alpha>1 → flatter kernel center, sharper falloff → penalizes inter-mode points more.
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

    if x.dim() > 2: x = x.reshape(x.shape[0], -1)
    if y.dim() > 2: y = y.reshape(y.shape[0], -1)
    n, m = x.shape[0], y.shape[0]

    # OLD: def rbf_kernel(a, b, bw,):
    # NEW: def rbf_kernel(a, b, bw, alpha):
    def rbf_kernel(a, b, bw, alpha):
        a_sq = (a ** 2).sum(dim=1, keepdim=True)
        b_sq = (b ** 2).sum(dim=1, keepdim=True)
        dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
        # OLD: return torch.exp(-dist_sq / (2 * bw ** 2))
        # NEW: generalized RBF — alpha=1 recovers standard Gaussian
        return torch.exp(-(dist_sq / (2 * bw ** 2)) ** alpha)

    if bandwidth is None:
        ss = min(1000, n, m)
        with torch.no_grad():
            x_sq = (x[:ss].detach() ** 2).sum(dim=1, keepdim=True)
            y_sq = (y[:ss] ** 2).sum(dim=1, keepdim=True)
            dists = x_sq + y_sq.T - 2 * torch.mm(x[:ss].detach(), y[:ss].T)
            dists = dists[dists > 0]
            bandwidth = (torch.sqrt(torch.median(dists) / 2) if len(dists) > 0
                         else torch.tensor(1.0, device=dev))
        bandwidth = bandwidth.detach() * bandwidth_scale

    # OLD: K_xx = rbf_kernel(x, x, bandwidth)
    # NEW: pass alpha through
    K_xx = rbf_kernel(x, x, bandwidth, kernel_alpha)
    K_yy = rbf_kernel(y, y, bandwidth, kernel_alpha)
    K_xy = rbf_kernel(x, y, bandwidth, kernel_alpha)

    xx_term = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) if n > 1 else 0.0
    yy_term = (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) if m > 1 else 0.0
    xy_term = 2 * K_xy.sum() / (n * m)
    mmd_sq = xx_term - xy_term + yy_term
    return torch.sqrt(mmd_sq.abs() + 1e-8)


def compute_swd(x, y, n_projections=None, tol=1e-3, min_projections=10, step=10, max_projections=500, seed=None):
    """
    Sliced Wasserstein Distance between two sets of embeddings.
    x: [n, d] — generated (grad flows through this)
    y: [m, d] — target (detached)

    If n_projections is given, uses exactly that many (original behavior).
    Otherwise, incrementally adds `step` projections until
    |SWD(n+step) - SWD(n)| < tol  (converged), capped at max_projections.
    """
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

    if x.dim() > 2: x = x.reshape(x.shape[0], -1)
    if y.dim() > 2: y = y.reshape(y.shape[0], -1)

    d = x.shape[1]

    def _swd_fixed(n_proj):
        gen_swd = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        projections = torch.randn(n_proj, d, device=dev, generator=gen_swd)
        projections = projections / projections.norm(dim=1, keepdim=True)

        x_proj = projections @ x.T  # [n_proj, n]
        y_proj = projections @ y.T  # [n_proj, m]

        x_sorted = x_proj.sort(dim=1).values
        y_sorted = y_proj.sort(dim=1).values

        if x_sorted.shape[1] != y_sorted.shape[1]:
            y_sorted = torch.nn.functional.interpolate(
                y_sorted.unsqueeze(0), size=x_sorted.shape[1], mode='linear', align_corners=False
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
        print(f"      [SWD] n_proj={n}  swd={curr_swd.item():.6f}  delta={delta.item():.6f}", flush=True)
        if delta.item() < tol:
            print(f"      [SWD] Converged at n_projections={n}", flush=True)
            return curr_swd
        prev_swd = curr_swd

    print(f"      [SWD] Reached max_projections={max_projections} without convergence", flush=True)
    return prev_swd

def evaluate_distribution_mmd(latent, architect_vae, architect_image_processor,
                                sprinter, clip_model, clip_processor,
                                all_clip_embeddings,eval_prompt, n_eval=10, device="cuda",seed=None):
    """
    Decode latent -> scribble PIL -> generate n_eval sprinter photos
    -> encode to CLIP -> compute MMD vs target distribution.
    Returns (mmd_scalar, eval_photos_list, clip_embs)
    """
    from clip_utils import encode_images_clip
    from image_utils import latent_to_pil

    # 1. Decode architect latent to scribble PIL
    with torch.no_grad():
        scribble_pil = latent_to_pil(latent, architect_vae, architect_image_processor)

    # 2. Generate sprinter photos from scribble
    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    eval_photos = []
    gen_eval = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    with torch.no_grad():
        for start in range(0, n_eval, 2):
            bs = min(2, n_eval - start)
            result = sprinter(
                prompt=[eval_prompt] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pil", return_dict=True,
                generator=gen_eval
            )
            eval_photos.extend(result.images)
    sprinter.vae.to(dtype=original_vae_dtype)

    # 3. Encode photos to CLIP
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in eval_photos]
    photo_tensor = torch.cat(tensors, dim=0).to(device)
    clip_model.to(device)
    with torch.no_grad():
        clip_embs = encode_images_clip(photo_tensor, clip_model, clip_processor)
    clip_model.to("cpu")

    # 4. MMD vs target
    mmd = compute_mmd(clip_embs, all_clip_embeddings).item()

    return mmd, eval_photos, clip_embs