import torch
import numpy as np


def compute_mmd(x, y, bandwidth=None):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()  # ← detach y; grad flows only through x

    if x.dim() > 2: x = x.reshape(x.shape[0], -1)
    if y.dim() > 2: y = y.reshape(y.shape[0], -1)
    n, m = x.shape[0], y.shape[0]

    def rbf_kernel(a, b, bw):
        a_sq = (a ** 2).sum(dim=1, keepdim=True)
        b_sq = (b ** 2).sum(dim=1, keepdim=True)
        dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
        return torch.exp(-dist_sq / (2 * bw ** 2))

    if bandwidth is None:
        ss = min(1000, n, m)
        with torch.no_grad():
            x_sq = (x[:ss].detach() ** 2).sum(dim=1, keepdim=True)
            y_sq = (y[:ss] ** 2).sum(dim=1, keepdim=True)
            dists = x_sq + y_sq.T - 2 * torch.mm(x[:ss].detach(), y[:ss].T)
            dists = dists[dists > 0]
            bandwidth = (torch.sqrt(torch.median(dists) / 2) if len(dists) > 0
                         else torch.tensor(1.0, device=dev))
        bandwidth = bandwidth.detach()  # ← detach bandwidth, it's just a scalar constant

    # These three calls are outside no_grad — grad flows through x
    K_xx = rbf_kernel(x, x, bandwidth)
    K_yy = rbf_kernel(y, y, bandwidth)
    K_xy = rbf_kernel(x, y, bandwidth)

    # Unbiased estimator terms; skip self-kernel term when n=1 or m=1
    # (0 off-diagonal elements → 0/0 division)
    xx_term = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) if n > 1 else 0.0
    yy_term = (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) if m > 1 else 0.0
    xy_term = 2 * K_xy.sum() / (n * m)
    mmd_sq = xx_term - xy_term + yy_term
    # Don't clamp — with small n the unbiased estimator can be slightly negative,
    # and clamp(min=0) kills the gradient (grad=0 when clamped).
    return mmd_sq

def evaluate_distribution_mmd(latent, architect_vae, architect_image_processor,
                                sprinter, clip_model, clip_processor,
                                all_clip_embeddings,eval_prompt, n_eval=10, device="cuda",):
    """
    Decode latent -> scribble PIL -> generate n_eval sprinter photos
    -> encode to CLIP -> compute MMD vs target distribution.
    Returns (mmd_scalar, eval_photos_list, clip_embs)
    """
    from clip_utils import encode_images_clip
    import torchvision.transforms.functional as TF
    from image_utils import latent_to_pil

    # 1. Decode architect latent to scribble PIL
    with torch.no_grad():
        scribble_pil = latent_to_pil(latent, architect_vae, architect_image_processor)

    # 2. Generate sprinter photos from scribble
    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    eval_photos = []
    with torch.no_grad():
        for start in range(0, n_eval, 2):
            bs = min(2, n_eval - start)
            result = sprinter(
                prompt=[eval_prompt] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=0.8,
                output_type="pil", return_dict=True,
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