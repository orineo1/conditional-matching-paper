import torch
import torch.nn.functional as F
import numpy as np


def compute_mmd(x, y, bandwidth=None):
    """Original MMD in raw latent space — kept for reference/ablation."""
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

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
        bandwidth = bandwidth.detach()

    K_xx = rbf_kernel(x, x, bandwidth)
    K_yy = rbf_kernel(y, y, bandwidth)
    K_xy = rbf_kernel(x, y, bandwidth)

    mmd_sq = ((K_xx.sum() - K_xx.trace()) / (n * (n - 1))
              - 2 * K_xy.sum() / (n * m)
              + (K_yy.sum() - K_yy.trace()) / (m * (m - 1)))
    return torch.clamp(mmd_sq, min=0.0)


def encode_images_clip(images_tensor, clip_model):
    """
    Encode a batch of images through CLIP vision encoder.

    Grad flows through this function (Option B) — do NOT wrap in no_grad.
    images_tensor: (B, 3, H, W) in [0, 1]
    clip_model: the CLIP model (frozen weights, but graph is live for grad)
    Returns: L2-normalized embeddings (B, D)
    """
    # Resize to CLIP expected input size
    imgs = F.interpolate(images_tensor, size=(224, 224),
                         mode="bilinear", align_corners=False)

    # CLIP normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=images_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=images_tensor.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    # vision_model + projection (weights frozen in models.py)
    embs = clip_model.vision_model(pixel_values=imgs).pooler_output
    embs = clip_model.visual_projection(embs)
    return F.normalize(embs, dim=-1)  # (B, D) unit vectors


def compute_clip_mmd(x_embs, y_embs):
    """
    MMD with linear kernel in CLIP embedding space.
    Linear kernel == dot product on unit vectors == cosine similarity.
    No bandwidth needed — CLIP embeddings are already normalized.

    x_embs: generated CLIP embeddings (N, D) — grad flows through
    y_embs: target CLIP embeddings    (M, D) — detached (precomputed)
    """
    n, m = x_embs.shape[0], y_embs.shape[0]

    K_xx = torch.mm(x_embs, x_embs.T)  # (N, N)
    K_yy = torch.mm(y_embs, y_embs.T)  # (M, M) — no grad needed
    K_xy = torch.mm(x_embs, y_embs.T)  # (N, M)

    mmd_sq = (
            (K_xx.sum() - K_xx.trace()) / (n * (n - 1))
            - 2 * K_xy.sum() / (n * m)
            + (K_yy.sum() - K_yy.trace()) / (m * (m - 1))
    )
    return torch.clamp(mmd_sq, min=0.0)