import torch
import numpy as np

def compute_mmd(x, y, bandwidth=None):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)
    dev = x.device
    x, y = x.float().to(dev), y.float().to(dev)
    if x.dim() > 2: x = x.reshape(x.shape[0], -1)
    if y.dim() > 2: y = y.reshape(y.shape[0], -1)
    n, m = x.shape[0], y.shape[0]

    def rbf_kernel(a, b, bw):
        a_sq = (a**2).sum(dim=1, keepdim=True)
        b_sq = (b**2).sum(dim=1, keepdim=True)
        dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
        return torch.exp(-dist_sq / (2 * bw**2))

    if bandwidth is None:
        ss = min(1000, n, m)
        with torch.no_grad():
            x_sq = (x[:ss]**2).sum(dim=1, keepdim=True)
            y_sq = (y[:ss]**2).sum(dim=1, keepdim=True)
            dists = x_sq + y_sq.T - 2 * torch.mm(x[:ss], y[:ss].T)
            dists = dists[dists > 0]
            bandwidth = torch.sqrt(torch.median(dists) / 2) if len(dists) > 0 else torch.tensor(1.0, device=dev)

    K_xx = rbf_kernel(x, x, bandwidth)
    K_yy = rbf_kernel(y, y, bandwidth)
    K_xy = rbf_kernel(x, y, bandwidth)
    mmd_sq = ((K_xx.sum() - K_xx.trace()) / (n*(n-1))
              - 2 * K_xy.sum() / (n*m)
              + (K_yy.sum() - K_yy.trace()) / (m*(m-1)))
    return torch.clamp(mmd_sq, min=0.0)