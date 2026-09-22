import torch
from torch import nn

#### MMD Loss #####
#
# Same kernel/bandwidth/U-statistic machinery as
# SD_cond_SD_controlnet/src/metrics.py's compute_mmd (single Gaussian RBF
# kernel, detached median-heuristic bandwidth, unbiased MMD^2_U U-statistic,
# eq:mmd-ustat-def in the theory appendix). compute_mmd returns the PLAIN
# squared statistic -- this is what actually gets used everywhere
# (gradients, adaptive step-size scaling, best-loss tracking): no sqrt, no
# +eps floor, matching SD_cond_SD_controlnet/src/metrics.py's compute_mmd
# exactly (byte-for-byte identical body). Use mmd_report() below to take the
# square root ONLY at points that print/log/plot a number for a human to
# read -- never inside the actual loss computation. Previously this module
# used a 5-bandwidth kernel mixture, a mean-distance bandwidth computed WITH
# autograd on (gradient leaked into the loss through sigma, not just through
# the samples), and a V-statistic (diagonal included, biased) -- none of
# which matched SD or the theory appendix's U-statistic.


def mmd_report(mmd_sq):
    """sqrt(|MMD^2_U|) for DISPLAY ONLY (printing/logging/plotting a human-
    readable MMD number) -- never feed this into a loss or a further
    computation; compute_mmd's own return value (the plain squared
    statistic) is what the algorithm actually uses everywhere."""
    if torch.is_tensor(mmd_sq):
        return torch.sqrt(mmd_sq.abs())
    return abs(mmd_sq) ** 0.5


def rbf_kernel(a, b, bw, alpha=1.0):
    """Generalised RBF kernel: exp(-(||a-b||^2 / 2bw^2)^alpha)."""
    a_sq = (a ** 2).sum(dim=1, keepdim=True)
    b_sq = (b ** 2).sum(dim=1, keepdim=True)
    dist_sq = a_sq + b_sq.T - 2 * torch.mm(a, b.T)
    return torch.exp(-(dist_sq / (2 * bw ** 2)) ** alpha)


def estimate_bandwidth(x, y, bandwidth_scale=1.0):
    """Median-heuristic RBF bandwidth between x and y (detached, no_grad) --
    sigma must never carry gradient from x into the loss."""
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


def compute_mmd(x, y, bandwidth=None, bandwidth_scale=1.0, kernel_alpha=1.0):
    """
    Unbiased MMD^2_U with a generalised RBF kernel -- identical estimator to
    SD_cond_SD_controlnet/src/metrics.py's compute_mmd. Computes the theory
    appendix's U-statistic exactly (eq:mmd-ustat-def), with NO sqrt applied:

        MMD^2_U(x,y) = 1/(n(n-1)) sum_{i!=i'} k(x_i,x_i')
                     - 2/(nm) sum_{i,j} k(x_i,y_j)
                     + 1/(m(m-1)) sum_{j!=j'} k(y_j,y_j')

    This plain squared value is what the algorithm actually uses (loss,
    gradients, any adaptive scaling downstream) -- use mmd_report() above to
    take a square root for display only. The unbiased estimator can be
    legitimately slightly negative when the two samples are very close;
    that's expected and left as-is (no abs()/floor here in the value used
    for computation).

    Args:
        x:               [n, d] generated samples (grad flows through).
        y:               [m, d] target samples (detached).
        bandwidth:       kernel bandwidth; estimated via median heuristic if None.
        bandwidth_scale: multiplicative scale applied after median estimation.
        kernel_alpha:    RBF exponent. 1 = standard Gaussian.

    Returns:
        Scalar MMD^2_U estimate (unbiased, squared -- NOT square-rooted).
    """
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
    return mmd_sq


class RBF(nn.Module):
    """Kept for backward compatibility with call sites that still pass an
    explicit `kernel=RBF()` to MMDLoss; MMDLoss no longer uses it internally
    (it calls compute_mmd directly, above), so any constructor args here are
    accepted but ignored."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class MMDLoss(nn.Module):
    """Thin nn.Module wrapper around compute_mmd, kept so existing call sites
    (`MMDLoss(kernel=RBF())` then `mmd_loss(X, Y)`) don't need to change."""

    def __init__(self, kernel=None, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, X, Y):
        return compute_mmd(X.to(self.device), Y.to(self.device))
