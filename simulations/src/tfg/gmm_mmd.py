"""Closed-form population MMD^2 between two Gaussian mixtures.

For a Gaussian kernel ``k(y, y') = exp(-||y - y'||^2 / (2 sigma^2))`` and
mixtures ``p = sum_i a_i N(m_i, S_i)``, ``q = sum_j b_j N(n_j, T_j)``:

    E_{y~p, z~q}[k(y, z)] = Z * sum_{i,j} a_i b_j N(m_i - n_j; 0, S_i + T_j + K)

with ``K = sigma^2 I`` and ``Z = (2 pi sigma^2)^{d/2}`` (the constant that turns
the normalised Gaussian density back into the un-normalised kernel).  So

    MMD^2(p, q) = E_pp[k] - 2 E_pq[k] + E_qq[k]

is available exactly, with no sampling, by the same machinery as
``tfg/gmm_l2.py`` -- only the covariance sum gains the kernel bandwidth.

This matters for the estimator study: the ground-truth gradient and the sampled
estimator must target the SAME functional, otherwise measured "bias" would just
be the gap between two different objectives.  Setting ``sigma = 0`` recovers the
L2 inner products exactly, so the same code covers both metrics.
"""

import math

import torch

from tfg.gmm_l2 import _as_covs, _as_means, _log_gaussian


def _prep(means, covs, weights, d=None, normalize=True):
    m = _as_means(means, d=d)
    K = m.shape[0]
    c = _as_covs(covs, K, m.shape[1])
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if w.shape[0] != K:
        raise ValueError("weight vector length does not match number of components")
    if normalize:
        w = w / w.sum()
    return m, c, w


def _inner(m1, c1, w1, m2, c2, w2, sigma):
    """``E_{y~p, z~q}[k(y, z)]`` for the un-normalised Gaussian kernel."""
    d = m1.shape[1]
    diff = m1.unsqueeze(1) - m2.unsqueeze(0)
    cov = c1.unsqueeze(1) + c2.unsqueeze(0)
    if sigma > 0:
        eye = torch.eye(d, dtype=torch.float64, device=cov.device)
        cov = cov + (sigma ** 2) * eye
    log_n = _log_gaussian(diff, cov)
    log_w = torch.log(w1).unsqueeze(1) + torch.log(w2).unsqueeze(0)
    total = log_n + log_w
    mx = total.max()
    if not torch.isfinite(mx):
        raise FloatingPointError(
            "gmm_mmd: non-finite log-density in the inner product.\n"
            f"  max log term    : {float(mx)!r}\n"
            f"  n non-finite    : {int((~torch.isfinite(total)).sum())} of {total.numel()}\n"
            f"  sigma           : {sigma!r}\n"
            "Returning zero here would silently break the autograd graph."
        )
    val = torch.exp(mx) * torch.exp(total - mx).sum()
    if sigma > 0:
        val = val * (2.0 * math.pi * sigma ** 2) ** (d / 2.0)
    return val


def population_mmd2(means_p, covs_p, weights_p, means_q, covs_q, weights_q,
                    sigma=1.0, normalize_weights=True):
    """Exact ``MMD^2(p, q)``. Differentiable, no sampling."""
    mp, cp, wp = _prep(means_p, covs_p, weights_p, normalize=normalize_weights)
    mq, cq, wq = _prep(means_q, covs_q, weights_q, d=mp.shape[1],
                       normalize=normalize_weights)
    pp = _inner(mp, cp, wp, mp, cp, wp, sigma)
    qq = _inner(mq, cq, wq, mq, cq, wq, sigma)
    pq = _inner(mp, cp, wp, mq, cq, wq, sigma)
    return pp - 2.0 * pq + qq


def kernel_mean_embedding(y, means_q, covs_q, weights_q, sigma=1.0):
    """``E_{z~q}[k(y, z)]`` at points ``y`` of shape ``(N, d)``.

    Lets a sampled estimator use the EXACT target distribution for the cross
    term, so that the only randomness in the estimator is the ``n`` conditional
    draws -- which is precisely the quantity ``n_t`` controls.
    """
    mq, cq, wq = _prep(means_q, covs_q, weights_q)
    d = mq.shape[1]
    y = y.reshape(-1, d)
    diff = y.unsqueeze(1) - mq.unsqueeze(0)                     # (N, K, d)
    cov = cq.unsqueeze(0).expand(y.shape[0], *cq.shape).clone()
    if sigma > 0:
        eye = torch.eye(d, dtype=torch.float64, device=cov.device)
        cov = cov + (sigma ** 2) * eye
    log_n = _log_gaussian(diff, cov)
    out = (torch.exp(log_n) * wq.unsqueeze(0)).sum(-1)
    if sigma > 0:
        out = out * (2.0 * math.pi * sigma ** 2) ** (d / 2.0)
    return out


def kernel_matrix(y, sigma=1.0):
    """Un-normalised Gaussian kernel matrix ``k(y_i, y_j)``."""
    d2 = torch.cdist(y, y, p=2) ** 2
    return torch.exp(-d2 / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Multi-bandwidth kernel matching the repository's RBF family
# ---------------------------------------------------------------------------
#
# LossFunctions.RBF (LossFunctions.py:19-23) computes
#
#     scaled = L2_distances / (bandwidth * bandwidth_multipliers[k])
#     K      = sum_k exp(-scaled)
#
# i.e. K(y, y') = sum_k exp( -||y - y'||^2 / (bw * m_k) ) with
# m_k = mul_factor^(k - n_kernels//2).  Matching that to the Gaussian form
# exp(-||y-y'||^2 / (2 sigma_k^2)) gives
#
#     sigma_k^2 = bw * m_k / 2
#
# Note this is the FIXED-bandwidth reading. The repository recomputes `bw`
# from the data on every call (the mean off-diagonal squared distance over the
# pooled X u Y), which makes the objective itself sample-dependent. That
# adaptive behaviour is a separate source of gradient noise and is studied as
# an ablation; the population reference below necessarily fixes the bandwidth,
# because a population quantity cannot depend on a sample.

def bandwidth_sigmas(bandwidth, n_kernels=5, mul_factor=2.0):
    """The ``sigma_k`` of the repository's kernel family, as a list."""
    ks = torch.arange(n_kernels, dtype=torch.float64) - (n_kernels // 2)
    mult = mul_factor ** ks
    return [float((bandwidth * float(m) / 2.0) ** 0.5) for m in mult]


def population_mmd2_multibandwidth(means_p, covs_p, weights_p,
                                   means_q, covs_q, weights_q,
                                   bandwidth=1.0, n_kernels=5, mul_factor=2.0,
                                   normalize_weights=True):
    """Exact population MMD^2 under the repository's summed RBF family.

    Because MMD^2 is linear in the kernel, the multi-bandwidth value is just
    the sum of the single-bandwidth values.
    """
    total = None
    for sigma in bandwidth_sigmas(bandwidth, n_kernels, mul_factor):
        term = population_mmd2(means_p, covs_p, weights_p,
                               means_q, covs_q, weights_q, sigma=sigma,
                               normalize_weights=normalize_weights)
        total = term if total is None else total + term
    return total


def kernel_mean_embedding_multibandwidth(y, means_q, covs_q, weights_q,
                                         bandwidth=1.0, n_kernels=5,
                                         mul_factor=2.0):
    total = None
    for sigma in bandwidth_sigmas(bandwidth, n_kernels, mul_factor):
        term = kernel_mean_embedding(y, means_q, covs_q, weights_q, sigma=sigma)
        total = term if total is None else total + term
    return total
