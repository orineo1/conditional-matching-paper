"""Exact closed-form L2 distance between two Gaussian mixtures.

The repository's only exact distributional metric is L1
(``dist_utils.compute_L1_distance_dynamic``), which is computed by adaptive
quadrature.  For Gaussian mixtures the *squared* L2 distance has a closed form
and needs no quadrature at all, because the product of two Gaussian densities
integrates to a single Gaussian density evaluation:

    int N(x; m1, S1) N(x; m2, S2) dx = N(m1; m2, S1 + S2)

Hence for p = sum_i a_i N(mu_i, Sigma_i) and q = sum_j b_j N(nu_j, Lambda_j):

    ||p - q||_2^2 = sum_{i,i'} a_i a_i'  N(mu_i;  mu_i',  Sigma_i + Sigma_i')
                  - 2 sum_{i,j}  a_i b_j  N(mu_i;  nu_j,   Sigma_i + Lambda_j)
                  + sum_{j,j'} b_j b_j'  N(nu_j;  nu_j',  Lambda_j + Lambda_j')

This is exact up to floating point, in any dimension, and is differentiable.

All computation is in float64.  The three double sums are evaluated in log
space and exponentiated with the global maximum factored out, so that
well-separated mixtures (where individual terms underflow) stay accurate.
"""

import math

import torch


def _as_means(means, d=None):
    """Normalise mixture means to shape ``(K, d)``.

    Accepts ``(K, d)``, ``(K, d, 1)`` and ``(K, 1, d)`` -- the repository's
    parameter files use the ``(K, d, 1)`` spelling, e.g.
    ``target_means = torch.tensor([[[5.0]], [[-5.0]]])``.
    """
    m = torch.as_tensor(means, dtype=torch.float64)
    if m.dim() == 1:
        m = m.reshape(-1, 1)
    elif m.dim() == 3:
        if m.shape[2] == 1:
            m = m.squeeze(2)
        elif m.shape[1] == 1:
            m = m.squeeze(1)
        else:
            raise ValueError(f"cannot interpret means of shape {tuple(m.shape)}")
    elif m.dim() != 2:
        raise ValueError(f"cannot interpret means of shape {tuple(m.shape)}")
    if d is not None and m.shape[1] != d:
        raise ValueError(f"means have dimension {m.shape[1]}, expected {d}")
    return m


def _as_covs(covs, K, d):
    """Normalise covariances to ``(K, d, d)``.

    Accepts ``(K, d, d)``, ``(K, d)`` (diagonal), ``(K,)`` / ``(K, 1, 1)``
    (scalar variance in 1-D), or a single shared ``(d, d)``.
    """
    c = torch.as_tensor(covs, dtype=torch.float64)
    if c.dim() == 0:
        c = c.reshape(1, 1, 1).expand(K, 1, 1)
    elif c.dim() == 1:
        if c.shape[0] == K and d == 1:
            c = c.reshape(K, 1, 1)
        elif c.shape[0] == d:
            c = torch.diag(c).unsqueeze(0).expand(K, d, d)
        else:
            raise ValueError(f"cannot interpret covariances of shape {tuple(c.shape)}")
    elif c.dim() == 2:
        if K == d and d > 1 and c.shape == (K, d):
            raise ValueError(
                f"ambiguous covariance of shape {tuple(c.shape)} with K == d == {d}: "
                "this could be K diagonal vectors or one shared full covariance. "
                "Pass an explicit (K, d, d) array to disambiguate."
            )
        if c.shape == (K, d) and d > 1:
            c = torch.stack([torch.diag(row) for row in c])
        elif c.shape == (d, d):
            c = c.unsqueeze(0).expand(K, d, d)
        elif c.shape == (K, 1) and d == 1:
            c = c.reshape(K, 1, 1)
        else:
            raise ValueError(f"cannot interpret covariances of shape {tuple(c.shape)}")
    if c.shape != (K, d, d):
        raise ValueError(f"covariances resolved to {tuple(c.shape)}, expected {(K, d, d)}")
    return c


def _log_gaussian(diff, cov_sum):
    """``log N(diff; 0, cov_sum)`` for a batch.

    ``diff`` is ``(..., d)`` and ``cov_sum`` is ``(..., d, d)``.
    """
    d = diff.shape[-1]
    L = torch.linalg.cholesky(cov_sum)
    sol = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
    maha = (sol.squeeze(-1) ** 2).sum(-1)
    log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (d * math.log(2.0 * math.pi) + log_det + maha)


def _cross_term(m1, c1, w1, m2, c2, w2):
    """``sum_{i,j} w1_i w2_j N(m1_i; m2_j, c1_i + c2_j)``, computed stably."""
    diff = m1.unsqueeze(1) - m2.unsqueeze(0)              # (K1, K2, d)
    cov = c1.unsqueeze(1) + c2.unsqueeze(0)               # (K1, K2, d, d)
    log_n = _log_gaussian(diff, cov)                      # (K1, K2)
    # Clamp before the log. Components far from the conditioning point can have
    # a conditional weight that underflows to exactly 0; log(0) = -inf is
    # harmless for the VALUE (exp(-inf) = 0) but makes the GRADIENT NaN. The
    # floor is far below any weight that contributes to the sum.
    tiny = torch.finfo(w1.dtype).tiny
    log_w = (torch.log(w1.clamp_min(tiny)).unsqueeze(1)
             + torch.log(w2.clamp_min(tiny)).unsqueeze(0))
    total = log_n + log_w
    m = total.max()
    if not torch.isfinite(m):
        raise FloatingPointError(
            "gmm_l2: non-finite log-density in the cross term.\n"
            f"  max log term      : {float(m)!r}\n"
            f"  n non-finite      : {int((~torch.isfinite(total)).sum())} of {total.numel()}\n"
            f"  log N range       : [{float(log_n.min())!r}, {float(log_n.max())!r}]\n"
            f"  log weight range  : [{float(log_w.min())!r}, {float(log_w.max())!r}]\n"
            f"  means A           : {m1.reshape(-1).tolist()[:8]}\n"
            f"  means B           : {m2.reshape(-1).tolist()[:8]}\n"
            "Likely causes: a zero or negative mixture weight (log -> -inf), a "
            "singular or non-PSD covariance, or a NaN in the inputs. Returning "
            "zero here would silently break the autograd graph and report a "
            "spuriously perfect distance, so this raises instead."
        )
    return torch.exp(m) * torch.exp(total - m).sum()


def gmm_l2_squared(means_p, covs_p, weights_p, means_q, covs_q, weights_q,
                   normalize_weights=True):
    """Exact ``||p - q||_2^2`` between two Gaussian mixtures.

    Returns a 0-dim float64 tensor.  Weights are renormalised to sum to 1 by
    default; pass ``normalize_weights=False`` to compare unnormalised mixtures.
    """
    mp = _as_means(means_p)
    d = mp.shape[1]
    mq = _as_means(means_q, d=d)
    Kp, Kq = mp.shape[0], mq.shape[0]
    cp = _as_covs(covs_p, Kp, d)
    cq = _as_covs(covs_q, Kq, d)

    wp = torch.as_tensor(weights_p, dtype=torch.float64).reshape(-1)
    wq = torch.as_tensor(weights_q, dtype=torch.float64).reshape(-1)
    if wp.shape[0] != Kp or wq.shape[0] != Kq:
        raise ValueError("weight vector length does not match number of components")
    if (wp < 0).any() or (wq < 0).any():
        raise ValueError("mixture weights must be non-negative")
    if normalize_weights:
        wp = wp / wp.sum()
        wq = wq / wq.sum()

    pp = _cross_term(mp, cp, wp, mp, cp, wp)
    qq = _cross_term(mq, cq, wq, mq, cq, wq)
    pq = _cross_term(mp, cp, wp, mq, cq, wq)

    val = pp - 2.0 * pq + qq
    # The exact value is non-negative; cancellation can push it a few ulps
    # below zero when p == q.
    return val.clamp_min(0.0)


def gmm_l2(means_p, covs_p, weights_p, means_q, covs_q, weights_q, **kw):
    """Exact ``||p - q||_2``."""
    return torch.sqrt(gmm_l2_squared(means_p, covs_p, weights_p,
                                     means_q, covs_q, weights_q, **kw))


# ---------------------------------------------------------------------------
# Numerical validation helpers (used by tests; not by the hot path)
# ---------------------------------------------------------------------------

def gmm_pdf(x, means, covs, weights):
    """Mixture density at points ``x`` of shape ``(N, d)``."""
    m = _as_means(means)
    d = m.shape[1]
    K = m.shape[0]
    c = _as_covs(covs, K, d)
    w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    w = w / w.sum()
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, d)

    diff = x.unsqueeze(1) - m.unsqueeze(0)                       # (N, K, d)
    cov = c.unsqueeze(0).expand(x.shape[0], K, d, d)
    log_n = _log_gaussian(diff, cov)                             # (N, K)
    return (torch.exp(log_n) * w.unsqueeze(0)).sum(-1)


def gmm_l2_squared_quadrature(means_p, covs_p, weights_p,
                              means_q, covs_q, weights_q,
                              lo=None, hi=None, n_points=200_001):
    """Reference ``||p - q||_2^2`` by 1-D trapezoidal quadrature.

    Only defined for ``d == 1``; used to validate the closed form.
    """
    mp, mq = _as_means(means_p), _as_means(means_q)
    if mp.shape[1] != 1:
        raise ValueError("quadrature reference is 1-D only")
    cp = _as_covs(covs_p, mp.shape[0], 1)
    cq = _as_covs(covs_q, mq.shape[0], 1)

    if lo is None or hi is None:
        sd = torch.sqrt(torch.cat([cp.reshape(-1), cq.reshape(-1)])).max()
        allm = torch.cat([mp.reshape(-1), mq.reshape(-1)])
        pad = 12.0 * sd
        lo = float(allm.min() - pad) if lo is None else lo
        hi = float(allm.max() + pad) if hi is None else hi

    grid = torch.linspace(lo, hi, n_points, dtype=torch.float64).reshape(-1, 1)
    p = gmm_pdf(grid, means_p, covs_p, weights_p)
    q = gmm_pdf(grid, means_q, covs_q, weights_q)
    return torch.trapezoid((p - q) ** 2, grid.reshape(-1))
