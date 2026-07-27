"""
Feasibility diagnostic for conditional matching in the 2D (dim(x)=1, dim(y)=1) GMM setting.

Motivation
----------
The 2D_cond_1D experiment always defines its target y-distribution as
P(Y|X=x_star) for a real x_star (see notebooks/Exp_2D_cond_1D.ipynb, cell "GMM
PARAMETERS"). That makes the matching problem feasible by construction: there
is always an x that reproduces the target exactly. This module lets us build
targets that are NOT of that form, and quantifies -- analytically, in closed
form, with no sampling noise and no trained network -- how far the *best
possible* x still is from the target.

Core object: the achievability curve
    D(x) = || P(Y|X=x) - target ||_2^2
computed exactly via dist_utils.gmm_l2_distance, swept over a grid of x.
  - If min_x D(x) ~= 0 (down to the numerical floor), a matching x* exists:
    the matching problem is feasible.
  - If min_x D(x) stays bounded well above that floor, no x reproduces the
    target: the matching problem is infeasible, and x* is only the closest
    achievable approximation.
"""

import torch
from scipy.optimize import minimize_scalar

import dist_utils


# ============================================================
# Achievable conditional P(Y|X=x) for the ground-truth joint GMM
# ============================================================

def conditional_gmm_at_x(mu_list, Sigma_list, alpha, x, threshold=0.01):
    """
    Exact P(Y|X=x) for the joint GMM (mu_list, Sigma_list, alpha), filtered
    to drop near-zero-weight components. Returns (means, vars, weights) in
    the same stacked-tensor format produced by dist_utils.compute_conditionals
    / compute_alpha, so the result can be fed directly into
    dist_utils.generate_mog_samples_not_differentiable or gmm_l2_distance.
    """
    x = torch.as_tensor(x, dtype=torch.float32).view(-1)
    cond_mu, cond_sigma = dist_utils.compute_conditionals(mu_list, Sigma_list, x)
    cond_w = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x)
    return dist_utils.filter_and_normalize(cond_mu, cond_sigma, cond_w, threshold=threshold)


# ============================================================
# Achievability curve D(x) = || P(Y|X=x) - target ||_2^2
# ============================================================

def gmm_l2_sq(mu_list, Sigma_list, alpha, target_means, target_vars, target_weights, x, threshold=0.01):
    """D(x) for a single x (float in, float out)."""
    m, s, w = conditional_gmm_at_x(mu_list, Sigma_list, alpha, x, threshold=threshold)
    return dist_utils.gmm_l2_distance(m, s, w, target_means, target_vars, target_weights)


def achievability_curve(mu_list, Sigma_list, alpha, target_means, target_vars, target_weights,
                         x_grid, threshold=0.01):
    """
    D(x) evaluated on a grid of x values. Returns a list of floats, same
    length as x_grid.
    """
    return [
        gmm_l2_sq(mu_list, Sigma_list, alpha, target_means, target_vars, target_weights, x, threshold)
        for x in x_grid
    ]


def find_best_x(mu_list, Sigma_list, alpha, target_means, target_vars, target_weights,
                 bounds, threshold=0.01, grid_n=400):
    """
    x* = argmin_x D(x) and D_min = D(x*).

    Two-stage search: coarse grid over `bounds` to locate the basin, then a
    bounded scalar refinement (Brent's method) around the grid minimum. D(x)
    is a smooth, typically multi-modal function of x (mixture weights move
    non-monotonically as components swap dominance), so the grid stage
    matters -- a purely local optimizer can get stuck in the wrong basin.
    """
    lo, hi = bounds
    x_grid = [lo + i * (hi - lo) / (grid_n - 1) for i in range(grid_n)]
    d_grid = achievability_curve(mu_list, Sigma_list, alpha, target_means, target_vars,
                                  target_weights, x_grid, threshold)
    best_idx = min(range(grid_n), key=lambda i: d_grid[i])
    x0 = x_grid[best_idx]
    step = (hi - lo) / (grid_n - 1)
    refine_lo, refine_hi = max(lo, x0 - 2 * step), min(hi, x0 + 2 * step)

    def obj(x):
        return gmm_l2_sq(mu_list, Sigma_list, alpha, target_means, target_vars, target_weights, x, threshold)

    res = minimize_scalar(obj, bounds=(refine_lo, refine_hi), method="bounded")
    x_star, d_min = float(res.x), float(res.fun)
    if d_min > d_grid[best_idx]:
        x_star, d_min = x0, d_grid[best_idx]
    return x_star, d_min, x_grid, d_grid


# ============================================================
# Target constructors
# ============================================================

def target_from_x(mu_list, Sigma_list, alpha, x_star, threshold=0.01):
    """
    Baseline, feasible-by-construction target: P(Y|X=x_star) itself. This is
    exactly what Exp_2D_cond_1D.ipynb uses, and should always yield
    D_min ~= 0 at x = x_star -- the calibration case for the diagnostic.
    """
    return conditional_gmm_at_x(mu_list, Sigma_list, alpha, x_star, threshold=threshold)


def target_mixture_of_two_x(mu_list, Sigma_list, alpha, x1, x2, mix_weight=0.5, threshold=0.01):
    """
    50/50 (or mix_weight/1-mix_weight) mixture of two real, far-apart
    conditionals P(Y|X=x1) and P(Y|X=x2). This is a distribution over y that
    a *pair* of x values can produce jointly, but generically no *single* x
    reproduces, because as x moves continuously from x1 to x2 the mixture
    weights over the underlying joint-GMM components shift smoothly instead
    of jumping to a fixed 2-point split.
    """
    m1, s1, w1 = conditional_gmm_at_x(mu_list, Sigma_list, alpha, x1, threshold=threshold)
    m2, s2, w2 = conditional_gmm_at_x(mu_list, Sigma_list, alpha, x2, threshold=threshold)

    means = torch.cat([m1, m2], dim=0)
    variances = torch.cat([s1, s2], dim=0)
    weights = torch.cat([mix_weight * w1, (1 - mix_weight) * w2], dim=0)
    weights = weights / weights.sum()
    return means, variances, weights


def target_shrink_variance(mu_list, Sigma_list, alpha, x_ref, scale, threshold=0.01):
    """
    Same means/weights as the real conditional P(Y|X=x_ref), but with every
    component variance scaled by `scale`. Pick `scale` well below the
    smallest variance achieved anywhere on the x-sweep (inspect the
    achievable-variance range first) to get a target that is provably
    outside the achievable family regardless of location.
    """
    m, s, w = conditional_gmm_at_x(mu_list, Sigma_list, alpha, x_ref, threshold=threshold)
    return m, s * scale, w


def target_custom_bimodal(mu_list, Sigma_list, alpha, comp_idx_pair, weight=0.5):
    """
    Hand-built target: an equal (or `weight`/1-weight) mixture of two of the
    *original joint-GMM components' own* (mu_y, Sigma_yy), taken directly
    from mu_list/Sigma_list rather than from any single conditional. Useful
    for constructing an explicitly adversarial two-mode target whose mode
    spacing / weight split you choose by hand, independent of what any
    P(Y|X=x) can produce.
    """
    i, j = comp_idx_pair
    means, variances = [], []
    for idx in (i, j):
        mu = mu_list[idx]
        Sigma = Sigma_list[idx]
        # y is the last component of the 2D joint (x is CONDITION_ON=1 dims)
        means.append(mu[-1:].view(1, 1))
        variances.append(Sigma[-1:, -1:].view(1, 1))
    means = torch.stack(means)
    variances = torch.stack(variances)
    weights = torch.tensor([weight, 1 - weight])
    return means, variances, weights


# ============================================================
# Achievable-variance range (used to calibrate target_shrink_variance)
# ============================================================

def achievable_variance_range(mu_list, Sigma_list, alpha, bounds, threshold=0.01, grid_n=200):
    """
    Min/max of the (weight-averaged) conditional variance of P(Y|X=x) as x
    sweeps `bounds`. Used to pick a `scale` for target_shrink_variance that
    is guaranteed to fall outside what any x on the sweep can produce.
    """
    lo, hi = bounds
    x_grid = [lo + i * (hi - lo) / (grid_n - 1) for i in range(grid_n)]
    avg_vars = []
    for x in x_grid:
        m, s, w = conditional_gmm_at_x(mu_list, Sigma_list, alpha, x, threshold=threshold)
        mean_y = (w.view(-1, 1, 1) * m).sum(dim=0)
        within = (w.view(-1, 1, 1) * s).sum(dim=0)
        between = (w.view(-1, 1, 1) * (m - mean_y) ** 2).sum(dim=0)
        avg_vars.append(float((within + between).view(-1)[0]))
    return min(avg_vars), max(avg_vars), x_grid, avg_vars


# ============================================================
# Density evaluation, for overlay plots
# ============================================================

def density_on_grid(means, variances, weights, y_grid):
    y = torch.as_tensor(y_grid, dtype=torch.float32).view(-1, 1)
    means_list = [means[i] for i in range(means.shape[0])]
    vars_list = [variances[i] for i in range(variances.shape[0])]
    return dist_utils.mog_multivariate_pdf(y, means_list, vars_list, weights).detach().numpy()
