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

import os

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema
from tqdm import trange

import dist_utils
import experiment_utils
import Optimization


# ============================================================
# GMM parameters for the 2D_cond_1D joint (shared with Exp_2D_cond_1D.ipynb)
# ============================================================

def ensure_gmm_params(params_dir, results_dir, experiment_name, global_seed):
    """
    Load the 2D_cond_1D joint-GMM parameters from params_dir/results_dir, or
    regenerate them if neither has a saved copy yet.

    Generation is fully deterministic (seeded, no randomness beforehand), and
    reproduces exactly the "GMM PARAMETERS" cell of Exp_2D_cond_1D.ipynb --
    the same mu_list/Sigma_list/alpha/x_star that experiment's checkpoints
    were trained against -- so this notebook does not require having run
    Exp_2D_cond_1D.ipynb first. The freshly generated parameters are saved to
    params_dir so later runs (and Exp_2D_cond_1D.ipynb itself) reuse them.
    """
    loaded = experiment_utils.load_gmm_params(params_dir, experiment_name)
    if loaded is None:
        loaded = experiment_utils.load_gmm_params(results_dir, experiment_name)
    if loaded is not None:
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
        mu_list    = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha      = alpha.float()
        return mu_list, Sigma_list, alpha, x_star

    print("[GMM] No saved parameters found -- regenerating (deterministic, seed={})".format(global_seed))
    experiment_utils.set_global_seed(global_seed)
    mu_list = [
        torch.tensor([-5,  5], dtype=torch.float64),
        torch.tensor([-5, -5], dtype=torch.float64),
        torch.tensor([ 5,  3], dtype=torch.float64),
        torch.tensor([ 5, -1], dtype=torch.float64),
        torch.tensor([ 0, -3], dtype=torch.float64),
        torch.tensor([-2,  4], dtype=torch.float64),
        torch.tensor([-2, -3], dtype=torch.float64),
        torch.tensor([ 1,  2], dtype=torch.float64),
        torch.tensor([-8,  1], dtype=torch.float64),
        torch.tensor([ 7,  5], dtype=torch.float64),
        torch.tensor([ 0, -5], dtype=torch.float64),
    ]
    Sigma_list = [
        torch.tensor([[0.5000, 0.1950],
                      [0.1950, 0.2000]], dtype=torch.float64)
    ] * len(mu_list)
    alpha = torch.tensor([1 / len(mu_list)] * len(mu_list), dtype=torch.float64)

    mu_list    = [mu.float() for mu in mu_list]
    Sigma_list = [cov.float() for cov in Sigma_list]
    alpha      = alpha.float()

    x_star = torch.tensor([-5])
    mu_temp, Sigma_temp = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
    temp_alpha          = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
    mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
        mu_temp, Sigma_temp, temp_alpha, threshold=0.01
    )

    experiment_utils.save_gmm_params(
        mu_list, Sigma_list, alpha,
        mog_means, mog_variances, weights, x_star,
        params_dir, experiment_name
    )
    return mu_list, Sigma_list, alpha, x_star


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
# Feasible-but-hard-landscape search: a genuine x* (D(x*)=0 by
# construction, via target_from_x) whose D(x) curve nonetheless has a
# decoy local minimum elsewhere -- distinct from the infeasibility cases
# above. Here a good match exists; the question is whether gradient-guided
# search (MLGD/MLGD-F) reliably finds it or gets trapped in the decoy.
# ============================================================

def find_decoy_candidate(mu_list, Sigma_list, alpha, bounds, n_candidates=25, grid_n=250,
                          min_x_gap=1.5, threshold=0.01):
    """
    For each of n_candidates real x* values swept over `bounds`, build
    target_from_x(x*) (feasible by construction) and scan its D(x) curve for
    a secondary local minimum at least min_x_gap away from x* -- a "decoy"
    that could mislead a local/gradient-guided search away from the true x*.

    Returns a list of (x_star_candidate, decoy_x, decoy_D) tuples, sorted by
    decoy_D ascending (most deceptive -- decoy nearly as good as the true
    zero at x_star -- first).
    """
    lo, hi = bounds
    results = []
    for x_star_c in np.linspace(lo, hi, n_candidates):
        target = target_from_x(mu_list, Sigma_list, alpha, torch.tensor([x_star_c]), threshold=threshold)
        x_grid = np.linspace(lo, hi, grid_n)
        d_grid = np.array(achievability_curve(mu_list, Sigma_list, alpha, *target, x_grid, threshold))
        local_min_idx = argrelextrema(d_grid, np.less_equal, order=8)[0]
        best_decoy = None
        for idx in local_min_idx:
            x_val, d_val = float(x_grid[idx]), float(d_grid[idx])
            if abs(x_val - x_star_c) > min_x_gap and (best_decoy is None or d_val < best_decoy[1]):
                best_decoy = (x_val, d_val)
        if best_decoy is not None:
            results.append((float(x_star_c), best_decoy[0], best_decoy[1]))
    results.sort(key=lambda r: r[2])
    return results


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


def mean_variance(means, variances, weights):
    """Weight-averaged mean/variance of a (means, variances, weights) GMM target."""
    w = weights.view(-1, 1, 1)
    mean = (w * means).sum(dim=0)
    var = (w * variances).sum(dim=0) + (w * (means - mean) ** 2).sum(dim=0)
    return float(mean.view(-1)[0]), float(var.view(-1)[0])


def safe_shrink_scale(mu_list, Sigma_list, alpha, x_ref, vmin, margin=0.5, threshold=0.01):
    """
    Scale factor for target_shrink_variance(x_ref, scale) that guarantees the
    resulting target variance sits below `vmin` (the achievable minimum from
    achievable_variance_range), with a safety margin.
    """
    _, base_var = mean_variance(*conditional_gmm_at_x(mu_list, Sigma_list, alpha, x_ref, threshold))
    return margin * vmin / base_var


# ============================================================
# One-call diagnostic: curve + optional density overlay + plot + summary
# ============================================================

def diagnose_target(mu_list, Sigma_list, alpha, target, bounds, label,
                     feasibility_floor=None, y_grid=None, save_dir=None, threshold=0.01):
    """
    Run the full analytic diagnostic for one target: find x*/D_min, plot the
    achievability curve (and, if y_grid is given, a target-vs-best-achievable
    density overlay), print a one-line verdict, and optionally save the figure.

    Returns a dict with x_star, d_min, x_grid, d_grid, feasible (bool or None
    if no floor was given) -- everything needed for a results table.
    """
    x_star, d_min, x_grid, d_grid = find_best_x(mu_list, Sigma_list, alpha, *target,
                                                 bounds=bounds, threshold=threshold)
    feasible = None if feasibility_floor is None else d_min <= feasibility_floor
    verdict = "" if feasible is None else ("FEASIBLE" if feasible else "INFEASIBLE")
    print(f"[{label}] x* = {x_star:.4f}   D_min = {d_min:.4g}" + (f"   -> {verdict}" if verdict else ""))

    ncols = 2 if y_grid is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4))
    ax0 = axes[0] if ncols > 1 else axes
    ax0.plot(x_grid, d_grid)
    ax0.axvline(x_star, color="r", ls="--", label=f"best x*={x_star:.2f}")
    ax0.set_xlabel("x"); ax0.set_ylabel("D(x)"); ax0.set_title(f"{label}: achievability curve")
    ax0.legend(); ax0.grid(True)

    if y_grid is not None:
        target_density = density_on_grid(*target, y_grid)
        achieved = target_from_x(mu_list, Sigma_list, alpha, torch.tensor([x_star]), threshold=threshold)
        achieved_density = density_on_grid(*achieved, y_grid)
        axes[1].plot(y_grid, target_density, label="target")
        axes[1].plot(y_grid, achieved_density, label=f"best achievable P(Y|X={x_star:.2f})")
        axes[1].set_xlabel("y"); axes[1].set_ylabel("density"); axes[1].set_title(f"{label}: target vs. closest achievable")
        axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    if save_dir is not None:
        fname = label.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=150)
    plt.show()

    return {"label": label, "x_star": x_star, "d_min": d_min, "feasible": feasible,
            "x_grid": x_grid, "d_grid": d_grid}


# ============================================================
# Cross-check with the paper's actual optimizer (MLGD / MLGD-F)
# ============================================================

def run_cross_check(model_uncond, model_cond, model_cm, target, mu_list, Sigma_list, alpha,
                     label, global_seed, device, n_attempts=10, nsamples=250, num_x_t=3):
    """
    Rerun Optimization.optimize_LGD (unmodified) on `target`, once with the
    diffusion conditional model (MLGD) and once with the consistency model
    (MLGD-F, CM=True). Returns (per-run DataFrame, per-method summary
    DataFrame) with the recovered x and final MMD loss.
    """
    rows = []
    for method, cond_model, cm_flag in [("MLGD", model_cond, False), ("MLGD-F", model_cm, True)]:
        for i in trange(n_attempts, desc=f"{label} | {method}"):
            experiment_utils.set_run_seed(global_seed, i)
            best_x_t, _, final_loss = Optimization.optimize_LGD(
                model_uncond, cond_model, *target, mu_list, Sigma_list, alpha,
                nsamples=nsamples, loss="MMD", device=device, CM=cm_flag, num_x_t=num_x_t
            )
            rows.append((method, i, float(best_x_t.view(-1)[0].item()), float(final_loss.item())))

    df = pd.DataFrame(rows, columns=["method", "run", "x_recovered", "final_mmd_loss"])
    summary = df.groupby("method").agg(
        x_mean=("x_recovered", "mean"), x_std=("x_recovered", "std"),
        loss_mean=("final_mmd_loss", "mean"), loss_std=("final_mmd_loss", "std"),
    )
    print(f"\n--- {label} ---")
    print(summary)
    return df, summary


def comparison_row(label, diagnosis, summary):
    """One row for the final results table: analytic vs. MLGD vs. MLGD-F."""
    return {
        "case": label,
        "analytic_x_star": diagnosis["x_star"],
        "analytic_D_min": diagnosis["d_min"],
        "MLGD_loss_mean": summary.loc["MLGD", "loss_mean"],
        "MLGD_x_mean": summary.loc["MLGD", "x_mean"],
        "MLGD-F_loss_mean": summary.loc["MLGD-F", "loss_mean"],
        "MLGD-F_x_mean": summary.loc["MLGD-F", "x_mean"],
    }
