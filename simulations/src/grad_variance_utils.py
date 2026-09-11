"""Shared helpers for lgd_vs_lgdcm_step_variance.py's gradient-variance/accuracy
diagnostic: the DDIM unroll, the closed-form population-reference-gradient
computation, and the variance/accuracy bookkeeping used at every captured
state.
"""
import numpy as np
import torch


def ddim_sample_kstep(model, nsamples, condition_x, K, device):
    """Differentiable, deterministic (eta=0) DDIM sampling from `model`, using
    an evenly-spaced K-step subsequence of its trained noise schedule
    (accelerated DDIM respacing) -- the standard way to control unroll depth
    without retraining. Gradients flow from the output back to `condition_x`.
    Returns (full_sample, y_only, n_steps_actually_taken).
    """
    model_dtype = next(model.parameters()).dtype
    T = model.diffusion_steps

    idx = torch.linspace(0, T - 1, K + 1).round().long()
    idx = torch.unique(idx, sorted=True).flip(0)  # descending, e.g. [99, ..., 0]
    n_steps = len(idx) - 1

    x = torch.randn(nsamples, model.nfeatures, device=device, dtype=model_dtype)
    cond = condition_x.to(device=device, dtype=model_dtype)

    for i in range(n_steps):
        t_cur = idx[i].item()
        t_next = idx[i + 1].item()
        t_batch = torch.full((nsamples, 1), t_cur, device=device, dtype=model_dtype)
        predicted_noise = model(x, t_batch, cond)

        alpha_bar_t = model.baralphas[t_cur]
        alpha_bar_prev = model.baralphas[t_next]

        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise  # eta=0 => sigma_t=0
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

    y = x[:, model.condition_on:]
    return x, y, n_steps


def analytic_target_samples(dist_utils, mu_list, Sigma_list, alpha, x_fixed, n_target, device):
    """Fixed target samples at x_fixed, drawn from the EXACT analytic
    conditional GMM (closed-form, ground truth) -- not model_cond's learned
    approximation of it. Returns a [n_target, y_dim] float tensor on `device`.
    """
    mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_fixed)
    w_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_fixed)
    return dist_utils.generate_mog_samples_not_differentiable(
        n_target, mu_cond, Sigma_cond, w_cond
    ).float().to(device)


def true_reference_gradient(dist_utils, mmd_loss, mu_list, Sigma_list, alpha, x_fixed, target_samples, grad_ref_n, device):
    """TRUE/population gradient at x_fixed: differentiate MMD(ref_samples(x),
    target_samples) w.r.t. x, where ref_samples(x) is a large, differentiable
    draw from the EXACT analytic conditional GMM at x (closed-form, no network
    forward -- not any model's learned approximation). Returns
    (grad_ref: np.ndarray, grad_ref_norm: float).
    """
    x_ref_leaf = x_fixed.clone().detach().to(device).requires_grad_(True)
    condi_mu, condi_sigma = dist_utils.compute_conditionals(mu_list, Sigma_list, x_ref_leaf)
    condi_mu = condi_mu.squeeze(-1)  # drop a spurious trailing dim; generate_mog_samples'
                                      # is_multivariate check looks at xi.shape[-1]
    condi_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_ref_leaf)
    ref_samples = dist_utils.generate_mog_samples(grad_ref_n, condi_mu, condi_sigma, condi_alpha, device=device)
    loss_ref = mmd_loss(ref_samples, target_samples)
    grad_ref = torch.autograd.grad(loss_ref, x_ref_leaf)[0].detach().cpu().numpy()
    grad_ref_norm = float(np.linalg.norm(grad_ref))
    return grad_ref, grad_ref_norm


def grad_stats_from_draws(grads):
    """grads: [n_draws, dim] array of independent gradient draws at one frozen
    point. Returns (mean_grad, stats) where stats has:
      normalized_variance    -- Var(grad)/||mean_grad||^2 (scale-free, but can mask a
                                 real dimension-dependent noise trend if ||mean_grad||
                                 itself grows with dimension).
      variance_trace         -- raw E[||g - mean_grad||^2] (sum over all coordinates --
                                 grows trivially with dimension just from having more
                                 terms in the sum, NOT itself evidence of a per-coordinate
                                 noise increase).
      variance_trace_per_dim -- variance_trace / dim, i.e. per-coordinate variance --
                                 the fair cross-dimension comparison: factors out the
                                 trivial "more coordinates -> bigger sum" effect.
      mean_grad_norm          -- ||mean_grad|| (for reference / sanity-checking whether
                                 the signal itself is what's scaling with dimension).
    """
    grads = np.asarray(grads)
    dim = grads.shape[1]
    mean_grad = grads.mean(axis=0)
    centered = grads - mean_grad
    variance_trace = float(np.mean(np.sum(centered ** 2, axis=1)))
    mean_grad_norm_sq = float(np.sum(mean_grad ** 2))
    normalized_variance = variance_trace / (mean_grad_norm_sq + 1e-12)
    stats = {
        "normalized_variance": normalized_variance,
        "variance_trace": variance_trace,
        "variance_trace_per_dim": variance_trace / dim,
        "mean_grad_norm": float(np.sqrt(mean_grad_norm_sq)),
    }
    return mean_grad, stats


def redraw_grad_stats(sampler_fn, x0_sample, target_samples, n_redraws, base_seed, mmd_loss, experiment_utils, device):
    """Redraw `sampler_fn(x_leaf) -> y_samples` n_redraws times at this ONE
    frozen x0_sample, against the fixed target_samples, backpropagating the
    MMD loss each time to get one gradient draw -- then hand the stacked
    gradients to grad_stats_from_draws. sampler_fn is either method's inner
    sampler (LGD's DDIM unroll or LGD-CM's multistep sample).
    Returns (mean_grad, stats, grads) -- grads is the raw
    [n_redraws, dim] array, for callers that want to keep it (e.g. for a
    per-trial dump in the output JSON); most callers can just ignore it.
    """
    grads = []
    for r in range(n_redraws):
        experiment_utils.set_run_seed(base_seed, r)
        x_leaf = x0_sample.clone().detach().to(device).requires_grad_(True)
        y_samples = sampler_fn(x_leaf)
        loss = mmd_loss(y_samples, target_samples)
        grad = torch.autograd.grad(loss, x_leaf)[0]
        grads.append(grad.detach().cpu().numpy().copy())

    grads = np.stack(grads, axis=0)
    mean_grad, stats = grad_stats_from_draws(grads)
    return mean_grad, stats, grads


def dist_to_ref_stats(mean_grad, grad_ref, grad_ref_norm):
    """Distance from an estimator's mean gradient to the TRUE reference
    gradient -- separates "did the estimate collapse/drift" (dist_to_ref_normalized
    -> 1) from "did it converge to the real value" (-> 0), which normalized_variance
    alone can't tell (a low-variance estimator can still be consistently wrong).
    Returns (dist_to_ref, dist_to_ref_normalized).
    """
    dist_to_ref = float(np.linalg.norm(mean_grad - grad_ref))
    dist_to_ref_normalized = dist_to_ref / (grad_ref_norm + 1e-12)
    return dist_to_ref, dist_to_ref_normalized
