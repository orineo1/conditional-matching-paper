"""
Minimal MoG (Mixture of Gaussians) distribution utilities for compare-methods pipeline.
Extracted from GlobalConditional/dist_utils.py — only what D-Flow, LGD, LGD-CM need.
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily


# ── Sampling ──────────────────────────────────────────────────────────────────

def generate_mog_samples(n, means, variances, weights, kernel_func=None, device=None):
    """Differentiable MoG sampler (reparameterization trick)."""
    device = means[0].device if isinstance(means[0], torch.Tensor) else torch.device("cpu")
    weights_t = weights.to(device) if isinstance(weights, torch.Tensor) else torch.tensor(weights, device=device)
    mix = Categorical(probs=weights_t)
    comp = MultivariateNormal(
        torch.stack([m.to(device) for m in means]),
        torch.stack([s.to(device) for s in variances]),
    )
    gmm = MixtureSameFamily(mix, comp)
    samples = gmm.sample((n,))
    if kernel_func is not None:
        samples = kernel_func(samples)
    return samples


def generate_mog_samples_not_differentiable(n, means, variances, weights, kernel_func=None, device=None):
    """Non-differentiable MoG sampler (faster, numpy-backed)."""
    device = means[0].device if isinstance(means[0], torch.Tensor) else torch.device("cpu")
    weights_np = weights.cpu().numpy() if isinstance(weights, torch.Tensor) else np.array(weights)
    weights_np = weights_np / weights_np.sum()

    k = len(means)
    counts = np.random.multinomial(n, weights_np)
    samples = []
    for i, cnt in enumerate(counts):
        if cnt == 0:
            continue
        mu = means[i].cpu().numpy() if isinstance(means[i], torch.Tensor) else np.array(means[i])
        sigma = variances[i].cpu().numpy() if isinstance(variances[i], torch.Tensor) else np.array(variances[i])
        s = np.random.multivariate_normal(mu, sigma, cnt)
        samples.append(s)

    result = np.vstack(samples)
    np.random.shuffle(result)
    result_t = torch.tensor(result, dtype=torch.float32, device=device)
    if kernel_func is not None:
        result_t = kernel_func(result_t)
    return result_t


# ── Conditional distribution utilities ───────────────────────────────────────

def compute_conditionals(mu_list, Sigma_list, x_cond):
    """
    Compute conditional means/covariances p(y | x = x_cond) for a MoG.
    Assumes first coordinate(s) are x, remaining are y.
    """
    x_cond = x_cond.float().view(-1) if isinstance(x_cond, torch.Tensor) else torch.tensor(x_cond, dtype=torch.float32).view(-1)
    d_x = x_cond.shape[0]

    mu_list_cond = []
    Sigma_list_cond = []

    for mu, Sigma in zip(mu_list, Sigma_list):
        mu = mu.float()
        Sigma = Sigma.float()

        mu_x = mu[:d_x]
        mu_y = mu[d_x:]
        Sigma_xx = Sigma[:d_x, :d_x]
        Sigma_yy = Sigma[d_x:, d_x:]
        Sigma_yx = Sigma[d_x:, :d_x]

        Sigma_xx_inv = torch.linalg.inv(Sigma_xx + 1e-6 * torch.eye(d_x, device=Sigma_xx.device))
        mu_y_given_x = mu_y + Sigma_yx @ Sigma_xx_inv @ (x_cond - mu_x)
        Sigma_y_given_x = Sigma_yy - Sigma_yx @ Sigma_xx_inv @ Sigma_yx.T
        # Ensure PSD
        Sigma_y_given_x = (Sigma_y_given_x + Sigma_y_given_x.T) / 2 + 1e-6 * torch.eye(Sigma_y_given_x.shape[0],
                                                                                       device=Sigma_y_given_x.device)

        mu_list_cond.append(mu_y_given_x)
        Sigma_list_cond.append(Sigma_y_given_x)

    return mu_list_cond, Sigma_list_cond


def compute_alpha(mu_list, Sigma_list, alpha, x_cond):
    """Compute conditional mixture weights p(component k | x = x_cond)."""
    x_cond = x_cond.float().view(-1) if isinstance(x_cond, torch.Tensor) else torch.tensor(x_cond, dtype=torch.float32).view(-1)
    d_x = x_cond.shape[0]

    log_probs = []
    for mu, Sigma in zip(mu_list, Sigma_list):
        mu = mu.float()
        Sigma = Sigma.float()
        mu_x = mu[:d_x]
        Sigma_xx = Sigma[:d_x, :d_x] + 1e-6 * torch.eye(d_x, device=Sigma.device)
        dist = MultivariateNormal(mu_x, Sigma_xx)
        log_probs.append(dist.log_prob(x_cond))

    log_probs = torch.stack(log_probs)
    alpha_t = alpha.float() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
    log_alpha = torch.log(alpha_t + 1e-12)
    log_weights = log_alpha + log_probs
    log_weights -= log_weights.logsumexp(0)
    return log_weights.exp()


def filter_and_normalize(mu_list_cond, Sigma_list_cond, alpha_cond, threshold=0.01):
    """Keep only components with weight > threshold and renormalize."""
    alpha_np = alpha_cond.cpu().numpy() if isinstance(alpha_cond, torch.Tensor) else np.array(alpha_cond)
    keep = [i for i, a in enumerate(alpha_np) if a > threshold]
    if not keep:
        keep = [int(np.argmax(alpha_np))]

    mu_out = [mu_list_cond[i] for i in keep]
    Sigma_out = [Sigma_list_cond[i] for i in keep]
    alpha_out = alpha_cond[keep]
    alpha_out = alpha_out / alpha_out.sum()
    return mu_out, Sigma_out, alpha_out


def mog_covariance(mu_list, Sigma_list, alpha):
    """Full MoG covariance matrix (for SWD normalization)."""
    alpha_t = alpha.float() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
    mu_stack = torch.stack([m.float() for m in mu_list])   # [K, d]
    mean = (alpha_t.unsqueeze(1) * mu_stack).sum(0)         # [d]

    d = mean.shape[0]
    cov = torch.zeros(d, d)
    for k, (mu, Sigma, a) in enumerate(zip(mu_list, Sigma_list, alpha_t)):
        diff = (mu.float() - mean).unsqueeze(1)             # [d, 1]
        cov += a * (Sigma.float() + diff @ diff.T)
    return cov


def warpper_L1_distance(x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights):
    """L1 distance between predicted x and the 'optimal' x (mode of marginal p(x))."""
    # Optimal x = weighted mean of component x-means
    alpha_t = alpha.float() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
    x_pred = x_pred.float().view(-1) if isinstance(x_pred, torch.Tensor) else torch.tensor(x_pred, dtype=torch.float32).view(-1)
    d_x = x_pred.shape[0]

    x_means = torch.stack([m.float()[:d_x] for m in mu_list])   # [K, d_x]
    optimal = (alpha_t.unsqueeze(1) * x_means).sum(0)
    return (x_pred - optimal).abs().sum().item()
