
import importlib
import Diffusion
from LossFunctions import MMDLoss,RBF
from dist_utils import generate_mog_samples,generate_mog_samples_not_differentiable
import ConsistencyModels
import dist_utils
import torch.nn.functional as F
import torch
import torch.optim as optim_cond
from tqdm.notebook import tqdm

#### Optimize LGD

import math
import torch
from torch import optim
from tqdm import tqdm
# Optimize LGD using Monte Carlo-based guidance
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda", FLAG=False,
                 zeta=1.0):          # <-- NEW: guidance strength (ζ)

    mmd_loss = MMDLoss(kernel=RBF())
    best_mmd_loss = float("inf")
    best_x0_sample = None

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)
    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)

    for t in pbar:
        x_t = x_t.detach().clone().requires_grad_(True)

        optimizer = optim.Adam([x_t], lr=0.05)
        optimizer.zero_grad()

        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
        current_var = model_uncond.betas[t].to(device)
        r_t = current_var / torch.sqrt(1 + current_var ** 2)
        log_mean_exp_loss = torch.tensor([0])

        # Skip guidance when zeta=0 (pure prior sampling)
        if zeta == 0.0:
            with torch.no_grad():
                x_t = x_t_minus_1.detach().clone()
            continue

        losses = []
        for j in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            condition = x0_sample.view(1, -1).repeat(nsamples, 1)
            target_samples, _,_ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
            if not CM:
                target_samples = target_samples[:, model_cond.condition_on:]

            mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
            loss_val = mmd_loss(target_samples, mog_samples)
            losses.append(-loss_val)
            if FLAG:
                pbar.set_description(f"Step {t} | x_t:{x_t}|x0_sample:{x0_sample} | loss_val:{loss_val}|log_mean_exp_loss: {log_mean_exp_loss.item():.4f}")

            if loss_val.item() < best_mmd_loss:
                best_mmd_loss = loss_val.item()
                best_x0_sample = x0_sample.detach().clone()

        log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        log_mean_exp_loss.backward()

        grad = x_t.grad.clone()
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - zeta * grad   # <-- only change: zeta * grad

    condition = x_t.view(1, -1).repeat(nsamples, 1)
    target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
    if not CM:
        target_samples = target_samples[:, model_cond.condition_on:]
    mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
    final_loss = mmd_loss(target_samples, mog_samples)

    x_t_final = x_t.detach().clone()
    del x_t, condition, target_samples, mog_samples
    torch.cuda.empty_cache() if device == "cuda" else None

    return x_t_final, x_t_final, final_loss.detach()


