
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
                 zeta=1.0,          # <-- NEW: guidance strength (ζ)
                 reuse_frac=0.0,    # <-- NEW: fraction of `nsamples` carried over from the previous step
                 adamdps=False):    # <-- NEW: AdamDPS gradient stabilization (arXiv:2603.16797), fixed beta1=0.9, beta2=0.999

    mmd_loss = MMDLoss(kernel=RBF())
    best_mmd_loss = float("inf")
    best_x0_sample = None

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)
    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)

    prev_target_samples = [None] * num_x_t   # one-step-old sample buffer, used when reuse_frac > 0
    ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
    m, v, adam_step = None, None, 0          # AdamDPS moment state, used when adamdps=True

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

            # reuse_frac: carry over up to reuse_frac*nsamples samples from the previous
            # step's fresh draws (clamped to what's available); rest generated fresh here.
            n_reuse = min(int(round(reuse_frac * nsamples)), prev_target_samples[j].shape[0]) \
                if prev_target_samples[j] is not None else 0
            n_new = nsamples - n_reuse

            if n_new > 0:
                condition = x0_sample.view(1, -1).repeat(n_new, 1)
                new_samples, _, _ = model_cond.sample(nsamples=n_new, condition_x=condition, device=device)
                if not CM:
                    new_samples = new_samples[:, model_cond.condition_on:]
            else:
                new_samples = None

            if n_reuse > 0:
                reused = prev_target_samples[j][:n_reuse]
                target_samples = torch.cat([reused, new_samples], dim=0) if new_samples is not None else reused
            else:
                target_samples = new_samples
            prev_target_samples[j] = new_samples.detach().clone() if new_samples is not None else None

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

        if adamdps:
            adam_step += 1
            m = grad.clone() if m is None else ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad
            v = grad ** 2 if v is None else ADAM_BETA2 * v + (1 - ADAM_BETA2) * grad ** 2
            m_hat = m / (1 - ADAM_BETA1 ** adam_step)
            v_hat = v / (1 - ADAM_BETA2 ** adam_step)
            grad = m_hat / (v_hat.sqrt() + ADAM_EPS)

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


