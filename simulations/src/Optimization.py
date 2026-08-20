
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
                 zeta=1.0,           # <-- guidance strength (ζ)
                 reuse_frac=0.0,     # <-- NEW: fraction of `nsamples` carried over from the previous step
                 momentum=0.0,       # <-- NEW: β1, EMA weight on past guidance gradients, 0 = no smoothing (old behavior)
                 beta2=None,         # <-- NEW: set (e.g. 0.999) to switch on full Adam-style adaptive scaling
                 adam_eps=1e-8,      # <-- NEW: denominator epsilon for the Adam-style update, only used if beta2 is set
                 return_history=False):   # <-- NEW: also return per-step diagnostics (grad norm, loss, n_reuse)

    mmd_loss = MMDLoss(kernel=RBF())
    best_mmd_loss = float("inf")
    best_x0_sample = None

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)
    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)

    # Buffer of previous step's target_samples per MC index j, used when reuse_frac > 0.
    # Always detached: the graph they were produced on is freed after the previous
    # step's backward(), so they contribute to the MMD *value* but zero gradient.
    prev_target_samples = [None] * num_x_t
    grad_ema = None      # first-moment (β1) EMA buffer over guidance gradients
    grad_sq_ema = None   # second-moment (β2) EMA buffer, only used in Adam mode (beta2 is not None)
    adam_step = 0        # counts applied updates, for Adam-style bias correction
    history = []

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
        n_reuse, n_new = 0, nsamples
        for j in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)

            if prev_target_samples[j] is not None:
                # Buffer now holds only the previous step's fresh draws (one step
                # old), so it can hold fewer than nsamples rows when reuse_frac > 0.5
                # (n_new_prev = nsamples * (1 - reuse_frac) < n_reuse needed here).
                # Clamp so target_samples always has exactly nsamples rows.
                n_reuse = min(int(round(reuse_frac * nsamples)), prev_target_samples[j].shape[0])
            else:
                n_reuse = 0
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

            # Buffer only the freshly generated portion, not the full reused+new
            # concatenation — otherwise staleness compounds across steps instead
            # of staying bounded to "one step old".
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

        # momentum == 0: no smoothing, identical to the original code (grad_to_apply == grad).
        # momentum > 0, beta2 is None: plain EMA/Polyak momentum on the gradient only.
        # momentum > 0, beta2 set: full Adam-style update — first AND second moment
        #   EMAs with bias correction, so each coordinate's step is also rescaled by
        #   its own recent gradient magnitude (adaptive), not just direction-smoothed.
        if momentum > 0.0 and beta2 is not None:
            adam_step += 1
            grad_ema = torch.zeros_like(grad) if grad_ema is None else grad_ema
            grad_sq_ema = torch.zeros_like(grad) if grad_sq_ema is None else grad_sq_ema
            grad_ema = momentum * grad_ema + (1 - momentum) * grad
            grad_sq_ema = beta2 * grad_sq_ema + (1 - beta2) * grad ** 2
            m_hat = grad_ema / (1 - momentum ** adam_step)
            v_hat = grad_sq_ema / (1 - beta2 ** adam_step)
            grad_to_apply = m_hat / (v_hat.sqrt() + adam_eps)
        elif momentum > 0.0:
            grad_ema = grad.clone() if grad_ema is None else momentum * grad_ema + (1 - momentum) * grad
            grad_to_apply = grad_ema
        else:
            grad_to_apply = grad

        if return_history:
            history.append({
                "t": t,
                "grad_norm": grad.norm().item(),
                "grad_ema_norm": grad_to_apply.norm().item(),
                "log_mean_exp_loss": log_mean_exp_loss.item(),
                "best_mmd_so_far": best_mmd_loss,
                "n_reuse": n_reuse,
                "n_new": n_new,
            })

        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - zeta * grad_to_apply

    condition = x_t.view(1, -1).repeat(nsamples, 1)
    target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
    if not CM:
        target_samples = target_samples[:, model_cond.condition_on:]
    mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
    final_loss = mmd_loss(target_samples, mog_samples)

    x_t_final = x_t.detach().clone()
    del x_t, condition, target_samples, mog_samples
    torch.cuda.empty_cache() if device == "cuda" else None

    if return_history:
        return x_t_final, x_t_final, final_loss.detach(), history
    return x_t_final, x_t_final, final_loss.detach()


