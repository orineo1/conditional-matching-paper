
import importlib
import Diffusion
from LossFunctions import MMDLoss,RBF
from dist_utils import generate_mog_samples,generate_mog_samples_not_differentiable
import ConsistencyModels
import  FlowMatching
from ConsistencyModels import ConsistencyModel
import dist_utils
import torch.nn.functional as F
import torch
import torch.optim as optim_cond
from tqdm.notebook import tqdm

######### Diffusion
def optimize_DPS(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda"):
    mmd_loss = MMDLoss(kernel=RBF())

    # Move initial tensor to CUDA and set requires_grad
    x_t = torch.randn(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.detach().clone().to(device).requires_grad_(True)
    optimizer = optim_cond.Adam([x_t], lr=0.05)
    # Move mog distribution parameters to device as well, if they are tensors
    if isinstance(mog_means, torch.Tensor):
        mog_means = mog_means.to(device)

    if isinstance(mog_variances, torch.Tensor):
        mog_variances = mog_variances.to(device)

    if isinstance(weights, torch.Tensor):
        weights = weights.to(device)

    best_mmd_loss = float("inf")
    best_pred_x0 = None

    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1))
    for t in pbar:


        # Make sure the model_uncond sampling runs on device
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)

        condition = pred_x0.view(1, -1).repeat(nsamples, 1).to(device)
        target_samples, _ = model_cond.sample(nsamples=nsamples, condition_x=condition,device=device)
        target_samples = target_samples.to(device)

        if not CM:
            target_samples = target_samples[:, model_cond.condition_on:]

        samples = generate_mog_samples(nsamples, mog_means, mog_variances, weights, tau=10.0).to(device)

        optimizer.zero_grad()
        loss_val = mmd_loss(target_samples, samples)
        loss_val.backward()

        current_var = model_uncond.betas[t].to(device)
        grad = x_t.grad.clone() * current_var

        pbar.set_description(
            f"Step {t} | pred_x0_model: {pred_x0} |  pred_x0 loss: {loss_val.item():.3f}"
        )

        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone().to(device) - grad
        x_t.requires_grad_()


        if loss_val.item() < best_mmd_loss:
            best_mmd_loss = loss_val.item()
            best_pred_x0 = pred_x0.detach().clone()

    x_t_final = x_t.detach().clone()

    del x_t,optimizer, grad, x_t_minus_1, pred_x0, condition, target_samples, loss_val
    torch.cuda.empty_cache() if device == "cuda" else None

    return x_t_final, best_pred_x0, best_mmd_loss


#### Optimize LGD

import math
import torch
from torch import optim
from tqdm import tqdm
# Optimize LGD using Monte Carlo-based guidance
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda",FLAG=False):

    mmd_loss = MMDLoss(kernel=RBF())  # Assumes MMDLoss and RBF are defined elsewhere
    best_mmd_loss = float("inf")
    best_x0_sample = None

    # Initialize starting point (change to match shape if needed)
    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)
    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)

    for t in pbar:
        x_t = x_t.detach().clone().requires_grad_(True)

        optimizer = optim.Adam([x_t], lr=0.05)
        optimizer.zero_grad()

        # DDIM step (get x_{t-1} and predicted x0)
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
        current_var = model_uncond.betas[t].to(device)  # Use unconditional model's betas
        r_t = current_var / torch.sqrt(1 + current_var ** 2)
        log_mean_exp_loss=torch.tensor([0])
        # Monte Carlo estimation of expected loss over x0 ~ N(pred_x0, r_t^2 * I)
        losses = []
        for j in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            condition = x0_sample.view(1, -1).repeat(nsamples, 1)
            target_samples, _,_ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
            if not CM:
                target_samples = target_samples[:, model_cond.condition_on:]

            mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
            loss_val = mmd_loss(target_samples, mog_samples)
            losses.append(-loss_val)  # negative for log-mean-exp
            if FLAG:
                pbar.set_description(f"Step {t} | x_t:{x_t}|x0_sample:{x0_sample} | loss_val:{loss_val}|log_mean_exp_loss: {log_mean_exp_loss.item():.4f}")

            if loss_val.item() < best_mmd_loss:
                best_mmd_loss = loss_val.item()
                best_x0_sample = x0_sample.detach().clone()


        # Compute final log-mean-exp loss
        log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        log_mean_exp_loss.backward()

        # Manual gradient update (avoiding optimizer.step to stay DDIM-compatible)
        grad = x_t.grad.clone()
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - grad

    # compute the final loss - proxy for the real loss
    condition = x_t.view(1, -1).repeat(nsamples, 1)
    target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
    if not CM:
        target_samples = target_samples[:, model_cond.condition_on:]
    mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
    final_loss = mmd_loss(target_samples, mog_samples)


    # Final output
    x_t_final = x_t.detach().clone()
    del x_t, condition, target_samples, mog_samples
    torch.cuda.empty_cache() if device == "cuda" else None

    return x_t_final, x_t_final, final_loss.detach()


def optimize_LGD_latent(model_uncond, model_cond, mog_means, mog_variances, weights,scaler2,encoder2, mu_list, Sigma_list, alpha,
               nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda", FLAG=False,factor_for_final_loss_calc=4):

  mmd_loss = MMDLoss(kernel=RBF())
  best_mmd_loss = float("inf")
  best_x0_sample = None

  # Initialize starting point
  # x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
  x_t = torch.randn(model_uncond.nfeatures, device=device, requires_grad=True)

  x_t = x_t.unsqueeze(0)
  pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)

  for t in pbar:
      x_t = x_t.detach().clone().requires_grad_(True)

      # optimizer = optim.AdamW([x_t], lr=.05, weight_decay=0.01)
      optimizer = optim.Adam([x_t], lr=0.1, betas=(0.5, 0.999), eps=1e-4)

      optimizer.zero_grad()

      # DDIM step (get x_{t-1} and predicted x0)
      x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
      current_var = model_uncond.betas[t].to(device)
      r_t = current_var / torch.sqrt(1 + current_var ** 2)

      log_mean_exp_loss = torch.tensor([0])
      # Monte Carlo estimation of expected loss over x0 ~ N(pred_x0, r_t^2 * I)
      losses = []
      for j in range(num_x_t):
          x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
          condition = x0_sample.view(1, -1).repeat(nsamples, 1)
          target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
          if not CM:
              target_samples = target_samples[:, model_cond.condition_on:]

          # Generate MoG samples in encoded space
          mog_samples_raw = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
          mog_samples_scaled = torch.tensor(scaler2.transform(mog_samples_raw.cpu().numpy()), device=device, dtype=torch.float32)
          mog_samples = encoder2(mog_samples_scaled)

          loss_val = mmd_loss(target_samples, mog_samples)
          losses.append(-loss_val)  # negative for log-mean-exp

          if loss_val.item() < best_mmd_loss:
              best_mmd_loss = loss_val.item()
              best_x0_sample = x0_sample.detach().clone()

      # Compute final log-mean-exp loss
      log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
      log_mean_exp_loss.backward()

      # Optimizer step
      optimizer.step()

      if FLAG:
          grad_norm = torch.norm(x_t.grad).item() if x_t.grad is not None else 0
          pbar.set_description(f"Step {t}| grad_norm:{grad_norm:.6f} | x_t:{x_t}|x0_sample:{x0_sample} | loss_val:{loss_val:.6f}|log_mean_exp_loss: {log_mean_exp_loss.item():.4f}")

  # compute the final loss - proxy for the real loss
  condition = x_t.view(1, -1).repeat(nsamples, 1)
  target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
  mog_samples_raw = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
  mog_samples_scaled = torch.tensor(scaler2.transform(mog_samples_raw.cpu().numpy()), device=device, dtype=torch.float32)
  mog_samples = encoder2(mog_samples_scaled)
  final_loss =  mmd_loss(target_samples, mog_samples)

  # Final output
  x_t_final = x_t.detach().clone()
  torch.cuda.empty_cache() if device == "cuda" else None
  return x_t_final, x_t_final, best_mmd_loss, best_x0_sample,final_loss

######### FlowMatching

#### DFlow
def find_samples_to_kl(numb_points, mog_means, mog_variances, weights, device="cuda"):
    if isinstance(mog_means, (list, tuple)):
        mog_means_tensor = torch.stack(mog_means, dim=0)
    elif isinstance(mog_means, torch.Tensor):
        mog_means_tensor = mog_means.unsqueeze(0) if mog_means.dim() == 0 else mog_means
    else:
        raise TypeError("mog_means must be a list, tuple, or tensor")

    if mog_means_tensor.dim() > 2 and mog_means_tensor.shape[-1] == 1:
        mog_means_tensor = mog_means_tensor.squeeze(-1)

    mog_means_tensor = mog_means_tensor.to(device)
    mog_samples = generate_mog_samples(numb_points, mog_means, mog_variances, weights).to(device)

    if mog_samples.dim() < mog_means_tensor.dim():
        mog_means_tensor = mog_means_tensor.squeeze(-1)

    Target = torch.cat([mog_means_tensor, mog_samples], dim=0).to(torch.float32)
    return Target





def compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method, mog_means, mog_variances, weights, device):
    if loss_method == "MMD":
        mmd_loss = MMDLoss(kernel=RBF())

        # Create conditioning tensor on device
        x_cond = x_1.reshape(-1).to(device) * torch.ones(
            samples.shape[0], vf_y_cond_x.condition_on, dtype=torch.float32, device=device
        )
        target_samples = vf_y_cond_x.sample(samples.shape[0], x_cond)
        target_samples = target_samples.to(device)
        samples = samples.to(device)

        loss = mmd_loss(target_samples, samples)

    else:
        raise ValueError(f"Unknown loss method: {loss_method}")

    return loss

def optimize_DFLOW(vf_y_cond_x, vf_X, device, mog_means, mog_variances, weights,
                   max_iter=100, FLAG=False, n_sample=50, loss_method="MMD", line_search_fn=None, REGULARIZE=False,
                    vf_x_cond_y=None):
    # Initialize x on the correct device with requires_grad
    x = torch.randn((1, vf_X.nfeatures), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn=line_search_fn)

    # Generate MoG samples on device
    samples = generate_mog_samples_not_differentiable(n_sample, mog_means, mog_variances, weights).to(device)

    best_x, best_loss = None, float('inf')

    def closure():
        optimizer.zero_grad()
        x_1 = vf_X.sample(num_sample=1, y=None, x_init=x)

        loss = compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method,
                            mog_means, mog_variances, weights, device)

        nonlocal best_x, best_loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()

        if FLAG:
            print(f"x_0: {x.detach().cpu().numpy()} | x_1: {x_1} | loss: {loss.item():.6f}")

        if REGULARIZE:
            norm_x = x.norm(p=2)  # ‖x‖
            d = x.shape[1]

            R = (d - 1) * torch.log(norm_x + 1e-6) - 0.5 * norm_x ** 2
            loss = loss + -1. * R

        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    best_x1 = vf_X.sample(num_sample=1, y=None, x_init=best_x.to(device))
    # compute the final loss - proxy for the real loss
    with torch.no_grad():
        final_loss = compute_loss(vf_y_cond_x, vf_X, best_x1, samples, loss_method,
                                  mog_means, mog_variances, weights, device)

    del optimizer, samples, x, best_x
    torch.cuda.empty_cache() if device == "cuda" else None

    return best_x1, final_loss


#
#
# def optimize_DFLOW(vf_y_cond_x, vf_X, device, mog_means, mog_variances, weights,
#                    max_iter=100, FLAG=False, n_sample=50, loss_method="MMD",
#                    line_search_fn=None, REGULARIZE=False, crazy_condition=False, vf_x_cond_y=None,
#                    max_restarts=3, restart_loss_threshold=0.5):
#     for attempt in range(max_restarts):
#         # Initialize x with requires_grad=True
#         x = torch.randn((1, vf_X.nfeatures), dtype=torch.float32, device=device, requires_grad=True)
#         optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn=line_search_fn)
#
#         samples = generate_mog_samples(n_sample, mog_means, mog_variances, weights, tau=10.0).to(device)
#         best_x, best_loss = None, float('inf')
#
#         def closure():
#             optimizer.zero_grad()
#             if crazy_condition:
#                 condition = dist_utils.generate_mog_samples(1, mog_means, mog_variances, weights, tau=10.0).to(torch.float32).to(device)
#                 x_cond = condition.reshape(-1).to(device) * torch.ones(1, vf_x_cond_y.condition_on, dtype=torch.float32, device=device)
#                 x_1 = vf_x_cond_y.sample(1, x_cond)
#             else:
#                 x_1 = vf_X.sample(num_sample=1, y=None, x_init=x)
#
#             loss = compute_loss(vf_y_cond_x, vf_X, x_1, samples, loss_method,
#                                 mog_means, mog_variances, weights, device)
#
#             nonlocal best_x, best_loss
#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 best_x = x.detach().clone()
#
#             if FLAG:
#                 print(f"Attempt {attempt+1} | x_0: {x.detach().cpu().numpy()} | x_1: {x_1} | loss: {loss.item():.6f}")
#             if REGULARIZE:
#                 norm_x = x.norm(p=2)
#                 d = x.shape[1]
#                 R = (d - 1) * torch.log(norm_x + 1e-6) - 0.5 * norm_x ** 2
#                 loss = loss - R  # Encourage Gaussian norm
#
#             loss.backward(retain_graph=True)
#             return loss
#
#         optimizer.step(closure)
#
#         # If success, break early
#         if best_loss <= restart_loss_threshold:
#             break
#         else:
#             if FLAG:
#                 print(f"Restarting (loss={best_loss:.4f} > {restart_loss_threshold})")
#
#         # Cleanup before retry
#         del optimizer, samples, x, best_x
#         torch.cuda.empty_cache() if device == "cuda" else None
#
#     # Final best sample
#     best_x1 = vf_X.sample(num_sample=1, y=None, x_init=best_x.to(device))
#     return best_x1, best_loss
