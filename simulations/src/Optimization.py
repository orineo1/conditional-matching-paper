
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
from witness_utils import apply_backsel, compute_witness_scores, witness_scenario_stats


def _reference_loss_term(x0_sample, mu_list, Sigma_list, alpha, mog_samples, grad_ref_n, device, mmd_loss):
    """
    The TRUE/population reference loss term for one j: sample grad_ref_n points from the
    EXACT analytic conditional GMM given x0_sample (known mu_list/Sigma_list/alpha, not
    model_cond's learned approximation), differentiably w.r.t. x0_sample, and score against
    mog_samples. No network forward -- cheap even at large grad_ref_n.
    """
    condi_mu, condi_sigma = dist_utils.compute_conditionals(mu_list, Sigma_list, x0_sample.view(-1))
    condi_mu = condi_mu.squeeze(-1)  # drop a spurious trailing dim; generate_mog_samples'
                                      # is_multivariate check looks at xi.shape[-1]
    condi_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x0_sample.view(-1))
    ref_samples = generate_mog_samples(grad_ref_n, condi_mu, condi_sigma, condi_alpha, device=device)
    loss_val_ref = mmd_loss(ref_samples, mog_samples)
    return -loss_val_ref


def _compute_step_gradients(log_mean_exp_loss, losses_full, losses_ref, x_t, num_x_t,
                             backsel_k, nsamples, normalize_by_k_frac):
    """
    Up to three SEPARATE autograd.grad calls (never one call over a list of outputs, which
    would sum their gradients together): the actual (possibly subsampled) update; the same-n
    full-batch comparison (only when losses_full was collected); the TRUE population reference
    (only when losses_ref was collected). All three come from the SAME underlying random draws
    at this step -- only *what's fed to the loss* differs, so differences in the resulting
    gradients isolate the selection effect.

    Returns a dict: grad, grad_full, grad_ref, grad_norm_error (||grad-grad_full||),
    grad_norm_error_vs_ref (||grad-grad_ref||), grad_full_norm_error_vs_ref
    (||grad_full-grad_ref||) -- entries are None where their inputs weren't collected.
    """
    need_full = losses_full is not None
    need_ref = losses_ref is not None
    grad = torch.autograd.grad(log_mean_exp_loss, x_t, retain_graph=(need_full or need_ref))[0]

    # Horvitz-Thompson-style rescaling: with backsel_k differentiable rows out of nsamples, the
    # raw grad above scales roughly linearly with k_frac (MMDLoss averages, not sums). Dividing
    # by k_frac makes its magnitude comparable across k_frac choices -- applied before the error
    # diagnostics below, so they reflect the normalized gradient (the one actually used for the
    # x_t update) against grad_full/grad_ref, which are never subsampled or rescaled.
    if normalize_by_k_frac and backsel_k is not None:
        k_frac = min(int(backsel_k), nsamples) / nsamples
        grad = grad / max(k_frac, 1e-12)

    result = {"grad": grad, "grad_full": None, "grad_ref": None,
              "grad_norm_error": None, "grad_norm_error_vs_ref": None,
              "grad_full_norm_error_vs_ref": None}

    if need_full:
        log_mean_exp_loss_full = -torch.logsumexp(torch.stack(losses_full), dim=0) + math.log(num_x_t)
        result["grad_full"] = torch.autograd.grad(log_mean_exp_loss_full, x_t, retain_graph=need_ref)[0]
        result["grad_norm_error"] = (grad - result["grad_full"]).norm().item()

    if need_ref:
        log_mean_exp_loss_ref = -torch.logsumexp(torch.stack(losses_ref), dim=0) + math.log(num_x_t)
        result["grad_ref"] = torch.autograd.grad(log_mean_exp_loss_ref, x_t, retain_graph=False)[0]
        result["grad_norm_error_vs_ref"] = (grad - result["grad_ref"]).norm().item()
        if result["grad_full"] is not None:
            result["grad_full_norm_error_vs_ref"] = (result["grad_full"] - result["grad_ref"]).norm().item()

    return result


def _build_history_entry(t, grad_results, log_mean_exp_loss, best_mmd_loss, step_backsel_info,
                          nsamples, diag_this_step, scenario_stats_list):
    """Assemble one per-step history entry: always the basics, plus (at diag steps) the
    scenario stats averaged over j and the grad-error fields from _compute_step_gradients."""
    grad = grad_results["grad"]
    entry = {
        "t": t,
        "grad_norm": grad.norm().item(),
        "log_mean_exp_loss": log_mean_exp_loss.item(),
        "best_mmd_so_far": best_mmd_loss,
        "n_selected": int(step_backsel_info["mask"].sum().item())
                      if step_backsel_info is not None else nsamples,
    }
    if diag_this_step:
        for key in ("witness_std", "witness_skew_proxy", "ess_raw"):
            entry[key] = sum(s[key] for s in scenario_stats_list) / len(scenario_stats_list)
        if grad_results["grad_norm_error"] is not None:
            entry["grad_norm_error"] = grad_results["grad_norm_error"]
        entry["grad_norm_error_vs_ref"] = grad_results["grad_norm_error_vs_ref"]
        if grad_results["grad_full_norm_error_vs_ref"] is not None:
            entry["grad_full_norm_error_vs_ref"] = grad_results["grad_full_norm_error_vs_ref"]
    return entry


# Optimize LGD using Monte Carlo-based guidance
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda", FLAG=False,
                 zeta=1.0,               # guidance strength (ζ); zeta=0 skips guidance entirely
                 use_inv_sqrt_alpha_scale=False,  # scale grad by 1/sqrt(model_uncond.alphas[t])
                                                   # instead of zeta (ignored when zeta == 0)
                 backsel_k=None,         # of `nsamples` target_samples, how many to backprop
                                         # through for the guidance loss (None = all)
                 backsel_rule="uniform", # 'uniform' | 'witness' (see witness_utils.py)
                 witness_floor=0.3,      # defensive-mixture floor for backsel_rule='witness'
                 backsel_replacement=False,  # sample backsel_k indices with/without replacement
                 backsel_generator=None,     # optional torch.Generator for reproducible selection
                 normalize_by_k_frac=False,  # rescale the applied gradient by 1/k_frac so its
                                             # magnitude is comparable across k_frac settings
                                             # (MMDLoss averages, so raw grad otherwise scales
                                             # ~linearly with k_frac); no-op when backsel_k is None
                 return_history=False,       # also return per-step diagnostics
                 diag_steps=None,            # t-values at which to log the extra gradient-error /
                                             # witness-scenario diagnostics (needs return_history=True)
                 grad_ref_n=2000):           # sample size for the analytic population reference
                                             # gradient at diag_steps (no network forward -- cheap)

    mmd_loss = MMDLoss(kernel=RBF())
    best_mmd_loss = float("inf")
    best_x0_sample = None

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)
    pbar = tqdm(range(model_uncond.diffusion_steps - 1, 0, -1)) if FLAG else range(model_uncond.diffusion_steps - 1, 0, -1)
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

        diag_this_step = return_history and diag_steps is not None and t in diag_steps

        losses = []
        losses_full = [] if (diag_this_step and backsel_k is not None) else None
        losses_ref = [] if diag_this_step else None
        step_backsel_info = None
        scenario_stats_list = [] if diag_this_step else None
        for j in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            condition = x0_sample.view(1, -1).repeat(nsamples, 1)
            target_samples, _,_ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
            if not CM:
                target_samples = target_samples[:, model_cond.condition_on:]

            mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

            # Witness-scenario diagnostics: computed on the FULL n target_samples, before any
            # subsampling, regardless of backsel_rule -- characterizes how heterogeneous this
            # step's mismatch is (see witness_utils.py). Plus the TRUE reference loss term.
            if diag_this_step:
                scores_diag = compute_witness_scores(target_samples.detach(), mog_samples)
                scenario_stats_list.append(witness_scenario_stats(scores_diag))
                losses_ref.append(_reference_loss_term(
                    x0_sample, mu_list, Sigma_list, alpha, mog_samples, grad_ref_n, device, mmd_loss
                ))

            # Backprop-subsampling: target_samples are already fully generated (forward cost is
            # fixed regardless of backsel_k here), so this isolates the pure statistical effect
            # of *which* samples carry gradient. backsel_k=None (default) keeps every row
            # differentiable, identical to the original behavior.
            if backsel_k is not None:
                # loss on the SAME raw target_samples draw, before subsampling -- gives
                # grad_full, the "what would the full-batch gradient have been" reference.
                if diag_this_step:
                    loss_val_full = mmd_loss(target_samples, mog_samples)
                    losses_full.append(-loss_val_full)
                target_samples, step_backsel_info = apply_backsel(
                    target_samples, mog_samples, backsel_k, rule=backsel_rule,
                    witness_floor=witness_floor, generator=backsel_generator,
                    replacement=backsel_replacement,
                )

            loss_val = mmd_loss(target_samples, mog_samples)
            losses.append(-loss_val)
            if FLAG:
                pbar.set_description(f"Step {t} | x_t:{x_t}|x0_sample:{x0_sample} | loss_val:{loss_val}|log_mean_exp_loss: {log_mean_exp_loss.item():.4f}")

            if loss_val.item() < best_mmd_loss:
                best_mmd_loss = loss_val.item()
                best_x0_sample = x0_sample.detach().clone()

        log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)

        grad_results = _compute_step_gradients(
            log_mean_exp_loss, losses_full, losses_ref, x_t, num_x_t,
            backsel_k, nsamples, normalize_by_k_frac,
        )
        grad = grad_results["grad"]

        if return_history:
            history.append(_build_history_entry(
                t, grad_results, log_mean_exp_loss, best_mmd_loss, step_backsel_info,
                nsamples, diag_this_step, scenario_stats_list,
            ))

        if use_inv_sqrt_alpha_scale:
            step_scale = 1.0 / torch.sqrt(model_uncond.alphas[t].to(device))
        else:
            step_scale = zeta
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - step_scale * grad

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
