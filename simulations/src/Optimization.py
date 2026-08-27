
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
# Optimize LGD using Monte Carlo-based guidance
def optimize_LGD(model_uncond, model_cond, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
                 nsamples=250, num_x_t=3, loss="MMD", CM=False, device="cuda", FLAG=False,
                 zeta=1.0,               # <-- guidance strength (ζ)
                 backsel_k=None,         # <-- NEW: of `nsamples` target_samples, how many to backprop
                                         #     through for the guidance loss (None = all, original behavior)
                 backsel_rule="uniform", # <-- NEW: 'uniform' | 'witness' (MMD witness-function importance
                                         #     sampling against this step's mog_samples; see witness_utils.py)
                 witness_floor=0.3,      # <-- NEW: defensive-mixture floor for backsel_rule='witness'
                 backsel_replacement=False,  # <-- NEW: sample the backsel_k indices with/without replacement
                 backsel_generator=None,     # <-- NEW: optional torch.Generator for reproducible selection
                 normalize_by_k_frac=False,  # <-- NEW: rescale the applied gradient by 1/k_frac (k_frac =
                                             #     backsel_k/nsamples, Horvitz-Thompson-style) so gradient
                                             #     magnitude is comparable across different k_frac settings
                                             #     instead of scaling ~linearly with k_frac (MMDLoss averages,
                                             #     not sums, so each differentiable row otherwise contributes
                                             #     a roughly fixed amount regardless of how many others are
                                             #     attached). No-op when backsel_k is None. Default False
                                             #     preserves original (unnormalized) behavior.
                 return_history=False,       # <-- NEW: also return per-step diagnostics
                 diag_steps=None,            # <-- NEW: list of t-values at which to log the extra
                                             #     gradient-error / witness-scenario diagnostics below
                                             #     (requires return_history=True; ignored otherwise)
                 grad_ref_n=2000):           # <-- NEW: sample size for the TRUE/population reference
                                             #     gradient at diag_steps, drawn from the exact analytic
                                             #     conditional GMM (mu_list/Sigma_list/alpha -- known
                                             #     ground truth, not model_cond's approximation). Cheap:
                                             #     closed-form reparameterized Gaussian sampling, no
                                             #     network forward, so grad_ref_n can be large without
                                             #     real cost.

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

            # Witness-scenario diagnostics: computed on the FULL n target_samples,
            # before any subsampling, regardless of backsel_rule -- characterizes
            # how heterogeneous this step's mismatch is (see witness_utils.py).
            if diag_this_step:
                scores_diag = compute_witness_scores(target_samples.detach(), mog_samples)
                scenario_stats_list.append(witness_scenario_stats(scores_diag))

                # TRUE/population reference gradient: sample from the EXACT
                # analytic conditional GMM given this x0_sample (known
                # mu_list/Sigma_list/alpha -- the actual ground truth this
                # experiment was built from, not model_cond's learned
                # approximation of it), differentiably w.r.t. x0_sample (hence
                # x_t) via compute_conditionals/compute_alpha + the
                # reparameterized generate_mog_samples. No network forward is
                # involved, so grad_ref_n can be large (near-zero MC noise)
                # basically for free -- this is what "the real gradient" means
                # here: not another finite-n approximation, but (up to
                # grad_ref_n's own already-small MC noise) the true one.
                condi_mu, condi_sigma = dist_utils.compute_conditionals(
                    mu_list, Sigma_list, x0_sample.view(-1)
                )
                condi_mu = condi_mu.squeeze(-1)  # compute_conditionals leaves a spurious
                                                  # trailing dim of 1 from an internal reshape;
                                                  # harmless for this repo's 1D-target
                                                  # experiments (sample_univariate_gaussian
                                                  # squeezes internally too) but generate_mog_
                                                  # samples' is_multivariate check looks at
                                                  # xi.shape[-1], so drop it explicitly here.
                condi_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x0_sample.view(-1))
                ref_samples = generate_mog_samples(
                    grad_ref_n, condi_mu, condi_sigma, condi_alpha, device=device
                )
                loss_val_ref = mmd_loss(ref_samples, mog_samples)
                losses_ref.append(-loss_val_ref)

            # Backprop-subsampling: target_samples are already fully generated
            # (forward cost is fixed regardless of backsel_k here -- unlike the SD
            # pipeline there's no per-sample generation cost to save), so this
            # isolates the pure statistical effect of *which* samples carry
            # gradient on guidance quality. backsel_k=None (default) keeps every
            # row differentiable, identical to the original behavior.
            if backsel_k is not None:
                # Gradient-error diagnostic: the loss computed on the SAME raw
                # target_samples draw, before subsampling -- gives grad_full, the
                # "what would the full-batch gradient have been" reference to
                # compare the actual (subsampled) grad against.
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

        # Three SEPARATE autograd.grad calls (never one call over a list of
        # outputs, which would sum their gradients together): the actual
        # (possibly subsampled) update; the same-n full-batch comparison
        # (only when backsel_k is not None); the TRUE population reference
        # (only at diag_steps). All three come from the SAME underlying random
        # draws at this step -- only *what's fed to the loss* differs, so
        # differences in the resulting gradients isolate the selection effect.
        need_full = losses_full is not None
        need_ref = losses_ref is not None
        grad = torch.autograd.grad(log_mean_exp_loss, x_t, retain_graph=(need_full or need_ref))[0]

        # Horvitz-Thompson-style rescaling: with backsel_k differentiable rows out
        # of nsamples, the raw grad above scales roughly linearly with k_frac (see
        # the module-level note on normalize_by_k_frac). Dividing by k_frac makes
        # its magnitude comparable across different k_frac choices -- applied here,
        # before the diagnostics below, so grad_norm_error* reflect the normalized
        # gradient (the one actually used for the x_t update) against grad_full /
        # grad_ref, which are never subsampled and so are never rescaled.
        if normalize_by_k_frac and backsel_k is not None:
            k_frac = min(int(backsel_k), nsamples) / nsamples
            grad = grad / max(k_frac, 1e-12)

        grad_norm_error = None
        grad_full = None
        if need_full:
            log_mean_exp_loss_full = -torch.logsumexp(torch.stack(losses_full), dim=0) + math.log(num_x_t)
            grad_full = torch.autograd.grad(log_mean_exp_loss_full, x_t, retain_graph=need_ref)[0]
            grad_norm_error = (grad - grad_full).norm().item()

        grad_norm_error_vs_ref = None
        grad_full_norm_error_vs_ref = None
        if need_ref:
            log_mean_exp_loss_ref = -torch.logsumexp(torch.stack(losses_ref), dim=0) + math.log(num_x_t)
            grad_ref = torch.autograd.grad(log_mean_exp_loss_ref, x_t, retain_graph=False)[0]
            grad_norm_error_vs_ref = (grad - grad_ref).norm().item()
            if grad_full is not None:
                grad_full_norm_error_vs_ref = (grad_full - grad_ref).norm().item()

        if return_history:
            entry = {
                "t": t,
                "grad_norm": grad.norm().item(),
                "log_mean_exp_loss": log_mean_exp_loss.item(),
                "best_mmd_so_far": best_mmd_loss,
                "n_selected": int(step_backsel_info["mask"].sum().item())
                              if step_backsel_info is not None else nsamples,
            }
            if diag_this_step:
                # Average the per-j (num_x_t) scenario stats for a single
                # per-step number; the grad_norm_error* fields are already
                # single values (computed on the combined log-mean-exp loss
                # across all j's).
                for key in ("witness_std", "witness_skew_proxy", "ess_raw"):
                    entry[key] = sum(s[key] for s in scenario_stats_list) / len(scenario_stats_list)
                if grad_norm_error is not None:
                    entry["grad_norm_error"] = grad_norm_error
                entry["grad_norm_error_vs_ref"] = grad_norm_error_vs_ref
                if grad_full_norm_error_vs_ref is not None:
                    entry["grad_full_norm_error_vs_ref"] = grad_full_norm_error_vs_ref
            history.append(entry)
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - zeta * grad

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


