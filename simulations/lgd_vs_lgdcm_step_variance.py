#!/usr/bin/env python
"""
lgd_vs_lgdcm_step_variance.py — per-step gradient variance/accuracy of LGD's
inner sampler (K-step DDIM unroll through the conditional diffusion model)
vs. LGD-CM's inner sampler (the consistency model's own ~14-step multistep
sampling procedure -- see ConsistencyModeliCT.sample: far fewer, coarser
jump-and-refine steps than LGD's fine-grained DDIM unroll, not a single
network call), along real
optimization trajectories rather than a single hand-picked point.

Extends gradient_variance_vs_unroll_depth.py's "freeze a state, redraw the
sampler's noise many times" methodology (same normalized_variance and
dist_to_ref_normalized metrics, same true/population reference-gradient
machinery) across EVERY step of one or more full trajectories, comparing the
two inner samplers directly instead of sweeping unroll depth K on one method.

Design:
  - States are captured from --n_trajectories independent UNGUIDED (zeta=0)
    trajectories of model_uncond, one state per outer diffusion step (every
    --step_stride-th step) -- unguided so the states themselves don't already
    depend on which inner sampler produced them (see
    backsel_state_gradient_variance.py on other branches for the same
    convention). Each trajectory uses a different seed.
  - At each captured (trajectory, step, x0_sample): draw --n_redraws times,
    independently for each of the two inner samplers:
      LGD:    a --k_lgd-step DDIM unroll through model_cond (default: the
              full schedule, matching what real LGD guidance actually runs
              each outer step).
      LGD-CM: one draw via the consistency model's own multistep sampling
              (ConsistencyModeliCT.sample -- ~14 network calls at default
              `ts`, each a coarse jump-and-refine step, not one forward pass).
    against the SAME fixed target-sample set (drawn once per state from the
    exact analytic conditional GMM at that state's x -- not model_cond's own
    approximation), and compute the MMD loss + its gradient w.r.t. x.
  - Per (trajectory, step, method): normalized_variance = Var(grad)/||mean_grad||^2
    across the redraws, and dist_to_ref_normalized = ||mean_grad - grad_ref|| /
    ||grad_ref||, where grad_ref is the TRUE population gradient at that state
    (closed-form, no network forward -- see gradient_variance_vs_unroll_depth.py).
  - Averaged (mean + std + SEM) across --n_trajectories at each step index,
    separately for LGD and LGD-CM -- this is the "run e.g. 10 trajectories,
    report the mean of each metric per step" output; per-trajectory raw rows
    are also kept in the output JSON so nothing is thrown away.

Cost warning: this is expensive by construction -- n_trajectories x
n_steps_sampled x n_redraws x 2 methods forward+backward passes, and LGD's
K-step unroll (default: the full schedule) is itself K network calls per
redraw. Use --smoke first, and raise --step_stride / lower --n_redraws /
--n_trajectories before scaling up.

Usage:
    python lgd_vs_lgdcm_step_variance.py --experiment_name 10D_cond_1D --smoke
    python lgd_vs_lgdcm_step_variance.py --experiment_name 10D_cond_1D \
        --n_trajectories 10 --step_stride 10 --n_redraws 30
"""
import os
import sys
import json
import math
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "src"))

ARCH = {
    "2D_cond_1D":  dict(nblocks=3, nunits=128, diffusion_steps=100, condition_on=1),
    "5D_cond_1D":  dict(nblocks=6, nunits=512, diffusion_steps=100, condition_on=4),
    "10D_cond_1D": dict(nblocks=8, nunits=512, diffusion_steps=100, condition_on=9),
}


def ddim_sample_kstep(model, nsamples, condition_x, K, device):
    """Differentiable, deterministic (eta=0) DDIM sampling from `model`, using
    an evenly-spaced K-step subsequence of its trained noise schedule
    (accelerated DDIM respacing). Gradients flow from the output back to
    `condition_x`. Returns (full_sample, y_only, n_steps_actually_taken).
    Identical to gradient_variance_vs_unroll_depth.py's version -- kept local
    here so this script has no import-order dependency on that one.
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


def capture_trajectory_states(model_uncond, seed, step_stride, condition_on, device, experiment_utils):
    """One UNGUIDED (zeta=0) DDIM trajectory of model_uncond, capturing
    x0_sample = pred_x0 + r_t * randn (matches optimize_LGD's own expression,
    so captured states look like real guidance-loop states) at every
    step_stride-th outer step, always including the last (t=1). Returns a
    list of (step_index, t, x0_sample) in trajectory order (noisy -> clean).
    """
    experiment_utils.set_run_seed(seed, 0)
    T = model_uncond.diffusion_steps
    all_steps = list(range(T - 1, 0, -1))  # descending, matches optimize_LGD's pbar

    x_t = torch.zeros(condition_on, device=device, requires_grad=False).unsqueeze(0)
    captured = []
    for step_idx, t in enumerate(all_steps):
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
        if step_idx % step_stride == 0 or step_idx == len(all_steps) - 1:
            current_var = model_uncond.betas[t].to(device)
            r_t = current_var / torch.sqrt(1 + current_var ** 2)
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            captured.append((step_idx, t, x0_sample.detach().clone()))
        x_t = x_t_minus_1.detach().clone()
    return captured


def run_guided_trajectory_states(model_uncond, cond_sampler_fn, condition_on, seed, step_stride,
                                 device, experiment_utils, mmd_loss, mog_means, mog_variances, weights,
                                 nsamples, zeta, dist_utils, mu_list, Sigma_list, alpha, num_x_t=1):
    """One REAL guided trajectory -- unlike capture_trajectory_states (zeta=0,
    neutral reference), x_t here evolves under an actual guidance correction
    computed from `cond_sampler_fn` (that method's OWN inner sampler: LGD's
    DDIM unroll or LGD-CM's multistep draw), exactly matching Optimization.
    optimize_LGD's math: the underlying denoising step is still driven by
    model_uncond, but each step's correction -zeta*grad comes from
    backpropagating the MMD loss between cond_sampler_fn(x0_sample) and fresh
    mog_samples through x0_sample = pred_x0 + r_t*randn. This lets you ask
    whether something inherent to a method's OWN drift (not just its sampler's
    isolated noise at a neutral point) explains the observed differences --
    each source method gets its own captured trajectory, evaluated later by
    BOTH samplers at the SAME self-generated states.

    num_x_t=1 (default) keeps the captured x0_sample unambiguous -- Optimization.
    optimize_LGD's own num_x_t>1 variant tracks the BEST of several candidates
    for its own bookkeeping, which this single-path capture doesn't reproduce;
    at num_x_t=1 there's only one candidate per step, so no such gap exists.

    Also computes final_l2_gmm exactly as the original optimization notebooks do
    (Exp_10D_cond_1D.ipynb etc., "MLGD"/"MLGD-F" cells): the CLOSED-FORM L2
    distance between the conditional GMM implied by the trajectory's FINAL x_t
    (after the last update, t=1) and the true conditional GMM at x_star --
    NOT any generated-sample-based metric. Specifically:
        x_pred_t = x_t_final.float().view(-1).cpu()
        mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x_pred_t)
        w_pred = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_pred_t)
        final_l2_gmm = dist_utils.gmm_l2_distance(mu_pred, Sigma_pred, w_pred, mog_means, mog_variances, weights)
    matching the notebooks' cells verbatim (same functions, same argument order).

    Returns (captured, final_l2_gmm): captured is a list of (step_index, t,
    x0_sample) exactly like capture_trajectory_states, so the rest of the
    pipeline is unchanged; final_l2_gmm is this one trajectory's downstream
    optimization outcome, for joining against its own per-step diagnostics.
    """
    experiment_utils.set_run_seed(seed, 0)
    T = model_uncond.diffusion_steps
    all_steps = list(range(T - 1, 0, -1))

    x_t = torch.zeros(condition_on, device=device, requires_grad=True).unsqueeze(0)
    captured = []
    for step_idx, t in enumerate(all_steps):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
        current_var = model_uncond.betas[t].to(device)
        r_t = current_var / torch.sqrt(1 + current_var ** 2)

        losses = []
        last_x0_sample = None
        for _ in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            last_x0_sample = x0_sample
            y_samples = cond_sampler_fn(x0_sample.view(-1))
            mog_samples = dist_utils.generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
            loss_val = mmd_loss(y_samples, mog_samples)
            losses.append(-loss_val)

        log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        grad = torch.autograd.grad(log_mean_exp_loss, x_t)[0]

        if step_idx % step_stride == 0 or step_idx == len(all_steps) - 1:
            captured.append((step_idx, t, last_x0_sample.detach().clone()))

        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - zeta * grad

    # Downstream outcome for this trajectory, computed exactly as the
    # optimization notebooks do (see docstring above).
    x_pred_t = x_t.detach().float().view(-1).cpu()
    mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x_pred_t)
    w_pred = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_pred_t)
    final_l2_gmm = dist_utils.gmm_l2_distance(mu_pred, Sigma_pred, w_pred, mog_means, mog_variances, weights)

    return captured, final_l2_gmm


def grad_stats_for_method(x0_sample, sampler_fn, target_samples, n_redraws, base_seed, mmd_loss, experiment_utils, device):
    """Redraw `sampler_fn` (LGD's K-step DDIM unroll or LGD-CM's own multistep sample)
    n_redraws times at this ONE frozen x0_sample, against the fixed
    target_samples. Returns (mean_grad, stats) where stats has:
      normalized_variance   -- Var(grad)/||mean_grad||^2 (scale-free, but can mask a
                                real dimension-dependent noise trend if ||mean_grad||
                                itself grows with dimension).
      variance_trace        -- raw E[||g - mean_grad||^2] (sum over condition_on
                                coordinates -- grows trivially with dimension just
                                from having more terms in the sum, NOT itself evidence
                                of a per-coordinate noise increase).
      variance_trace_per_dim -- variance_trace / condition_on, i.e. per-coordinate
                                variance -- the fair cross-dimension comparison: factors
                                out the trivial "more coordinates -> bigger sum" effect,
                                isolating whether a TYPICAL coordinate's noise actually
                                gets worse as dimension grows.
      mean_grad_norm         -- ||mean_grad|| (for reference / sanity-checking whether
                                the signal itself is what's scaling with dimension).
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
    condition_on = grads.shape[1]
    mean_grad = grads.mean(axis=0)
    centered = grads - mean_grad
    variance_trace = float(np.mean(np.sum(centered ** 2, axis=1)))
    mean_grad_norm_sq = float(np.sum(mean_grad ** 2))
    normalized_variance = variance_trace / (mean_grad_norm_sq + 1e-12)
    stats = {
        "normalized_variance": normalized_variance,
        "variance_trace": variance_trace,
        "variance_trace_per_dim": variance_trace / condition_on,
        "mean_grad_norm": float(np.sqrt(mean_grad_norm_sq)),
    }
    return mean_grad, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment_name", type=str, default="10D_cond_1D", choices=list(ARCH.keys()))
    p.add_argument("--n_trajectories", type=int, default=10,
                   help="Independent unguided trajectories (different seeds) to average over.")
    p.add_argument("--step_stride", type=int, default=10,
                   help="Capture (and analyze) every step_stride-th outer diffusion step "
                        "instead of every single one -- controls cost. 1 = every step.")
    p.add_argument("--n_redraws", type=int, default=30,
                   help="Independent redraws of each inner sampler per (trajectory, step).")
    p.add_argument("--nsamples", type=int, default=250,
                   help="Batch size for the inner MMD estimator (both methods).")
    p.add_argument("--n_target", type=int, default=None,
                   help="Fixed target-sample count per state (defaults to --nsamples).")
    p.add_argument("--k_lgd", type=int, default=None,
                   help="Unroll depth for LGD's inner DDIM sampler (default: the experiment's "
                        "full diffusion_steps, matching what real LGD guidance actually runs "
                        "each outer step).")
    p.add_argument("--grad_ref_n", type=int, default=2000,
                   help="Sample size for the TRUE/population reference gradient at each state "
                        "(closed-form, no network forward -- cheap even at this size).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; also selects which pretrained checkpoints to load. "
                        "Trajectory i uses seed + i.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for a quick sanity check: 2 trajectories, step_stride=40, "
                        "5 redraws, nsamples=32.")
    p.add_argument("--state_source", type=str, default="unguided", choices=["unguided", "own"],
                   help="'unguided' (default): states from a neutral zeta=0 reference trajectory, "
                        "shared fairly between both samplers -- isolates the sampler mechanism "
                        "alone. 'own': for EACH method, run n_trajectories REAL guided "
                        "trajectories using that method's own inner sampler for the actual "
                        "-zeta*grad correction, then evaluate BOTH samplers at those "
                        "self-generated states -- checks whether something inherent to a "
                        "method's own drift (not just its sampler's isolated noise at a "
                        "neutral point) explains the observed differences. Also records each "
                        "trajectory's final_l2_gmm downstream outcome in this mode.")
    p.add_argument("--zeta", type=float, default=1.0,
                   help="Guidance strength for --state_source=own (ignored otherwise) -- matches "
                        "Optimization.optimize_LGD's zeta parameter.")
    p.add_argument("--num_x_t", type=int, default=1,
                   help="Candidate x0_samples per step for --state_source=own (ignored "
                        "otherwise). Default 1 keeps the captured state unambiguous -- see "
                        "run_guided_trajectory_states's docstring.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--plot", action="store_true", help="Also save a per-step comparison PNG.")
    args = p.parse_args()

    if args.smoke:
        args.n_trajectories = 2
        args.step_stride = 40
        args.n_redraws = 5
        args.nsamples = 32

    ARCH_CFG = ARCH[args.experiment_name]
    n_target = args.n_target or args.nsamples
    k_lgd = args.k_lgd or ARCH_CFG["diffusion_steps"]
    condition_on = ARCH_CFG["condition_on"]

    BASE_DIR = _HERE
    PARAMS_DIR = os.path.join(BASE_DIR, "params")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", args.experiment_name)
    RESULTS_DIR = args.output_dir or os.path.join(BASE_DIR, "results", args.experiment_name, "lgd_vs_lgdcm_step_variance")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import experiment_utils
    import dist_utils
    import Diffusion
    from ConsistencyModels import ConsistencyModeliCT
    from LossFunctions import MMDLoss, RBF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StepVar] experiment={args.experiment_name} n_trajectories={args.n_trajectories} "
          f"step_stride={args.step_stride} n_redraws={args.n_redraws} k_lgd={k_lgd} device={device}")

    experiment_utils.set_global_seed(args.seed)

    loaded = experiment_utils.load_gmm_params(PARAMS_DIR, args.experiment_name)
    if loaded is None:
        raise FileNotFoundError(f"No GMM params found under {PARAMS_DIR} for '{args.experiment_name}'.")
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
    mu_list = [mu.float() for mu in mu_list]
    Sigma_list = [cov.float() for cov in Sigma_list]
    alpha = alpha.float()
    nfeatures_full = mu_list[0].shape[0]

    # ── models: uncond (state capture), cond diffusion (LGD), CM (LGD-CM) ───
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=condition_on, nblocks=ARCH_CFG["nblocks"], nunits=ARCH_CFG["nunits"],
        condition=False, diffusion_steps=ARCH_CFG["diffusion_steps"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_uncond, "Diffusion_uncond", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load/download the pretrained Diffusion_uncond checkpoint.")
    model_uncond.to(device).eval()

    model_cond = Diffusion.DiffusionModel(
        nfeatures=nfeatures_full, nblocks=ARCH_CFG["nblocks"], nunits=ARCH_CFG["nunits"],
        condition=True, condition_on=condition_on, diffusion_steps=ARCH_CFG["diffusion_steps"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_cond, "Diffusion_cond", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load/download the pretrained Diffusion_cond checkpoint.")
    model_cond.to(device).eval()

    model_cm = ConsistencyModeliCT(
        nfeatures=nfeatures_full - condition_on, condition_on=condition_on,
        nunits=ARCH_CFG["nunits"], depth=ARCH_CFG["nblocks"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_cm, "CM", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load/download the pretrained CM checkpoint.")
    model_cm.to(device).eval()

    mmd_loss = MMDLoss(kernel=RBF())

    # Stateless closures (only depend on the fixed models/k_lgd/nsamples, not on which
    # state they're evaluated at) -- hoisted above the loop so run_guided_trajectory_states
    # can reuse them as a method's OWN inner sampler when --state_source=own.
    def lgd_sampler(x_leaf, _k=k_lgd):
        cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)
        _, y, _ = ddim_sample_kstep(model_cond, args.nsamples, cond, _k, device)
        return y

    def lgdcm_sampler(x_leaf):
        cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)
        y, _, _ = model_cm.sample(nsamples=args.nsamples, condition_x=cond, device=device)
        return y

    sampler_by_name = {"LGD": lgd_sampler, "LGD-CM": lgdcm_sampler}
    sources = ["unguided"] if args.state_source == "unguided" else ["LGD", "LGD-CM"]

    # ── main loop: sources x trajectories x captured steps x {LGD, LGD-CM} ──
    raw_rows = []       # one row per (state_source, trajectory, step, evaluated method)
    final_outcomes = []  # one row per (state_source, trajectory) -- only when state_source=own
    for source in sources:
        for traj_i in range(args.n_trajectories):
            traj_seed = args.seed + traj_i
            if source == "unguided":
                states = capture_trajectory_states(
                    model_uncond, traj_seed, args.step_stride, condition_on, device, experiment_utils
                )
            else:
                states, final_l2_gmm = run_guided_trajectory_states(
                    model_uncond, sampler_by_name[source], condition_on, traj_seed, args.step_stride,
                    device, experiment_utils, mmd_loss, mog_means, mog_variances, weights,
                    args.nsamples, args.zeta, dist_utils, mu_list, Sigma_list, alpha,
                    num_x_t=args.num_x_t,
                )
                final_outcomes.append({
                    "state_source": source, "trajectory": traj_i, "final_l2_gmm": final_l2_gmm,
                })
                print(f"[StepVar] source={source} trajectory {traj_i}: final_l2_gmm={final_l2_gmm:.6f}")
            print(f"[StepVar] source={source} trajectory {traj_i} (seed={traj_seed}): "
                  f"{len(states)} states captured")

            for step_idx, t, x0_sample in states:
                x_fixed = x0_sample.view(-1)

                # Fixed target samples for this state: drawn once from the exact analytic
                # conditional GMM at x_fixed (ground truth, not model_cond's approximation).
                mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_fixed)
                w_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_fixed)
                target_samples = dist_utils.generate_mog_samples_not_differentiable(
                    n_target, mu_cond, Sigma_cond, w_cond
                ).float().to(device)

                # TRUE/population reference gradient at this state (closed-form, independent
                # of which inner sampler is being evaluated).
                x_ref_leaf = x_fixed.clone().detach().to(device).requires_grad_(True)
                condi_mu, condi_sigma = dist_utils.compute_conditionals(mu_list, Sigma_list, x_ref_leaf)
                condi_mu = condi_mu.squeeze(-1)
                condi_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_ref_leaf)
                ref_samples = dist_utils.generate_mog_samples(args.grad_ref_n, condi_mu, condi_sigma, condi_alpha, device=device)
                loss_ref = mmd_loss(ref_samples, target_samples)
                grad_ref = torch.autograd.grad(loss_ref, x_ref_leaf)[0].detach().cpu().numpy()
                grad_ref_norm = float(np.linalg.norm(grad_ref))

                for method_name, sampler_fn, seed_offset in (
                    ("LGD", lgd_sampler, 0),
                    ("LGD-CM", lgdcm_sampler, 500),
                ):
                    mean_grad, stats = grad_stats_for_method(
                        x_fixed, sampler_fn, target_samples, args.n_redraws,
                        traj_seed * 100_000 + step_idx + seed_offset,
                        mmd_loss, experiment_utils, device,
                    )
                    dist_to_ref = float(np.linalg.norm(mean_grad - grad_ref))
                    dist_to_ref_normalized = dist_to_ref / (grad_ref_norm + 1e-12)
                    raw_rows.append({
                        "state_source": source, "trajectory": traj_i, "step_index": step_idx,
                        "t": t, "method": method_name,
                        **stats,
                        "dist_to_ref": dist_to_ref, "dist_to_ref_normalized": dist_to_ref_normalized,
                        "grad_ref_norm": grad_ref_norm,
                    })
                print(f"  [{source} traj {traj_i}] step {step_idx:>3} (t={t:>3}) | "
                      f"LGD nv={raw_rows[-2]['normalized_variance']:.4e} "
                      f"var/dim={raw_rows[-2]['variance_trace_per_dim']:.4e} "
                      f"dist={raw_rows[-2]['dist_to_ref_normalized']:.4f} | "
                      f"LGD-CM nv={raw_rows[-1]['normalized_variance']:.4e} "
                  f"var/dim={raw_rows[-1]['variance_trace_per_dim']:.4e} "
                  f"dist={raw_rows[-1]['dist_to_ref_normalized']:.4f}",
                  flush=True)

    # ── aggregate: mean/std/sem per (state_source, step_index, method) ───────
    # Nested by state_source because in --state_source=own mode, "LGD" states
    # and "LGD-CM" states come from two DIFFERENT trajectory distributions --
    # pooling them into one (step_index, method) bucket the way the old flat
    # aggregation did would silently average across two different populations
    # of states. In --state_source=unguided mode there's only one source
    # ("unguided"), so this is a strict superset of the old structure -- just
    # nested one level deeper.
    methods = ["LGD", "LGD-CM"]
    METRICS = ["normalized_variance", "dist_to_ref_normalized",
               "variance_trace", "variance_trace_per_dim", "mean_grad_norm"]
    aggregated = {}
    for source in sources:
        source_rows = [r for r in raw_rows if r["state_source"] == source]
        step_indices = sorted(set(r["step_index"] for r in source_rows))
        agg_source = {m: {metric: [] for metric in METRICS} for m in methods}
        agg_source["step_index"] = step_indices
        agg_source["t"] = [next(r["t"] for r in source_rows if r["step_index"] == s) for s in step_indices]

        for s in step_indices:
            for m in methods:
                vals_by_metric = {
                    metric: [r[metric] for r in source_rows if r["step_index"] == s and r["method"] == m]
                    for metric in METRICS
                }
                for metric in METRICS:
                    vals = np.array(vals_by_metric[metric])
                    agg_source[m][metric].append({
                        "mean": float(vals.mean()), "std": float(vals.std()),
                        "sem": float(vals.std() / max(1, np.sqrt(len(vals) - 1)) if len(vals) > 1 else 0.0),
                        "n": len(vals),
                    })
        aggregated[source] = agg_source

    out = {
        "experiment": args.experiment_name,
        "seed": args.seed,
        "n_trajectories": args.n_trajectories,
        "step_stride": args.step_stride,
        "n_redraws": args.n_redraws,
        "nsamples": args.nsamples,
        "n_target": n_target,
        "k_lgd": k_lgd,
        "grad_ref_n": args.grad_ref_n,
        "state_source": args.state_source,
        "sources": sources,
        "zeta": args.zeta,
        "num_x_t": args.num_x_t,
        "raw_rows": raw_rows,
        "final_outcomes": final_outcomes,
        "aggregated": aggregated,
    }
    suffix = "" if args.state_source == "unguided" else f"_{args.state_source}"
    out_path = os.path.join(
        RESULTS_DIR, f"{args.experiment_name}_lgd_vs_lgdcm_step_variance{suffix}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[StepVar] Saved results to {out_path}")

    if args.plot:
        _make_plot(aggregated, sources, args.experiment_name, out_path.replace(".json", ".png"))


def _make_plot(aggregated, sources, experiment_name, save_path):
    """One row of 3 panels per state_source (1 row for --state_source=unguided,
    2 rows -- LGD-states and LGD-CM-states -- for --state_source=own), each row
    identical in layout to the original single-row plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LGD": "steelblue", "LGD-CM": "crimson"}
    panels = (
        ("normalized_variance", "Per-step gradient variance (scale-free)", "Var(grad) / ||mean_grad||²"),
        ("variance_trace_per_dim", "Per-step gradient variance (per coordinate)",
         "Var(grad) / condition_on  (fair across dimensions)"),
        ("dist_to_ref_normalized", "Per-step gradient accuracy", "||mean_grad − grad_ref|| / ||grad_ref||"),
    )

    n_rows = len(sources)
    fig, axes = plt.subplots(n_rows, 3, figsize=(17, 4.5 * n_rows), squeeze=False)
    for row, source in enumerate(sources):
        agg = aggregated[source]
        ts = agg["t"]
        n_traj = None
        for col, (metric, title, ylabel) in enumerate(panels):
            ax = axes[row][col]
            for method in ("LGD", "LGD-CM"):
                means = np.array([e["mean"] for e in agg[method][metric]])
                sems = np.array([e["sem"] for e in agg[method][metric]])
                n_traj = agg[method][metric][0]["n"] if agg[method][metric] else n_traj
                ax.plot(ts, means, marker="o", color=colors[method], label=method)
                ax.fill_between(ts, np.clip(means - sems, 1e-9, None), means + sems, color=colors[method], alpha=0.2)
            ax.set_xlabel("timestep t (noisy → clean, right to left)")
            ax.set_ylabel(ylabel)
            row_title = title if source == "unguided" else f"{title}\n(states from {source}'s own trajectory)"
            ax.set_title(row_title)
            ax.invert_xaxis()
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
    fig.suptitle(f"{experiment_name}: LGD vs LGD-CM inner-sampler gradient quality along real trajectories\n"
                 f"(mean ± SEM over {n_traj} trajectories per step; state_source={'/'.join(sources)})")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[StepVar] Saved plot to {save_path}")


if __name__ == "__main__":
    main()
