#!/usr/bin/env python
"""
lgd_vs_lgdcm_step_variance.py — per-step gradient variance/accuracy of LGD's
inner sampler (K-step DDIM unroll through the conditional diffusion model)
vs. LGD-CM's inner sampler (one-shot consistency-model draw), along real
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
      LGD-CM: one draw through the consistency model.
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


def grad_stats_for_method(x0_sample, sampler_fn, target_samples, n_redraws, base_seed, mmd_loss, experiment_utils, device):
    """Redraw `sampler_fn` (LGD's K-step unroll or LGD-CM's one-shot draw)
    n_redraws times at this ONE frozen x0_sample, against the fixed
    target_samples. Returns (mean_grad, normalized_variance)."""
    grads = []
    for r in range(n_redraws):
        experiment_utils.set_run_seed(base_seed, r)
        x_leaf = x0_sample.clone().detach().to(device).requires_grad_(True)
        y_samples = sampler_fn(x_leaf)
        loss = mmd_loss(y_samples, target_samples)
        grad = torch.autograd.grad(loss, x_leaf)[0]
        grads.append(grad.detach().cpu().numpy().copy())

    grads = np.stack(grads, axis=0)
    mean_grad = grads.mean(axis=0)
    centered = grads - mean_grad
    variance_trace = float(np.mean(np.sum(centered ** 2, axis=1)))
    mean_grad_norm_sq = float(np.sum(mean_grad ** 2))
    normalized_variance = variance_trace / (mean_grad_norm_sq + 1e-12)
    return mean_grad, normalized_variance


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

    # ── main loop: trajectories x captured steps x {LGD, LGD-CM} ────────────
    raw_rows = []  # one row per (trajectory, step, method)
    for traj_i in range(args.n_trajectories):
        traj_seed = args.seed + traj_i
        states = capture_trajectory_states(
            model_uncond, traj_seed, args.step_stride, condition_on, device, experiment_utils
        )
        print(f"[StepVar] trajectory {traj_i} (seed={traj_seed}): {len(states)} states captured")

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

            def lgd_sampler(x_leaf, _k=k_lgd):
                cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)
                _, y, _ = ddim_sample_kstep(model_cond, args.nsamples, cond, _k, device)
                return y

            def lgdcm_sampler(x_leaf):
                cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)
                y, _, _ = model_cm.sample(nsamples=args.nsamples, condition_x=cond, device=device)
                return y

            for method_name, sampler_fn, seed_offset in (
                ("LGD", lgd_sampler, 0),
                ("LGD-CM", lgdcm_sampler, 500),
            ):
                mean_grad, normalized_variance = grad_stats_for_method(
                    x_fixed, sampler_fn, target_samples, args.n_redraws,
                    traj_seed * 100_000 + step_idx + seed_offset,
                    mmd_loss, experiment_utils, device,
                )
                dist_to_ref = float(np.linalg.norm(mean_grad - grad_ref))
                dist_to_ref_normalized = dist_to_ref / (grad_ref_norm + 1e-12)
                raw_rows.append({
                    "trajectory": traj_i, "step_index": step_idx, "t": t, "method": method_name,
                    "normalized_variance": normalized_variance,
                    "dist_to_ref": dist_to_ref, "dist_to_ref_normalized": dist_to_ref_normalized,
                    "grad_ref_norm": grad_ref_norm,
                })
            print(f"  [traj {traj_i}] step {step_idx:>3} (t={t:>3}) | "
                  f"LGD nv={raw_rows[-2]['normalized_variance']:.4e} dist={raw_rows[-2]['dist_to_ref_normalized']:.4f} | "
                  f"LGD-CM nv={raw_rows[-1]['normalized_variance']:.4e} dist={raw_rows[-1]['dist_to_ref_normalized']:.4f}",
                  flush=True)

    # ── aggregate: mean/std/sem per (step_index, method) across trajectories ─
    step_indices = sorted(set(r["step_index"] for r in raw_rows))
    methods = ["LGD", "LGD-CM"]
    METRICS = ["normalized_variance", "dist_to_ref_normalized"]
    aggregated = {m: {metric: [] for metric in METRICS} for m in methods}
    aggregated["step_index"] = step_indices
    aggregated["t"] = [next(r["t"] for r in raw_rows if r["step_index"] == s) for s in step_indices]

    for s in step_indices:
        for m in methods:
            vals_by_metric = {
                metric: [r[metric] for r in raw_rows if r["step_index"] == s and r["method"] == m]
                for metric in METRICS
            }
            for metric in METRICS:
                vals = np.array(vals_by_metric[metric])
                aggregated[m][metric].append({
                    "mean": float(vals.mean()), "std": float(vals.std()),
                    "sem": float(vals.std() / max(1, np.sqrt(len(vals) - 1)) if len(vals) > 1 else 0.0),
                    "n": len(vals),
                })

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
        "raw_rows": raw_rows,
        "aggregated": aggregated,
    }
    out_path = os.path.join(RESULTS_DIR, f"{args.experiment_name}_lgd_vs_lgdcm_step_variance_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[StepVar] Saved results to {out_path}")

    if args.plot:
        _make_plot(aggregated, args.experiment_name, out_path.replace(".json", ".png"))


def _make_plot(aggregated, experiment_name, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_indices = aggregated["step_index"]
    ts = aggregated["t"]
    colors = {"LGD": "steelblue", "LGD-CM": "crimson"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title, ylabel in (
        (axes[0], "normalized_variance", "Per-step gradient variance", "Var(grad) / ||mean_grad||²"),
        (axes[1], "dist_to_ref_normalized", "Per-step gradient accuracy", "||mean_grad − grad_ref|| / ||grad_ref||"),
    ):
        for method in ("LGD", "LGD-CM"):
            means = np.array([e["mean"] for e in aggregated[method][metric]])
            sems = np.array([e["sem"] for e in aggregated[method][metric]])
            ax.plot(ts, means, marker="o", color=colors[method], label=method)
            ax.fill_between(ts, means - sems, means + sems, color=colors[method], alpha=0.2)
        ax.set_xlabel("timestep t (noisy → clean, right to left)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.invert_xaxis()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{experiment_name}: LGD vs LGD-CM inner-sampler gradient quality along one trajectory\n"
                 f"(mean ± SEM over {aggregated[method][metric][0]['n']} trajectories per step)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[StepVar] Saved plot to {save_path}")


if __name__ == "__main__":
    main()
