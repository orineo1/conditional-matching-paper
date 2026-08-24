"""
compare_antithetic_gradient.py -- Does antithetic noise reduce the variance of
optimize_LGD's Monte Carlo guidance gradient? (arXiv:2506.06185, applied to 2D_cond_1D)

Checks two conditional samplers optimize_LGD can use for target_samples:
    lgd     -- model_cond, the conditional diffusion model. Called with
               eta=0.0, so its only randomness is one initial Gaussian draw
               per row (deterministic DDIM afterward).
    lgd_cm  -- model_cm, the consistency model. Unlike lgd, it draws fresh
               noise at every one of its ~15 sampling steps, so antithetic
               pairing here negates the *entire* per-step noise trajectory
               for the paired row, not just a single initial draw.
...and two MMD kernels: rbf (Gaussian) and energy (negative Euclidean
distance -- equivalent to energy distance, no bandwidth to tune).

For every (model, kernel) combination, at every (t, x_t) point checked, the
mechanism is always the same: x_t -> one DDIM step -> pred_x0 -> a noisy
x0_sample -> sample y given x0_sample. Antithetic pairing is always applied
at that last step (drawing y | x0_sample), never anywhere else:
    A (baseline): nsamples independent draws.
    B (antithetic): nsamples/2 draws z paired with their negation -z.
against an analytic reference gradient computed from the known GMM (no
diffusion/CM sampling involved at all, since the true P(Y|X) is known in
closed form for this experiment).

x_t points come from one or more full unconditional reverse-diffusion
trajectories (--n_x_t independent seeds), each read out at one or more
timesteps (--n_t, evenly spaced, or --t_values / --t for explicit ones) --
so points at different t along the same trajectory are the nested,
correlated x_t's optimize_LGD actually sees, not independently resampled
ones. Every (model, kernel) pair is evaluated at the identical set of
points, so "lgd vs lgd" across points, "lgd_cm vs lgd_cm" across points,
and "lgd vs lgd_cm" at the same point are all directly comparable.

Example:
    python compare_antithetic_gradient.py --experiment 2D_cond_1D \\
        --model both --kernel both --n_t 5 --n_x_t 3 --n_trials 200
"""

import os
import sys
import json
import argparse
from functools import partial

import numpy as np
import torch

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dist_utils
import experiment_utils
from LossFunctions import MMDLoss, RBF, EnergyKernel
from sweep_common import EXPERIMENT_CONFIGS, load_or_generate_gmm_params, load_or_train_models

# Must match ConsistencyModeliCT.sample()'s default `ts` schedule.
CM_TS = [80, 40, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.062, 0.031, 0.015, 0.007, 0.002]

KERNEL_BUILDERS = {
    "rbf": lambda device: RBF(device=device),
    "energy": lambda device: EnergyKernel(device=device),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="2D_cond_1D", choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--model", default="both", choices=["lgd", "lgd_cm", "both"],
                    help="Which conditional sampler(s) to check: LGD (diffusion), LGD-CM (consistency model), or both")
    p.add_argument("--kernel", default="both", choices=["rbf", "energy", "both"],
                    help="Which MMD kernel(s) to check")
    p.add_argument("--t", type=int, default=None,
                    help="Single diffusion timestep to freeze x_t at (overrides --n_t/--t_values)")
    p.add_argument("--t_values", default=None,
                    help="Comma-separated explicit timesteps, e.g. '20,50,80' (overrides --n_t)")
    p.add_argument("--n_t", type=int, default=1,
                    help="Number of timesteps to sweep, evenly spaced across the trajectory")
    p.add_argument("--n_x_t", type=int, default=1,
                    help="Number of independent reverse-diffusion trajectories (x_t draws) to check per timestep")
    p.add_argument("--nsamples", type=int, default=250, help="Sample budget for estimators A and B (must be even)")
    p.add_argument("--n_trials", type=int, default=200, help="Independent repetitions per estimator per point")
    p.add_argument("--n_ref", type=int, default=5000, help="Sample size for the analytic reference gradient")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def resolve_t_values(args, model_uncond):
    """Explicit --t_values, else --t alone, else --n_t evenly spaced points, else the old single mid-trajectory default."""
    if args.t_values:
        return sorted({int(v) for v in args.t_values.split(",")}, reverse=True)
    if args.t is not None:
        return [args.t]
    if args.n_t <= 1:
        return [model_uncond.diffusion_steps // 2]
    lo, hi = 1, model_uncond.diffusion_steps - 1
    ts = np.linspace(lo, hi, args.n_t)
    return sorted({int(round(v)) for v in ts}, reverse=True)


def freeze_x_t_trajectory(model_uncond, t_values, device, seed):
    """
    One full unconditional reverse trajectory (deterministic DDIM, eta=0), read
    out at every requested timestep. x_t's from the same trajectory at
    different t are the nested, correlated points optimize_LGD actually walks
    through -- not independently resampled ones.
    """
    experiment_utils.set_run_seed(seed, 0)
    t_start = model_uncond.diffusion_steps - 1
    _, xt, _ = model_uncond.sample(nsamples=1, condition_x=None, device=device, eta=0.0, t_start=t_start, t_end=0)
    return {t: xt[t_start - t].detach() for t in t_values}


def pred_x0_and_x0_sample(x_t, t, model_uncond, r_t_z0, device):
    """One DDIM step from x_t, then the LGD-style noisy x0 candidate. x_t must require grad."""
    _, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
    return pred_x0 + r_t_z0


def reference_gradient(x_t_frozen, t, model_uncond, mu_list, Sigma_list, alpha,
                        mog_means, mog_variances, weights, r_t_z0, mmd_loss, n_ref, device):
    """Analytic gradient: replace the sampler's samples with samples from the true P(Y|X=x0_sample)."""
    x_t = x_t_frozen.clone().requires_grad_(True)
    x0_sample = pred_x0_and_x0_sample(x_t, t, model_uncond, r_t_z0, device)

    mog_means_cond, mog_variances_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x0_sample.reshape(-1, 1))
    weights_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x0_sample.reshape(-1, 1))
    target_samples = dist_utils.generate_mog_samples(n_ref, mog_means_cond, mog_variances_cond, weights_cond)

    mog_samples = dist_utils.generate_mog_samples_not_differentiable(n_ref, mog_means, mog_variances, weights)
    loss = mmd_loss(target_samples, mog_samples)
    loss_value = loss.item()
    loss.backward()
    return x_t.grad.clone(), loss_value


def sample_targets_lgd(model_cond, condition, nsamples, device, antithetic):
    """LGD sampler: model_cond.sample(), antithetic-paired on its one initial noise draw."""
    if antithetic:
        half = nsamples // 2
        z = torch.randn(half, model_cond.nfeatures, device=device)
        init_noise = torch.cat([z, -z], dim=0)
    else:
        init_noise = None

    target_samples, _, _ = model_cond.sample(
        nsamples=nsamples, condition_x=condition, device=device, init_noise=init_noise,
    )
    return target_samples[:, model_cond.condition_on:]


def sample_targets_lgd_cm(model_cm, condition, nsamples, device, antithetic):
    """LGD-CM sampler: model_cm.sample(), antithetic-paired on its whole per-step noise trajectory."""
    if antithetic:
        half = nsamples // 2
        z = torch.randn(len(CM_TS), half, model_cm.nfeatures, device=device)
        init_noise = torch.cat([z, -z], dim=1)
    else:
        init_noise = None

    target_samples, _, _ = model_cm.sample(
        nsamples=nsamples, condition_x=condition, ts=CM_TS, device=device, init_noise=init_noise,
    )
    return target_samples


def mc_gradient(x_t_frozen, t, model_uncond, sample_targets, mog_means, mog_variances, weights,
                 r_t_z0, mmd_loss, nsamples, device, antithetic):
    """The gradient optimize_LGD actually takes (num_x_t=1), via sample_targets(condition, ...)."""
    x_t = x_t_frozen.clone().requires_grad_(True)
    x0_sample = pred_x0_and_x0_sample(x_t, t, model_uncond, r_t_z0, device)
    condition = x0_sample.view(1, -1).repeat(nsamples, 1)

    target_samples = sample_targets(condition, nsamples, device, antithetic)

    mog_samples = dist_utils.generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)
    loss = mmd_loss(target_samples, mog_samples)
    loss_value = loss.item()
    loss.backward()
    return x_t.grad.clone(), target_samples.detach(), loss_value


def pair_correlation(target_samples):
    """
    Pearson correlation between the (z, -z)-paired halves of an antithetic
    target_samples batch (row i in the first half is paired with row
    i + nsamples/2, matching how mc_gradient built init_noise). Near -1 would
    match the paper's antisymmetry claim; near 0 means pairing bought nothing.
    """
    half = target_samples.shape[0] // 2
    a, b = target_samples[:half], target_samples[half:]
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    num = (a * b).sum(dim=0)
    denom = torch.sqrt((a ** 2).sum(dim=0) * (b ** 2).sum(dim=0)) + 1e-12
    return (num / denom).mean().item()


def estimator_stats(grads, reference):
    """grads: [n_trials, dim] tensor. Returns bias/variance/MSE against the reference gradient."""
    mean = grads.mean(dim=0)
    bias = (mean - reference).norm().item()
    variance = grads.var(dim=0, unbiased=True).sum().item()
    mse = ((grads - reference) ** 2).sum(dim=1).mean().item()
    return {"bias_norm": bias, "variance": variance, "mse": mse}


def scalar_stats(values, reference):
    """values: [n_trials] tensor of scalar MMD losses. Same bias/variance/MSE, for a scalar."""
    values = values.float()
    bias = (values.mean() - reference).item()
    variance = values.var(unbiased=True).item()
    mse = ((values - reference) ** 2).mean().item()
    return {"bias": bias, "variance": variance, "mse": mse}


def run_point(label, sample_targets, x_t_frozen, t, traj_idx, model_uncond, mu_list, Sigma_list, alpha,
              mog_means, mog_variances, weights, r_t_z0, mmd_loss, args):
    """Runs the full A-vs-B comparison at one (model, kernel, t, trajectory) point. Returns a result dict."""
    device = x_t_frozen.device
    print(f"[{label}] [Point] t={t} traj={traj_idx}", flush=True)

    reference, reference_loss = reference_gradient(
        x_t_frozen, t, model_uncond, mu_list, Sigma_list, alpha,
        mog_means, mog_variances, weights, r_t_z0, mmd_loss, args.n_ref, device,
    )

    grads_A, grads_B, correlations, losses_A, losses_B = [], [], [], [], []
    for i in range(args.n_trials):
        grad_A, _, loss_A = mc_gradient(
            x_t_frozen, t, model_uncond, sample_targets, mog_means, mog_variances, weights,
            r_t_z0, mmd_loss, args.nsamples, device, antithetic=False,
        )
        grad_B, samples_B, loss_B = mc_gradient(
            x_t_frozen, t, model_uncond, sample_targets, mog_means, mog_variances, weights,
            r_t_z0, mmd_loss, args.nsamples, device, antithetic=True,
        )
        grads_A.append(grad_A)
        grads_B.append(grad_B)
        losses_A.append(loss_A)
        losses_B.append(loss_B)
        correlations.append(pair_correlation(samples_B))

    grads_A = torch.cat(grads_A, dim=0)
    grads_B = torch.cat(grads_B, dim=0)
    reference_flat = reference.view(-1)
    errors_A = (grads_A - reference_flat).norm(dim=1)
    errors_B = (grads_B - reference_flat).norm(dim=1)

    stats_A = estimator_stats(grads_A, reference_flat)
    stats_B = estimator_stats(grads_B, reference_flat)
    variance_reduction = stats_A["variance"] / stats_B["variance"] if stats_B["variance"] > 0 else float("inf")
    mean_pair_correlation = float(np.mean(correlations))
    std_pair_correlation = float(np.std(correlations))

    loss_stats_A = scalar_stats(torch.tensor(losses_A), reference_loss)
    loss_stats_B = scalar_stats(torch.tensor(losses_B), reference_loss)
    loss_variance_reduction = (
        loss_stats_A["variance"] / loss_stats_B["variance"] if loss_stats_B["variance"] > 0 else float("inf")
    )

    print(f"[{label}] [Point t={t} traj={traj_idx}] var_reduction={variance_reduction:.4f} "
          f"loss_var_reduction={loss_variance_reduction:.4f} mean_pair_corr={mean_pair_correlation:.4f}", flush=True)

    trials = [
        {
            "trial": i,
            "grad_A": grads_A[i].tolist(), "error_A": errors_A[i].item(), "loss_A": losses_A[i],
            "grad_B": grads_B[i].tolist(), "error_B": errors_B[i].item(), "loss_B": losses_B[i],
            "pair_correlation_B": correlations[i],
        }
        for i in range(args.n_trials)
    ]

    return {
        "t": t, "traj_idx": traj_idx,
        "reference_grad": reference_flat.tolist(), "reference_loss": reference_loss,
        "pure_noise": stats_A, "antithetic": stats_B,
        "variance_reduction_factor": variance_reduction,
        "pure_noise_loss": loss_stats_A, "antithetic_loss": loss_stats_B,
        "loss_variance_reduction_factor": loss_variance_reduction,
        "mean_pair_correlation": mean_pair_correlation,
        "std_pair_correlation": std_pair_correlation,
        "trials": trials,
    }


def save_results(label, points, results_dir, args):
    """Aggregates all points for one (model, kernel) pair into one JSON + one pooled plot."""
    def summarize(key):
        values = [pt[key] for pt in points]
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    summary = {
        "variance_reduction_factor": summarize("variance_reduction_factor"),
        "loss_variance_reduction_factor": summarize("loss_variance_reduction_factor"),
        "mean_pair_correlation": summarize("mean_pair_correlation"),
    }
    print(f"[{label}] [Summary across {len(points)} point(s)] "
          f"var_reduction mean={summary['variance_reduction_factor']['mean']:.4f} "
          f"loss_var_reduction mean={summary['loss_variance_reduction_factor']['mean']:.4f} "
          f"pair_corr mean={summary['mean_pair_correlation']['mean']:.4f}")

    t_values = sorted({pt["t"] for pt in points}, reverse=True)
    out = {
        "label": label, "experiment": args.experiment, "nsamples": args.nsamples,
        "n_trials": args.n_trials, "n_ref": args.n_ref, "seed": args.seed,
        "t_values": t_values, "n_x_t": args.n_x_t,
        "summary": summary, "points": points,
    }
    tag = f"{label}_nt{len(t_values)}_nxt{args.n_x_t}_n{args.nsamples}_seed{args.seed}"
    json_path = os.path.join(results_dir, f"antithetic_gradient_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{label}] [Results] Saved JSON to {json_path}")

    errors_A = np.array([trial["error_A"] for pt in points for trial in pt["trials"]])
    errors_B = np.array([trial["error_B"] for pt in points for trial in pt["trials"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color_A, color_B = "tab:blue", "tab:orange"
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        suptitle = f"{label}  {args.experiment}  {len(t_values)} t-value(s)  {args.n_x_t} trajector(y/ies)  nsamples={args.nsamples}"

        ax = axes[0]
        ax.hist(errors_A, bins=30, alpha=0.6, color=color_A, label="pure noise (A)")
        ax.hist(errors_B, bins=30, alpha=0.6, color=color_B, label="antithetic (B)")
        ax.set_xlabel("||grad - reference||")
        ax.set_ylabel("count")
        ax.set_title("Histogram (pooled across points)")
        ax.legend()

        ax = axes[1]
        for errors, color, elabel in [(errors_A, color_A, "pure noise (A)"), (errors_B, color_B, "antithetic (B)")]:
            sorted_errors = np.sort(errors)
            ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax.step(sorted_errors, ecdf, where="post", color=color, label=elabel)
        ax.set_xlabel("||grad - reference||")
        ax.set_ylabel("empirical CDF")
        ax.set_title("eCDF (pooled across points)")
        ax.legend()

        ax = axes[2]
        ax.boxplot([errors_A, errors_B], showfliers=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["pure noise (A)", "antithetic (B)"])
        ax.set_ylabel("||grad - reference||")
        ax.set_title("Boxplot (pooled across points)")

        fig.suptitle(suptitle)
        fig.tight_layout()
        plot_path = os.path.join(results_dir, f"antithetic_gradient_{tag}.png")
        fig.savefig(plot_path, dpi=150)
        print(f"[{label}] [Results] Saved plot to {plot_path}")
    except ImportError:
        print(f"[{label}] [Results] matplotlib not available, skipping plot")


def main():
    args = parse_args()
    cfg = EXPERIMENT_CONFIGS[args.experiment]
    assert args.nsamples % 2 == 0, "--nsamples must be even for antithetic pairing"

    base_dir = args.base_dir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    params_dir = os.path.join(base_dir, "params")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.experiment)
    results_dir = args.output_dir or os.path.join(base_dir, "results", args.experiment)
    for d in (checkpoint_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    experiment_utils.set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = load_or_generate_gmm_params(
        cfg, params_dir, results_dir, args.experiment, args.seed
    )
    model_uncond, model_cond, model_cm = load_or_train_models(
        cfg, mu_list, Sigma_list, alpha, checkpoint_dir, args.experiment, args.seed, device, args.force_retrain
    )

    t_values = resolve_t_values(args, model_uncond)
    samplers = {
        "lgd": partial(sample_targets_lgd, model_cond),
        "lgd_cm": partial(sample_targets_lgd_cm, model_cm),
    }
    models_to_run = ["lgd", "lgd_cm"] if args.model == "both" else [args.model]
    kernels_to_run = ["rbf", "energy"] if args.kernel == "both" else [args.kernel]

    print(f"[Setup] experiment={args.experiment} models={models_to_run} kernels={kernels_to_run} "
          f"t_values={t_values} n_x_t={args.n_x_t} nsamples={args.nsamples} n_trials={args.n_trials}", flush=True)

    points_by_label = {(m, k): [] for m in models_to_run for k in kernels_to_run}
    mmd_losses = {k: MMDLoss(kernel=KERNEL_BUILDERS[k](device)) for k in kernels_to_run}

    for traj_idx in range(args.n_x_t):
        traj_seed = args.seed + traj_idx
        x_t_by_t = freeze_x_t_trajectory(model_uncond, t_values, device, traj_seed)
        for t in t_values:
            x_t_frozen = x_t_by_t[t]
            # Draw #1 (x0_sample's own perturbation) is held fixed per point: we're isolating
            # Draw #2 only. Shared across models/kernels so they're compared at the same x0_sample.
            current_var = model_uncond.betas[t].to(device)
            r_t = current_var / torch.sqrt(1 + current_var ** 2)
            r_t_z0 = r_t * torch.randn(1, model_uncond.nfeatures, device=device)

            for model_label in models_to_run:
                for kernel_label in kernels_to_run:
                    label = f"{model_label}_{kernel_label}"
                    point = run_point(
                        label, samplers[model_label], x_t_frozen, t, traj_idx, model_uncond,
                        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                        r_t_z0, mmd_losses[kernel_label], args,
                    )
                    points_by_label[(model_label, kernel_label)].append(point)

    for (model_label, kernel_label), points in points_by_label.items():
        save_results(f"{model_label}_{kernel_label}", points, results_dir, args)


if __name__ == "__main__":
    main()
