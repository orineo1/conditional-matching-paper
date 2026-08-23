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

For each sampler, we compare two ways of drawing its noise:
    A (baseline): nsamples independent draws.
    B (antithetic): nsamples/2 draws z paired with their negation -z.
against an analytic reference gradient computed from the known GMM (no
diffusion/CM sampling involved at all, since the true P(Y|X) is known in
closed form for this experiment).

Example:
    python compare_antithetic_gradient.py --experiment 2D_cond_1D --model both --n_trials 200
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
from LossFunctions import MMDLoss, RBF
from sweep_common import EXPERIMENT_CONFIGS, load_or_generate_gmm_params, load_or_train_models

# Must match ConsistencyModeliCT.sample()'s default `ts` schedule.
CM_TS = [80, 40, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.062, 0.031, 0.015, 0.007, 0.002]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="2D_cond_1D", choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--model", default="both", choices=["lgd", "lgd_cm", "both"],
                    help="Which conditional sampler(s) to check: LGD (diffusion), LGD-CM (consistency model), or both")
    p.add_argument("--t", type=int, default=None, help="Diffusion timestep to freeze x_t at (default: mid-trajectory)")
    p.add_argument("--nsamples", type=int, default=250, help="Sample budget for estimators A and B (must be even)")
    p.add_argument("--n_trials", type=int, default=200, help="Independent repetitions per estimator")
    p.add_argument("--n_ref", type=int, default=5000, help="Sample size for the analytic reference gradient")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def freeze_x_t(model_uncond, t, device, seed):
    """Draw one unconditional sample and stop the reverse trajectory at timestep t."""
    experiment_utils.set_run_seed(seed, 0)
    x_t, _, _ = model_uncond.sample(
        nsamples=1, condition_x=None, device=device, eta=0.0,
        t_start=model_uncond.diffusion_steps - 1, t_end=t,
    )
    return x_t.detach()


def pred_x0_and_x0_sample(x_t, t, model_uncond, r_t_z0, device):
    """One DDIM step from x_t, then the LGD-style noisy x0 candidate. x_t must require grad."""
    _, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
    return pred_x0 + r_t_z0


def reference_gradient(x_t_frozen, t, model_uncond, mu_list, Sigma_list, alpha,
                        mog_means, mog_variances, weights, r_t_z0, mmd_loss, n_ref, device):
    """Analytic gradient: replace model_cond's samples with samples from the true P(Y|X=x0_sample)."""
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


def run_comparison(model_label, sample_targets, x_t_frozen, t, model_uncond, mu_list, Sigma_list, alpha,
                    mog_means, mog_variances, weights, r_t_z0, mmd_loss, args, results_dir):
    """Runs the full A-vs-B comparison for one sampler (LGD or LGD-CM) and writes its JSON + plot."""
    print(f"[Setup] model={model_label} experiment={args.experiment} t={t} nsamples={args.nsamples} "
          f"n_trials={args.n_trials} n_ref={args.n_ref}", flush=True)

    reference, reference_loss = reference_gradient(
        x_t_frozen, t, model_uncond, mu_list, Sigma_list, alpha,
        mog_means, mog_variances, weights, r_t_z0, mmd_loss, args.n_ref, x_t_frozen.device,
    )
    print(f"[{model_label}] [Reference] grad={reference.view(-1).tolist()} loss={reference_loss:.6f}", flush=True)

    grads_A, grads_B, correlations, losses_A, losses_B = [], [], [], [], []
    for i in range(args.n_trials):
        grad_A, _, loss_A = mc_gradient(
            x_t_frozen, t, model_uncond, sample_targets, mog_means, mog_variances, weights,
            r_t_z0, mmd_loss, args.nsamples, x_t_frozen.device, antithetic=False,
        )
        grad_B, samples_B, loss_B = mc_gradient(
            x_t_frozen, t, model_uncond, sample_targets, mog_means, mog_variances, weights,
            r_t_z0, mmd_loss, args.nsamples, x_t_frozen.device, antithetic=True,
        )
        grads_A.append(grad_A)
        grads_B.append(grad_B)
        losses_A.append(loss_A)
        losses_B.append(loss_B)
        correlations.append(pair_correlation(samples_B))
        if (i + 1) % max(1, args.n_trials // 10) == 0:
            print(f"[{model_label}] [Trial] {i + 1}/{args.n_trials}", flush=True)

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

    losses_A_t = torch.tensor(losses_A)
    losses_B_t = torch.tensor(losses_B)
    loss_stats_A = scalar_stats(losses_A_t, reference_loss)
    loss_stats_B = scalar_stats(losses_B_t, reference_loss)
    loss_variance_reduction = (
        loss_stats_A["variance"] / loss_stats_B["variance"] if loss_stats_B["variance"] > 0 else float("inf")
    )

    print(f"[{model_label}] [A: pure noise]  bias={stats_A['bias_norm']:.6f} var={stats_A['variance']:.6f} "
          f"mse={stats_A['mse']:.6f}")
    print(f"[{model_label}] [B: antithetic]  bias={stats_B['bias_norm']:.6f} var={stats_B['variance']:.6f} "
          f"mse={stats_B['mse']:.6f}")
    print(f"[{model_label}] [Variance reduction factor A/B] {variance_reduction:.4f}")
    print(f"[{model_label}] [Paired-sample correlation] mean={mean_pair_correlation:.4f} "
          f"std={std_pair_correlation:.4f} (near -1 = antisymmetry holds, near 0 = pairing bought nothing)")
    print(f"[{model_label}] [MMD loss A: pure noise]  bias={loss_stats_A['bias']:.6f} "
          f"var={loss_stats_A['variance']:.6f} mse={loss_stats_A['mse']:.6f}")
    print(f"[{model_label}] [MMD loss B: antithetic]  bias={loss_stats_B['bias']:.6f} "
          f"var={loss_stats_B['variance']:.6f} mse={loss_stats_B['mse']:.6f}")
    print(f"[{model_label}] [MMD loss variance reduction factor A/B] {loss_variance_reduction:.4f} "
          f"(this is the scalar the paper's control-variate argument is about; "
          f"the gradient is a higher-dimensional, nonlinear function of the same samples)")

    # Per-trial breakdown, so any run can be inspected individually to check
    # whether the aggregate variance is driven by a handful of outlier trials.
    trials = [
        {
            "trial": i,
            "grad_A": grads_A[i].tolist(), "error_A": errors_A[i].item(), "loss_A": losses_A[i],
            "grad_B": grads_B[i].tolist(), "error_B": errors_B[i].item(), "loss_B": losses_B[i],
            "pair_correlation_B": correlations[i],
        }
        for i in range(args.n_trials)
    ]

    out = {
        "model": model_label, "experiment": args.experiment, "t": t, "nsamples": args.nsamples,
        "n_trials": args.n_trials, "n_ref": args.n_ref, "seed": args.seed,
        "reference_grad": reference_flat.tolist(), "reference_loss": reference_loss,
        "pure_noise": stats_A, "antithetic": stats_B,
        "variance_reduction_factor": variance_reduction,
        "pure_noise_loss": loss_stats_A, "antithetic_loss": loss_stats_B,
        "loss_variance_reduction_factor": loss_variance_reduction,
        "mean_pair_correlation": mean_pair_correlation,
        "std_pair_correlation": std_pair_correlation,
        "trials": trials,
    }
    tag = f"{model_label}_t{t}_n{args.nsamples}_seed{args.seed}"
    json_path = os.path.join(results_dir, f"antithetic_gradient_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{model_label}] [Results] Saved JSON to {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        errors_A = errors_A.cpu().numpy()
        errors_B = errors_B.cpu().numpy()
        color_A, color_B = "tab:blue", "tab:orange"

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        suptitle = f"{model_label}  {args.experiment}  t={t}  nsamples={args.nsamples}"

        ax = axes[0]
        ax.hist(errors_A, bins=30, alpha=0.6, color=color_A, label="pure noise (A)")
        ax.hist(errors_B, bins=30, alpha=0.6, color=color_B, label="antithetic (B)")
        ax.set_xlabel("||grad - reference||")
        ax.set_ylabel("count")
        ax.set_title("Histogram")
        ax.legend()

        ax = axes[1]
        for errors, color, label in [(errors_A, color_A, "pure noise (A)"), (errors_B, color_B, "antithetic (B)")]:
            sorted_errors = np.sort(errors)
            ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax.step(sorted_errors, ecdf, where="post", color=color, label=label)
        ax.set_xlabel("||grad - reference||")
        ax.set_ylabel("empirical CDF")
        ax.set_title("eCDF")
        ax.legend()

        ax = axes[2]
        ax.boxplot([errors_A, errors_B], showfliers=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["pure noise (A)", "antithetic (B)"])
        ax.set_ylabel("||grad - reference||")
        ax.set_title("Boxplot")

        fig.suptitle(suptitle)
        fig.tight_layout()
        plot_path = os.path.join(results_dir, f"antithetic_gradient_{tag}.png")
        fig.savefig(plot_path, dpi=150)
        print(f"[{model_label}] [Results] Saved plot to {plot_path}")
    except ImportError:
        print(f"[{model_label}] [Results] matplotlib not available, skipping plot")


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
    mmd_loss = MMDLoss(kernel=RBF())

    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = load_or_generate_gmm_params(
        cfg, params_dir, results_dir, args.experiment, args.seed
    )
    model_uncond, model_cond, model_cm = load_or_train_models(
        cfg, mu_list, Sigma_list, alpha, checkpoint_dir, args.experiment, args.seed, device, args.force_retrain
    )

    t = args.t if args.t is not None else model_uncond.diffusion_steps // 2
    x_t_frozen = freeze_x_t(model_uncond, t, device, args.seed)

    # Draw #1 (x0_sample's own perturbation) is held fixed: we're isolating Draw #2 only.
    # Shared across models so LGD and LGD-CM are compared at the exact same x0_sample.
    current_var = model_uncond.betas[t].to(device)
    r_t = current_var / torch.sqrt(1 + current_var ** 2)
    r_t_z0 = r_t * torch.randn(1, model_uncond.nfeatures, device=device)

    samplers = {
        "lgd": partial(sample_targets_lgd, model_cond),
        "lgd_cm": partial(sample_targets_lgd_cm, model_cm),
    }
    models_to_run = ["lgd", "lgd_cm"] if args.model == "both" else [args.model]

    for model_label in models_to_run:
        run_comparison(
            model_label, samplers[model_label], x_t_frozen, t, model_uncond, mu_list, Sigma_list, alpha,
            mog_means, mog_variances, weights, r_t_z0, mmd_loss, args, results_dir,
        )


if __name__ == "__main__":
    main()
