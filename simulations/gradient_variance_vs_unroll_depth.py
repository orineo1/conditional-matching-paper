#!/usr/bin/env python
"""
Direct test of the paper's claim that unrolling deeper diffusion chains
injects more noise into the guidance gradient (Reviewer Q1: "Why does
MLGD-F outperform LGD in high dimensions despite lower conditional
fidelity? The authors attribute this to gradient variance from unrolled
diffusion chains. Is there a more direct way to test this hypothesis?").

Design (isolates the inner sampler only -- no outer diffusion loop, no
optimization, nothing about x is ever updated):
  - One model, one checkpoint: the pretrained *conditional* diffusion model
    model_cond (P(Y|X=x)), trained for the given synthetic experiment.
  - One fixed conditioning point x (defaults to the experiment's x_star).
  - One fixed target sample set, drawn once from the analytic ground-truth
    conditional at x and reused for every trial and every K.
  - For each K in K_VALUES, run the inner MMD-guidance estimator 100 times:
    each run redraws only the sampler's internal noise (the initial Gaussian
    state of the K-step DDIM chain), computes y-samples via a K-step
    accelerated DDIM unroll conditioned on x, forms the MMD loss against the
    fixed target samples, and backpropagates to get grad = d(loss)/dx.
  - Report Var(grad) (trace of the empirical covariance across the 100
    gradient draws) normalized by ||mean(grad)||^2, for each K.

If normalized variance rises with K, deeper unrolled chains do inject more
noise into the guidance signal -- directly, without any confound from
whether a given K is a more or less *accurate* sampler (accuracy plays no
role in this measurement: we never compare to ground truth, only to the
estimator's own spread across noise draws for a fixed x).

Usage:
    python gradient_variance_vs_unroll_depth.py --experiment_name 2D_cond_1D
"""
import os
import sys
import json
import argparse

import numpy as np
import torch


def ddim_sample_kstep(model, nsamples, condition_x, K, device):
    """Differentiable, deterministic (eta=0) DDIM sampling from `model`,
    using an evenly-spaced K-step subsequence of the model's trained noise
    schedule (accelerated DDIM respacing) -- the standard way to control
    unroll depth without retraining. Gradients flow from the output back to
    `condition_x`. Returns (full_sample, y_only, n_steps_actually_taken).
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment_name", type=str, default="2D_cond_1D",
                    choices=["2D_cond_1D", "5D_cond_1D", "10D_cond_1D"])
    p.add_argument("--k_values", type=str, default="10,25,40,60,80,100",
                    help="Comma-separated list of unroll depths (number of DDIM steps) to test")
    p.add_argument("--n_trials", type=int, default=100,
                    help="Number of independent gradient draws per K (noise redrawn each time)")
    p.add_argument("--nsamples", type=int, default=250,
                    help="Batch size for the inner MMD estimator (matches the paper's MC budget)")
    p.add_argument("--n_target", type=int, default=None,
                    help="Number of fixed target samples (defaults to --nsamples)")
    p.add_argument("--seed", type=int, default=42,
                    help="Global seed; also selects which pretrained checkpoint file to load")
    p.add_argument("--x_star", type=float, nargs="*", default=None,
                    help="Override the fixed conditioning point x (defaults to the saved x_star)")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    ARCH = {
        "2D_cond_1D":  dict(nblocks=3, nunits=128, diffusion_steps=100, condition_on=1),
        "5D_cond_1D":  dict(nblocks=6, nunits=512, diffusion_steps=100, condition_on=4),
        "10D_cond_1D": dict(nblocks=8, nunits=512, diffusion_steps=100, condition_on=9),
    }[args.experiment_name]

    K_values = [int(k) for k in args.k_values.split(",")]
    n_target = args.n_target or args.nsamples

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARAMS_DIR = os.path.join(BASE_DIR, "params")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", args.experiment_name)
    RESULTS_DIR = args.output_dir or os.path.join(BASE_DIR, "results", args.experiment_name, "gradient_variance")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    import experiment_utils
    import dist_utils
    import Diffusion
    from LossFunctions import MMDLoss, RBF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradVar] experiment={args.experiment_name} K_values={K_values} "
          f"n_trials={args.n_trials} nsamples={args.nsamples} device={device}")

    experiment_utils.set_global_seed(args.seed)

    loaded = experiment_utils.load_gmm_params(PARAMS_DIR, args.experiment_name)
    if loaded is None:
        raise FileNotFoundError(f"No GMM params found under {PARAMS_DIR} for '{args.experiment_name}'.")
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
    mu_list = [mu.float() for mu in mu_list]
    Sigma_list = [cov.float() for cov in Sigma_list]
    alpha = alpha.float()

    condition_on = ARCH["condition_on"]
    nfeatures_full = mu_list[0].shape[0]  # model_cond denoises the full (x, y) vector jointly

    if args.x_star is not None:
        x_fixed = torch.tensor(args.x_star, dtype=torch.float32)
        if x_fixed.numel() != condition_on:
            raise ValueError(f"--x_star must have {condition_on} value(s), got {x_fixed.numel()}")
    else:
        x_fixed = x_star.float().view(-1)
    print(f"[GradVar] fixed conditioning point x = {x_fixed.tolist()}")

    # ── pretrained conditional diffusion model P(Y|X=x) ─────────────────────
    model_cond = Diffusion.DiffusionModel(
        nfeatures=nfeatures_full, nblocks=ARCH["nblocks"], nunits=ARCH["nunits"],
        condition=True, condition_on=condition_on, diffusion_steps=ARCH["diffusion_steps"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_cond, "Diffusion_cond", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError(
            "Could not load or download the pretrained conditional diffusion model checkpoint "
            f"({args.experiment_name}_Diffusion_cond_seed{args.seed}.pt). Train it via "
            f"notebooks/Exp_{args.experiment_name}.ipynb (cell training model_cond) if you don't "
            "want to rely on the HuggingFace fallback download."
        )
    model_cond.to(device)
    model_cond.eval()

    # ── fixed target samples: drawn once from the analytic ground-truth conditional at x_fixed ──
    mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_fixed)
    w_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_fixed)
    target_samples = dist_utils.generate_mog_samples_not_differentiable(
        n_target, mu_cond, Sigma_cond, w_cond
    ).float().to(device)

    mmd_loss = MMDLoss(kernel=RBF())
    condition_x = x_fixed.view(1, -1).repeat(args.nsamples, 1)

    results = {}
    for K in K_values:
        grads = []
        actual_steps = None
        for trial in range(args.n_trials):
            experiment_utils.set_run_seed(args.seed, trial)  # only the sampler's noise draw changes across trials

            x_leaf = x_fixed.clone().detach().to(device).requires_grad_(True)
            cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)

            _, y_samples, actual_steps = ddim_sample_kstep(model_cond, args.nsamples, cond, K, device)
            loss = mmd_loss(y_samples, target_samples)
            loss.backward()

            grads.append(x_leaf.grad.detach().cpu().numpy().copy())

        grads = np.stack(grads, axis=0)  # [n_trials, condition_on]
        mean_grad = grads.mean(axis=0)
        centered = grads - mean_grad
        variance_trace = float(np.mean(np.sum(centered ** 2, axis=1)))  # E[||g - ḡ||^2]
        mean_grad_norm_sq = float(np.sum(mean_grad ** 2))
        normalized_variance = variance_trace / (mean_grad_norm_sq + 1e-12)

        results[K] = {
            "K_requested": K,
            "K_actual_steps": actual_steps,
            "mean_grad": mean_grad.tolist(),
            "mean_grad_norm": float(np.sqrt(mean_grad_norm_sq)),
            "variance_trace": variance_trace,
            "normalized_variance": normalized_variance,
            "grads": grads.tolist(),
        }
        print(f"  K={K:>3} (actual {actual_steps:>3} steps) | "
              f"||ḡ||={results[K]['mean_grad_norm']:.6f} | "
              f"Var(g)={variance_trace:.6e} | "
              f"Var(g)/||ḡ||^2={normalized_variance:.6e}")

    out = {
        "experiment": args.experiment_name,
        "x_fixed": x_fixed.tolist(),
        "n_trials": args.n_trials,
        "nsamples": args.nsamples,
        "n_target": n_target,
        "seed": args.seed,
        "results_by_K": results,
    }

    out_path = os.path.join(RESULTS_DIR, f"{args.experiment_name}_gradient_variance_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[GradVar] Saved results to {out_path}")

    Ks_sorted = sorted(results.keys())
    nv = [results[k]["normalized_variance"] for k in Ks_sorted]
    monotonic_nondecreasing = all(nv[i] <= nv[i + 1] * 1.0 for i in range(len(nv) - 1))
    print(f"[GradVar] Normalized variance by K: {list(zip(Ks_sorted, nv))}")
    print(f"[GradVar] Strictly nondecreasing across the full K sweep: {monotonic_nondecreasing}")


if __name__ == "__main__":
    main()
