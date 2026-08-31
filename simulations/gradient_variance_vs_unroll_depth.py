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
  - One or more fixed conditioning points x (--x_conds; defaults to just the
    experiment's x_star), each analyzed independently and reported separately.
  - For each x, one fixed target sample set, drawn once from the analytic
    ground-truth conditional at that x and reused for every trial and every K.
  - For each K in K_VALUES, run the inner MMD-guidance estimator 100 times:
    each run redraws only the sampler's internal noise (the initial Gaussian
    state of the K-step DDIM chain), computes y-samples via a K-step
    accelerated DDIM unroll conditioned on x, forms the MMD loss against the
    fixed target samples, and backpropagates to get grad = d(loss)/dx.
  - Report Var(grad) (trace of the empirical covariance across the 100
    gradient draws) normalized by ||mean(grad)||^2, for each K.
  - Also compute the TRUE/population reference gradient at each x: the
    exact analytic conditional GMM at x (differentiable w.r.t. x via
    compute_conditionals/compute_alpha), sampled at --grad_ref_n and scored
    by MMD against the SAME fixed target samples -- no network forward, so
    this is cheap even at grad_ref_n >> nsamples. ||mean(grad) - grad_ref||
    at each K then tells you whether the estimator's mean is actually near
    the true gradient, or has instead plateaued near zero (or some other
    wrong value) as K grows -- distinguishing "high variance but roughly
    right on average" from "the estimator itself is biased/collapsed".

Var(grad)/||mean(grad)||^2 answers "is deeper unrolling noisier"; distance
to the true reference gradient answers the separate question "is deeper
unrolling's *mean* estimate even correct" -- both matter and neither implies
the other (a low-variance estimator can still be consistently wrong).

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
    p.add_argument("--x_conds", action="append", nargs="+", type=float, default=None,
                    help="A conditioning point x to analyze -- pass this flag once per point "
                         "to sweep several (e.g. --x_conds -5 --x_conds 0 --x_conds 5 for a "
                         "1D-conditioning experiment). Each occurrence needs exactly "
                         "condition_on values. Mutually exclusive with --n_random_conds. "
                         "Defaults to just the experiment's saved x_star.")
    p.add_argument("--n_random_conds", type=int, default=None,
                    help="Instead of fixed --x_conds, draw this many conditioning points "
                         "at random from the GMM's own marginal distribution over x (the "
                         "first condition_on coordinates of a joint draw) -- avoids "
                         "hand-picking points, and is what makes a 2D/5D/10D comparison fair "
                         "(the same fixed-point choice doesn't generalize across dimensions). "
                         "Mutually exclusive with --x_conds.")
    p.add_argument("--grad_ref_n", type=int, default=2000,
                    help="Sample size for the TRUE/population reference gradient computed at "
                         "each x (closed-form differentiable sampling from the exact analytic "
                         "conditional GMM, no network forward -- cheap even at this size). Used "
                         "to report each K's distance to the true gradient, not just its own "
                         "spread across trials.")
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

    if args.x_conds is not None and args.n_random_conds is not None:
        raise ValueError("--x_conds and --n_random_conds are mutually exclusive.")

    if args.x_conds is not None:
        x_fixed_list = [torch.tensor(x, dtype=torch.float32) for x in args.x_conds]
        for xf in x_fixed_list:
            if xf.numel() != condition_on:
                raise ValueError(f"--x_conds must have {condition_on} value(s) per occurrence, got {xf.numel()}")
    elif args.n_random_conds is not None:
        # Sample from the GMM's own marginal over x: a joint draw's first condition_on
        # coordinates (the joint GMM already puts x in the leading coordinates -- same
        # convention as target_samples[:, model_cond.condition_on:] elsewhere). This is
        # the actual distribution x is drawn from during real guidance, unlike a hand-picked
        # point -- and is what makes a 2D/5D/10D comparison meaningful, since a fixed choice
        # like x=0 doesn't mean the same thing (same density, same "difficulty") in each.
        joint_samples = dist_utils.generate_mog_samples_not_differentiable(
            args.n_random_conds, mu_list, Sigma_list, alpha
        ).float()
        x_fixed_list = [joint_samples[i, :condition_on] for i in range(args.n_random_conds)]
    else:
        x_fixed_list = [x_star.float().view(-1)]
    print(f"[GradVar] conditioning points to analyze: {[xf.tolist() for xf in x_fixed_list]}")

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

    mmd_loss = MMDLoss(kernel=RBF())

    results_by_x = {}
    for x_fixed in x_fixed_list:
        x_key = "_".join(f"{v:.4g}" for v in x_fixed.tolist())
        print(f"[GradVar] === conditioning point x = {x_fixed.tolist()} ===")

        # fixed target samples: drawn once from the analytic ground-truth conditional at x_fixed
        mu_cond, Sigma_cond = dist_utils.compute_conditionals(mu_list, Sigma_list, x_fixed)
        w_cond = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_fixed)
        target_samples = dist_utils.generate_mog_samples_not_differentiable(
            n_target, mu_cond, Sigma_cond, w_cond
        ).float().to(device)

        # TRUE/population reference gradient at this x: differentiate MMD(ref_samples(x),
        # target_samples) w.r.t. x, where ref_samples(x) is a large, differentiable draw from
        # the EXACT analytic conditional GMM at x (not model_cond's learned approximation).
        # Independent of K -- computed once per x, not once per K.
        x_ref_leaf = x_fixed.clone().detach().to(device).requires_grad_(True)
        condi_mu, condi_sigma = dist_utils.compute_conditionals(mu_list, Sigma_list, x_ref_leaf)
        condi_mu = condi_mu.squeeze(-1)  # drop a spurious trailing dim; generate_mog_samples'
                                          # is_multivariate check looks at xi.shape[-1]
        condi_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_ref_leaf)
        ref_samples = dist_utils.generate_mog_samples(args.grad_ref_n, condi_mu, condi_sigma, condi_alpha, device=device)
        loss_ref = mmd_loss(ref_samples, target_samples)
        grad_ref = torch.autograd.grad(loss_ref, x_ref_leaf)[0].detach().cpu().numpy()
        grad_ref_norm = float(np.linalg.norm(grad_ref))
        print(f"  x={x_fixed.tolist()} | TRUE reference grad: ||grad_ref||={grad_ref_norm:.6f}")

        results_by_K = {}
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

            # Distance from the estimator's mean to the TRUE reference gradient: separates
            # "did the estimate collapse near zero" (mean_grad_norm -> 0 while grad_ref_norm
            # stays large, so dist_to_ref_normalized -> 1) from "did it converge to the real
            # value" (dist_to_ref_normalized -> 0), which normalized_variance alone can't tell.
            dist_to_ref = float(np.linalg.norm(mean_grad - grad_ref))
            dist_to_ref_normalized = dist_to_ref / (grad_ref_norm + 1e-12)

            results_by_K[K] = {
                "K_requested": K,
                "K_actual_steps": actual_steps,
                "mean_grad": mean_grad.tolist(),
                "mean_grad_norm": float(np.sqrt(mean_grad_norm_sq)),
                "variance_trace": variance_trace,
                "normalized_variance": normalized_variance,
                "dist_to_ref": dist_to_ref,
                "dist_to_ref_normalized": dist_to_ref_normalized,
                "grads": grads.tolist(),
            }
            print(f"  x={x_fixed.tolist()} | K={K:>3} (actual {actual_steps:>3} steps) | "
                  f"||ḡ||={results_by_K[K]['mean_grad_norm']:.6f} | "
                  f"Var(g)={variance_trace:.6e} | "
                  f"Var(g)/||ḡ||^2={normalized_variance:.6e} | "
                  f"||ḡ-grad_ref||={dist_to_ref:.6f} (normalized {dist_to_ref_normalized:.4f})")

        Ks_sorted = sorted(results_by_K.keys())
        nv = [results_by_K[k]["normalized_variance"] for k in Ks_sorted]
        monotonic_nondecreasing = all(nv[i] <= nv[i + 1] * 1.0 for i in range(len(nv) - 1))
        print(f"[GradVar] x={x_fixed.tolist()} | normalized variance by K: {list(zip(Ks_sorted, nv))}")
        print(f"[GradVar] x={x_fixed.tolist()} | strictly nondecreasing across the full K sweep: {monotonic_nondecreasing}")

        results_by_x[x_key] = {
            "x_fixed": x_fixed.tolist(),
            "grad_ref": grad_ref.tolist(),
            "grad_ref_norm": grad_ref_norm,
            "results_by_K": results_by_K,
        }

    out = {
        "experiment": args.experiment_name,
        "n_trials": args.n_trials,
        "nsamples": args.nsamples,
        "n_target": n_target,
        "grad_ref_n": args.grad_ref_n,
        "seed": args.seed,
        "results_by_x": results_by_x,
    }

    out_path = os.path.join(RESULTS_DIR, f"{args.experiment_name}_gradient_variance_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[GradVar] Saved results to {out_path}")


if __name__ == "__main__":
    main()
