#!/usr/bin/env python
"""
Plot the output of gradient_variance_vs_unroll_depth.py: normalized
gradient variance (and its raw components) as a function of unroll depth K.

Usage:
    python plot_gradient_variance.py --experiment_name 2D_cond_1D
"""
import os
import json
import argparse

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", type=str, default="2D_cond_1D")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(BASE_DIR, "results", args.experiment_name, "gradient_variance")
    out_dir = args.out_dir or results_dir

    in_path = os.path.join(results_dir, f"{args.experiment_name}_gradient_variance_seed{args.seed}.json")
    with open(in_path) as f:
        data = json.load(f)

    by_K = data["results_by_K"]
    Ks = sorted(int(k) for k in by_K.keys())
    norm_var = [by_K[str(k)]["normalized_variance"] for k in Ks]
    var_trace = [by_K[str(k)]["variance_trace"] for k in Ks]
    grad_norm = [by_K[str(k)]["mean_grad_norm"] for k in Ks]
    actual_steps = [by_K[str(k)]["K_actual_steps"] for k in Ks]

    print(f"{'K':>6} {'actual_steps':>14} {'||ḡ||':>12} {'Var(g)':>14} {'Var(g)/||ḡ||^2':>18}")
    for k, a, g, v, nv in zip(Ks, actual_steps, grad_norm, var_trace, norm_var):
        print(f"{k:>6} {a:>14} {g:>12.6f} {v:>14.6e} {nv:>18.6e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(Ks, norm_var, marker="o", color="tab:red")
    axes[0].set_xlabel("K (DDIM unroll depth)")
    axes[0].set_ylabel("Var(g) / ||ḡ||²")
    axes[0].set_title("Normalized gradient variance vs K")

    axes[1].plot(Ks, var_trace, marker="o", color="tab:blue")
    axes[1].set_xlabel("K (DDIM unroll depth)")
    axes[1].set_ylabel("Var(g) = E[||g - ḡ||²]")
    axes[1].set_title("Raw gradient variance vs K")

    axes[2].plot(Ks, grad_norm, marker="o", color="tab:green")
    axes[2].set_xlabel("K (DDIM unroll depth)")
    axes[2].set_ylabel("||ḡ||")
    axes[2].set_title("Mean gradient magnitude vs K")

    fig.suptitle(
        f"Inner-sampler gradient variance vs unroll depth — {args.experiment_name}\n"
        f"(x fixed at {data['x_fixed']}, {data['n_trials']} noise-only trials per K)"
    )
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{args.experiment_name}_gradient_variance.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
