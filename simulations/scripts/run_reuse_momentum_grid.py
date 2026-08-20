"""
2D grid search over optimize_LGD's reuse_frac x momentum, comparing every
grid point back to the "regular" baseline (reuse_frac=0.0, momentum=0.0 —
full fresh samples every step, no gradient smoothing) on all four metrics:
L2 GMM, L2 to x*, MMD, and wall time. Runnable standalone / via sbatch.

For one experiment (2D_cond_1D / 5D_cond_1D / 10D_cond_1D) and a single
fixed num_x_t, sweeps every (reuse_frac, momentum) combination — default
reuse_frac in {0.0..0.9 step 0.1}, momentum in {0.0, 0.9} (no smoothing vs.
the standard Adam beta1) — for the LGD and/or LGD-CM methods. When
momentum > 0, beta2 (default 0.999, the standard Adam value) is also
passed to optimize_LGD, switching on the full Adam-style adaptive update
(see Optimization.py); momentum == 0 always means the raw, unsmoothed
gradient regardless of beta2.

Writes out, per experiment:
  - a JSON file with every run's final_loss (MMD), l2_gmm, l2_x, time
  - a summary CSV (mean/std per (method, reuse_frac, momentum), plus each
    point's delta from the (0.0, 0.0) baseline)
  - PNG heatmaps: L2 GMM, L2 to x*, MMD, wall time — one grid per method,
    reuse_frac x momentum, with the baseline cell outlined
  - PNG baseline-comparison bar charts (same four metrics)

Example:
    python run_reuse_momentum_grid.py --experiment 5D_cond_1D --num_x_t 3 \
        --n_runs 25 --methods LGD LGD-CM
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dist_utils
import Optimization
import experiment_utils
from sweep_common import EXPERIMENT_CONFIGS, load_or_generate_gmm_params, load_or_train_models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--num_x_t", type=int, default=3, help="Fixed num_x_t for this sweep (not swept).")
    p.add_argument("--reuse_fracs", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--momentums", type=float, nargs="+", default=[0.0, 0.9])
    p.add_argument("--beta2", type=float, default=0.999,
                    help="Adam beta2, used whenever momentum > 0 (see Optimization.optimize_LGD).")
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD", "LGD-CM"])
    p.add_argument("--n_runs", type=int, default=25)
    p.add_argument("--nsamples", type=int, default=250, help="NSAMPLES_IN_OPTIM_FOR_MMD")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None,
                    help="Defaults to simulations/ (parent of this script's directory).")
    return p.parse_args()


def run_grid_point(model_uncond, cond_model, CM_flag, num_x_t, reuse_frac, momentum, beta2,
                    n_runs, nsamples, global_seed, device, mu_list, Sigma_list, alpha,
                    mog_means, mog_variances, weights, x_star, label):
    final_loss_list, l2_gmm_list, l2_x_list, times = [], [], [], []
    for i in range(n_runs):
        run_seed = experiment_utils.set_run_seed(global_seed, i)

        start_time = time.time()
        best_x_t, _, final_loss = Optimization.optimize_LGD(
            model_uncond, cond_model, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha,
            nsamples=nsamples, loss="MMD", device=device,
            num_x_t=num_x_t, CM=CM_flag,
            reuse_frac=reuse_frac, momentum=momentum,
            beta2=(beta2 if momentum > 0.0 else None),
        )
        elapsed = time.time() - start_time
        best_x_t = best_x_t.reshape(-1, 1)

        x_pred_t = best_x_t.float().view(-1).cpu()
        mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x_pred_t)
        w_pred = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_pred_t)
        l2_gmm = dist_utils.gmm_l2_distance(mu_pred, Sigma_pred, w_pred, mog_means, mog_variances, weights)
        l2_x = (x_pred_t - x_star.float().cpu()).pow(2).sum().sqrt().item()

        final_loss_list.append(final_loss.item())
        l2_gmm_list.append(l2_gmm)
        l2_x_list.append(l2_x)
        times.append(elapsed)

        print(f"[{label} | reuse_frac={reuse_frac:.1f} momentum={momentum:.2f} | {i + 1}/{n_runs}] "
              f"seed={run_seed} | L2 GMM: {l2_gmm:.6f} | L2 to x*: {l2_x:.6f} "
              f"| MMD: {final_loss.item():.6f} | time: {elapsed:.2f}s", flush=True)

    return {"final_loss": final_loss_list, "l2_gmm": l2_gmm_list, "l2_x": l2_x_list, "times": times}


METRICS = [("l2_gmm", "L2 GMM"), ("l2_x", "L2 to x*"), ("final_loss", "MMD"), ("times", "Wall time (s)")]


def make_heatmaps(results, methods, reuse_fracs, momentums, plots_dir, experiment_name, num_x_t):
    os.makedirs(plots_dir, exist_ok=True)
    baseline_rf, baseline_m = reuse_fracs[0], momentums[0]  # (0.0, 0.0) by default grids

    for method in methods:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, (key, label) in zip(axes.flat, METRICS):
            grid = np.array([[np.mean(results[method][(rf, m)][key]) for rf in reuse_fracs] for m in momentums])
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(reuse_fracs)))
            ax.set_xticklabels([f"{rf:.1f}" for rf in reuse_fracs])
            ax.set_yticks(range(len(momentums)))
            ax.set_yticklabels([f"{m:.2f}" for m in momentums])
            ax.set_xlabel("reuse_frac")
            ax.set_ylabel("momentum")
            ax.set_title(f"{label} mean")
            fig.colorbar(im, ax=ax, shrink=0.8)

            # Outline the baseline cell (0.0, 0.0) so it's easy to spot on every panel.
            if baseline_rf in reuse_fracs and baseline_m in momentums:
                bi, bj = reuse_fracs.index(baseline_rf), momentums.index(baseline_m)
                ax.add_patch(plt.Rectangle((bi - 0.5, bj - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))

        fig.suptitle(f"{experiment_name} | {method} | num_x_t={num_x_t} "
                      f"(baseline reuse_frac={baseline_rf}, momentum={baseline_m} outlined in red)")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"grid_heatmaps_{method}.png"), dpi=150)
        plt.close(fig)


def make_baseline_comparison(results, methods, reuse_fracs, momentums, plots_dir, experiment_name, num_x_t):
    baseline_rf, baseline_m = reuse_fracs[0], momentums[0]

    for method in methods:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        baseline_r = results[method][(baseline_rf, baseline_m)]
        for ax, (key, label) in zip(axes.flat, METRICS):
            grid_points = [(rf, m) for m in momentums for rf in reuse_fracs]
            means = [np.mean(results[method][pt][key]) for pt in grid_points]
            stds = [np.std(results[method][pt][key]) for pt in grid_points]
            xlabels = [f"rf={rf:.1f}\nm={m:.2f}" for (rf, m) in grid_points]

            colors = ["red" if pt == (baseline_rf, baseline_m) else "steelblue" for pt in grid_points]
            x_pos = np.arange(len(grid_points))
            ax.bar(x_pos, means, yerr=stds, color=colors, capsize=2)
            ax.axhline(np.mean(baseline_r[key]), color="red", linestyle="--", linewidth=1,
                       label=f"baseline mean ({baseline_rf}, {baseline_m})")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
            ax.set_ylabel(label)
            ax.set_title(f"{label}: every grid point vs baseline")
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=7)

        fig.suptitle(f"{experiment_name} | {method} | num_x_t={num_x_t}")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"grid_vs_baseline_{method}.png"), dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    cfg = EXPERIMENT_CONFIGS[args.experiment]

    base_dir = args.base_dir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    params_dir = os.path.join(base_dir, "params")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.experiment)
    results_dir = os.path.join(base_dir, "results", args.experiment)
    plots_dir = os.path.join(results_dir, "plots")
    for d in (checkpoint_dir, results_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    env_info = experiment_utils.get_environment_info()
    experiment_utils.print_environment_info(env_info)

    experiment_utils.set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = load_or_generate_gmm_params(
        cfg, params_dir, results_dir, args.experiment, args.seed
    )
    print(f"x_star = {x_star} | number of conditional modes after filtering: {len(mog_means)}")

    model_uncond, model_cond, model_cm = load_or_train_models(
        cfg, mu_list, Sigma_list, alpha, checkpoint_dir, args.experiment,
        args.seed, device, args.force_retrain,
    )

    method_models = {
        "LGD":    dict(cond_model=model_cond, CM_flag=False),
        "LGD-CM": dict(cond_model=model_cm,   CM_flag=True),
    }
    methods = [m for m in args.methods if m in method_models]

    reuse_fracs = args.reuse_fracs
    momentums = args.momentums
    grid_points = [(rf, m) for m in momentums for rf in reuse_fracs]

    results = {}
    for method in methods:
        results[method] = {}
        for rf, m in grid_points:
            results[method][(rf, m)] = run_grid_point(
                model_uncond, method_models[method]["cond_model"], method_models[method]["CM_flag"],
                args.num_x_t, rf, m, args.beta2, args.n_runs, args.nsamples, args.seed, device,
                mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
                label=method,
            )

    out = {
        "experiment": args.experiment,
        "seed": args.seed,
        "environment": env_info,
        "meta": {
            "num_x_t": args.num_x_t,
            "nsamples_in_optim_for_mmd": args.nsamples,
            "n_runs": args.n_runs,
            "reuse_fracs": reuse_fracs,
            "momentums": momentums,
            "beta2": args.beta2,
            "methods": methods,
            "baseline": {"reuse_frac": reuse_fracs[0], "momentum": momentums[0]},
            "x_star": x_star.detach().cpu().tolist() if isinstance(x_star, torch.Tensor) else x_star,
        },
        "results": {
            method: {f"rf{rf}_m{m}": results[method][(rf, m)] for (rf, m) in grid_points}
            for method in methods
        },
    }

    json_path = os.path.join(results_dir, f"grid_numxt{args.num_x_t}_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Results] Saved JSON to {json_path}")

    baseline_rf, baseline_m = reuse_fracs[0], momentums[0]
    summary_rows = []
    for method in methods:
        baseline_means = {key: np.mean(results[method][(baseline_rf, baseline_m)][key]) for key, _ in METRICS}
        for rf, m in grid_points:
            r = results[method][(rf, m)]
            row = {
                "method": method, "reuse_frac": rf, "momentum": m,
                "L2_GMM_mean": np.mean(r["l2_gmm"]), "L2_GMM_std": np.std(r["l2_gmm"]),
                "L2_x_mean": np.mean(r["l2_x"]), "L2_x_std": np.std(r["l2_x"]),
                "MMD_mean": np.mean(r["final_loss"]), "MMD_std": np.std(r["final_loss"]),
                "Time_mean_s": np.mean(r["times"]), "Time_std_s": np.std(r["times"]),
            }
            row["L2_GMM_delta_vs_baseline"] = row["L2_GMM_mean"] - baseline_means["l2_gmm"]
            row["L2_x_delta_vs_baseline"] = row["L2_x_mean"] - baseline_means["l2_x"]
            row["MMD_delta_vs_baseline"] = row["MMD_mean"] - baseline_means["final_loss"]
            row["Time_speedup_vs_baseline"] = baseline_means["times"] / row["Time_mean_s"]
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(results_dir, f"grid_numxt{args.num_x_t}_seed{args.seed}_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[Results] Saved summary CSV to {csv_path}")

    make_heatmaps(results, methods, reuse_fracs, momentums, plots_dir, args.experiment, args.num_x_t)
    make_baseline_comparison(results, methods, reuse_fracs, momentums, plots_dir, args.experiment, args.num_x_t)
    print(f"[Results] Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
