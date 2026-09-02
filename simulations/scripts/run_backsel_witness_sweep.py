"""
Sweep over n (nsamples) x k/n (backsel_k as a fraction of nsamples) x selection
rule (uniform vs. MMD witness-function importance sampling), comparing both
against the "full" (no subsampling, backsel_k=None) baseline on the same four
metrics as the rest of this repo's grid scripts: L2 GMM, L2 to x*, MMD, and wall
time. Answers: under which (n, k/n) does witness selection actually help vs.
uniform, and how close does either get to the full-gradient baseline?

For one experiment (2D_cond_1D / 5D_cond_1D / 10D_cond_1D) and a fixed num_x_t,
sweeps every (nsamples, k_frac, rule) combination for the LGD and/or LGD-CM
methods. k_frac=1.0 collapses to the same "full" run for both rules (backprop
through every sample -- there is nothing to select), so it is computed ONCE per
(method, nsamples) and reused, rather than re-run per rule.

Writes out, per experiment:
  - a JSON file with every run's final_loss (MMD), l2_gmm, l2_x, time
  - a summary CSV (mean/std per (method, nsamples, rule, k_frac), each point's
    delta from the "full" (k_frac=1.0) baseline at the same nsamples, and the
    witness-vs-uniform delta at matching (nsamples, k_frac))
  - PNG curve plots: per method, one figure per metric, rows = nsamples values,
    x-axis = k_frac, one line per rule + a dashed "full" baseline reference
  - PNG heatmaps of (witness - uniform) delta over the (nsamples, k_frac) grid,
    one per metric -- negative (for loss-type metrics) = witness helped

For per-step gradient-variance diagnostics (uniform vs. witness vs. full, at a
handful of frozen trajectory states, redrawn 200+ times each) see the separate,
purpose-built scripts/backsel_state_gradient_variance.py instead -- this sweep
only ever measures the combined end-to-end effect of many steps together.

One optional diagnostic axis, off by default (no extra cost unless requested):

  --alpha_list A1 A2 ...   Sweeps witness_floor over these values for rule=
    'witness' (uniform doesn't use a floor). --witness_floor's value is always
    included (it's the "canonical" alpha that feeds the main JSON/plots/summary,
    so those stay well-defined regardless of what else is swept). Writes an
    extra *_alpha_sweep.csv when more than one value is given.

Example:
    python run_backsel_witness_sweep.py --experiment 5D_cond_1D --num_x_t 3 \
        --n_runs 25 --nsamples_list 50 100 250 500 --k_fracs 0.1 0.2 0.5 1.0 \
        --alpha_list 0.0 0.15 0.3 0.5
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
from gmm_experiment_setup import EXPERIMENT_CONFIGS, load_or_generate_gmm_params, load_or_train_models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--num_x_t", type=int, default=3, help="Fixed num_x_t for this sweep (not swept).")
    p.add_argument("--nsamples_list", type=int, nargs="+", default=[50, 100, 250, 500],
                   help="The 'n' axis: values of nsamples to sweep.")
    p.add_argument("--k_fracs", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0],
                   help="The 'proportion' axis: backsel_k / nsamples. 1.0 = no "
                        "subsampling (the 'full' baseline, computed once per "
                        "nsamples and shared across rules).")
    p.add_argument("--rules", nargs="+", choices=["uniform", "witness"], default=["uniform", "witness"])
    p.add_argument("--witness_floor", type=float, default=0.3,
                   help="Defensive-mixture floor for rule='witness' (see witness_utils.py). Also "
                        "the canonical alpha used for plots/JSON/summary when --alpha_list is not "
                        "given (or is given but doesn't otherwise include this value -- it's always "
                        "added to the swept set).")
    p.add_argument("--alpha_list", type=float, nargs="+", default=None,
                   help="Sweep witness_floor over these values (rule='witness' only -- 'uniform' "
                        "doesn't use a floor). Defaults to just [--witness_floor], i.e. no extra "
                        "sweep. --witness_floor's value is always included even if you list others.")
    p.add_argument("--backsel_replacement", action="store_true",
                   help="Sample the backsel_k indices with replacement (default: without)")
    p.add_argument("--normalize_by_k_frac", action="store_true",
                   help="Rescale the applied (subsampled) gradient by 1/k_frac (k_frac = "
                        "backsel_k/nsamples), so gradient magnitude is comparable across "
                        "different k_frac values instead of scaling ~linearly with k_frac. "
                        "No-op at k_frac=1.0 (the 'full' baseline). Default: off (original, "
                        "unnormalized behavior).")
    p.add_argument("--use_inv_sqrt_alpha_scale", action="store_true",
                   help="Scale the guidance gradient by 1/sqrt(model_uncond.alphas[t]) instead "
                        "of the constant zeta. Default: off (use zeta).")
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD", "LGD-CM"])
    p.add_argument("--n_runs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None,
                   help="Defaults to simulations/ (parent of this script's directory).")
    return p.parse_args()


def run_grid_point(model_uncond, cond_model, CM_flag, num_x_t, nsamples, backsel_k, backsel_rule,
                   witness_floor, backsel_replacement, n_runs, global_seed, device,
                   mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star, label,
                   normalize_by_k_frac=False, use_inv_sqrt_alpha_scale=False):
    """Returns metrics = {"final_loss", "l2_gmm", "l2_x", "times"} -- what feeds the JSON/summary
    CSV/plots. For per-step gradient diagnostics, use backsel_state_gradient_variance.py instead."""
    final_loss_list, l2_gmm_list, l2_x_list, times = [], [], [], []
    for i in range(n_runs):
        run_seed = experiment_utils.set_run_seed(global_seed, i)
        backsel_generator = torch.Generator().manual_seed(run_seed)

        start_time = time.time()
        best_x_t, _, final_loss = Optimization.optimize_LGD(
            model_uncond, cond_model, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha,
            nsamples=nsamples, loss="MMD", device=device,
            num_x_t=num_x_t, CM=CM_flag,
            backsel_k=backsel_k, backsel_rule=backsel_rule,
            witness_floor=witness_floor, backsel_replacement=backsel_replacement,
            backsel_generator=backsel_generator, normalize_by_k_frac=normalize_by_k_frac,
            use_inv_sqrt_alpha_scale=use_inv_sqrt_alpha_scale,
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

        print(f"[{label} | n={nsamples} k={backsel_k} | {i + 1}/{n_runs}] "
              f"seed={run_seed} | L2 GMM: {l2_gmm:.6f} | L2 to x*: {l2_x:.6f} "
              f"| MMD: {final_loss.item():.6f} | time: {elapsed:.2f}s", flush=True)

    return {"final_loss": final_loss_list, "l2_gmm": l2_gmm_list, "l2_x": l2_x_list, "times": times}


METRICS = [("l2_gmm", "L2 GMM"), ("l2_x", "L2 to x*"), ("final_loss", "MMD")]


def make_curve_plots(results, methods, nsamples_list, k_fracs, rules, plots_dir, experiment_name, num_x_t):
    os.makedirs(plots_dir, exist_ok=True)
    colors = {"uniform": "steelblue", "witness": "crimson"}

    for method in methods:
        fig, axes = plt.subplots(len(nsamples_list), len(METRICS),
                                 figsize=(5 * len(METRICS), 4 * len(nsamples_list)), squeeze=False)
        for row, n in enumerate(nsamples_list):
            baseline = results[method][n]["full"]
            for col, (key, label) in enumerate(METRICS):
                ax = axes[row][col]
                baseline_mean = np.mean(baseline[key])
                ax.axhline(baseline_mean, color="black", linestyle="--", linewidth=1.2,
                          label="full (no subsampling)")
                for rule in rules:
                    means = [np.mean(results[method][n][rule][kf][key]) for kf in k_fracs]
                    stds = [np.std(results[method][n][rule][kf][key]) for kf in k_fracs]
                    ax.errorbar(k_fracs, means, yerr=stds, marker="o", capsize=3,
                               color=colors.get(rule, "gray"), label=rule)
                ax.set_xlabel("k / n (backsel_k / nsamples)")
                ax.set_ylabel(label)
                ax.set_title(f"n={n} | {label}")
                ax.grid(True, alpha=0.3)
                if row == 0 and col == len(METRICS) - 1:
                    ax.legend(fontsize=8)
        fig.suptitle(f"{experiment_name} | {method} | num_x_t={num_x_t} "
                     f"-- uniform vs. witness backsel, at fixed nsamples per row")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"curves_{method}.png"), dpi=150)
        plt.close(fig)


def make_delta_heatmaps(results, methods, nsamples_list, k_fracs, plots_dir, experiment_name, num_x_t):
    """Heatmap of (witness_mean - uniform_mean) over the (nsamples, k_frac) grid.
    Negative (for these loss-type metrics, lower=better) means witness beat uniform."""
    os.makedirs(plots_dir, exist_ok=True)
    sub_k_fracs = [kf for kf in k_fracs if kf < 1.0]  # k_frac=1.0 has no uniform/witness distinction
    if not sub_k_fracs:
        return

    for method in methods:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5))
        for ax, (key, label) in zip(axes, METRICS):
            grid = np.array([
                [np.mean(results[method][n]["witness"][kf][key]) - np.mean(results[method][n]["uniform"][kf][key])
                 for kf in sub_k_fracs]
                for n in nsamples_list
            ])
            vmax = np.abs(grid).max() if grid.size else 1.0
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(sub_k_fracs)))
            ax.set_xticklabels([f"{kf:.2f}" for kf in sub_k_fracs])
            ax.set_yticks(range(len(nsamples_list)))
            ax.set_yticklabels([str(n) for n in nsamples_list])
            ax.set_xlabel("k / n")
            ax.set_ylabel("nsamples (n)")
            ax.set_title(f"{label}: witness - uniform\n(blue = witness better)")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{experiment_name} | {method} | num_x_t={num_x_t}")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"delta_heatmap_{method}.png"), dpi=150)
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
    # Included in every output filename below so two concurrent jobs against the
    # same results_dir/num_x_t/seed (e.g. one running --methods LGD, the other
    # --methods LGD-CM, split across two machines) never overwrite each other's
    # JSON/CSVs -- those were previously named only by (num_x_t, seed).
    methods_tag = "-".join(methods)
    nsamples_list = args.nsamples_list
    k_fracs = sorted(set(args.k_fracs) | {1.0})  # always include the 1.0 baseline point
    rules = args.rules
    # alpha sweep (witness rule only): witness_floor's own value is always included, so
    # plots/JSON/summary (which key off a single "canonical" witness result per kf) stay
    # well-defined even if --alpha_list doesn't happen to list it.
    alpha_list = sorted(set(args.alpha_list or [args.witness_floor]) | {args.witness_floor})
    canonical_alpha = args.witness_floor

    results = {}
    alpha_metrics_rows = []  # every (method, n, kf, alpha) witness point's summary stats,
                             # captured inline as they're computed -- feeds the alpha-sweep CSV
    for method in methods:
        results[method] = {}
        for n in nsamples_list:
            results[method][n] = {"full": None}
            results[method][n]["full"] = run_grid_point(
                model_uncond, method_models[method]["cond_model"], method_models[method]["CM_flag"],
                args.num_x_t, n, None, "uniform", args.witness_floor, args.backsel_replacement,
                args.n_runs, args.seed, device,
                mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
                label=f"{method}-full", normalize_by_k_frac=args.normalize_by_k_frac,
                use_inv_sqrt_alpha_scale=args.use_inv_sqrt_alpha_scale,
            )
            for rule in rules:
                results[method][n][rule] = {}
                for kf in k_fracs:
                    if kf >= 1.0:
                        results[method][n][rule][kf] = results[method][n]["full"]
                        continue
                    k = max(1, round(kf * n))
                    if rule == "witness":
                        # Sweep alpha_list for witness; the canonical alpha's result is what
                        # feeds the existing JSON/summary/plots (unchanged when alpha_list is
                        # just [witness_floor], i.e. the default).
                        for a in alpha_list:
                            metrics_a = run_grid_point(
                                model_uncond, method_models[method]["cond_model"], method_models[method]["CM_flag"],
                                args.num_x_t, n, k, "witness", a, args.backsel_replacement,
                                args.n_runs, args.seed, device,
                                mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
                                label=f"{method}-witness-alpha{a}", normalize_by_k_frac=args.normalize_by_k_frac,
                                use_inv_sqrt_alpha_scale=args.use_inv_sqrt_alpha_scale,
                            )
                            if a == canonical_alpha:
                                results[method][n][rule][kf] = metrics_a
                            alpha_metrics_rows.append({
                                "method": method, "nsamples": n, "k_frac": kf, "alpha": a,
                                "L2_GMM_mean": np.mean(metrics_a["l2_gmm"]), "L2_GMM_std": np.std(metrics_a["l2_gmm"]),
                                "L2_x_mean": np.mean(metrics_a["l2_x"]), "L2_x_std": np.std(metrics_a["l2_x"]),
                                "MMD_mean": np.mean(metrics_a["final_loss"]), "MMD_std": np.std(metrics_a["final_loss"]),
                            })
                    else:
                        results[method][n][rule][kf] = run_grid_point(
                            model_uncond, method_models[method]["cond_model"], method_models[method]["CM_flag"],
                            args.num_x_t, n, k, rule, args.witness_floor, args.backsel_replacement,
                            args.n_runs, args.seed, device,
                            mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
                            label=f"{method}-{rule}", normalize_by_k_frac=args.normalize_by_k_frac,
                            use_inv_sqrt_alpha_scale=args.use_inv_sqrt_alpha_scale,
                        )

    out = {
        "experiment": args.experiment,
        "seed": args.seed,
        "environment": env_info,
        "meta": {
            "num_x_t": args.num_x_t,
            "n_runs": args.n_runs,
            "nsamples_list": nsamples_list,
            "k_fracs": k_fracs,
            "rules": rules,
            "witness_floor": args.witness_floor,
            "alpha_list": alpha_list,
            "backsel_replacement": args.backsel_replacement,
            "normalize_by_k_frac": args.normalize_by_k_frac,
            "use_inv_sqrt_alpha_scale": args.use_inv_sqrt_alpha_scale,
            "methods": methods,
            "x_star": x_star.detach().cpu().tolist() if isinstance(x_star, torch.Tensor) else x_star,
        },
        "results": {
            method: {
                str(n): {
                    "full": results[method][n]["full"],
                    **{rule: {str(kf): results[method][n][rule][kf] for kf in k_fracs} for rule in rules},
                }
                for n in nsamples_list
            }
            for method in methods
        },
    }

    json_path = os.path.join(results_dir, f"witness_sweep_numxt{args.num_x_t}_seed{args.seed}_{methods_tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Results] Saved JSON to {json_path}")

    summary_rows = []
    for method in methods:
        for n in nsamples_list:
            baseline_means = {key: np.mean(results[method][n]["full"][key]) for key, _ in METRICS}
            for rule in rules:
                for kf in k_fracs:
                    r = results[method][n][rule][kf]
                    row = {
                        "method": method, "nsamples": n, "rule": rule, "k_frac": kf,
                        "backsel_k": ("full" if kf >= 1.0 else max(1, round(kf * n))),
                        "L2_GMM_mean": np.mean(r["l2_gmm"]), "L2_GMM_std": np.std(r["l2_gmm"]),
                        "L2_x_mean": np.mean(r["l2_x"]), "L2_x_std": np.std(r["l2_x"]),
                        "MMD_mean": np.mean(r["final_loss"]), "MMD_std": np.std(r["final_loss"]),
                        "Time_mean_s": np.mean(r["times"]), "Time_std_s": np.std(r["times"]),
                    }
                    row["L2_GMM_delta_vs_full"] = row["L2_GMM_mean"] - baseline_means["l2_gmm"]
                    row["L2_x_delta_vs_full"] = row["L2_x_mean"] - baseline_means["l2_x"]
                    row["MMD_delta_vs_full"] = row["MMD_mean"] - baseline_means["final_loss"]
                    summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # witness-vs-uniform delta at matching (n, k_frac), for a quick "did it help" scan
    if "witness" in rules and "uniform" in rules:
        piv = summary_df.pivot_table(index=["method", "nsamples", "k_frac"], columns="rule",
                                     values=["MMD_mean", "L2_GMM_mean", "L2_x_mean"])
        wu_rows = []
        for (method, n, kf), _ in piv.iterrows():
            if kf >= 1.0:
                continue
            wu_rows.append({
                "method": method, "nsamples": n, "k_frac": kf,
                "MMD_witness_minus_uniform": piv.loc[(method, n, kf), ("MMD_mean", "witness")]
                                            - piv.loc[(method, n, kf), ("MMD_mean", "uniform")],
                "L2_GMM_witness_minus_uniform": piv.loc[(method, n, kf), ("L2_GMM_mean", "witness")]
                                                - piv.loc[(method, n, kf), ("L2_GMM_mean", "uniform")],
                "L2_x_witness_minus_uniform": piv.loc[(method, n, kf), ("L2_x_mean", "witness")]
                                             - piv.loc[(method, n, kf), ("L2_x_mean", "uniform")],
            })
        wu_df = pd.DataFrame(wu_rows)
        wu_csv_path = os.path.join(results_dir, f"witness_sweep_numxt{args.num_x_t}_seed{args.seed}_{methods_tag}_witness_vs_uniform.csv")
        wu_df.to_csv(wu_csv_path, index=False)
        print(f"[Results] Saved witness-vs-uniform delta table to {wu_csv_path}")
        print(wu_df.to_string(index=False))

    csv_path = os.path.join(results_dir, f"witness_sweep_numxt{args.num_x_t}_seed{args.seed}_{methods_tag}_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[Results] Saved summary CSV to {csv_path}")

    # Alpha sweep (witness_floor), only meaningful when --alpha_list listed more than one value.
    # alpha_metrics_rows was captured inline in the grid-building loop above, as each
    # (method, n, kf, alpha) witness point was computed.
    if "witness" in rules and len(alpha_list) > 1:
        alpha_csv_path = os.path.join(
            results_dir, f"witness_sweep_numxt{args.num_x_t}_seed{args.seed}_{methods_tag}_alpha_sweep.csv"
        )
        alpha_df = pd.DataFrame(alpha_metrics_rows)
        alpha_df.to_csv(alpha_csv_path, index=False)
        print(f"[Results] Saved alpha-sweep summary to {alpha_csv_path}")

    make_curve_plots(results, methods, nsamples_list, k_fracs, rules, plots_dir, args.experiment, args.num_x_t)
    if "witness" in rules and "uniform" in rules:
        make_delta_heatmaps(results, methods, nsamples_list, k_fracs, plots_dir, args.experiment, args.num_x_t)
    print(f"[Results] Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
