"""
Minimal grid: optimize_LGD's reuse_frac x adamdps, vs. the baseline
(reuse_frac=0.0, adamdps=False -- full fresh samples every step, raw
gradient). adamdps=True applies AdamDPS gradient stabilization
(arXiv:2603.16797), fixed beta1=0.9, beta2=0.999.

Example:
    python run_reuse_adamdps_grid.py --experiment 5D_cond_1D --n_runs 25
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import pandas as pd

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
    p.add_argument("--num_x_t", type=int, default=3)
    p.add_argument("--reuse_fracs", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--adamdps", type=int, nargs="+", default=[0, 1], choices=[0, 1])
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD", "LGD-CM"])
    p.add_argument("--n_runs", type=int, default=25)
    p.add_argument("--nsamples", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None)
    return p.parse_args()


def run_point(model_uncond, cond_model, CM_flag, num_x_t, reuse_frac, adamdps, n_runs, nsamples,
              seed, device, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star, label):
    final_loss, l2_gmm, l2_x, times = [], [], [], []
    for i in range(n_runs):
        run_seed = experiment_utils.set_run_seed(seed, i)
        t0 = time.time()
        best_x_t, _, loss = Optimization.optimize_LGD(
            model_uncond, cond_model, mog_means, mog_variances, weights, mu_list, Sigma_list, alpha,
            nsamples=nsamples, device=device, num_x_t=num_x_t, CM=CM_flag,
            reuse_frac=reuse_frac, adamdps=bool(adamdps),
        )
        elapsed = time.time() - t0
        x_pred = best_x_t.float().view(-1).cpu()
        mu_p, Sigma_p = dist_utils.compute_conditionals(mu_list, Sigma_list, x_pred)
        w_p = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_pred)

        final_loss.append(loss.item())
        l2_gmm.append(dist_utils.gmm_l2_distance(mu_p, Sigma_p, w_p, mog_means, mog_variances, weights))
        l2_x.append((x_pred - x_star.float().cpu()).pow(2).sum().sqrt().item())
        times.append(elapsed)

        print(f"[{label} | reuse_frac={reuse_frac:.1f} adamdps={bool(adamdps)} | {i+1}/{n_runs}] "
              f"seed={run_seed} L2_GMM={l2_gmm[-1]:.6f} L2_x={l2_x[-1]:.6f} MMD={final_loss[-1]:.6f} "
              f"time={elapsed:.2f}s", flush=True)

    return {"final_loss": final_loss, "l2_gmm": l2_gmm, "l2_x": l2_x, "times": times}


def main():
    args = parse_args()
    cfg = EXPERIMENT_CONFIGS[args.experiment]

    base_dir = args.base_dir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    params_dir = os.path.join(base_dir, "params")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.experiment)
    results_dir = os.path.join(base_dir, "results", args.experiment)
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
    method_models = {"LGD": (model_cond, False), "LGD-CM": (model_cm, True)}
    methods = [m for m in args.methods if m in method_models]

    results = {}
    for method in methods:
        cond_model, CM_flag = method_models[method]
        results[method] = {}
        for rf in args.reuse_fracs:
            for ad in args.adamdps:
                results[method][(rf, ad)] = run_point(
                    model_uncond, cond_model, CM_flag, args.num_x_t, rf, ad, args.n_runs, args.nsamples,
                    args.seed, device, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
                    label=method,
                )

    json_path = os.path.join(results_dir, f"reuse_adamdps_grid_numxt{args.num_x_t}_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump({
            "experiment": args.experiment, "seed": args.seed,
            "meta": {"num_x_t": args.num_x_t, "n_runs": args.n_runs,
                      "reuse_fracs": args.reuse_fracs, "adamdps": args.adamdps, "methods": methods},
            "results": {m: {f"rf{rf}_ad{ad}": results[m][(rf, ad)]
                             for rf in args.reuse_fracs for ad in args.adamdps} for m in methods},
        }, f, indent=2)
    print(f"[Results] Saved JSON to {json_path}")

    rows = []
    for method in methods:
        baseline = results[method][(0.0, 0)]
        baseline_means = {k: np.mean(baseline[k]) for k in ("l2_gmm", "l2_x", "final_loss", "times")}
        for rf in args.reuse_fracs:
            for ad in args.adamdps:
                r = results[method][(rf, ad)]
                rows.append({
                    "method": method, "reuse_frac": rf, "adamdps": bool(ad),
                    "L2_GMM_mean": np.mean(r["l2_gmm"]), "L2_GMM_std": np.std(r["l2_gmm"]),
                    "L2_x_mean": np.mean(r["l2_x"]), "L2_x_std": np.std(r["l2_x"]),
                    "MMD_mean": np.mean(r["final_loss"]), "MMD_std": np.std(r["final_loss"]),
                    "Time_mean_s": np.mean(r["times"]), "Time_std_s": np.std(r["times"]),
                    "L2_GMM_delta_vs_baseline": np.mean(r["l2_gmm"]) - baseline_means["l2_gmm"],
                    "L2_x_delta_vs_baseline": np.mean(r["l2_x"]) - baseline_means["l2_x"],
                    "Time_speedup_vs_baseline": baseline_means["times"] / np.mean(r["times"]),
                })
    csv_path = os.path.join(results_dir, f"reuse_adamdps_grid_numxt{args.num_x_t}_seed{args.seed}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Results] Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
