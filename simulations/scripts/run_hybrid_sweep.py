"""
Hybrid sample-reuse sweep for optimize_LGD, runnable standalone / via sbatch.

For one experiment (2D_cond_1D / 5D_cond_1D / 10D_cond_1D) and a single
fixed num_x_t, sweeps `reuse_frac` (the fraction of the per-step MMD
sample batch carried over from the previous diffusion step, see
Optimization.optimize_LGD) over a grid — default 0%, 10%, ..., 90% — for
the LGD and/or LGD-CM methods, runs N_RUNS seeded optimization attempts
per point, and writes out:

  - a JSON file with every run's final_loss (MMD), l2_gmm, l2_x, time
  - a summary CSV (mean/std per (method, reuse_frac))
  - PNG plots: L2 GMM, L2 to x*, MMD, and wall time vs reuse_frac

Note: `reuse_frac > 0.5` is self-limiting under optimize_LGD's one-step
lookback buffer — the achieved reuse fraction converges to ~(1 - reuse_frac)
in that regime (see Optimization.py). The sweep still runs those points;
the summary/plots will show the ceiling rather than hide it.

Example:
    python run_hybrid_sweep.py --experiment 5D_cond_1D --num_x_t 3 \
        --n_runs 25 --methods LGD LGD-CM
"""

import os
import sys
import time
import json
import argparse
from functools import partial

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Diffusion
import ConsistencyModels
import dist_utils
import Optimization
import experiment_utils
from ConsistencyModels import ConsistencyModeliCT


EXPERIMENT_CONFIGS = {
    "2D_cond_1D": dict(
        NBLOCKS=3, NUNITS=128, NBLOCKS_CM=3, NUNITS_CM=128,
        NEPOCHS=20_000, BATCH_SIZE=1_024, NEPOCHS_CM=20_000, BATCH_SIZE_CM=1_024,
        DIFFUSION_STEPS=100, CONDITION_ON=1,
    ),
    "5D_cond_1D": dict(
        NBLOCKS=6, NUNITS=512, NBLOCKS_CM=6, NUNITS_CM=512,
        NEPOCHS=40_000, BATCH_SIZE=4_096, NEPOCHS_CM=40_000, BATCH_SIZE_CM=4_096,
        DIFFUSION_STEPS=100, CONDITION_ON=4,
    ),
    "10D_cond_1D": dict(
        NBLOCKS=8, NUNITS=512, NBLOCKS_CM=8, NUNITS_CM=512,
        NEPOCHS=40_000, BATCH_SIZE=4_096, NEPOCHS_CM=40_000, BATCH_SIZE_CM=4_096,
        DIFFUSION_STEPS=100, CONDITION_ON=9,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--num_x_t", type=int, default=3, help="Fixed num_x_t for this sweep (not swept).")
    p.add_argument("--reuse_fracs", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD", "LGD-CM"])
    p.add_argument("--n_runs", type=int, default=25)
    p.add_argument("--nsamples", type=int, default=250, help="NSAMPLES_IN_OPTIM_FOR_MMD")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None,
                    help="Defaults to simulations/ (parent of this script's directory).")
    return p.parse_args()


def load_or_generate_gmm_params(cfg, params_dir, results_dir, experiment_name, global_seed):
    loaded = experiment_utils.load_gmm_params(params_dir, experiment_name)
    if loaded is None:
        loaded = experiment_utils.load_gmm_params(results_dir, experiment_name)

    if loaded is not None:
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
        mu_list = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha = alpha.float()
        print(f"[GMM] Loaded existing parameters for {experiment_name}")
        return mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star

    print(f"[GMM] No saved parameters found for {experiment_name}, generating fresh...")
    experiment_utils.set_global_seed(global_seed)

    if experiment_name == "2D_cond_1D":
        mu_list = [
            torch.tensor([-5, 5], dtype=torch.float64), torch.tensor([-5, -5], dtype=torch.float64),
            torch.tensor([5, 3], dtype=torch.float64), torch.tensor([5, -1], dtype=torch.float64),
            torch.tensor([0, -3], dtype=torch.float64), torch.tensor([-2, 4], dtype=torch.float64),
            torch.tensor([-2, -3], dtype=torch.float64), torch.tensor([1, 2], dtype=torch.float64),
            torch.tensor([-8, 1], dtype=torch.float64), torch.tensor([7, 5], dtype=torch.float64),
            torch.tensor([0, -5], dtype=torch.float64),
        ]
        Sigma_list = [torch.tensor([[0.5000, 0.1950], [0.1950, 0.2000]], dtype=torch.float64)] * len(mu_list)
        alpha = torch.tensor([1 / len(mu_list)] * len(mu_list), dtype=torch.float64)
        mu_list = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha = alpha.float()
        x_star = torch.tensor([-5])
        mu_temp, Sigma_temp = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
        temp_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
        mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
            mu_temp, Sigma_temp, temp_alpha, threshold=0.01
        )
    else:
        dim_data = cfg["CONDITION_ON"] + 1
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = \
            dist_utils.get_param_mog_with_target(
                dim_data=dim_data, num_components=4, device="cpu",
                conditional_modes=2, distanceOrScale="Distance",
            )
        mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
            mog_means, mog_variances, weights, threshold=0.001
        )
        mu_list = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha = alpha.float()

    experiment_utils.save_gmm_params(
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
        params_dir, experiment_name,
    )
    return mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star


def load_or_train_models(cfg, mu_list, Sigma_list, alpha, checkpoint_dir, experiment_name,
                          global_seed, device, force_retrain):
    condition_on = cfg["CONDITION_ON"]

    X = dist_utils.generate_mog_samples(1_000, mu_list, Sigma_list, alpha).float().to(device)
    nfeatures = X.shape[1] - condition_on

    # Consistency Model — P(Y|X=x)
    data_generator_cm = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
    )
    model_cm = ConsistencyModeliCT(
        nfeatures=nfeatures, condition_on=condition_on,
        nunits=cfg["NUNITS_CM"], depth=cfg["NBLOCKS_CM"],
    )
    loaded_cm = experiment_utils.load_checkpoint_with_hf_fallback(
        model_cm, "CM", checkpoint_dir, experiment_name, global_seed, device
    ) if not force_retrain else False
    if not loaded_cm:
        experiment_utils.set_global_seed(global_seed)
        model_cm.train_model(
            X=None, nepochs=cfg["NEPOCHS_CM"], batch_size=cfg["BATCH_SIZE_CM"],
            device=device, condition=condition_on,
            data_generator=data_generator_cm, use_improved_training=True,
        )
        experiment_utils.save_model_checkpoint(model_cm, "CM", checkpoint_dir, experiment_name, global_seed)

    # Conditional diffusion — P(Y|X=x)
    data_generator_diff_cond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha, kernel_func=None,
    )
    model_cond = Diffusion.DiffusionModel(
        nfeatures=nfeatures + condition_on, nblocks=cfg["NBLOCKS"], nunits=cfg["NUNITS"],
        condition=True, condition_on=condition_on, diffusion_steps=cfg["DIFFUSION_STEPS"],
    )
    loaded_cond = experiment_utils.load_checkpoint_with_hf_fallback(
        model_cond, "Diffusion_cond", checkpoint_dir, experiment_name, global_seed, device
    ) if not force_retrain else False
    if not loaded_cond:
        experiment_utils.set_global_seed(global_seed)
        model_cond.train_model(
            None, data_generator=data_generator_diff_cond,
            nepochs=cfg["NEPOCHS"], batch_size=cfg["BATCH_SIZE"], condition_on=condition_on,
        )
        experiment_utils.save_model_checkpoint(model_cond, "Diffusion_cond", checkpoint_dir, experiment_name, global_seed)

    # Unconditional diffusion — P(X=x)
    data_generator_diff_uncond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
        kernel_func=lambda X: X[:, :condition_on],
    )
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=condition_on, nblocks=cfg["NBLOCKS"], nunits=cfg["NUNITS"],
        condition=False, diffusion_steps=cfg["DIFFUSION_STEPS"],
    )
    loaded_uncond = experiment_utils.load_checkpoint_with_hf_fallback(
        model_uncond, "Diffusion_uncond", checkpoint_dir, experiment_name, global_seed, device
    ) if not force_retrain else False
    if not loaded_uncond:
        experiment_utils.set_global_seed(global_seed)
        model_uncond.train_model(
            None, data_generator=data_generator_diff_uncond,
            nepochs=cfg["NEPOCHS"], batch_size=cfg["BATCH_SIZE"], condition_on=condition_on,
        )
        experiment_utils.save_model_checkpoint(model_uncond, "Diffusion_uncond", checkpoint_dir, experiment_name, global_seed)

    return model_uncond, model_cond, model_cm


def run_sweep_point(model_uncond, cond_model, CM_flag, num_x_t, reuse_frac, n_runs,
                     nsamples, global_seed, device, mu_list, Sigma_list, alpha,
                     mog_means, mog_variances, weights, x_star, label):
    final_loss_list, l2_gmm_list, l2_x_list, times = [], [], [], []
    for i in range(n_runs):
        run_seed = experiment_utils.set_run_seed(global_seed, i)

        start_time = time.time()
        best_x_t, _, final_loss = Optimization.optimize_LGD(
            model_uncond, cond_model, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha,
            nsamples=nsamples, loss="MMD", device=device,
            num_x_t=num_x_t, CM=CM_flag, reuse_frac=reuse_frac,
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

        print(f"[{label} | reuse_frac={reuse_frac:.1f} | {i + 1}/{n_runs}] seed={run_seed} "
              f"| L2 GMM: {l2_gmm:.6f} | L2 to x*: {l2_x:.6f} "
              f"| MMD: {final_loss.item():.6f} | time: {elapsed:.2f}s", flush=True)

    return {"final_loss": final_loss_list, "l2_gmm": l2_gmm_list, "l2_x": l2_x_list, "times": times}


def make_plots(results, methods, reuse_fracs, plots_dir, experiment_name, num_x_t):
    os.makedirs(plots_dir, exist_ok=True)
    metrics = [("l2_gmm", "L2 GMM"), ("l2_x", "L2 to x*"), ("final_loss", "MMD"), ("times", "Wall time (s)")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        for method in methods:
            means = [np.mean(results[method][rf][key]) for rf in reuse_fracs]
            stds = [np.std(results[method][rf][key]) for rf in reuse_fracs]
            ax.errorbar(reuse_fracs, means, yerr=stds, marker="o", capsize=3, label=method)
        ax.set_xlabel("reuse_frac")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs reuse_frac (mean ± std, n={len(next(iter(results[methods[0]].values()))[key])})")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{experiment_name} | num_x_t={num_x_t}")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "hybrid_sweep_summary.png"), dpi=150)
    plt.close(fig)

    # Boxplots (distribution, not just mean/std) — one file per method.
    for method in methods:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, (key, ylabel) in zip(axes.flat, metrics):
            ax.boxplot([results[method][rf][key] for rf in reuse_fracs],
                       labels=[f"{rf:.1f}" for rf in reuse_fracs])
            ax.set_xlabel("reuse_frac")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{method}: {ylabel} distribution")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{experiment_name} | {method} | num_x_t={num_x_t}")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"hybrid_sweep_boxplots_{method}.png"), dpi=150)
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

    results = {}
    for method in methods:
        results[method] = {}
        for rf in args.reuse_fracs:
            results[method][rf] = run_sweep_point(
                model_uncond, method_models[method]["cond_model"], method_models[method]["CM_flag"],
                args.num_x_t, rf, args.n_runs, args.nsamples, args.seed, device,
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
            "reuse_fracs": args.reuse_fracs,
            "methods": methods,
            "x_star": x_star.detach().cpu().tolist() if isinstance(x_star, torch.Tensor) else x_star,
        },
        "results": {
            method: {str(rf): results[method][rf] for rf in args.reuse_fracs}
            for method in methods
        },
    }

    json_path = os.path.join(results_dir, f"hybrid_sweep_numxt{args.num_x_t}_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Results] Saved JSON to {json_path}")

    summary_rows = []
    for method in methods:
        for rf in args.reuse_fracs:
            r = results[method][rf]
            summary_rows.append({
                "method": method, "reuse_frac": rf,
                "L2_GMM_mean": np.mean(r["l2_gmm"]), "L2_GMM_std": np.std(r["l2_gmm"]),
                "L2_x_mean": np.mean(r["l2_x"]), "L2_x_std": np.std(r["l2_x"]),
                "MMD_mean": np.mean(r["final_loss"]), "MMD_std": np.std(r["final_loss"]),
                "Time_mean_s": np.mean(r["times"]), "Time_std_s": np.std(r["times"]),
            })
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(results_dir, f"hybrid_sweep_numxt{args.num_x_t}_seed{args.seed}_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[Results] Saved summary CSV to {csv_path}")

    make_plots(results, methods, args.reuse_fracs, plots_dir, args.experiment, args.num_x_t)
    print(f"[Results] Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
