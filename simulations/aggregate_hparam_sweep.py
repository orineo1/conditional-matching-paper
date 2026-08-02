#!/usr/bin/env python
"""
Aggregate the results of run_hparam_sweep.py into a CSV table and a figure
showing L2-GMM distance (and wall-clock time) vs. nsamples and num_x_t.

Usage:
    python aggregate_hparam_sweep.py --experiment_name 2D_cond_1D \
        [--sweep_tag hparam_sweep] [--out_dir results/2D_cond_1D/hparam_sweep]
"""
import os
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_sweep(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        h = r["hparams"]
        s = r["summary"]
        rows.append({
            "nsamples": h["nsamples"],
            "num_x_t": h["num_x_t"],
            "n_attempts": h["n_attempts"],
            "l2_gmm_mean": s["l2_gmm_mean"],
            "l2_gmm_std": s["l2_gmm_std"],
            "l2_x_mean": s["l2_x_mean"],
            "l2_x_std": s["l2_x_std"],
            "final_loss_mean": s["final_loss_mean"],
            "final_loss_std": s["final_loss_std"],
            "time_mean": s["time_mean"],
            "time_std": s["time_std"],
            "file": os.path.basename(path),
        })
    if not rows:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", type=str, default="2D_cond_1D")
    p.add_argument("--sweep_tag", type=str, default="hparam_sweep")
    p.add_argument("--out_dir", type=str, default=None,
                    help="Where to write the CSV/figure (defaults to the sweep results dir)")
    p.add_argument("--default_nsamples", type=int, default=250)
    p.add_argument("--default_num_x_t", type=int, default=3)
    args = p.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(BASE_DIR, "results", args.experiment_name, args.sweep_tag)
    out_dir = args.out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)

    df = load_sweep(results_dir)
    df = df.sort_values(["num_x_t", "nsamples"]).reset_index(drop=True)

    csv_path = os.path.join(out_dir, f"{args.experiment_name}_hparam_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    print(df.to_string(index=False))

    # nsamples sweep: rows where num_x_t == default_num_x_t
    ns_df = df[df["num_x_t"] == args.default_num_x_t].sort_values("nsamples")
    # num_x_t sweep: rows where nsamples == default_nsamples
    xt_df = df[df["nsamples"] == args.default_nsamples].sort_values("num_x_t")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.errorbar(ns_df["nsamples"], ns_df["l2_gmm_mean"], yerr=ns_df["l2_gmm_std"],
                marker="o", capsize=3)
    ax.axvline(args.default_nsamples, color="gray", linestyle="--", alpha=0.5, label="paper default")
    ax.set_xscale("log")
    ax.set_xlabel("nsamples (MC samples for MMD loss)")
    ax.set_ylabel("L2-GMM distance")
    ax.set_title(f"L2-GMM vs nsamples (num_x_t={args.default_num_x_t})")
    ax.legend()

    ax = axes[0, 1]
    ax.errorbar(xt_df["num_x_t"], xt_df["l2_gmm_mean"], yerr=xt_df["l2_gmm_std"],
                marker="o", capsize=3, color="tab:orange")
    ax.axvline(args.default_num_x_t, color="gray", linestyle="--", alpha=0.5, label="paper default")
    ax.set_xlabel("num_x_t (resampled candidates per step)")
    ax.set_ylabel("L2-GMM distance")
    ax.set_title(f"L2-GMM vs num_x_t (nsamples={args.default_nsamples})")
    ax.legend()

    ax = axes[1, 0]
    ax.errorbar(ns_df["nsamples"], ns_df["time_mean"], yerr=ns_df["time_std"],
                marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("nsamples")
    ax.set_ylabel("wall-clock time per run (s)")
    ax.set_title("Cost vs nsamples")

    ax = axes[1, 1]
    ax.errorbar(xt_df["num_x_t"], xt_df["time_mean"], yerr=xt_df["time_std"],
                marker="o", capsize=3, color="tab:orange")
    ax.set_xlabel("num_x_t")
    ax.set_ylabel("wall-clock time per run (s)")
    ax.set_title("Cost vs num_x_t")

    fig.suptitle(f"MLGD-F hyperparameter sensitivity — {args.experiment_name}")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{args.experiment_name}_hparam_sweep.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
