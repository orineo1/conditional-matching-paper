"""Runnable comparison of the two x0_hat estimators (mode vs sampled) for
twisted-SMC guidance of masked discrete diffusion on the sanity task.

Usage (from repo root):
    python discrete_x/run_sanity_compare.py --seeds 0 1 2 3 4 --beta 10

Outputs to --outdir:
    metrics.json                    all per-run results + aggregates
    log_{estimator}_seed{S}.jsonl   per-step ESS / weight-var / L range / decode
    ess_trajectories.png            ESS per step, both estimators, all seeds
    loss_trajectories.png           min L(x0_hat) across particles per step
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sanity_task import SanityTask
from smc import run_smc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"mode": "tab:blue", "sampled": "tab:orange"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_particles", type=int, default=512)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--beta", type=float, default=200.0)
    ap.add_argument("--n_dec", type=int, default=8)
    ap.add_argument("--ess_frac", type=float, default=0.5)
    ap.add_argument("--beta_anneal", action="store_true")
    ap.add_argument("--n_dec_early", type=int, default=None)
    ap.add_argument("--task_seed", type=int, default=0)
    ap.add_argument("--outdir", default="output/discrete_x_sanity")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    task = SanityTask(task_seed=args.task_seed)
    print(f"Ground-truth prompt x_true : '{task.decode(task.x_true)}'")
    print(f"L(x_true)                  = {task.loss(task.x_true):.5f}")

    x_bf, l_bf = task.brute_force_optimum()
    bf_matches_true = bool(np.all(x_bf == task.x_true))
    print(
        f"Brute-force optimum        : '{task.decode(x_bf)}' "
        f"(L={l_bf:.5f}, == x_true: {bf_matches_true})"
    )

    all_results = {"mode": [], "sampled": []}
    all_records = {"mode": [], "sampled": []}
    for estimator in ["mode", "sampled"]:
        for seed in args.seeds:
            log_path = os.path.join(
                args.outdir, f"log_{estimator}_seed{seed}.jsonl"
            )
            res, recs = run_smc(
                task,
                estimator,
                n_particles=args.n_particles,
                T=args.T,
                beta=args.beta,
                n_dec=args.n_dec,
                seed=seed,
                ess_frac=args.ess_frac,
                log_path=log_path,
                beta_anneal=args.beta_anneal,
                n_dec_early=args.n_dec_early,
            )
            all_results[estimator].append(res)
            all_records[estimator].append(recs)
            print(
                f"[{estimator:7s} seed {seed}] L_best={res['L_best']:.5f} "
                f"mmd2_eval={res['mmd2_eval']:.5f} "
                f"slot_acc={res['slot_acc_vs_true']:.2f} "
                f"exact={res['exact_recovery']} "
                f"resamples={res['n_resamples']} "
                f"time={res['wall_clock_s']:.2f}s  -> '{res['x_best_decode']}'"
            )

    # ---- aggregates ----
    agg = {}
    for est, results in all_results.items():
        agg[est] = {
            "L_best_mean": float(np.mean([r["L_best"] for r in results])),
            "L_best_std": float(np.std([r["L_best"] for r in results])),
            "mmd2_eval_mean": float(np.mean([r["mmd2_eval"] for r in results])),
            "mmd2_eval_std": float(np.std([r["mmd2_eval"] for r in results])),
            "slot_acc_mean": float(
                np.mean([r["slot_acc_vs_true"] for r in results])
            ),
            "exact_recovery_rate": float(
                np.mean([r["exact_recovery"] for r in results])
            ),
            "wall_clock_mean_s": float(
                np.mean([r["wall_clock_s"] for r in results])
            ),
            "n_resamples_mean": float(
                np.mean([r["n_resamples"] for r in results])
            ),
        }

    print("\n=== Summary (mean over seeds) ===")
    header = (
        f"{'estimator':10s} {'L_best':>14s} {'MMD2_eval':>14s} "
        f"{'slot_acc':>9s} {'exact%':>7s} {'time(s)':>8s} {'resamp':>7s}"
    )
    print(header)
    for est in ["mode", "sampled"]:
        a = agg[est]
        print(
            f"{est:10s} {a['L_best_mean']:.5f}±{a['L_best_std']:.5f} "
            f"{a['mmd2_eval_mean']:.5f}±{a['mmd2_eval_std']:.5f} "
            f"{a['slot_acc_mean']:9.2f} {100*a['exact_recovery_rate']:6.0f}% "
            f"{a['wall_clock_mean_s']:8.2f} {a['n_resamples_mean']:7.1f}"
        )

    metrics = {
        "config": vars(args),
        "x_true": task.x_true.tolist(),
        "x_true_decode": task.decode(task.x_true),
        "L_x_true": task.loss(task.x_true),
        "brute_force_x": x_bf.tolist(),
        "brute_force_decode": task.decode(x_bf),
        "brute_force_L": l_bf,
        "brute_force_matches_true": bf_matches_true,
        "runs": all_results,
        "aggregates": agg,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- plots ----
    steps = list(range(args.T - 1, -1, -1))  # state index t-1 after each step

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for est in ["mode", "sampled"]:
        trajs = np.array([[r["ess"] for r in recs] for recs in all_records[est]])
        for row in trajs:
            ax.plot(steps, row, color=COLORS[est], alpha=0.25, lw=1)
        ax.plot(
            steps, trajs.mean(axis=0), color=COLORS[est], lw=2.5,
            label=f"{est} (mean of {len(args.seeds)} seeds)",
        )
    ax.axhline(
        args.ess_frac * args.n_particles, color="gray", ls="--", lw=1,
        label=f"resample threshold ({args.ess_frac:.0%} N)",
    )
    ax.set_xlabel("diffusion state t (T-1 → 0)")
    ax.set_ylabel("ESS")
    ax.set_title(
        f"ESS trajectories (N={args.n_particles}, T={args.T}, "
        f"beta={args.beta}, n_dec={args.n_dec})"
    )
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "ess_trajectories.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for est in ["mode", "sampled"]:
        lmin = np.array(
            [[r["L_min"] for r in recs] for recs in all_records[est]]
        )
        for row in lmin:
            ax.plot(steps, row, color=COLORS[est], alpha=0.25, lw=1)
        ax.plot(
            steps, lmin.mean(axis=0), color=COLORS[est], lw=2.5,
            label=f"{est} min-L (mean)",
        )
    ax.axhline(
        l_bf, color="black", ls=":", lw=1.5,
        label=f"brute-force optimum L={l_bf:.4f}",
    )
    ax.set_xlabel("diffusion state t (T-1 → 0)")
    ax.set_ylabel("min L(x0_hat) across particles")
    ax.set_title("Best estimated loss across particles per step")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "loss_trajectories.png"), dpi=150)
    plt.close(fig)

    print(f"\nWrote metrics, logs, and plots to {args.outdir}/")


if __name__ == "__main__":
    main()
