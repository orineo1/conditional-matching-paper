#!/usr/bin/env python
"""
Matched-compute, PAIRED downstream comparison of Witness vs. Uniform backsel
selection: fixed sample budget (k_frac held at one value, not swept), exact
per-row rescaling wired into the actual guidance step, and every restart run
with identical randomness for both rules so only the selection rule differs.

Unlike run_backsel_witness_sweep.py (a (nsamples, k_frac, rule) grid, each
cell's R runs independently seeded per rule -- so Witness's run i and
Uniform's run i do NOT share x_T, target draws, or sampler noise), this
script:

  1. Fixes k_frac (default 0.5) -- one head-to-head, not a sweep.
  2. Uses backsel_ht_rescale=True (Optimization.optimize_LGD / witness_utils.
     apply_backsel_ht) so BOTH rules' applied gradient is an exact unbiased
     estimator of the full-batch gradient before zeta*grad -- Uniform gets the
     n/k Horvitz-Thompson correction, Witness gets its own per-row
     counts_i/(k*p_i) importance-sampling correction (NOT the same flat
     factor). See apply_backsel_ht's docstring for why a flat correction is
     wrong for Witness. Without this, a comparison is measuring the
     rescaling's own shrinkage, not the selection rule.
  3. PAIRS every restart: for restart r, both rules' optimize_LGD call reset
     the global RNG to the SAME seed immediately beforehand and use a
     freshly-seeded (same seed) backsel_generator -- since neither x_T, the
     per-step target draws (mog_samples, fixed target distribution, drawn
     independent of x_t), nor model_cond.sample's internal noise depend on
     backsel_generator, they are bit-identical between the two runs; only the
     actual backsel SELECTION differs, and its downstream effect on x_t (and
     hence which state subsequent steps evaluate at) is exactly what's being
     measured.
  4. Reports PAIRED differences (Witness - Uniform), not two separate means:
     mean diff, a 95% CI (percentile bootstrap over the R paired diffs),
     a Wilcoxon signed-rank test p-value (robust to outliers/non-normality,
     unlike a t-test), and a win-count (restarts where Witness beat Uniform).
  5. Reports both "All R" and "Top-10" (by default, the 10 restarts with the
     lowest min(Witness final_loss, Uniform final_loss) -- i.e. the pairs
     where guidance actually reached a good mode, mirroring experiment_utils.
     top10_stats's "best runs" filter while keeping the restarts PAIRED across
     rules; override the ranking criterion with --top10_rank_by).

Usage:
    python witness_vs_uniform_paired_downstream_test.py --experiment 5D_cond_1D
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from scipy.stats import wilcoxon

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dist_utils
import Optimization
import experiment_utils
from gmm_experiment_setup import EXPERIMENT_CONFIGS, load_or_generate_gmm_params, load_or_train_models

METRICS = [("final_loss", "MMD"), ("l2_gmm", "L2 GMM"), ("l2_x", "L2 to x*")]


def run_one(cond_model, CM_flag, num_x_t, nsamples, backsel_k, rule, witness_floor, witness_temperature,
            run_seed, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
            x_star, device, model_uncond):
    """One optimize_LGD call with the HT-exact rescaling wired in, plus this
    repo's standard three downstream metrics (matches run_backsel_witness_
    sweep.py's run_grid_point body exactly, so results are directly comparable
    to the existing sweep/table conventions)."""
    experiment_utils.set_run_seed(run_seed, 0)  # reset the GLOBAL RNG (x_T, target
    # draws, sampler noise) to the SAME state every time this is called with the
    # same run_seed -- the pairing mechanism. run_idx=0 because run_seed itself
    # already encodes the restart index (see main()); a second offset here would
    # just be redundant, not wrong, but keeping one seed->one state is clearer.
    backsel_generator = torch.Generator().manual_seed(run_seed)

    best_x_t, _, final_loss = Optimization.optimize_LGD(
        model_uncond, cond_model, mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, loss="MMD", device=device,
        num_x_t=num_x_t, CM=CM_flag,
        backsel_k=backsel_k, backsel_rule=rule, witness_floor=witness_floor,
        witness_temperature=witness_temperature,
        backsel_generator=backsel_generator,
        backsel_ht_rescale=True, normalize_by_k_frac=False,
    )
    best_x_t = best_x_t.reshape(-1, 1)
    x_pred_t = best_x_t.float().view(-1).cpu()
    mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x_pred_t)
    w_pred = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_pred_t)
    l2_gmm = dist_utils.gmm_l2_distance(mu_pred, Sigma_pred, w_pred, mog_means, mog_variances, weights)
    l2_x = (x_pred_t - x_star.float().cpu()).pow(2).sum().sqrt().item()
    return {"final_loss": final_loss.item(), "l2_gmm": l2_gmm, "l2_x": l2_x}


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    diffs = np.asarray(diffs)
    R = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = diffs[rng.integers(0, R, size=R)]
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_stats(witness_vals, uniform_vals):
    """Witness - Uniform paired analysis for one metric over one restart subset."""
    w = np.asarray(witness_vals, dtype=float)
    u = np.asarray(uniform_vals, dtype=float)
    diffs = w - u
    mean_w, mean_u = float(w.mean()), float(u.mean())
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_ci(diffs)

    if np.allclose(diffs, 0):
        p_value = 1.0
    else:
        try:
            _, p_value = wilcoxon(diffs)
            p_value = float(p_value)
        except ValueError:
            # all-zero or too-few-nonzero differences after scipy's internal handling
            p_value = float("nan")

    win_count = int((diffs < 0).sum())  # Witness beat Uniform (lower = better for all 3 metrics here)
    return {
        "mean_witness": mean_w, "mean_uniform": mean_u,
        "mean_paired_diff": mean_diff, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "p_value_wilcoxon": p_value, "win_count": win_count, "n": len(diffs),
    }


def rank_indices_for_top10(records, rank_by, metric="final_loss"):
    """metric selects WHICH per-restart quantity ranks the pairs (final_loss,
    l2_gmm, or l2_x -- all "lower is better"); rank_by selects HOW the two
    rules' values for that metric are combined into one per-restart score."""
    vals_w = np.array([r["witness"][metric] for r in records])
    vals_u = np.array([r["uniform"][metric] for r in records])
    if rank_by == "min":
        score = np.minimum(vals_w, vals_u)
    elif rank_by == "max":
        score = np.maximum(vals_w, vals_u)
    elif rank_by == "witness":
        score = vals_w
    elif rank_by == "uniform":
        score = vals_u
    elif rank_by == "mean":
        score = (vals_w + vals_u) / 2.0
    else:
        raise ValueError(f"unknown top10_rank_by {rank_by!r}")
    k = min(10, len(records))
    return np.argsort(score)[:k]


def print_table(title, records, indices):
    sub = [records[i] for i in indices]
    print(f"\n=== {title} (n={len(sub)}) ===")
    header = f"{'metric':>10} {'mean(W)':>10} {'mean(U)':>10} {'mean diff':>11} {'95% CI':>22} {'p (Wilcoxon)':>13} {'win-count':>10}"
    print(header)
    print("-" * len(header))
    table_rows = []
    for key, label in METRICS:
        w_vals = [s["witness"][key] for s in sub]
        u_vals = [s["uniform"][key] for s in sub]
        stats = paired_stats(w_vals, u_vals)
        ci_str = f"[{stats['ci95_lo']:.4f}, {stats['ci95_hi']:.4f}]"
        p_str = f"{stats['p_value_wilcoxon']:.4g}" if not np.isnan(stats["p_value_wilcoxon"]) else "n/a"
        print(f"{label:>10} {stats['mean_witness']:>10.4f} {stats['mean_uniform']:>10.4f} "
              f"{stats['mean_paired_diff']:>11.4f} {ci_str:>22} {p_str:>13} "
              f"{stats['win_count']:>4}/{stats['n']:<5}")
        table_rows.append({"metric": label, **stats})
    return table_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD-CM"])
    p.add_argument("--num_x_t", type=int, default=3)
    p.add_argument("--nsamples", type=int, default=250)
    p.add_argument("--k_frac", type=float, default=0.5,
                   help="backsel_k / nsamples, FIXED (not swept) -- one head-to-head comparison.")
    p.add_argument("--witness_floor", type=float, default=0.3)
    p.add_argument("--witness_temperature", type=float, default=1.0,
                   help="Sharpens (T<1) or flattens (T>1) witness selection toward "
                        "|scores|^(1/T); T=1 (default) is the original plain-|scores| weighting.")
    p.add_argument("--n_restarts", type=int, default=25, help="R, number of paired restarts (>=25 recommended).")
    p.add_argument("--top10_metric", choices=["final_loss", "l2_gmm", "l2_x"], default="l2_gmm",
                   help="WHICH per-restart quantity ranks the Top-10 subset (default: L2 distance to "
                        "the target GMM).")
    p.add_argument("--top10_rank_by", choices=["min", "max", "mean", "witness", "uniform"], default="min",
                   help="HOW the two rules' --top10_metric values are combined into one per-restart "
                        "score (default: the pair's best-of-both value, i.e. restarts where guidance "
                        "actually found a good mode on at least one rule).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None)
    args = p.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    base_dir = args.base_dir or os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
    params_dir = os.path.join(base_dir, "params")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.experiment)
    results_dir = os.path.join(base_dir, "results", args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backsel_k = max(1, round(args.k_frac * args.nsamples))
    print(f"[PairedWvU] experiment={args.experiment} methods={args.methods} "
          f"nsamples={args.nsamples} k_frac={args.k_frac} (k={backsel_k}) "
          f"n_restarts={args.n_restarts} device={device}")

    experiment_utils.set_global_seed(args.seed)
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = load_or_generate_gmm_params(
        cfg, params_dir, results_dir, args.experiment, args.seed,
    )
    model_uncond, model_cond, model_cm = load_or_train_models(
        cfg, mu_list, Sigma_list, alpha, checkpoint_dir, args.experiment,
        args.seed, device, args.force_retrain,
    )
    model_uncond.to(device).eval()
    model_cond.to(device).eval()
    if model_cm is not None:
        model_cm.to(device).eval()

    method_models = {"LGD": (model_cond, False), "LGD-CM": (model_cm, True)}

    all_out = {}
    for method in args.methods:
        cond_model, CM_flag = method_models[method]
        records = []
        for r in range(args.n_restarts):
            run_seed = args.seed * 1000 + r  # one seed per restart, shared by both rules
            witness_res = run_one(
                cond_model, CM_flag, args.num_x_t, args.nsamples, backsel_k, "witness",
                args.witness_floor, args.witness_temperature, run_seed, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights, x_star, device, model_uncond,
            )
            uniform_res = run_one(
                cond_model, CM_flag, args.num_x_t, args.nsamples, backsel_k, "uniform",
                args.witness_floor, args.witness_temperature, run_seed, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights, x_star, device, model_uncond,
            )
            records.append({"restart": r, "seed": run_seed, "witness": witness_res, "uniform": uniform_res})
            print(f"[{method} | {r + 1}/{args.n_restarts}] seed={run_seed} | "
                  f"MMD  W={witness_res['final_loss']:.6f} U={uniform_res['final_loss']:.6f} | "
                  f"L2gmm W={witness_res['l2_gmm']:.6f} U={uniform_res['l2_gmm']:.6f} | "
                  f"L2x W={witness_res['l2_x']:.6f} U={uniform_res['l2_x']:.6f}", flush=True)

        all_idx = np.arange(len(records))
        top10_idx = rank_indices_for_top10(records, args.top10_rank_by, metric=args.top10_metric)

        table_all = print_table(f"{method}: All {len(records)}", records, all_idx)
        table_top10 = print_table(
            f"{method}: Top-10 (ranked by {args.top10_rank_by} of {args.top10_metric})", records, top10_idx
        )

        all_out[method] = {
            "backsel_k": backsel_k, "nsamples": args.nsamples, "k_frac": args.k_frac,
            "n_restarts": args.n_restarts, "top10_rank_by": args.top10_rank_by,
            "top10_metric": args.top10_metric,
            "records": records,
            "table_all": table_all,
            "table_top10": table_top10,
        }

    out = {
        "experiment": args.experiment, "seed": args.seed,
        "witness_floor": args.witness_floor, "witness_temperature": args.witness_temperature,
        "results": all_out,
    }
    out_path = os.path.join(
        results_dir,
        f"{args.experiment}_witness_vs_uniform_paired_n{args.nsamples}_kfrac{args.k_frac:g}_"
        f"R{args.n_restarts}_seed{args.seed}.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[PairedWvU] saved {out_path}")


if __name__ == "__main__":
    main()
