#!/usr/bin/env python
"""
Matched-compute, PAIRED downstream comparison of backsel selection "arms" --
Uniform, and one or more Witness temperatures (e.g. Witness with no
temperature, T=1, vs. Witness sharpened toward high-witness samples, T<1) --
at a FIXED sample budget (k_frac held at one value, not swept), with the
exact per-row rescaling wired into the actual guidance step, and every
restart run with identical randomness across ALL arms so only the
selection rule/temperature differs.

Unlike run_backsel_witness_sweep.py (a (nsamples, k_frac, rule) grid, each
cell's R runs independently seeded per rule -- so different arms' run i do
NOT share x_T, target draws, or sampler noise), this script:

  1. Fixes k_frac (default 0.5) -- one head-to-head, not a sweep.
  2. Uses backsel_ht_rescale=True (Optimization.optimize_LGD / witness_utils.
     apply_backsel_ht) so every arm's applied gradient is an exact unbiased
     estimator of the full-batch gradient before zeta*grad -- Uniform gets
     the n/k Horvitz-Thompson correction, each Witness temperature gets its
     own per-row counts_i/(k*p_i) importance-sampling correction (NOT the
     same flat factor, and not the same across temperatures either, since
     p_i itself depends on witness_temperature). Without this, a comparison
     is measuring the rescaling's own shrinkage, not the selection itself.
  3. PAIRS every restart across ALL arms: for restart r, every arm's
     optimize_LGD call resets the global RNG to the SAME seed immediately
     beforehand and uses a freshly-seeded (same seed) backsel_generator --
     since neither x_T, the per-step target draws (mog_samples, fixed target
     distribution, drawn independent of x_t), nor model_cond.sample's
     internal noise depend on backsel_generator, they are bit-identical
     across every arm at that restart; only the actual backsel SELECTION
     differs, and its downstream effect on x_t (hence which state
     subsequent steps evaluate at) is exactly what's being measured.
  4. Reports PAIRED differences for every pair of arms (not just Witness vs
     Uniform): mean diff, a 95% CI (percentile bootstrap over the R paired
     diffs), a Wilcoxon signed-rank p-value (robust to outliers/non-
     normality, unlike a t-test), and a win-count.
  5. Reports both "All R" and "Top-10" per comparison pair (by default, the
     10 restarts with the best min(--top10_metric) across that PAIR's two
     arms -- i.e. restarts where guidance actually reached a good mode on at
     least one of the two, mirroring experiment_utils.top10_stats's "best
     runs" filter while keeping restarts PAIRED; override with
     --top10_rank_by / --top10_metric).

Arms: always includes "uniform", plus one "witness_T<value>" arm per value
in --witness_temperatures (default just [1.0], i.e. plain Witness vs
Uniform -- pass e.g. --witness_temperatures 1.0 0.3 for a three-way
Uniform / no-temperature-Witness / temperature-Witness comparison; every
pair among the resulting arms is reported).

Usage:
    python witness_vs_uniform_paired_downstream_test.py --experiment 5D_cond_1D \
        --witness_temperatures 1.0 0.3
"""
import os
import sys
import json
import argparse
import itertools

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


RESCALE_MODES = ["ht", "raw", "kfrac"]


def run_one(cond_model, CM_flag, num_x_t, nsamples, backsel_k, rule, witness_floor, witness_temperature,
            rescale_mode, run_seed, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
            x_star, device, model_uncond):
    """One optimize_LGD call, plus this repo's standard three downstream
    metrics (matches run_backsel_witness_sweep.py's run_grid_point body
    exactly, so results are directly comparable to the existing sweep/table
    conventions). rescale_mode picks which correction (if any) is applied to
    the selected samples' gradient before zeta*grad:
      'ht'    -- apply_backsel_ht: exact per-row Horvitz-Thompson/importance-
                 sampling correction (n/k for uniform, counts_i/(k*p_i) for
                 witness) -- unbiased for the full-batch gradient under
                 EITHER rule. The only mode that answers "which rule
                 actually picks a better subset", isolated from any rescale
                 artifact.
      'raw'   -- apply_backsel's original behavior: NO rescaling at all.
                 Selected rows keep their raw gradient; the applied gradient
                 is smaller (by roughly k/n in magnitude) than the full-batch
                 one and, for 'witness', additionally biased in DIRECTION
                 whenever p_i isn't uniform (see apply_backsel_ht's
                 docstring). This is the repo's original/production
                 backsel_k behavior -- include it to see what the existing
                 (uncorrected) pipeline's comparison actually looks like.
      'kfrac' -- apply_backsel + a flat post-hoc 1/k_frac rescale
                 (normalize_by_k_frac=True). Exact for 'uniform' (a global
                 constant commutes with autograd summation) but NOT exact
                 for 'witness' (needs a per-row weight, not one flat
                 factor) -- included as the "partially corrected" middle
                 ground between 'raw' and 'ht'.
    """
    if rescale_mode not in RESCALE_MODES:
        raise ValueError(f"unknown rescale_mode {rescale_mode!r} (known: {RESCALE_MODES})")

    experiment_utils.set_run_seed(run_seed, 0)  # reset the GLOBAL RNG (x_T, target
    # draws, sampler noise) to the SAME state every time this is called with the
    # same run_seed -- the pairing mechanism, generalized across however many
    # arms are run at this restart (all of them reset to the SAME run_seed).
    backsel_generator = torch.Generator().manual_seed(run_seed)

    best_x_t, _, final_loss = Optimization.optimize_LGD(
        model_uncond, cond_model, mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, loss="MMD", device=device,
        num_x_t=num_x_t, CM=CM_flag,
        backsel_k=backsel_k, backsel_rule=rule, witness_floor=witness_floor,
        witness_temperature=witness_temperature,
        backsel_generator=backsel_generator,
        backsel_ht_rescale=(rescale_mode == "ht"),
        normalize_by_k_frac=(rescale_mode == "kfrac"),
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


def paired_stats(a_vals, b_vals):
    """(a - b) paired analysis for one metric over one restart subset."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    diffs = a - b
    mean_a, mean_b = float(a.mean()), float(b.mean())
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

    win_count = int((diffs < 0).sum())  # arm "a" beat arm "b" (lower = better for all 3 metrics here)
    return {
        "mean_a": mean_a, "mean_b": mean_b,
        "mean_paired_diff": mean_diff, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "p_value_wilcoxon": p_value, "win_count": win_count, "n": len(diffs),
    }


def rank_indices_for_top10(records, rank_by, metric, label_a, label_b):
    """metric selects WHICH per-restart quantity ranks the pair (final_loss,
    l2_gmm, or l2_x -- all "lower is better"); rank_by selects HOW the two
    arms' values for that metric are combined into one per-restart score."""
    vals_a = np.array([r["arms"][label_a][metric] for r in records])
    vals_b = np.array([r["arms"][label_b][metric] for r in records])
    if rank_by == "min":
        score = np.minimum(vals_a, vals_b)
    elif rank_by == "max":
        score = np.maximum(vals_a, vals_b)
    elif rank_by == "a":
        score = vals_a
    elif rank_by == "b":
        score = vals_b
    elif rank_by == "mean":
        score = (vals_a + vals_b) / 2.0
    else:
        raise ValueError(f"unknown top10_rank_by {rank_by!r}")
    k = min(10, len(records))
    return np.argsort(score)[:k]


def print_table(title, records, indices, label_a, label_b):
    sub = [records[i] for i in indices]
    print(f"\n=== {title}: {label_a} vs {label_b} (n={len(sub)}) ===")
    header = (f"{'metric':>10} {'mean(' + label_a + ')':>16} {'mean(' + label_b + ')':>16} "
              f"{'mean diff':>11} {'95% CI':>22} {'p (Wilcoxon)':>13} {'win-count':>10}")
    print(header)
    print("-" * len(header))
    table_rows = []
    for key, label in METRICS:
        a_vals = [s["arms"][label_a][key] for s in sub]
        b_vals = [s["arms"][label_b][key] for s in sub]
        stats = paired_stats(a_vals, b_vals)
        ci_str = f"[{stats['ci95_lo']:.4f}, {stats['ci95_hi']:.4f}]"
        p_str = f"{stats['p_value_wilcoxon']:.4g}" if not np.isnan(stats["p_value_wilcoxon"]) else "n/a"
        print(f"{label:>10} {stats['mean_a']:>16.4f} {stats['mean_b']:>16.4f} "
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
    p.add_argument("--witness_temperatures", type=float, nargs="+", default=[1.0],
                   help="One Witness arm per value, labeled witness_T<value>. Each sharpens (T<1) "
                        "or flattens (T>1) selection toward |scores|^(1/T); T=1 is the original "
                        "plain-|scores| weighting (i.e. 'no temperature'). Pass multiple values "
                        "(e.g. 1.0 0.3) for a full Uniform / no-temperature-Witness / "
                        "temperature-Witness comparison -- every pair of arms is reported.")
    p.add_argument("--rescale_modes", nargs="+", choices=RESCALE_MODES, default=["ht"],
                   help="Which gradient-rescale correction(s) to run, each producing its own set of "
                        "arms (suffixed _<mode> when more than one is given): 'ht' (default; exact "
                        "per-row correction, unbiased for either rule -- see run_one's docstring), "
                        "'raw' (no rescaling at all -- apply_backsel's original/production behavior, "
                        "biased in magnitude for both rules and in direction for witness), 'kfrac' "
                        "(flat post-hoc 1/k_frac correction -- exact for uniform, not for witness). "
                        "Pass multiple (e.g. ht raw) to see how much the correction itself matters.")
    p.add_argument("--n_restarts", type=int, default=25, help="R, number of paired restarts (>=25 recommended).")
    p.add_argument("--top10_metric", choices=["final_loss", "l2_gmm", "l2_x"], default="l2_gmm",
                   help="WHICH per-restart quantity ranks the Top-10 subset (default: L2 distance to "
                        "the target GMM).")
    p.add_argument("--top10_rank_by", choices=["min", "max", "mean", "a", "b"], default="min",
                   help="HOW a comparison pair's --top10_metric values are combined into one "
                        "per-restart score (default: the pair's best-of-both value, i.e. restarts "
                        "where guidance actually found a good mode on at least one arm).")
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

    # arms: always "uniform", plus one "witness_T<value>" per --witness_temperatures entry,
    # each repeated per --rescale_modes entry (suffixed _<mode> when more than one mode is given).
    multi_mode = len(args.rescale_modes) > 1
    arms = []
    for mode in args.rescale_modes:
        suffix = f"_{mode}" if multi_mode else ""
        arms.append((f"uniform{suffix}", "uniform", 1.0, mode))
        for T in args.witness_temperatures:
            arms.append((f"witness_T{T:g}{suffix}", "witness", T, mode))
    arm_labels = [label for label, _, _, _ in arms]
    comparisons = list(itertools.combinations(arm_labels, 2))

    print(f"[PairedArms] experiment={args.experiment} methods={args.methods} "
          f"nsamples={args.nsamples} k_frac={args.k_frac} (k={backsel_k}) "
          f"n_restarts={args.n_restarts} device={device}")
    print(f"[PairedArms] arms={arm_labels} -> comparisons={comparisons}")

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
            run_seed = args.seed * 1000 + r  # one seed per restart, shared by every arm
            arm_results = {}
            for label, rule, temperature, mode in arms:
                arm_results[label] = run_one(
                    cond_model, CM_flag, args.num_x_t, args.nsamples, backsel_k, rule,
                    args.witness_floor, temperature, mode, run_seed, mu_list, Sigma_list, alpha,
                    mog_means, mog_variances, weights, x_star, device, model_uncond,
                )
            records.append({"restart": r, "seed": run_seed, "arms": arm_results})
            line = f"[{method} | {r + 1}/{args.n_restarts}] seed={run_seed} | "
            line += " | ".join(
                f"{label}: MMD={arm_results[label]['final_loss']:.6f} "
                f"L2gmm={arm_results[label]['l2_gmm']:.6f} L2x={arm_results[label]['l2_x']:.6f}"
                for label in arm_labels
            )
            print(line, flush=True)

        all_idx = np.arange(len(records))
        tables_all, tables_top10 = {}, {}
        for label_a, label_b in comparisons:
            key = f"{label_a}_vs_{label_b}"
            top10_idx = rank_indices_for_top10(records, args.top10_rank_by, args.top10_metric, label_a, label_b)
            tables_all[key] = print_table(f"{method}: All {len(records)}", records, all_idx, label_a, label_b)
            tables_top10[key] = print_table(
                f"{method}: Top-10 (ranked by {args.top10_rank_by} of {args.top10_metric})",
                records, top10_idx, label_a, label_b,
            )

        all_out[method] = {
            "backsel_k": backsel_k, "nsamples": args.nsamples, "k_frac": args.k_frac,
            "n_restarts": args.n_restarts, "top10_rank_by": args.top10_rank_by,
            "top10_metric": args.top10_metric, "arms": arm_labels, "comparisons": comparisons,
            "records": records,
            "tables_all": tables_all,
            "tables_top10": tables_top10,
        }

    out = {
        "experiment": args.experiment, "seed": args.seed,
        "witness_floor": args.witness_floor, "witness_temperatures": args.witness_temperatures,
        "rescale_modes": args.rescale_modes,
        "results": all_out,
    }
    temps_tag = "_".join(f"{T:g}" for T in args.witness_temperatures)
    modes_tag = "_".join(args.rescale_modes)
    out_path = os.path.join(
        results_dir,
        f"{args.experiment}_witness_vs_uniform_paired_n{args.nsamples}_kfrac{args.k_frac:g}_"
        f"T{temps_tag}_{modes_tag}_R{args.n_restarts}_seed{args.seed}.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[PairedArms] saved {out_path}")


if __name__ == "__main__":
    main()
