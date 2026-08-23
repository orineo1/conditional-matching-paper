"""Experiment 2 -- LGD versus Adam.

QUESTION
    LGD spends M_t = 3 spatial perturbations, each costing n_t conditional
    generations, so its conditional cost is 3x. Can temporal momentum stabilise
    the MMD gradient WITHOUT that multiplicative spatial cost?

COMPARISON
    2x2 factorial at fixed n:  {no LGD (M=1), LGD (M=3)} x {no momentum, Adam}.
    no-LGD costs n per step; LGD costs 3n. Every individual MMD uses at most n
    conditional samples in all four cells.

METRICS
    exact GMM L2 (primary), failure-penalised mean, success rate, paired mean
    difference with bootstrap 95% CI and paired permutation p, conditional
    calls, runtime.

    python experiments/exp2_lgd_vs_adam.py --n 8 --restarts 100
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (fixed_bandwidth, load, paired_stats,      # noqa: E402
                     penalised_score, save, target_set)
from _guided import evaluate, run, summarise                    # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0,
                    help="restart-index block; use a disjoint block for held-out runs")
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params = load("2D")
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=a.seed)
    mu = unconditional_model(params, seed=SEEDS[0])
    print(f"n={a.n}  restarts={a.restarts}  offset={a.offset}  "
          f"fixed bandwidth={bw:.6f}  adam_rho={a.adam_rho}")

    arms, byL2 = {}, {}
    for spatial in ("no_lgd", "lgd"):
        for temporal in ("none", "adam"):
            runs = []
            for r in range(a.offset, a.offset + a.restarts):
                x, info = run(mc, mu, S_G, bw, a.n, spatial, temporal, r,
                              adam_rho=a.adam_rho)
                runs.append(evaluate(x, params, info))
            s = summarise(runs, a.restarts)
            s["score"] = penalised_score(runs)
            arms[f"{spatial}/{temporal}"] = {"summary": s, "runs": runs}
            byL2[f"{spatial}/{temporal}"] = [
                min(r["L2"], 2.0) if not r["diverged"] else 2.0 for r in runs]
            print(f"  {spatial:<7}{temporal:<6} score={s['score']:.4f} "
                  f"L2mean={s['L2_mean']:.4f} med={s['L2_median']:.4f} "
                  f"succ={s['success_rate']:.0%} div={s['diverged']} "
                  f"calls={s['conditional_calls_mean']:.0f} {s['seconds_total']:.0f}s")

    comps = {}
    for pair, label in (
            (("no_lgd/none", "no_lgd/adam"), "no-LGD: none vs Adam"),
            (("lgd/none", "lgd/adam"), "LGD: none vs Adam"),
            (("lgd/none", "no_lgd/adam"), "LGD/none vs no-LGD/Adam (1/3 cost)"),
            (("no_lgd/none", "lgd/none"), "no-LGD/none vs LGD/none (3x cost)")):
        comps[label] = paired_stats(byL2[pair[0]], byL2[pair[1]])
        c = comps[label]
        print(f"  {label:<40} mean={c['mean_diff']:+.4f} "
              f"CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
              f"wins={c['wins_for_b']}/{c['n']} p={c['perm_p']:.4f}")

    save(f"exp2_lgd_vs_adam_n{a.n}{a.tag}",
         {"config": vars(a), "bandwidth": bw, "arms": arms, "comparisons": comps})


if __name__ == "__main__":
    main()
