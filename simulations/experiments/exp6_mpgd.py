"""Experiment 6 -- MPGD-style guidance, with and without momentum.

QUESTION
    All previous experiments differentiate the guidance loss back through the
    denoiser to x_t (DPS/LGD style). MPGD instead differentiates w.r.t. the
    clean estimate x_{0|t}, treated as a leaf, and moves x_{0|t} directly --
    no backpropagation through the network. Two questions:

      (a) does MPGD-style guidance help at all here?
      (b) does momentum on top of MPGD help, given that momentum helps the
          x_t-style gradient only at small n in 2D?

COMPARISON
    2x2: guidance target {x_t, x0 (MPGD)} x temporal {none, Adam}, no LGD,
    at fixed n. Everything else identical: same target set, bandwidth, seeds,
    restarts, conditional cost.

    In TFG terms the x0 arm is the mu branch with N_iter = 1 and rho = 0
    (Ye et al. 2024, Theorem 3.2 identifies exactly this subspace as MPGD).

    python experiments/exp6_mpgd.py --n 8 --restarts 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (fixed_bandwidth, load, paired_stats,      # noqa: E402
                     penalised_score, save, target_set)
from _guided import evaluate, run, summarise                    # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--mu-strength", type=float, default=1.0)
    ap.add_argument("--setting", default="2D")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params = load(a.setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0])
    mu = unconditional_model(params, seed=SEEDS[0])
    print(f"setting={a.setting} n={a.n} restarts={a.restarts} bw={bw:.4f} "
          f"adam_rho={a.adam_rho} mu_strength={a.mu_strength}")

    arms, scores = {}, {}
    for target in ("x_t", "x0"):
        for temporal in ("none", "adam"):
            key = f"{target}/{temporal}"
            runs = []
            for r in range(a.offset, a.offset + a.restarts):
                x, info = run(mc, mu, S_G, bw, a.n, "no_lgd", temporal, r,
                              adam_rho=a.adam_rho, guidance_target=target,
                              mu_strength=a.mu_strength)
                runs.append(evaluate(x, params, info))
            s = summarise(runs, a.restarts)
            s["score"] = penalised_score(runs)
            arms[key] = {"summary": s, "runs": runs}
            scores[key] = [min(q["L2"], 2.0) if not q["diverged"] else 2.0
                           for q in runs]
            print(f"  {key:<12} score={s['score']:.4f} L2mean={s['L2_mean']:.4f} "
                  f"med={s['L2_median']:.4f} succ={s['success_rate']:.0%} "
                  f"div={s['diverged']} calls={s['conditional_calls_mean']:.0f}")

    comps = {}
    for pair, label in (
            (("x_t/none", "x0/none"), "x_t vs MPGD (no momentum)"),
            (("x0/none", "x0/adam"), "MPGD: none vs Adam"),
            (("x_t/none", "x_t/adam"), "x_t: none vs Adam"),
            (("x_t/adam", "x0/adam"), "x_t+Adam vs MPGD+Adam")):
        comps[label] = paired_stats(scores[pair[0]], scores[pair[1]])
        c = comps[label]
        print(f"  {label:<32} mean={c['mean_diff']:+.4f} "
              f"CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
              f"wins={c['wins_for_b']}/{c['n']} p={c['perm_p']:.4f}")

    save(f"exp6_mpgd_{a.setting}_n{a.n}{a.tag}",
         {"config": vars(a), "bandwidth": bw, "arms": arms, "comparisons": comps})


if __name__ == "__main__":
    main()
