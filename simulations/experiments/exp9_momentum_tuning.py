"""Experiment 9 -- tune PURE momentum properly, in the regime it is meant for.

WHY THIS EXISTS
    Experiment 3 [calibrated] found full Adam worse than no momentum, and the
    mechanism was normalisation: dividing by sqrt(v_hat) discards gradient
    magnitude, which on this multimodal landscape is signal. That is an argument
    against normalisation, NOT against momentum. Accumulation without division
    is a separate rule and has never been tuned.

    Two mistakes in the first attempt at it, both corrected here:

    1. It was calibrated at n = 128. Momentum's mechanism is variance reduction
       of a noisy gradient estimate, so its benefit should be largest where the
       estimate is noisiest -- SMALL n. Tuning at n = 128 selects for
       basin-finding, which is not what momentum is for, and finds nothing.
    2. Only beta1 in {0.5, 0.9} at a fixed zeta. beta1 and zeta are coupled: an
       EMA changes the effective step size, so the useful (beta1, zeta) pairs lie
       on a ridge and a one-dimensional slice through it can miss the ridge
       entirely.

DESIGN
    Joint (beta1, zeta) grid at the small n where momentum should help, scored
    against the separately calibrated no-momentum baseline. Tuning block at
    --offset 1000, disjoint from the 0..99 block used for reported results; any
    winner must be re-run held-out before it is reported.

    python experiments/exp9_momentum_tuning.py --n 8 --restarts 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (fixed_bandwidth, load, paired_stats,      # noqa: E402
                     penalised_score, save, target_set)
from _guided import evaluate, run                               # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="2D")
    ap.add_argument("--n", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--beta1-grid", type=float, nargs="*",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9, 0.95])
    ap.add_argument("--zeta-grid", type=float, nargs="*",
                    default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--zeta-baseline", type=float, default=8.0,
                    help="separately calibrated no-momentum step (Exp 5B rule)")
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=1000)
    ap.add_argument("--step-clip", default="noise")
    ap.add_argument("--x-init", default="randn")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.offset < 100:
        raise SystemExit("--offset < 100 would overlap the held-out block")

    params = load(a.setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0])
    mu = unconditional_model(params, seed=SEEDS[0])
    print(f"{a.setting} bw={bw:.3f} restarts={a.restarts} offset={a.offset} "
          f"(TUNING BLOCK)  baseline zeta={a.zeta_baseline}")

    def cell(n, temporal, zeta, beta1=0.9):
        runs = []
        for r in range(a.offset, a.offset + a.restarts):
            x, i = run(mc, mu, S_G, bw, n, "no_lgd", temporal, r, zeta=zeta,
                       beta1=beta1, step_clip=a.step_clip, x_init=a.x_init)
            runs.append(evaluate(x, params, i))
        return (penalised_score(runs),
                [min(q["L2"], 2.0) if not q["diverged"] else 2.0 for q in runs],
                sum(not q["diverged"] and q["abs_err"] < 0.5
                    for q in runs) / len(runs))

    rows = []
    for n in a.n:
        b_score, b_scores, b_succ = cell(n, "none", a.zeta_baseline)
        print(f"\n== n={n} ==  baseline none: score={b_score:.4f} succ={b_succ:.0%}")
        best = None
        for b1 in a.beta1_grid:
            for z in a.zeta_grid:
                sc, scores, succ = cell(n, "momentum", z, beta1=b1)
                d = paired_stats(b_scores, scores)
                beats = d["ci95"][0] > 0
                rows.append({"n": n, "beta1": b1, "zeta": z, "score": sc,
                             "success_rate": succ, "vs_baseline": d,
                             "beats_baseline": beats})
                if best is None or sc < best["score"]:
                    best = rows[-1]
                flag = "  BEATS BASELINE" if beats else ""
                print(f"  b1={b1:<5} zeta={z:<6} score={sc:.4f} succ={succ:.0%} "
                      f"Delta={d['mean_diff']:+.4f} "
                      f"CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                      f"p={d['perm_p']:.4f}{flag}")
        print(f"  best: b1={best['beta1']} zeta={best['zeta']} "
              f"score={best['score']:.4f} vs baseline {b_score:.4f}")
        rows.append({"n": n, "baseline": {"score": b_score, "success": b_succ,
                                          "zeta": a.zeta_baseline}})

    save(f"exp9_momentum_tuning_{a.setting}{a.tag}",
         {"config": vars(a), "bandwidth": bw, "rows": rows,
          "note": "TUNING BLOCK -- re-run any winner held-out before reporting"})


if __name__ == "__main__":
    main()
