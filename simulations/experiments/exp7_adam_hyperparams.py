"""Experiment 7 -- is the inherited Adam configuration the right one here?

QUESTION
    Experiments 2, 3, 5 and 6 all use beta1 = 0.9, beta2 = 0.995, delta = 1e-8.
    Those are the official AdamDPS defaults (arXiv:2603.16797, utils/configs.py),
    tuned for a POINTWISE likelihood guidance loss. Our loss is an MMD between a
    sample of the model conditional and a target set, whose gradient has a very
    different noise structure. Nothing so far establishes that the inherited
    constants are the right ones for it -- and Experiment 5A already found that
    beta1 = 0.9 is actively worse than beta1 = 0 on the dim(Y) benchmark.

    This sweeps (beta1, beta2) on the paper's canonical 2D setting, at the sample
    counts where momentum was found to matter, to see whether the operative
    constant is beta2 (normalisation) or beta1 (accumulation), and whether the
    default is anywhere near optimal.

DESIGN
    Tuning block only. Restarts run at --offset 1000 by default, disjoint from
    the 0..99 block used for every reported result, so nothing here can leak into
    the held-out numbers. Any winner must be re-run on the held-out block before
    it is reported as a result.

    The no-momentum baseline is run in the same block for reference, so each cell
    is read as "does this configuration beat plain gradient guidance", not just
    as a ranking among Adam variants.

COMPARISON
    {beta1} x {beta2} x {n}, no LGD, x_t guidance, rho fixed.

    python experiments/exp7_adam_hyperparams.py --restarts 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (fixed_bandwidth, load, paired_stats,      # noqa: E402
                     penalised_score, save, target_set)
from _guided import evaluate, run, summarise                    # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402

DEFAULT_B1 = 0.9
DEFAULT_B2 = 0.995


def scores(runs):
    return [min(q["L2"], 2.0) if not q["diverged"] else 2.0 for q in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="2D", choices=["2D", "5D", "10D"])
    ap.add_argument("--n-grid", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--beta1-grid", type=float, nargs="*",
                    default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--beta2-grid", type=float, nargs="*",
                    default=[0.9, 0.99, 0.995, 0.999])
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=1000,
                    help="tuning block; keep disjoint from the reported 0..99")
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.offset < 100:
        raise SystemExit("--offset < 100 would overlap the held-out block")

    params = load(a.setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0])
    mu = unconditional_model(params, seed=SEEDS[0])
    print(f"setting={a.setting} bandwidth={bw:.4f} rho={a.adam_rho} "
          f"restarts={a.restarts} offset={a.offset}  (TUNING BLOCK)")

    rows = []
    for n in a.n_grid:
        def cell(temporal, **kw):
            runs = []
            for r in range(a.offset, a.offset + a.restarts):
                x, info = run(mc, mu, S_G, bw, n, "no_lgd", temporal, r,
                              adam_rho=a.adam_rho, **kw)
                runs.append(evaluate(x, params, info))
            s = summarise(runs, a.restarts)
            s["score"] = penalised_score(runs)
            return s, scores(runs)

        base_s, base_scores = cell("none")
        print(f"\n== n={n} ==  baseline none: score={base_s['score']:.4f} "
              f"succ={base_s['success_rate']:.0%}")

        best = None
        for b1 in a.beta1_grid:
            for b2 in a.beta2_grid:
                s, sc = cell("adam", beta1=b1, beta2=b2)
                d = paired_stats(base_scores, sc)
                beats = d["ci95"][0] > 0
                is_default = (b1 == DEFAULT_B1 and b2 == DEFAULT_B2)
                rows.append({"n": n, "beta1": b1, "beta2": b2,
                             "score": s["score"],
                             "success_rate": s["success_rate"],
                             "vs_none": d, "beats_none": beats,
                             "is_default": is_default})
                if best is None or s["score"] < best["score"]:
                    best = rows[-1]
                print(f"  b1={b1:<4} b2={b2:<6} score={s['score']:.4f} "
                      f"succ={s['success_rate']:.0%} "
                      f"Delta={d['mean_diff']:+.4f} "
                      f"CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                      f"p={d['perm_p']:.4f}"
                      f"{'  BEATS NONE' if beats else ''}"
                      f"{'  <-- DEFAULT' if is_default else ''}")

        dflt = next(r for r in rows if r["n"] == n and r["is_default"])
        rows.append({"n": n, "baseline_none": base_s})
        print(f"  best: b1={best['beta1']} b2={best['beta2']} "
              f"score={best['score']:.4f}  vs default {dflt['score']:.4f}")

    save(f"exp7_adam_hyperparams_{a.setting}{a.tag}",
         {"config": vars(a), "bandwidth": bw, "rows": rows,
          "note": "TUNING BLOCK ONLY -- re-run any winner on offset 0 before reporting"})


if __name__ == "__main__":
    main()
