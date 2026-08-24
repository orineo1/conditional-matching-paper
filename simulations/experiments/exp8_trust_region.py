"""Experiment 8 -- the noise-level trust region (trust_noise1), THROUGH THE ENGINE.

QUESTION
    Does bounding the guidance step by the current noise level,
    ||Delta_t|| <= sqrt(1 - alphabar_t)  (TFGConfig.temporal.step_clip="noise",
    step_tau=1), improve the no-LGD estimator at equal conditional cost, and does
    the gain transfer across the 2D/5D/10D settings with the same constant?

DESIGN
    Runs through ``tfg.engine.GeneralizedTFG`` via the campaign runner
    (``experiments/model-optimization/estimator/engine_runner.py``), NOT through
    ``_guided.py``; the runner is proven bit-identical to ``_guided.run`` for the
    baseline (tests/test_engine_matches_guided.py). Paired restarts (same tape
    seeds) of baseline vs trust_noise1, no-LGD, no momentum, n in {4, 8, 16, 32};
    optionally LGD/none and the budget-matched baselines at larger n for the Pareto
    statement. ``--offset`` selects a disjoint restart block (the held-out block
    used in VERIFICATION.md is offset 1000, 100 restarts).

METRICS
    failure-penalised mean exact GMM L2, paired mean diff + bootstrap CI +
    permutation p, success rate, conditional generator draws (cm_samples),
    wall time per restart.

    python experiments/exp8_trust_region.py --setting 2D --restarts 100 --offset 1000
    python experiments/exp8_trust_region.py --setting 10D --restarts 2 --n-grid 4 8   # smoke
"""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SIM.parents[0] / "experiments" / "model-optimization" / "estimator"))
from _common import PENALTY, paired_stats, penalised_score, save   # noqa: E402
from _guided import evaluate                                        # noqa: E402
from engine_runner import build_models, run_engine                  # noqa: E402

RULE = "trust_noise1"


def run_cell(params, S_G, bw, mc, mu, n, spatial, candidate, restarts, offset):
    runs, t0 = [], time.perf_counter()
    for r in range(offset, offset + restarts):
        x, info = run_engine(mc, mu, S_G, bw, n, spatial, "none", r, candidate=candidate)
        ev = evaluate(x, params, info)
        ev["cm_samples"] = info["cm_samples"]
        runs.append(ev)
    wall = time.perf_counter() - t0
    fin = [q for q in runs if not q["diverged"]]
    return {"summary": {"score": penalised_score(runs),
                        "success_rate": sum(1 for q in fin if q["abs_err"] < 0.5) / restarts,
                        "diverged": len(runs) - len(fin),
                        "cm_samples": st.mean(q["cm_samples"] for q in runs),
                        "seconds_per_run": wall / restarts},
            "scores": [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in runs],
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="2D", choices=["2D", "5D", "10D"])
    ap.add_argument("--n-grid", type=int, nargs="*", default=[4, 8, 16, 32])
    ap.add_argument("--pareto-n", type=int, nargs="*", default=[64, 96],
                    help="extra baseline-only n for the budget-matched frontier")
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0,
                    help="restart block; 1000 is the held-out block of VERIFICATION.md")
    ap.add_argument("--lgd", action="store_true", help="also run LGD/none at n in --n-grid")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params, S_G, bw, mc, mu = build_models(a.setting)
    print(f"{a.setting}  restarts={a.restarts} offset={a.offset}  rule={RULE}  bandwidth={bw:.6f}")
    cells, comps = {}, {}
    for n in a.n_grid:
        base = run_cell(params, S_G, bw, mc, mu, n, "no_lgd", "baseline", a.restarts, a.offset)
        cand = run_cell(params, S_G, bw, mc, mu, n, "no_lgd", RULE, a.restarts, a.offset)
        cells[f"no_lgd/none/baseline/n{n}"] = base
        cells[f"no_lgd/none/{RULE}/n{n}"] = cand
        c = paired_stats(base["scores"], cand["scores"])
        comps[f"n{n}"] = c
        print(f"  n={n:<3} baseline={base['summary']['score']:.4f} {RULE}={cand['summary']['score']:.4f} "
              f"diff={c['mean_diff']:+.4f} CI[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] "
              f"wins={c['wins_for_b']}/{c['n']} p={c['perm_p']:.4f} calls={cand['summary']['cm_samples']:.0f}")
        if a.lgd:
            cells[f"lgd/none/baseline/n{n}"] = run_cell(params, S_G, bw, mc, mu, n, "lgd",
                                                         "baseline", a.restarts, a.offset)
    for n in a.pareto_n:
        cells[f"no_lgd/none/baseline/n{n}"] = run_cell(params, S_G, bw, mc, mu, n, "no_lgd",
                                                        "baseline", a.restarts, a.offset)
    for k, v in cells.items():
        s = v["summary"]
        print(f"  {k:<32} score={s['score']:.4f} succ={s['success_rate']:.0%} "
              f"calls={s['cm_samples']:.0f} {s['seconds_per_run']:.2f}s/run")
    save(f"exp8_trust_region_{a.setting}{a.tag}",
         {"config": vars(a), "rule": {"step_clip": "noise", "step_tau": 1.0},
          "bandwidth": bw, "cells": cells, "comparisons": comps})


if __name__ == "__main__":
    main()
