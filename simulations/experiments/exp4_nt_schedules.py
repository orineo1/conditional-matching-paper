"""Experiment 4 -- n_t schedules under matched total sample budget.

QUESTION
    Does spending the conditional budget unevenly across diffusion steps beat
    spending it uniformly? Local diagnostics say the MMD gradient is noisiest
    at small n and that noise matters most early, which motivates ramping n_t up
    as the trajectory approaches the data.

COMPARISON
    constant n_t = n  vs  time-increasing  vs  noise-increasing, each with and
    without Adam.

MATCHED COMPUTE -- read this before interpreting anything
    An increasing schedule 1 -> n_max spends about HALF of what constant n_max
    spends (sum_t n_t ~ 0.5 * (T-1) * n_max). Comparing it against constant
    n_max is a cheaper method against a more expensive baseline. This script
    therefore reports sum_t n_t for every arm and additionally runs a constant
    baseline matched to each schedule's realised budget, so the comparison at
    equal cost is available directly.

    python experiments/exp4_nt_schedules.py --n-max 16 --restarts 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (fixed_bandwidth, load, paired_stats,      # noqa: E402
                     penalised_score, save, target_set)
from _guided import evaluate, n_for_step, run, summarise        # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402
from tfg.schedule import DiffusionSchedule                      # noqa: E402


def budget(T, n_max, schedule):
    sched = DiffusionSchedule(T=T)
    return sum(n_for_step(t, sched, n_max, schedule) for t in range(T - 1, 0, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params = load("2D")
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0])
    mu = unconditional_model(params, seed=SEEDS[0])
    T = mu.diffusion_steps

    budgets = {s: budget(T, a.n_max, s) for s in ("constant", "time", "noise")}
    # constant arm matched to the increasing schedules' realised budget
    matched_n = max(1, round(budgets["time"] / (T - 1)))
    budgets["constant_matched"] = budget(T, matched_n, "constant")
    print(f"n_max={a.n_max}  T={T}  budgets (sum_t n_t): {budgets}")
    print(f"matched-budget constant uses n_t = {matched_n}")

    arms, scores = {}, {}
    plan = [("constant", a.n_max), ("time", a.n_max), ("noise", a.n_max),
            ("constant", matched_n)]
    for schedule, n_max in plan:
        for temporal in ("none", "adam"):
            key = f"{schedule}(n={n_max})/{temporal}"
            runs = []
            for r in range(a.offset, a.offset + a.restarts):
                x, info = run(mc, mu, S_G, bw, n_max, "no_lgd", temporal, r,
                              schedule=schedule, adam_rho=a.adam_rho)
                runs.append(evaluate(x, params, info))
            s = summarise(runs, a.restarts)
            s["score"] = penalised_score(runs)
            s["sum_n_t"] = budget(T, n_max, schedule)
            arms[key] = {"summary": s, "runs": runs}
            scores[key] = [min(q["L2"], 2.0) if not q["diverged"] else 2.0
                           for q in runs]
            print(f"  {key:<28} score={s['score']:.4f} L2mean={s['L2_mean']:.4f} "
                  f"succ={s['success_rate']:.0%} sum_n_t={s['sum_n_t']:>6} "
                  f"calls={s['conditional_calls_mean']:.0f}")

    comps = {}
    for tmp in ("none", "adam"):
        base = f"constant(n={matched_n})/{tmp}"
        for sch in ("time", "noise"):
            k = f"{sch}(n={a.n_max})/{tmp}"
            label = f"[{tmp}] matched-constant vs {sch} (equal budget)"
            comps[label] = paired_stats(scores[base], scores[k])
            c = comps[label]
            print(f"  {label:<52} mean={c['mean_diff']:+.4f} "
                  f"CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] p={c['perm_p']:.4f}")

    save(f"exp4_nt_schedules_nmax{a.n_max}{a.tag}",
         {"config": vars(a), "budgets": budgets, "matched_n": matched_n,
          "bandwidth": bw, "arms": arms, "comparisons": comps})


if __name__ == "__main__":
    main()
