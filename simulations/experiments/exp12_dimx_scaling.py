"""Experiment 12 -- does Adam help once dim(X) is genuinely high?

THE OVERSIGHT THIS CORRECTS
    Experiments 3, 9, 10 and 11 all run on the 2D setting, where **dim(X) = 1**.
    Adam's second moment is PER-COORDINATE adaptive scaling. With a scalar design
    variable there are no coordinates to scale across, so m_hat / sqrt(v_hat)
    collapses to a signed step of near-constant magnitude -- which is why those
    experiments read the mechanism as "discards gradient magnitude". In 1-D that
    is all normalisation can possibly do.

    That makes every negative result so far potentially an artifact of dim(X) = 1
    rather than a property of adaptive moments. In a high-dimensional design space
    the second moment does something entirely different and genuinely useful:
    equalising coordinates whose gradient scales differ by orders of magnitude.
    No amount of dim(Y) manipulation could have exposed this, which is why the
    dim(Y) investigation kept returning nulls.

THE LADDER
    dim(X) = 1 (2D setting), 4 (5D), 9 (10D), and 784 (MNIST, a real image
    manifold -- run separately, not by this script). All keep dim(Y) = 1.
    If the deficit shrinks along the ladder, the negative results are a
    low-dimensional artifact and the real setting is where this must be decided.

PROTOCOL
    Per-arm zeta calibration at every setting, since gradient scale changes with
    the setting and one arm's step size is not valid for the other (the error
    documented in the README PROTOCOL CORRECTION, made once in each direction).

    python experiments/exp12_dimx_scaling.py --restarts 60
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
    ap.add_argument("--settings", nargs="*", default=["2D", "5D", "10D"])
    ap.add_argument("--n", type=int, nargs="*", default=[8])
    ap.add_argument("--zeta-grid", type=float, nargs="*",
                    default=[0.03125, 0.125, 0.5, 2.0, 8.0, 32.0])
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--calib-restarts", type=int, default=16)
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--step-clip", default="noise")
    ap.add_argument("--x-init", default="randn")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows = []
    for setting in a.settings:
        params = load(setting)
        S_G = target_set(params)
        bw = fixed_bandwidth(S_G)
        mc = conditional_model(params, seed=SEEDS[0])
        mu = unconditional_model(params, seed=SEEDS[0])
        dim_x = int(params["x_star"].reshape(-1).numel())
        xs = params["x_star"].reshape(-1).double()
        print(f"\n== {setting}: dim(X)={dim_x} bw={bw:.3f} ==")

        def basin(arm, zeta, n, reps):
            hits = 0
            for r in range(reps):
                x, _ = run(mc, mu, S_G, bw, n, "no_lgd", arm, r,
                           adam_rho=a.adam_rho, zeta=zeta,
                           step_clip=a.step_clip, x_init=a.x_init)
                if float((x.reshape(-1).double() - xs).norm()) < 0.5 * dim_x ** 0.5:
                    hits += 1
            return hits / reps

        zetas = {}
        for arm in ("none", "adam"):
            scored = [(basin(arm, z, a.calib_n, a.calib_restarts), -z, z)
                      for z in a.zeta_grid]
            best = max(scored)
            zetas[arm] = best[2]
            print(f"  calibrated {arm}: zeta={best[2]} basin={best[0]:.0%}")

        for n in a.n:
            cells = {}
            for arm in ("none", "adam"):
                runs = []
                for r in range(a.restarts):
                    x, info = run(mc, mu, S_G, bw, n, "no_lgd", arm, r,
                                  adam_rho=a.adam_rho, zeta=zetas[arm],
                                  step_clip=a.step_clip, x_init=a.x_init)
                    runs.append(evaluate(x, params, info))
                cells[arm] = {
                    "score": penalised_score(runs),
                    "success": sum(not q["diverged"] and q["abs_err"] < 0.5
                                   for q in runs) / len(runs),
                    "scores": [min(q["L2"], 2.0) if not q["diverged"] else 2.0
                               for q in runs]}
            d = paired_stats(cells["none"]["scores"], cells["adam"]["scores"])
            adam_better = d["ci95"][0] > 0
            rows.append({"setting": setting, "dim_x": dim_x, "n": n,
                         "zeta_none": zetas["none"], "zeta_adam": zetas["adam"],
                         "score_none": cells["none"]["score"],
                         "score_adam": cells["adam"]["score"],
                         "success_none": cells["none"]["success"],
                         "success_adam": cells["adam"]["success"],
                         "delta": d, "adam_better": adam_better})
            print(f"  n={n}: none={cells['none']['score']:.4f} "
                  f"adam={cells['adam']['score']:.4f} "
                  f"Delta={d['mean_diff']:+.4f} "
                  f"CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                  f"p={d['perm_p']:.4f} "
                  f"succ {cells['none']['success']:.0%}/{cells['adam']['success']:.0%}"
                  f"{'   ADAM BETTER' if adam_better else ''}")

    print("\ndim(X) ladder, Delta = S_none - S_Adam (positive = Adam better):")
    for r in rows:
        print(f"  dim(X)={r['dim_x']:<4} Delta={r['delta']['mean_diff']:+.4f} "
              f"p={r['delta']['perm_p']:.4f}")
    save(f"exp12_dimx_scaling{a.tag}", {"config": vars(a), "rows": rows})


if __name__ == "__main__":
    main()
