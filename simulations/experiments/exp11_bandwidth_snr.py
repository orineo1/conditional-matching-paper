"""Experiment 11 -- does momentum help once the MMD gradient is genuinely low-SNR?

WHY BANDWIDTH AND NOT DIMENSION
    The dimensional route is a dead end on this benchmark, for structural reasons
    documented in the README: extra coordinates either repeat one scalar signal
    (primary construction) or attenuate the kernel uniformly (nuisance
    construction), and in both cases signal and noise scale together, leaving
    SNR flat. Raw dimension is not what makes an MMD estimate hard.

    What makes it hard is STRUCTURE RELATIVE TO THE KERNEL BANDWIDTH. Measured
    relative gradient noise c_emp at d = 1, varying only the bandwidth:

        bw/median   1.0    0.3    0.1
        c (n=8)     0.78   3.83   3.65

    Narrowing the kernel by 3-10x produces a noisier gradient than ANY dimension
    setting reached, and it does so with REAL finite-sample noise rather than the
    injected perturbation of Experiment 10. It is also the better analogue of the
    deployed setting: in CLIP embedding space the target has fine structure
    relative to a median-heuristic bandwidth, whereas this Gaussian toy at median
    bandwidth is smooth.

    So this is the experiment that actually tests the efficiency hypothesis --
    "fewer samples means a noisier gradient, and momentum should pay off there" --
    in a regime where the premise genuinely holds.

PROTOCOL
    Both arms are calibrated SEPARATELY at every bandwidth, because the gradient
    magnitude changes with the kernel and one arm's zeta is not valid for the
    other. That error is documented in the README PROTOCOL CORRECTION and has now
    been made in both directions once, so it is guarded here by construction.

INTERPRETATION, FIXED IN ADVANCE
    * Adam overtakes as bandwidth narrows -> the efficiency hypothesis is right
      and our earlier negatives were a statement about the smooth regime only.
    * Adam still loses at c ~ 4-7 -> momentum does not help even when the
      gradient is genuinely low-SNR, and the negative result is regime-independent
      on this objective.

    python experiments/exp11_bandwidth_snr.py --restarts 60
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
    ap.add_argument("--bw-ratios", type=float, nargs="*", default=[1.0, 0.3, 0.1])
    ap.add_argument("--n", type=int, nargs="*", default=[8])
    ap.add_argument("--zeta-grid", type=float, nargs="*",
                    default=[0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0])
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--calib-restarts", type=int, default=16)
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--step-clip", default="noise")
    ap.add_argument("--x-init", default="randn")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params = load(a.setting)
    S_G = target_set(params)
    bw_med = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0])
    mu = unconditional_model(params, seed=SEEDS[0])
    xstar = float(params["x_star"].reshape(-1)[0])
    print(f"{a.setting} median bandwidth={bw_med:.3f}  restarts={a.restarts}")

    def basin(bw, arm, zeta, n, reps):
        hits = 0
        for r in range(reps):
            x, _ = run(mc, mu, S_G, bw, n, "no_lgd", arm, r, adam_rho=a.adam_rho,
                       zeta=zeta, step_clip=a.step_clip, x_init=a.x_init)
            if abs(float(x.reshape(-1)[0]) - xstar) < 0.5:
                hits += 1
        return hits / reps

    rows = []
    for ratio in a.bw_ratios:
        bw = bw_med * ratio
        zetas = {}
        for arm in ("none", "adam"):
            scored = [(basin(bw, arm, z, a.calib_n, a.calib_restarts), -z, z)
                      for z in a.zeta_grid]
            best = max(scored)
            zetas[arm] = best[2]
            print(f"  bw={bw:.3f} ({ratio}x)  calibrated {arm}: "
                  f"zeta={best[2]} basin={best[0]:.0%}")
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
            adam_better = d["ci95"][0] > 0     # Delta = none - adam; >0 = Adam better
            rows.append({"bw_ratio": ratio, "bw": bw, "n": n,
                         "zeta_none": zetas["none"], "zeta_adam": zetas["adam"],
                         "score_none": cells["none"]["score"],
                         "score_adam": cells["adam"]["score"],
                         "success_none": cells["none"]["success"],
                         "success_adam": cells["adam"]["success"],
                         "delta": d, "adam_better": adam_better})
            print(f"    n={n}: none={cells['none']['score']:.4f} "
                  f"adam={cells['adam']['score']:.4f} "
                  f"Delta={d['mean_diff']:+.4f} "
                  f"CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                  f"p={d['perm_p']:.4f} "
                  f"succ {cells['none']['success']:.0%}/{cells['adam']['success']:.0%}"
                  f"{'   ADAM BETTER' if adam_better else ''}")

    print("\nAdam better anywhere:", any(r["adam_better"] for r in rows))
    save(f"exp11_bandwidth_snr_{a.setting}{a.tag}",
         {"config": vars(a), "bandwidth_median": bw_med, "rows": rows})


if __name__ == "__main__":
    main()
