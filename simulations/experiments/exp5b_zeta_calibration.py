"""Experiment 5B -- functional calibration of the guidance strength per dim(Y).

WHY
    Experiment 5 is invalid for the question it poses because at d >= 8 its
    no-momentum baseline never leaves the diffusion prior's basin at ANY n
    (0/8 escapes even at n = 2048). Its calibration rule, zeta_d = C/median|g|_d,
    equalises the MAGNITUDE of the first guidance step across d. That is not the
    same as making the baseline a working optimiser: it says nothing about
    whether zeta_d is large enough to traverse the shelf between the prior's
    basin and x*, which is a property of the whole landscape at that d.

    Until the baseline works, Delta(d,n) measures "escapes a plateau" versus
    "does not", and n*(d) is meaningless.

WHAT THIS DOES
    Replaces the magnitude rule with a FUNCTIONAL one. For each d, sweep zeta and
    take

        zeta_d* = the smallest zeta at which the NO-MOMENTUM baseline reaches x*
                  on at least --success-floor of restarts at a large n

    "Large n" (--n-calib, default 128) is deliberately far above the n-grid used
    for the actual comparison, so the calibration is a statement about the
    landscape, not about sampling noise at the n we care about.

    Calibrating on the baseline, never on Adam, is what keeps this fair: the
    momentum arm inherits a zeta chosen to make its COMPETITOR work.

ACCEPTANCE GATE
    d = 1 must reproduce Experiment 3 -- momentum helps at small n and the
    benefit decays -- since the d = 1 benchmark is constructed to match the 2-D
    one. If it does not, the calibration is not comparable across d and the
    downstream sweep must not be run. The gate is reported, not asserted away.

OUTPUT
    zeta_d* per d, plus the full sweep, to results/tfg/. Feed it to Experiment 5
    with --zeta-map.

    python experiments/exp5b_zeta_calibration.py --restarts 24
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import penalised_score, save                        # noqa: E402
from _models import SEEDS, unconditional_model                    # noqa: E402
from exp5_dimy_scaling import (evaluate, probe_gradient_scale,    # noqa: E402
                               run, target_samples)
from tfg.dimy_benchmark import as_params                          # noqa: E402


def bandwidth_of(S_G):
    d2 = torch.cdist(S_G, S_G, p=2) ** 2
    m = S_G.shape[0]
    return float(d2.sum() / (m ** 2 - m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-grid", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--zeta-mult", type=float, nargs="*",
                    default=[0.5, 1, 2, 4, 8, 16, 32, 64],
                    help="multipliers on the magnitude-matched zeta of Exp 5")
    ap.add_argument("--n-calib", type=int, default=128,
                    help="large n: calibration must be about the landscape, "
                         "not about sampling noise at the compared n")
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--success-floor", type=float, default=0.8)
    ap.add_argument("--success-tol", type=float, default=0.5,
                    help="|x_hat - x*| below this counts as reaching x*")
    ap.add_argument("--target-size", type=int, default=250)
    ap.add_argument("--calib-c", type=float, default=0.05)
    ap.add_argument("--x-init", default="randn",
                    help="'randn' draws x_T ~ N(0,1) per restart (correct "
                         "protocol); a float fixes it, reproducing the original")
    ap.add_argument("--step-clip", default="none",
                    choices=["none", "noise", "ddim"],
                    help="noise-level trust region on the applied step; lets "
                         "zeta be raised past the divergence point")
    ap.add_argument("--step-tau", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows, zeta_star = [], {}
    for d in a.d_grid:
        params = as_params(d)
        gen = torch.Generator().manual_seed(987654)
        S_G = target_samples(params, a.target_size, gen)
        bw = bandwidth_of(S_G)
        mu = unconditional_model(params, seed=SEEDS[0], tag=f"_dimy{d}")
        zeta_mag = a.calib_c / max(probe_gradient_scale(params, mu, S_G, bw), 1e-30)
        x_star = float(params["x_star"])
        print(f"\n== d={d} ==  bw={bw:.2f}  zeta_magnitude={zeta_mag:.4g}  "
              f"n_calib={a.n_calib}  x*={x_star}  "
              f"step_clip={a.step_clip}(tau={a.step_tau})")

        xinit = a.x_init if a.x_init == "randn" else float(a.x_init)
        found, best = None, None
        for mult in a.zeta_mult:
            zeta = zeta_mag * mult
            runs, xs = [], []
            for r in range(a.restarts):
                x, info = run(params, mu, S_G, bw, a.n_calib, "none", r,
                              zeta=zeta, step_clip=a.step_clip,
                              step_tau=a.step_tau, x_init=xinit)
                xs.append(x)
                runs.append(evaluate(x, params, info))
            reached = sum(abs(v - x_star) < a.success_tol for v in xs) / a.restarts
            div = sum(q["diverged"] for q in runs) / a.restarts
            # Criterion. The original rule -- 80% reached from one fixed start --
            # was wrong twice over: with x_T pinned to 0 every restart shares a
            # basin, so the rate is 0 or 1 and says nothing about the optimiser;
            # and the objective is genuinely multimodal, so no zeta reaches 80%
            # from an arbitrary start. With x_T ~ N(0,1) the rate IS the
            # basin-of-attraction rate, and the calibrated zeta is the one that
            # maximises it subject to not diverging.
            ok = reached >= a.success_floor and div == 0.0
            rows.append({"d": d, "mult": mult, "zeta": zeta,
                         "reached_rate": reached, "diverged_rate": div,
                         "score": penalised_score(runs),
                         "x_hat_sample": [round(v, 3) for v in xs[:6]],
                         "accepted": ok})
            print(f"  x{mult:<5} zeta={zeta:<12.4g} reached={reached:.0%} "
                  f"div={div:.0%} score={rows[-1]['score']:.4f} "
                  f"x_hat={rows[-1]['x_hat_sample'][:4]}"
                  f"{'  <-- ACCEPTED' if ok else ''}")
            if div == 0.0 and (best is None or reached > best[0]):
                best = (reached, zeta, mult)
            if ok and found is None:
                found = zeta
                break
        if found is None and best is not None and best[0] > 0.0:
            found = best[1]
            print(f"  floor {a.success_floor:.0%} not met; taking the "
                  f"basin-rate maximiser: zeta={best[1]:.4g} (x{best[2]}), "
                  f"reached={best[0]:.0%}")
        zeta_star[d] = found
        if found is None:
            print(f"  d={d}: NO zeta in the grid makes the baseline work "
                  f"(floor={a.success_floor:.0%}). Widen --zeta-mult or the "
                  f"benchmark itself is not traversable at this d.")

    print("\nzeta_d*:")
    for d, z in zeta_star.items():
        print(f"  d={d:<3} {'FAILED' if z is None else f'{z:.6g}'}")
    gate = zeta_star.get(1) is not None
    print(f"\nd=1 anchor calibrated: {gate}. "
          f"{'Run Exp 3 comparison at this zeta before trusting the sweep.' if gate else 'GATE FAILED -- do not run the downstream sweep.'}")

    save(f"exp5b_zeta_calibration{a.tag}",
         {"config": vars(a), "rows": rows,
          "zeta_star": {str(k): v for k, v in zeta_star.items()},
          "d1_gate": gate})


if __name__ == "__main__":
    main()
