"""Experiment 5A -- why does Adam cross the plateau?

Experiment 5 established that at dim(Y) >= 8 plain gradient guidance never
leaves the prior's basin (x_hat ~ 0.39 at every n, including n = 2048) while
Adam reaches x* = -5 on most restarts. Two mechanisms could explain it, and
Adam applies both at once:

  ACCUMULATION (beta1)  a small but persistent gradient is summed across
                        diffusion steps until the total displacement is enough
                        to cross the shelf.

  NORMALISATION (beta2) the update is divided by sqrt(v_hat), so a tiny
                        gradient produces a FULL-SIZE step. A 1e-3 signal moves
                        x by ~rho per step instead of by 1e-3.

The arithmetic favours normalisation: a 1e-3 gradient summed over 99 steps
gives 0.1, but the shelf is ~5 wide. Normalisation alone gives ~99*rho.

DESIGN
    Three arms, identical in every other respect:
      none              raw gradient
      adam              beta1 = 0.9, beta2 = 0.995     (both mechanisms)
      normalise_only    beta1 = 0.0, beta2 = 0.995     (normalisation only)

    Reported per arm: escape rate (x_hat < -3), how many escapes land near x*,
    and the exact GMM L2.

    An accumulation-only arm was dropped: with beta2 -> 1 the update still
    divides by a frozen sqrt(v_hat), which is normalisation by a constant, so
    it does not isolate accumulation and is not a usable signal.

    python experiments/exp5a_plateau_mechanism.py --d 8 --restarts 40
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import paired_stats, penalised_score, save          # noqa: E402
from _models import SEEDS, unconditional_model                    # noqa: E402
from exp5_dimy_scaling import (evaluate, probe_gradient_scale,    # noqa: E402
                               run, target_samples)
from tfg.dimy_benchmark import as_params                          # noqa: E402

ARMS = {
    "none":            dict(temporal="none"),
    "adam":            dict(temporal="adam", beta1=0.9, beta2=0.995),
    "normalise_only":  dict(temporal="adam", beta1=0.0, beta2=0.995),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params = as_params(a.d)
    gen = torch.Generator().manual_seed(987654)
    S_G = target_samples(params, 250, gen)
    d2 = torch.cdist(S_G, S_G, p=2) ** 2
    m = S_G.shape[0]
    bw = float(d2.sum() / (m ** 2 - m))
    mu = unconditional_model(params, seed=SEEDS[0], tag=f"_dimy{a.d}")
    zeta = 0.05 / max(probe_gradient_scale(params, mu, S_G, bw), 1e-30)
    x_star = float(params["x_star"])
    print(f"d={a.d} n={a.n} restarts={a.restarts} bw={bw:.2f} zeta={zeta:.3f} "
          f"x*={x_star}")

    out = {}
    for name, kw in ARMS.items():
        runs, xs = [], []
        for r in range(a.restarts):
            x, info = run(params, mu, S_G, bw, a.n, kw["temporal"], r,
                          a.adam_rho, zeta=zeta,
                          **{k: v for k, v in kw.items() if k != "temporal"})
            xs.append(x)
            runs.append(evaluate(x, params, info))
        escaped = [v for v in xs if v < -3]
        near = [v for v in xs if abs(v - x_star) < 0.5]
        out[name] = {"score": penalised_score(runs),
                     "escape_rate": len(escaped) / a.restarts,
                     "near_optimum_rate": len(near) / a.restarts,
                     "x_hat_sample": [round(v, 3) for v in xs[:8]],
                     "scores": [min(q["L2"], 2.0) if not q["diverged"] else 2.0
                                for q in runs]}
        print(f"  {name:<16} score={out[name]['score']:.4f} "
              f"escaped={out[name]['escape_rate']:.0%} "
              f"near_x*={out[name]['near_optimum_rate']:.0%} "
              f"x_hat={out[name]['x_hat_sample'][:5]}")

    comps = {}
    for a_, b_, label in (("none", "normalise_only", "none vs normalise-only"),
                          ("none", "adam", "none vs full Adam"),
                          ("normalise_only", "adam", "normalise-only vs full Adam")):
        comps[label] = paired_stats(out[a_]["scores"], out[b_]["scores"])
        c = comps[label]
        print(f"  {label:<32} mean={c['mean_diff']:+.4f} "
              f"CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] p={c['perm_p']:.4f}")

    verdict = ("normalisation suffices; beta1 momentum is not required"
               if out["normalise_only"]["escape_rate"] >= out["adam"]["escape_rate"]
               else "beta1 momentum contributes")
    print(f"\nmechanism: {verdict}")
    save(f"exp5a_plateau_mechanism_d{a.d}{a.tag}",
         {"config": vars(a), "zeta": zeta, "bandwidth": bw, "arms": out,
          "comparisons": comps, "verdict": verdict})


if __name__ == "__main__":
    main()
