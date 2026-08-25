"""Experiment 10 -- replicate the AdamDPS synthetic protocol on our benchmark.

MOTIVATION (from the AdamDPS paper, arXiv:2603.16797v2, Section 4)
    AdamDPS's own synthetic study uses a 2-D GMM in which the likelihood score is
    available in closed form. The authors state that the setting is therefore
    "free of these noise sources", and that they "simulate them by adding to the
    guidance term Gaussian noise of magnitude zeta * ||grad_x log p(y|x_0|t)||".
    Their Figure 1 then plots KL divergence against that coefficient and shows
    AdamDPS "achieving lower KL divergence with the target distribution AS ZETA
    INCREASES".

    So the paper's claim is explicitly conditional: adaptive moments buy
    robustness TO GRADIENT NOISE. Where the guidance gradient is not noise-
    limited, the figure predicts little or no benefit.

WHY THIS MATTERS HERE
    Experiments 3 and 9 [calibrated] find Adam, and momentum alone, WORSE than
    plain guidance on our benchmark at every step size tested. That is not
    prima facie a contradiction: our binding difficulty looks like basin
    selection on a multimodal objective, not estimator noise.

    But there is a genuine tension. If noise were the whole story, Adam should
    close the gap at our NOISIEST setting, small n. It does not. That suggests
    our finite-sample MMD noise differs in kind from their additive perturbation:
    theirs is magnitude-proportional and direction-preserving in expectation,
    whereas resampling the conditional set can swing the gradient DIRECTION
    between basins, and momentum then averages incompatible directions.

WHAT THIS RUNS
    Their protocol exactly: take our (comparatively clean) gradient and add
    Gaussian noise of magnitude noise_coeff * ||grad||, sweep the coefficient,
    and compare no-momentum against Adam at each level. Both arms see the SAME
    noise realisation at every (restart, t), so the comparison is paired.

INTERPRETATION, FIXED IN ADVANCE
    * Adam overtakes plain guidance as the coefficient rises -> our benchmark
      reproduces their Figure 1, both results stand, and our negative result is a
      statement about REGIME (we are not noise-limited) rather than about Adam.
    * Adam never overtakes -> our benchmark genuinely disagrees with theirs, and
      the difference must lie in the objective (multimodal distribution matching
      vs. their reconstruction likelihood), which is a reportable finding.

    Each arm keeps its own separately calibrated zeta, since applying one arm's
    step size to the other is the error documented in the PROTOCOL CORRECTION.

    python experiments/exp10_guidance_noise.py --n 8 --restarts 60
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
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--noise-grid", type=float, nargs="*",
                    default=[0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--zeta-none", type=float, default=8.0)
    ap.add_argument("--zeta-adam", type=float, default=0.125)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--step-clip", default="noise")
    ap.add_argument("--x-init", default="randn")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    params = load(a.setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0])
    mu = unconditional_model(params, seed=SEEDS[0])
    print(f"{a.setting} n={a.n} bw={bw:.3f} restarts={a.restarts} "
          f"zeta none/adam = {a.zeta_none}/{a.zeta_adam}")
    print("AdamDPS Fig.1 protocol: grad <- grad + c*||grad||*eps\n")

    rows, crossover = [], None
    for c in a.noise_grid:
        cells = {}
        for arm, z in (("none", a.zeta_none), ("adam", a.zeta_adam)):
            runs = []
            for r in range(a.offset, a.offset + a.restarts):
                x, i = run(mc, mu, S_G, bw, a.n, "no_lgd", arm, r,
                           adam_rho=a.adam_rho, zeta=z, noise_coeff=c,
                           step_clip=a.step_clip, x_init=a.x_init)
                runs.append(evaluate(x, params, i))
            cells[arm] = {
                "score": penalised_score(runs),
                "success": sum(not q["diverged"] and q["abs_err"] < 0.5
                               for q in runs) / len(runs),
                "scores": [min(q["L2"], 2.0) if not q["diverged"] else 2.0
                           for q in runs]}
        d = paired_stats(cells["none"]["scores"], cells["adam"]["scores"])
        # Delta = S_none - S_Adam and LOWER score is better, so Adam is better
        # exactly when Delta > 0. Testing ci95[1] < 0 flags the OPPOSITE case
        # (Adam significantly worse) and inverts every label.
        adam_wins = d["ci95"][0] > 0      # CI entirely on Adam's side
        if adam_wins and crossover is None:
            crossover = c
        rows.append({"noise_coeff": c,
                     "score_none": cells["none"]["score"],
                     "score_adam": cells["adam"]["score"],
                     "success_none": cells["none"]["success"],
                     "success_adam": cells["adam"]["success"],
                     "delta": d, "adam_significantly_better": adam_wins})
        print(f"  c={c:<5} none={cells['none']['score']:.4f} "
              f"adam={cells['adam']['score']:.4f} "
              f"Delta={d['mean_diff']:+.4f} "
              f"CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
              f"p={d['perm_p']:.4f} "
              f"succ {cells['none']['success']:.0%}/{cells['adam']['success']:.0%}"
              f"{'   ADAM WINS' if adam_wins else ''}")

    print(f"\ncrossover (first c where Adam significantly better): {crossover}")
    print("reproduces AdamDPS Figure 1" if crossover is not None
          else "does NOT reproduce AdamDPS Figure 1 on this benchmark")
    save(f"exp10_guidance_noise_{a.setting}_n{a.n}{a.tag}",
         {"config": vars(a), "bandwidth": bw, "rows": rows,
          "crossover": crossover})


if __name__ == "__main__":
    main()
