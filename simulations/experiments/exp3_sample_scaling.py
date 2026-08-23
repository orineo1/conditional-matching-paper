"""Experiment 3 -- sample (and dimension) scaling of the momentum benefit.

QUESTION
    Where does Adam stop helping? Define

        Delta(n) = S_none(n) - S_Adam(n)          (S = failure-penalised mean L2)

    and n* = the largest n at which the 95% lower confidence bound on Delta is
    still positive. Hypothesis: the useful regime is small n, and n* grows with
    the dimension of the distribution being matched.

    NOTE ON DIMENSION. The paper's 5D and 10D settings scale dim(X) and keep
    dim(Y) = 1 (verified from params/{5D,10D}_cond_1D_gmm_params.pt, whose
    mog_means is (2,1,1)). MMD sample complexity lives in dim(Y), so those
    settings do NOT provide a dim(Y) sweep. This script therefore sweeps n at
    the available settings and reports dim(X) and dim(Y) explicitly; a genuine
    dim(Y) sweep needs a new benchmark and is NOT claimed here.

COMPARISON
    no-LGD x {none, Adam} across n, per setting.

    python experiments/exp3_sample_scaling.py --restarts 100
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
    ap.add_argument("--setting", default="2D", choices=["2D", "5D", "10D"])
    ap.add_argument("--params-file", default=None,
                    help="explicit parameter file; overrides --setting. Use this "
                         "to run the legacy .txt file through identical code.")
    ap.add_argument("--n-grid", type=int, nargs="*", default=[4, 8, 16, 32])
    ap.add_argument("--restarts", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.params_file:
        from tfg import oracle as _o
        params = _o.load_params(a.params_file)
    else:
        params = load(a.setting)
    print(f"params: {params['source']}")
    dim_x = int(params["x_star"].reshape(-1).numel())
    dim_y = int(params["target_means"].reshape(params["target_means"].shape[0], -1).shape[1])
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    tag = a.tag or ("_" + Path(str(params["source"])).stem if a.params_file else "")
    mc = conditional_model(params, seed=SEEDS[0], tag=tag)
    mu = unconditional_model(params, seed=SEEDS[0], tag=tag)
    print(f"setting={a.setting}  dim(X)={dim_x}  dim(Y)={dim_y}  "
          f"bandwidth={bw:.6f}  restarts={a.restarts}")

    rows, n_star = [], None
    for n in a.n_grid:
        cells = {}
        for temporal in ("none", "adam"):
            runs = []
            for r in range(a.offset, a.offset + a.restarts):
                x, info = run(mc, mu, S_G, bw, n, "no_lgd", temporal, r,
                              adam_rho=a.adam_rho)
                runs.append(evaluate(x, params, info))
            s = summarise(runs, a.restarts)
            s["score"] = penalised_score(runs)
            cells[temporal] = {"summary": s,
                               "scores": [min(q["L2"], 2.0) if not q["diverged"]
                                          else 2.0 for q in runs]}
        delta = paired_stats(cells["none"]["scores"], cells["adam"]["scores"])
        supported = delta["ci95"][0] > 0
        if supported:
            n_star = n
        rows.append({"n": n, "dim_x": dim_x, "dim_y": dim_y,
                     "score_none": cells["none"]["summary"]["score"],
                     "score_adam": cells["adam"]["summary"]["score"],
                     "delta": delta, "supported": supported,
                     "summaries": {k: v["summary"] for k, v in cells.items()}})
        print(f"  n={n:<4} none={rows[-1]['score_none']:.4f} "
              f"adam={rows[-1]['score_adam']:.4f} "
              f"Delta={delta['mean_diff']:+.4f} "
              f"CI[{delta['ci95'][0]:+.4f},{delta['ci95'][1]:+.4f}] "
              f"p={delta['perm_p']:.4f} {'SUPPORTED' if supported else ''}")

    print(f"\nn* ({a.setting}, dim(Y)={dim_y}) = {n_star}")
    save(f"exp3_sample_scaling_{a.setting}{a.tag}",
         {"config": vars(a), "dim_x": dim_x, "dim_y": dim_y,
          "bandwidth": bw, "rows": rows, "n_star": n_star})


if __name__ == "__main__":
    main()
