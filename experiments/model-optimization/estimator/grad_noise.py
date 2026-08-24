"""Agent 4 -- gradient-noise measurement for the regime explanation.

At fixed points x_t of baseline (no_lgd/none, n=8) trajectories, draw K
independent conditional-noise sets of size n and compute the rho-branch
gradient g = d(MMD^2)/dx_t for each. Report, per (n, t): |E g|, sd(g),
SNR = |E g| / sd(g), the probability the draw's sign disagrees with the mean
sign, the raw step |g| vs the Adam step (~rho=0.4 per coordinate once v_hat has
warmed up), and the distance to x*.

    python experiments/model-optimization/estimator/grad_noise.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine_runner import build_models, run_engine, repository_schedule, CMSampler, DistributionalLoss  # noqa
from _models import PAPER_TS  # noqa
from tfg.noise_tape import NoiseTape  # noqa

K = 64
NS = (4, 8, 32)
TS = (90, 70, 50, 30, 15, 5, 1)
RESTARTS = range(8)


def main():
    params, S_G, bw, mc, mu = build_models("2D")
    sch = repository_schedule(mu)
    loss = DistributionalLoss(S_G, bandwidth="fixed", bandwidth_value=bw)
    x_star = float(params["x_star"].reshape(-1)[0])
    out = {}
    for r in RESTARTS:
        x, info = run_engine(mc, mu, S_G, bw, 8, "no_lgd", "none", r, trace_steps=True)
        for t in TS:
            # x_t entering step t is x_prev of step t+1 (or 0 at t = T)
            xt = (info["x_prev_trace"][(t + 1, 1)] if t < sch.T else torch.zeros(1, 1))
            for n in NS:
                tape = NoiseTape(seed=10_000 + r, dtype=torch.float32)
                smp = CMSampler(mc, PAPER_TS, tape, source="tape")
                gs = []
                for k in range(K):
                    keys = [("gn", k, i) for i in range(n)]
                    xx = xt.clone().requires_grad_(True)
                    eps = mu(xx, torch.full([1, 1], t), None)
                    x0 = (xx - sch.sqrt_one_minus_ab(t) * eps) / sch.sqrt_ab(t)
                    m2 = loss(smp(x0, keys))
                    g, = torch.autograd.grad(m2, xx)
                    gs.append(float(g))
                mean, sd = st.mean(gs), st.pstdev(gs)
                sign_dis = sum(1 for g in gs if (g > 0) != (mean > 0)) / K
                out.setdefault(f"n={n}", {}).setdefault(f"t={t}", []).append(
                    {"restart": r, "x_t": float(xt), "dist_to_xstar": abs(float(xt) - x_star),
                     "mean_g": mean, "sd_g": sd, "snr": abs(mean) / sd if sd > 0 else float("inf"),
                     "sign_disagree": sign_dis, "abs_mean_g": abs(mean),
                     "median_abs_g": st.median(abs(g) for g in gs)})
    summary = {}
    for nk, d in out.items():
        for tk, rows in d.items():
            summary.setdefault(nk, {})[tk] = {
                "snr_median": st.median(r["snr"] for r in rows),
                "sign_disagree_mean": st.mean(r["sign_disagree"] for r in rows),
                "abs_mean_g_median": st.median(r["abs_mean_g"] for r in rows),
                "sd_g_median": st.median(r["sd_g"] for r in rows),
                "dist_median": st.median(r["dist_to_xstar"] for r in rows)}
    (HERE / "grad_noise.json").write_text(json.dumps({"summary": summary, "raw": out}, indent=1))
    print(f"{'n':>3} {'t':>3} {'SNR med':>8} {'P(sign flip)':>12} {'|E g| med':>10} {'sd g med':>9} {'dist med':>8}")
    for nk in summary:
        for tk, s in summary[nk].items():
            print(f"{nk[2:]:>3} {tk[2:]:>3} {s['snr_median']:8.3f} {s['sign_disagree_mean']:12.3f} "
                  f"{s['abs_mean_g_median']:10.4f} {s['sd_g_median']:9.4f} {s['dist_median']:8.3f}")


if __name__ == "__main__":
    main()
