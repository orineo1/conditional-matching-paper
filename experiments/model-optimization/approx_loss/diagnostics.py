"""Diagnostics of approximate MMD objectives against the exact repository loss.

Settings
  synthetic : d=1 (paper 2D GMM target, m=250), d=8,16 (tfg.dimy_benchmark,
              m=250), n in {4,8,32}; X = n conditional draws P(Y|X=x) at a
              random x (what the guidance loop actually feeds the loss).
  CLIP-like : d=768, m=120 targets = two Gaussian clusters on the unit
              sphere; X = n draws from a third, shifted cluster; n in {8,32}.

Per (setting, n, candidate, size D, feature seed) vs the exact reference:
  rel loss error, gradient cosine, rel gradient-norm error, wall time of
  forward+backward (median of >=7 after warm-up), analytic peak intermediate
  memory, hardware-independent flop-ish cost model; plus across-feature-seed
  dispersion (CV of the loss, mean pairwise gradient cosine).

Outputs diagnostics.csv, DIAGNOSTICS.md and png plots next to this file.
Run:  /Users/stolk/miniconda3/bin/python diagnostics.py [--quick]
"""
import argparse
import csv
import resource
import statistics as st
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import approx_mmd as A  # noqa: E402

sys.path.insert(0, str(A._SIM / "experiments"))
import _common  # noqa: E402
from tfg import oracle  # noqa: E402
from tfg.dimy_benchmark import as_params  # noqa: E402

torch.set_default_dtype(torch.float64)
DT = torch.float64


# ------------------------------------------------------------------ problems
def synth_problem(d, m=250):
    if d == 1:
        params = _common.load("2D")
        Y = _common.target_set(params, size=m).double()
    else:
        params = as_params(d)
        gen = torch.Generator().manual_seed(987654)
        tm, tw = params["target_means"], params["target_weights"]
        L = torch.linalg.cholesky(params["target_variances"][0])
        idx = torch.multinomial(tw, m, replacement=True, generator=gen)
        Y = tm[idx] + torch.randn(m, d, generator=gen) @ L.T
    bw = _common.fixed_bandwidth(Y)
    gmm = (params["target_means"], torch.stack([torch.as_tensor(v, dtype=DT) for v in params["target_variances"]]),
           params["target_weights"])

    def draw_X(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = (torch.rand(1, generator=g) * 14 - 7).reshape(1, 1)   # x in [-7, 7]
        cm, cc, w = oracle.conditional_params(
            x.double(), params["mu_list"], params["Sigma_list"], params["alpha"])
        cc = cc if cc.dim() == 2 else cc[0]
        Lc = torch.linalg.cholesky(cc)
        idx = torch.multinomial(w.detach(), n, replacement=True, generator=g)
        eps = torch.randn(n, cm.shape[1], generator=g)
        return (cm[idx] + eps @ Lc.T).detach().reshape(n, -1)

    return dict(name=f"synth_d{d}", d=d, Y=Y, bw=bw, gmm=gmm, draw_X=draw_X)


def clip_problem(d=768, m=120, seed=11):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(3, d, generator=g)
    mu = mu / mu.norm(dim=1, keepdim=True)
    std = 0.7 / d ** 0.5                       # clusters ~ 0.7 wide on the sphere
    pts = []
    for k in range(2):
        pts.append(mu[k] + std * torch.randn(m // 2, d, generator=g))
    Y = torch.cat(pts)
    Y = Y / Y.norm(dim=1, keepdim=True)
    bw = _common.fixed_bandwidth(Y)

    def draw_X(n, seed):
        gg = torch.Generator().manual_seed(seed)
        c = 0.5 * mu[0] + 0.5 * mu[2]
        X = c + std * torch.randn(n, d, generator=gg)
        return (X / X.norm(dim=1, keepdim=True)).detach()

    return dict(name="clip_d768", d=d, Y=Y, bw=bw, gmm=None, draw_X=draw_X)


# ------------------------------------------------------------------ helpers
def grad_of(obj, X):
    X = X.detach().clone().requires_grad_(True)
    L = obj.loss(X)
    g, = torch.autograd.grad(L, X)
    return float(L), g.reshape(-1)


def timed(obj, X, reps=7):
    X = X.detach().clone().requires_grad_(True)
    for _ in range(2):
        L = obj.loss(X)
        torch.autograd.grad(L, X)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        L = obj.loss(X)
        torch.autograd.grad(L, X)
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-300))


def peak_mem_mb(name, n, m, d, D, nk=5):
    """Analytic largest intermediate (float64) of one forward pass, MB."""
    b = 8
    if name == "reference":
        v = nk * (n + m) ** 2 + (n + m) ** 2 * d       # 5 exps + cdist temp
    elif name == "exact_cachedYY":
        v = nk * n * m + n * m * d
    elif name in ("rff", "orf"):
        v = nk * n * D * 2 + n * d
    elif name == "nystrom":
        v = nk * n * D + D * D
    elif name == "subsample":
        v = nk * n * D
    elif name == "sliced_w2":
        v = n * D + m * D
    elif name == "population_gmm":
        v = nk * n * n + n * 11 * d * d
    elif name == "tab_kme_1d":
        v = nk * n * n + 2048
    else:
        v = 0
    return v * b / 1e6


# ------------------------------------------------------------------ main
def candidates(prob, quick):
    d, m = prob["d"], prob["Y"].shape[0]
    Ds = [16, 64, 256, 1024] if not quick else [16, 256]
    out = [("reference", None, False), ("exact_cachedYY", None, False)]
    for D in Ds:
        out.append(("rff", D, True))
    if d >= 64:
        for D in Ds:
            out.append(("orf", D, True))
    for L in [x for x in Ds if x <= m]:
        out.append(("nystrom", L, True))
    for B in [x for x in Ds if x < m]:
        out.append(("subsample", B, True))
    out.append(("sliced_w2", 1 if d == 1 else 32, d > 1))
    if prob["gmm"] is not None:
        out.append(("population_gmm", None, False))
    if d == 1:
        out.append(("tab_kme_1d", 2048, False))
    return out


def build(name, prob, D, seed):
    Y, bw = prob["Y"], prob["bw"]
    if name == "population_gmm":
        return A.PopulationGMMMMD(*prob["gmm"], bw)
    kw = {}
    if name in ("rff", "orf"):
        kw = dict(D=D, seed=seed)
    elif name == "nystrom":
        kw = dict(L=D, seed=seed)
    elif name == "subsample":
        kw = dict(B=D, seed=seed)
    elif name == "sliced_w2":
        kw = dict(P=D, seed=seed)
    elif name == "tab_kme_1d":
        kw = dict(G=D)
    return A.make(name, Y, bw, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--x-draws", type=int, default=4)
    ap.add_argument("--feat-seeds", type=int, default=6)
    a = ap.parse_args()
    if a.quick:
        a.x_draws, a.feat_seeds = 2, 3

    probs = [synth_problem(1), synth_problem(8), synth_problem(16), clip_problem()]
    n_grid = {"synth": [4, 8, 32], "clip": [8, 32]}
    rows = []
    for prob in probs:
        ns = n_grid["clip" if prob["name"].startswith("clip") else "synth"]
        ref = A.Reference(prob["Y"], prob["bw"])
        m = prob["Y"].shape[0]
        print(f"== {prob['name']}  d={prob['d']} m={m} bw={prob['bw']:.4g}")
        for n in ns:
            Xs = [prob["draw_X"](n, 1000 + i) for i in range(a.x_draws)]
            refs = [grad_of(ref, X) for X in Xs]
            t_ref = timed(ref, Xs[0])
            for name, D, random in candidates(prob, a.quick):
                seeds = range(a.feat_seeds) if random else [0]
                objs = [build(name, prob, D, s) for s in seeds]
                per_seed = []      # (loss_err, cos, gnorm_err) averaged over X
                losses_by_X = []
                grads_by_X = []
                for X, (Lr, gr) in zip(Xs, refs):
                    ls, gs = [], []
                    for o in objs:
                        L, g = grad_of(o, X)
                        ls.append(L)
                        gs.append(g)
                    losses_by_X.append(ls)
                    grads_by_X.append(gs)
                    per_seed.append([
                        (abs(L - Lr) / max(abs(Lr), 1e-300),
                         cos(g, gr),
                         abs(float(g.norm()) - float(gr.norm())) / max(float(gr.norm()), 1e-300))
                        for L, g in zip(ls, gs)])
                loss_err = st.mean(v[0] for ps in per_seed for v in ps)
                gcos = st.mean(v[1] for ps in per_seed for v in ps)
                gnerr = st.mean(v[2] for ps in per_seed for v in ps)
                # across-seed dispersion
                if len(seeds) > 1:
                    cv = st.mean(st.pstdev(ls) / max(abs(st.mean(ls)), 1e-300)
                                 for ls in losses_by_X)
                    pair = []
                    for gs in grads_by_X:
                        for i in range(len(gs)):
                            for j in range(i + 1, len(gs)):
                                pair.append(cos(gs[i], gs[j]))
                    seed_cos = st.mean(pair)
                else:
                    cv, seed_cos = 0.0, 1.0
                wall = timed(objs[0], Xs[0])
                row = dict(setting=prob["name"], d=prob["d"], m=m, n=n,
                           candidate=name, D=D if D is not None else "",
                           feat_seeds=len(seeds), x_draws=a.x_draws,
                           loss_rel_err=loss_err, grad_cos=gcos,
                           grad_norm_rel_err=gnerr, seed_loss_cv=cv,
                           seed_grad_cos=seed_cos, wall_ms=wall * 1e3,
                           wall_ratio_vs_ref=wall / t_ref,
                           cost_model=objs[0].cost_per_call(n),
                           cost_ratio_vs_cachedYY=objs[0].cost_per_call(n)
                           / A.ReferenceCachedYY.cost_per_call(
                               type("o", (), dict(m=m, d=prob["d"], nk=5))(), n),
                           peak_mem_mb=peak_mem_mb(name, n, m, prob["d"], D or 0),
                           rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
                rows.append(row)
                print(f"  n={n:3d} {name:15s} D={str(D):5s} lerr={loss_err:.2e} "
                      f"gcos={gcos:.5f} gnerr={gnerr:.2e} cv={cv:.2e} "
                      f"t={wall*1e3:.3f}ms ({wall/t_ref:.2f}x ref) cost={row['cost_ratio_vs_cachedYY']:.2f}x")
    write(rows)


def write(rows):
    cols = list(rows[0].keys())
    with open(HERE / "diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    try:
        plots(rows)
    except Exception as e:      # plotting is optional
        print("plots skipped:", e)
    md(rows)


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = sorted({r["setting"] for r in rows}, key=lambda s: (s[0], len(s), s))
    fig, axes = plt.subplots(2, len(settings), figsize=(4.2 * len(settings), 7.5), squeeze=False)
    for j, s in enumerate(settings):
        for name, mk in [("rff", "o"), ("orf", "s"), ("nystrom", "^"), ("subsample", "v")]:
            sub = [r for r in rows if r["setting"] == s and r["candidate"] == name]
            if not sub:
                continue
            for n in sorted({r["n"] for r in sub}):
                pts = sorted([(int(r["D"]), r) for r in sub if r["n"] == n])
                xs = [p[0] for p in pts]
                axes[0, j].plot(xs, [1 - p[1]["grad_cos"] for p in pts], marker=mk,
                                label=f"{name} n={n}")
                axes[1, j].plot(xs, [p[1]["wall_ratio_vs_ref"] for p in pts], marker=mk,
                                label=f"{name} n={n}")
        for i in range(2):
            axes[i, j].set_xscale("log", base=2)
            axes[i, j].set_yscale("log")
            axes[i, j].set_xlabel("D (features / landmarks / targets)")
        axes[0, j].set_title(s)
        axes[0, j].set_ylabel("1 - grad cosine")
        axes[1, j].set_ylabel("wall / reference wall")
        axes[1, j].axhline(1.0, color="k", lw=0.8, ls="--")
        axes[0, j].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(HERE / "diagnostics_grad_cos_and_time.png", dpi=130)


def md(rows):
    lines = ["# Approximate-MMD diagnostics", "",
             "Generated by `diagnostics.py`. Columns: rel. loss error, gradient cosine vs the exact",
             "reference, rel. gradient-norm error, across-feature-seed loss CV and mean pairwise",
             "gradient cosine, wall (fwd+bwd, median) and ratio to the repository `MMDLoss`, and",
             "the hardware-independent cost ratio to the exact loss with the target block cached.",
             "", "![plot](diagnostics_grad_cos_and_time.png)", ""]
    for s in sorted({r["setting"] for r in rows}, key=lambda s: (s[0], len(s), s)):
        sub = [r for r in rows if r["setting"] == s]
        lines += [f"## {s}  (d={sub[0]['d']}, m={sub[0]['m']})", "",
                  "| n | candidate | D | loss rel err | grad cos | grad-norm rel err | seed loss CV | seed grad cos | wall ms | wall/ref | cost/cachedYY |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(f"| {r['n']} | {r['candidate']} | {r['D']} | {r['loss_rel_err']:.2e} | "
                         f"{r['grad_cos']:.5f} | {r['grad_norm_rel_err']:.2e} | {r['seed_loss_cv']:.2e} | "
                         f"{r['seed_grad_cos']:.4f} | {r['wall_ms']:.3f} | {r['wall_ratio_vs_ref']:.2f} | "
                         f"{r['cost_ratio_vs_cachedYY']:.2f} |")
        lines.append("")
    (HERE / "DIAGNOSTICS.md").write_text("\n".join(lines))
    print("written DIAGNOSTICS.md, diagnostics.csv")


if __name__ == "__main__":
    main()
