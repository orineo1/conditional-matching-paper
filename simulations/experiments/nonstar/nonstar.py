"""
nonstar.py — inverse design when NO design point realises the requested target.

The question: a user asks for outputs distributed as N(y*=4, sigma_t^2), but no
setting x of the design variable produces that distribution. What does guided
generation do? Does a distributional (MMD) objective still land somewhere
sensible, and does a pointwise objective (y - y*)^2 land somewhere else?

Construction (all analytic; joint GMM over (x, y), diagonal covariances):

      y=4 ─   (B)●                          <-- requested value y* = 4
      y=3 ─                  ●(C)
      y=2 ─   (A)●
              x = -2         x = +2
      weights (1/4, 1/4, 1/2), sigma_x = 0.7, sigma_y = 0.2

    p(y | x_bi = -2)  = 50/50 bimodal on {2, 4}  -> EXACTLY right half the time
    p(y | x_uni = +2) = N(3, 0.2^2)              -> never right, always close

  The tension: E[(y-4)^2] = 2.04 at x_bi but only 1.04 at x_uni, so a pointwise
  squared-error objective is provably minimised at x_uni -- the point that never
  produces the requested value. A distributional objective sees that only x_bi
  puts any mass at y* and prefers x_bi at every sigma_t.

Arms (identical capped guided DDIM loop over x; they differ ONLY in the loss):
    point_sq  n=1 sample/step,  loss (y - 4)^2
    mmd       n=32 samples/step, multi-bandwidth-RBF MMD vs a fixed 250-sample
              draw of N(4, sigma_t^2)
    unguided  prior only (calibrates "just follows the prior")

Readout: per restart, REGRET = D(p(y|x_hat), target) - min_x D(p(y|x), target),
for D = population multi-bandwidth-RBF MMD^2 (closed form) and exact 1-D W2
(numeric quantiles). 4 cells (sigma_t in {0.1, 0.25, 0.5, 1.0}) x 3 arms x
2 seeds x 40 restarts.

    python nonstar.py --verify    # analytic checks, no run
    python nonstar.py --run       # full run (~2 min CPU) -> results.json
    python nonstar.py --figure    # results.json -> figure.png

Physics (kernel, sampler, DDIM loop, landscape formulas) is copied unchanged
from experiments/infeasible_target/exp_infeasible.py, geometry="sketch".
"""

import os, json, math, time, argparse
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
FIGURE = os.path.join(HERE, "figure.png")

# ── Construction constants ───────────────────────────────────────────────────
X_BI, X_UNI = -2.0, 2.0                       # the two design points
SIGX = 0.7                                    # x-spread of each component
M_X = torch.tensor([X_BI, X_BI, X_UNI])       # component x-means
MU_Y = torch.tensor([2.0, 4.0, 3.0])          # component y-means
SIG_Y = torch.tensor([0.2, 0.2, 0.2])         # component y-stds
LOG_ALPHA = torch.log(torch.tensor([0.25, 0.25, 0.50]))
Y_STAR = 4.0                                  # requested target mean
ST_LIST = [0.1, 0.25, 0.5, 1.0]               # requested target stds (the cells)

T_STEPS = 100
N_MMD_BATCH, N_TARGET = 32, 250               # samples/step and target draw size
ARMS = ["point_sq", "mmd", "unguided"]

# Cosine noise schedule (s=0.008), as in the project's Diffusion.py
_s = 0.008
_sched = torch.cos((torch.arange(0, T_STEPS, dtype=torch.float32) / T_STEPS + _s)
                   / (1 + _s) * torch.pi / 2) ** 2
BARALPHAS = _sched / _sched[0]
BETAS = 1 - BARALPHAS / torch.cat([BARALPHAS[0:1], BARALPHAS[:-1]])
ALPHAS = 1 - BETAS


# ── MMD with a multi-bandwidth RBF kernel (verbatim from src/LossFunctions.py) ─
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.mults = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def forward(self, X):
        d2 = torch.cdist(X, X, p=2) ** 2
        n = d2.shape[0]
        # adaptive bandwidth: mean pairwise squared distance of the POOLED sample
        b = self.bandwidth if self.bandwidth is not None else d2.sum() / (n ** 2 - n)
        return torch.exp(-d2[None] / (b * self.mults[:, None, None])).sum(0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=None):
        super().__init__()
        self.kernel = kernel or RBF()

    def forward(self, X, Y):
        K, n = self.kernel(torch.vstack([X, Y])), X.shape[0]
        return K[:n, :n].mean() - 2 * K[:n, n:].mean() + K[n:, n:].mean()


MULTS64 = 2.0 ** (torch.arange(5, dtype=torch.float64) - 2)   # same, float64


# ── The conditional p(y|x) ───────────────────────────────────────────────────
_SQRT2PI = math.sqrt(2.0 * math.pi)


def cond_logits(x):
    """Log unnormalised mixture weights w_k(x) of p(y|x); differentiable in x."""
    return LOG_ALPHA - (x.reshape(()) - M_X) ** 2 / (2 * SIGX ** 2)


def sample_cond_diff(x, n):
    """Implicit-reparameterisation oracle sampler (exact in 1-D): the forward
    pass draws exact hard mixture samples; the backward pass carries the
    pathwise derivative dy/dx = -(dF/dx)/p(y|x) through the CDF trick."""
    xs = x.reshape(())
    with torch.no_grad():
        probs0 = torch.softmax(cond_logits(xs), dim=0)
        idx = torch.multinomial(probs0, n, replacement=True)
        y0 = MU_Y[idx] + SIG_Y[idx] * torch.randn(n)
    w = torch.softmax(cond_logits(xs), dim=0)               # differentiable copy
    z = (y0.unsqueeze(-1) - MU_Y) / SIG_Y
    Fv = (w * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).sum(-1)
    pdf = (probs0 * torch.exp(-0.5 * z ** 2) / (_SQRT2PI * SIG_Y)).sum(-1).clamp_min(1e-8)
    return (y0 - (Fv - Fv.detach()) / pdf).unsqueeze(-1)    # [n, 1]


def cond_mean_var(x):
    """Closed-form E[y|x], Var(y|x)."""
    w = torch.softmax(cond_logits(torch.as_tensor(float(x))), dim=0)
    m = (w * MU_Y).sum()
    return m.item(), ((w * (SIG_Y ** 2 + MU_Y ** 2)).sum() - m ** 2).item()


# ── Analytic diffusion prior over x (the marginal GMM is Gaussian-mixture) ────
def pred_x0_analytic(x_t, t):
    """Exact posterior mean E[x_0 | x_t] for the GMM marginal under the cosine
    schedule -- stands in for a trained denoiser, so the prior is exact."""
    abar = BARALPHAS[t]
    var_marg = abar * SIGX ** 2 + (1 - abar)
    w = torch.softmax(LOG_ALPHA - (x_t.reshape(()) - torch.sqrt(abar) * M_X) ** 2
                      / (2 * var_marg), dim=0)
    prec = 1 / SIGX ** 2 + abar / (1 - abar)
    mu_post = (M_X / SIGX ** 2 + torch.sqrt(abar) * x_t.reshape(()) / (1 - abar)) / prec
    return (w * mu_post).sum().reshape(1, 1)


def ddim_step(x_t, t):
    abar_t = BARALPHAS[t]
    abar_prev = BARALPHAS[t - 1] if t > 0 else torch.tensor(1.0)
    px0 = pred_x0_analytic(x_t, t)
    noise = (x_t - torch.sqrt(abar_t) * px0) / torch.sqrt(1 - abar_t)
    return torch.sqrt(abar_prev) * px0 + torch.sqrt(1 - abar_prev) * noise, px0


# ── Ground-truth divergence landscapes D(x) ──────────────────────────────────
def _mix_E_kernel(w1, mu1, s1, w2, mu2, s2, c):
    """E[exp(-d^2/c)] between two Gaussian mixtures, via the Gaussian integral
    E[exp(-d^2/c)] = sqrt(c/(c+2S)) exp(-mu^2/(c+2S)) for d ~ N(mu, S)."""
    MU, S = mu1[:, None] - mu2[None, :], s1[:, None] ** 2 + s2[None, :] ** 2
    return (torch.outer(w1, w2) * torch.sqrt(c / (c + 2 * S))
            * torch.exp(-MU ** 2 / (c + 2 * S))).sum()


def pop_mmd2(x, st, bandwidth=None):
    """Population multi-bandwidth-RBF MMD^2(p(y|x), N(Y_STAR, st^2)), closed form.
    bandwidth=None uses the population analog of the adaptive rule the empirical
    arm sees: b = 2 Var(M) for the pooled mixture M = (32 p + 250 q)/282."""
    w = torch.softmax(cond_logits(torch.as_tensor(float(x))), dim=0).double()
    mu, sg = MU_Y.double(), SIG_Y.double()
    if bandwidth is None:
        mP, vP = cond_mean_var(x)
        lam = N_MMD_BATCH / (N_MMD_BATCH + N_TARGET)
        mM = lam * mP + (1 - lam) * Y_STAR
        b = 2.0 * (lam * (vP + mP ** 2) + (1 - lam) * (st ** 2 + Y_STAR ** 2) - mM ** 2)
    else:
        b = float(bandwidth)
    tw = torch.tensor([1.0], dtype=torch.float64)
    tm = torch.tensor([Y_STAR], dtype=torch.float64)
    ts = torch.tensor([st], dtype=torch.float64)
    tot = 0.0
    for m in MULTS64:                              # sum over the 5 bandwidths
        c = torch.tensor(b, dtype=torch.float64) * m
        tot += (_mix_E_kernel(w, mu, sg, w, mu, sg, c)
                - 2 * _mix_E_kernel(w, mu, sg, tw, tm, ts, c)
                + _mix_E_kernel(tw, tm, ts, tw, tm, ts, c)).item()
    return tot


_NQ = 1000
_QS = (torch.arange(_NQ, dtype=torch.float64) + 0.5) / _NQ
_GAUSS_Q = math.sqrt(2.0) * torch.erfinv(2 * _QS - 1)


def w2_dist(x, st):
    """Exact 1-D W2(p(y|x), N(Y_STAR, st^2)) = ||Q_P - Q_Q||_2 over quantiles.
    Q_P for the mixture by vectorised bisection (60 iters on [-4, 10])."""
    w = torch.softmax(cond_logits(torch.as_tensor(float(x))), dim=0).double()
    mu, sg = MU_Y.double(), SIG_Y.double()
    lo = torch.full((_NQ,), -4.0, dtype=torch.float64)
    hi = torch.full((_NQ,), 10.0, dtype=torch.float64)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        F = (w * 0.5 * (1.0 + torch.erf((mid[:, None] - mu) / sg / math.sqrt(2.0)))).sum(-1)
        lo, hi = torch.where(F < _QS, mid, lo), torch.where(F < _QS, hi, mid)
    qP = 0.5 * (lo + hi)
    return math.sqrt(((qP - (Y_STAR + st * _GAUSS_Q)) ** 2).mean().item())


def compute_landscape(st, n_grid=401):
    xs = [-4.0 + 8.0 * i / (n_grid - 1) for i in range(n_grid)]
    dm, dw = [pop_mmd2(x, st) for x in xs], [w2_dist(x, st) for x in xs]
    im, iw = min(range(n_grid), key=dm.__getitem__), min(range(n_grid), key=dw.__getitem__)
    return {"xs": xs, "d_mmd2": dm, "w2": dw,
            "x_opt_mmd": xs[im], "dmin_mmd2": dm[im],
            "x_opt_w2": xs[iw], "w2min": dw[iw],
            "d_mmd2_bi": pop_mmd2(X_BI, st), "d_mmd2_uni": pop_mmd2(X_UNI, st)}


# ── The guided loop (identical for all arms except the loss) ─────────────────
def optimize_arm(arm, target_samples):
    x_t = torch.randn(1, 1)                                   # x_T ~ N(0, 1)
    if arm == "unguided":
        with torch.no_grad():
            for t in range(T_STEPS - 1, 0, -1):
                x_t, _ = ddim_step(x_t, t)
        return x_t.reshape(()).item()

    mmd_loss = MMDLoss()
    for t in range(T_STEPS - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_t_minus_1, pred_x0 = ddim_step(x_t, t)
        r_t = BETAS[t] / torch.sqrt(1 + BETAS[t] ** 2)        # LGD sampling noise
        x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)

        if arm == "point_sq":
            loss_val = (sample_cond_diff(x0_sample, 1).reshape(()) - Y_STAR) ** 2
        else:                                                  # arm == "mmd"
            loss_val = mmd_loss(sample_cond_diff(x0_sample, N_MMD_BATCH), target_samples)

        # -logsumexp(-loss) aggregation of optimize_LGD, with a single x_t
        grad = torch.autograd.grad(-torch.logsumexp(torch.stack([-loss_val]), 0), x_t)[0]
        with torch.no_grad():
            if torch.isnan(grad).any():
                grad = torch.zeros_like(grad)
            delta = grad / torch.sqrt(ALPHAS[t])               # 1/sqrt(alpha_t) scaling
            cap = torch.sqrt(1.0 - BARALPHAS[t])               # trust region, tau = 1
            if delta.norm() > cap:
                delta = delta * (cap / delta.norm())
            x_t = x_t_minus_1.detach().clone() - delta
    return x_t.detach().reshape(()).item()


# ── Run ──────────────────────────────────────────────────────────────────────
def run(seeds=(42, 1042), n_restarts=40):
    out = {"meta": {"y_star": Y_STAR, "x_bi": X_BI, "x_uni": X_UNI,
                    "sigma_t": ST_LIST, "arms": ARMS, "seeds": list(seeds),
                    "n_restarts": n_restarts, "T_STEPS": T_STEPS,
                    "n_mmd_batch": N_MMD_BATCH, "n_target": N_TARGET},
           "cells": []}
    for ci, st in enumerate(ST_LIST):
        t0 = time.time()
        land = compute_landscape(st)
        print(f"[cell {ci}] sigma_t={st}: x_opt(MMD)={land['x_opt_mmd']:+.2f} "
              f"Dmin={land['dmin_mmd2']:.4f} (bi {land['d_mmd2_bi']:.4f} / "
              f"uni {land['d_mmd2_uni']:.4f})  x_opt(W2)={land['x_opt_w2']:+.2f} "
              f"[{time.time() - t0:.1f}s]", flush=True)
        rows = []
        for seed in seeds:
            # one fixed target draw per (seed, cell), shared by all restarts
            g = torch.Generator().manual_seed(seed * 7919 + ci)
            target = (Y_STAR + st * torch.randn(N_TARGET, generator=g)).unsqueeze(-1)
            for arm in ARMS:
                t1 = time.time()
                for i in range(n_restarts):
                    torch.manual_seed(seed + i)      # paired inits across arms
                    x_hat = optimize_arm(arm, target)
                    rows.append({"arm": arm, "seed": seed, "restart": i,
                                 "x_hat": x_hat, "basin_bi": int(x_hat < 0),
                                 "regret_mmd2": pop_mmd2(x_hat, st) - land["dmin_mmd2"],
                                 "regret_w2": w2_dist(x_hat, st) - land["w2min"]})
                print(f"    {arm:9s} seed {seed}: {n_restarts} restarts "
                      f"[{time.time() - t1:.1f}s]", flush=True)
        out["cells"].append({"cell": ci, "sigma_t": st, "landscape": land,
                             "rows": rows, "summary": summarize(rows)})
        for arm, s in out["cells"][-1]["summary"].items():
            print(f"  -> {arm:9s} bi {100 * s['frac_bi']:3.0f}%  "
                  f"regret(MMD^2) {s['regret_mmd2']:.3f}  regret(W2) {s['regret_w2']:.3f}")
    with open(RESULTS, "w") as f:
        json.dump(out, f)
    print(f"[saved] {RESULTS}")
    return out


def summarize(rows):
    s = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        s[arm] = {"n": len(r),
                  "frac_bi": sum(x["basin_bi"] for x in r) / len(r),
                  "regret_mmd2": sum(x["regret_mmd2"] for x in r) / len(r),
                  "regret_w2": sum(x["regret_w2"] for x in r) / len(r),
                  "mean_x": sum(x["x_hat"] for x in r) / len(r)}
    return s


# ── Verification (--verify) ──────────────────────────────────────────────────
def verify():
    print("Conditionals (analytic vs 100k empirical):")
    for name, x in (("x_bi ", X_BI), ("x_uni", X_UNI)):
        w = torch.softmax(cond_logits(torch.tensor(x)), dim=0)
        m, v = cond_mean_var(x)
        torch.manual_seed(0)
        idx = torch.multinomial(w, 100_000, replacement=True)
        ys = MU_Y[idx] + SIG_Y[idx] * torch.randn(100_000)
        print(f"  {name} x={x:+.1f}  w={[round(t, 5) for t in w.tolist()]}  "
              f"E[y|x]={m:+.4f} (emp {ys.mean():+.4f})  "
              f"std={math.sqrt(v):.4f} (emp {ys.std():.4f})")

    # The pointwise objective J(x) = Var(y|x) + (E[y|x] - y*)^2. Here E[y|x] = 3
    # for EVERY x (modes 2 and 4 average to the unimodal mean), so J is driven
    # purely by the conditional variance -> minimised at x_uni.
    def J(x):
        m, v = cond_mean_var(x)
        return v + (m - Y_STAR) ** 2
    print(f"\nPointwise objective E[(y-4)^2]:  x_bi {J(X_BI):.4f} (expect 2.04)   "
          f"x_uni {J(X_UNI):.4f} (expect 1.04)  -> pointwise prefers x_uni")
    assert abs(J(X_BI) - 2.04) < 1e-5 and abs(J(X_UNI) - 1.04) < 1e-5

    # closed-form population MMD^2 against an empirical fixed-bandwidth MMD
    g = torch.Generator().manual_seed(1)
    st, b = 0.5, 1.7
    w = torch.softmax(cond_logits(torch.tensor(0.5)), dim=0)
    idx = torch.multinomial(w, 4000, replacement=True, generator=g)
    P = (MU_Y[idx] + SIG_Y[idx] * torch.randn(4000, generator=g)).unsqueeze(-1)
    Q = (Y_STAR + st * torch.randn(4000, generator=g)).unsqueeze(-1)
    emp = MMDLoss(RBF(bandwidth=b))(P, Q).item()
    ana = pop_mmd2(0.5, st, bandwidth=b)
    print(f"\nMMD^2 closed form vs empirical (x=0.5, b={b}): {ana:.5f} vs {emp:.5f} "
          f"({100 * abs(ana - emp) / ana:.1f}% -- 4k-sample Monte-Carlo error)")
    assert abs(ana - emp) / ana < 0.03

    # W2 machinery against the Gaussian-pair closed form (a-b)^2 + (s-t)^2
    num = w2_dist(X_UNI, 0.5)
    ref = math.sqrt((3.0 - Y_STAR) ** 2 + (0.2 - 0.5) ** 2)
    print(f"W2 numeric vs Gaussian closed form at x_uni: {num:.5f} vs {ref:.5f}")
    assert abs(num - ref) < 5e-3

    print("\nLandscape argmins (population MMD^2 / exact W2):")
    for st in ST_LIST:
        L = compute_landscape(st, n_grid=161)
        print(f"  sigma_t={st:<5} MMD^2: x_opt={L['x_opt_mmd']:+.2f} "
              f"(bi {L['d_mmd2_bi']:.4f} < uni {L['d_mmd2_uni']:.4f})   "
              f"W2: x_opt={L['x_opt_w2']:+.2f}")
        assert L["d_mmd2_bi"] < L["d_mmd2_uni"], "MMD must prefer x_bi"
    print("\nall checks passed.")


# ── Figure (--figure) ────────────────────────────────────────────────────────
C = {"mmd": "#0072B2", "point_sq": "#D55E00", "unguided": "#808080"}
LBL = {"mmd": "MMD", "point_sq": "pointwise", "unguided": "unguided"}


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(RESULTS))
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 18})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # LEFT: the joint p(x, y), the requested value, the two design points
    torch.manual_seed(0)
    idx = torch.multinomial(torch.exp(LOG_ALPHA), 6000, replacement=True)
    xs = M_X[idx] + SIGX * torch.randn(6000)
    ys = MU_Y[idx] + SIG_Y[idx] * torch.randn(6000)
    axL.scatter(xs, ys, s=4, c="#b0b0b0", alpha=0.45, linewidths=0, rasterized=True)
    axL.axhline(Y_STAR, color="#009E73", lw=3, ls="--", zorder=3)
    axL.text(3.85, Y_STAR + 0.12, "requested  y* = 4", color="#009E73",
             fontsize=16, fontweight="bold", ha="right", va="bottom")
    for x, txt, col in ((X_BI, "x_bi = -2\nhits y* half the time", "#0072B2"),
                        (X_UNI, "x_uni = +2\nnever hits, always close", "#D55E00")):
        axL.axvline(x, color=col, lw=2, ls=":", zorder=3)
        axL.text(x, 0.75, txt, color=col, fontsize=15, fontweight="bold",
                 ha="center", va="bottom",
                 bbox=dict(fc="white", ec=col, lw=1.6, boxstyle="round,pad=0.35"))
    axL.set_xlim(-4.2, 4.2); axL.set_ylim(0.5, 5.4)
    axL.set_xlabel("design variable  x"); axL.set_ylabel("outcome  y")
    axL.set_title("The joint p(x, y): no x gives N(4, $\\sigma_t^2$)", pad=12)

    # RIGHT: where each arm lands, per requested target width
    rows_y, labels = [], []
    for ci, cell in enumerate(d["cells"][::-1]):          # sigma_t large -> small
        for ai, arm in enumerate(["unguided", "point_sq", "mmd"]):
            y0 = ci * 3.6 + ai
            rows_y.append((y0, arm, cell))
            labels.append((y0, LBL[arm]))
    torch.manual_seed(1)
    for y0, arm, cell in rows_y:
        xh = [r["x_hat"] for r in cell["rows"] if r["arm"] == arm]
        jit = (torch.rand(len(xh)) - 0.5) * 0.62
        axR.scatter(xh, y0 + jit.numpy(), s=26, c=C[arm], alpha=0.55, linewidths=0)
        s = cell["summary"][arm]
        axR.text(6.4, y0, f"{100 * s['frac_bi']:3.0f}%", color=C[arm], fontsize=15,
                 fontweight="bold", va="center", ha="right", family="monospace")
        axR.text(8.9, y0, f"{s['regret_mmd2']:.3f}", color=C[arm], fontsize=15,
                 va="center", ha="right", family="monospace")
    for y0, lab in labels:
        axR.text(-5.1, y0, lab, fontsize=14, va="center", ha="right", color="#333333")
    for ci, cell in enumerate(d["cells"][::-1]):
        axR.text(-11.4, ci * 3.6 + 1, f"$\\sigma_t$ = {cell['sigma_t']:g}", fontsize=16,
                 fontweight="bold", va="center", ha="left")
        if ci:
            axR.axhline(ci * 3.6 - 1.3, color="#dddddd", lw=1)
    axR.axvline(X_BI, color="#0072B2", lw=2, ls=":")
    axR.axvline(X_UNI, color="#D55E00", lw=2, ls=":")
    top = len(d["cells"]) * 3.6 - 1.05
    axR.text(X_BI, top, "x_bi", color="#0072B2", fontsize=15, fontweight="bold", ha="center")
    axR.text(X_UNI, top, "x_uni", color="#D55E00", fontsize=15, fontweight="bold", ha="center")
    axR.text(6.4, top, "% at x_bi", fontsize=14, ha="right", fontweight="bold")
    axR.text(8.9, top, "regret", fontsize=14, ha="right", fontweight="bold")
    axR.set_xlim(-11.6, 9.1); axR.set_ylim(-1.5, top + 1.3)
    axR.set_yticks([])
    axR.set_xticks([-4, -2, 0, 2, 4])
    axR.spines[["left", "right", "top"]].set_visible(False)
    axR.spines["bottom"].set_bounds(-4.8, 4.8)
    axR.set_xlabel("where the guided run landed,  $\\hat{x}$")
    axR.set_title("Landings over 80 restarts per arm", pad=12)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGURE, dpi=150)
    print(f"[saved] {FIGURE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true", help="analytic checks, then exit")
    p.add_argument("--run", action="store_true", help="full run -> results.json (~2 min)")
    p.add_argument("--figure", action="store_true", help="results.json -> figure.png")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1042])
    p.add_argument("--n_restarts", type=int, default=40)
    a = p.parse_args()
    if not (a.verify or a.run or a.figure):
        p.error("pass one of --verify / --run / --figure")
    if a.verify:
        verify()
    if a.run:
        run(seeds=a.seeds, n_restarts=a.n_restarts)
    if a.figure:
        figure()
