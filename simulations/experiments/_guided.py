"""The guided optimisation loop shared by experiments 2-4.

Follows the repository's ``Optimization.optimize_LGD`` structure exactly:
x_T = 0, DDIM eta = 0, loop t = T-1 .. 1, guidance applied as
``x_{t-1} = DDIM(x_t) - ghat``. What varies is only:

  spatial   M_t = 1 (no LGD)  |  M_t = 3 (LGD)
  temporal  none              |  Adam (AdamDPS moments)
  n_t       constant          |  scheduled

Aggregation for M_t > 1 is the TFG/LGD form ``-log((1/M) sum_j exp(-MMD^2_j))``;
outputs from different perturbations are NEVER pooled into one MMD.

Cost per diffusion step: ``C_t = M_t * n_t``. Each individual MMD evaluation
uses at most ``n_t`` conditional samples.
"""
import math
import sys
import time
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PENALTY, SUCCESS_TOL, key_seed          # noqa: E402
from _models import PAPER_TS                                 # noqa: E402
from LossFunctions import MMDLoss, RBF                       # noqa: E402
from tfg import oracle                                       # noqa: E402
from tfg.adam_guidance import AdamGuidance                   # noqa: E402
from tfg.n_schedule import n_at                              # noqa: E402
from tfg.schedule import DiffusionSchedule                   # noqa: E402

M_LGD = 3


def allocate(n, spatial):
    """Perturbation sample counts. A/B cost n; C costs 3n."""
    if spatial == "no_lgd":
        return [int(n)]
    if spatial == "lgd":
        return [int(n)] * M_LGD
    raise ValueError(f"unknown spatial mode {spatial!r}")


def n_for_step(t, sched, n_max, schedule):
    if schedule == "constant":
        return int(n_max)
    return int(n_at(t, sched, n_max, 1.0, schedule))     # "time" | "noise"


def run(model_cond, model_uncond, S_G, bandwidth, n_max, spatial, temporal,
        restart, schedule="constant", adam_rho=0.4, device="cpu",
        guidance_target="x_t", mu_strength=1.0):
    """``guidance_target`` selects where the guidance gradient is taken.

    ``x_t``  DPS/LGD style: differentiate through the denoiser back to x_t and
             add the correction to the DDIM iterate. This is what Experiments
             2-4 use and what ``Optimization.optimize_LGD`` does.
    ``x0``   MPGD style: differentiate w.r.t. x_{0|t} treated as a LEAF, move
             x_{0|t} itself, then rebuild x_{t-1} from the moved clean estimate.
             Avoids backpropagating through the denoiser entirely. In TFG terms
             this is the mu branch with N_iter = 1 and rho = 0.
    """
    T = model_uncond.diffusion_steps
    sched = DiffusionSchedule(T=T)
    mmd = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"), device="cpu")
    adam = (AdamGuidance(beta1=0.9, beta2=0.995, delta=1e-8, rho=adam_rho,
                         inv_sqrt_alpha=False) if temporal == "adam" else None)

    x = torch.zeros(1, model_uncond.nfeatures, device=device)
    calls, n_hist, diverged = 0, [], False
    t0 = time.perf_counter()
    for t in range(T - 1, 0, -1):
        x = x.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = model_uncond.sample_ddim_step(
            x, t, condition_x=None, device=device, eta=0.0)
        cur_var = model_uncond.betas[t].to(device)
        r_t = cur_var / torch.sqrt(1 + cur_var ** 2)

        n = n_for_step(t, sched, n_max, schedule)
        n_hist.append(n)
        # For MPGD the guidance is a function of x_{0|t} as a leaf.
        x0_leaf = pred_x0.detach().clone().requires_grad_(True)
        base_x0 = x0_leaf if guidance_target == "x0" else pred_x0
        alloc = allocate(n, spatial)
        terms = []
        for j, n_j in enumerate(alloc):
            torch.manual_seed(key_seed("cond", restart, t, j))
            x0 = base_x0 + (r_t * torch.randn_like(base_x0) if len(alloc) > 1 else 0.0)
            cond = x0.reshape(1, -1).repeat(n_j, 1)
            y, _, _ = model_cond.sample(nsamples=n_j, condition_x=cond,
                                        device=device, ts=PAPER_TS)
            calls += n_j
            terms.append(-mmd(y, S_G))
        loss = (-terms[0] if len(terms) == 1
                else -torch.logsumexp(torch.stack(terms), 0) + math.log(len(terms)))

        if guidance_target == "x0":
            # MPGD: gradient w.r.t. the clean estimate, applied to it directly.
            g, = torch.autograd.grad(loss, x0_leaf, allow_unused=True)
            g = torch.zeros_like(x0_leaf) if g is None else g
            upd = g.detach() if adam is None else adam.step(g)
            with torch.no_grad():
                ab_t = model_uncond.baralphas[t].to(device)
                ab_prev = model_uncond.baralphas[t - 1].to(device)
                x0_moved = x0_leaf.detach() - mu_strength * upd
                eps_eff = (x.detach() - ab_t.sqrt() * x0_moved) / (1 - ab_t).sqrt()
                x = ab_prev.sqrt() * x0_moved + (1 - ab_prev).sqrt() * eps_eff
        else:
            g, = torch.autograd.grad(loss, x, allow_unused=True)
            g = torch.zeros_like(x) if g is None else g
            upd = g.detach() if adam is None else adam.step(g)
            with torch.no_grad():
                x = x_prev.detach().clone() - upd
        if not torch.isfinite(x).all() or float(x.abs().max()) > 50.0:
            diverged = True
            break

    return x.detach().reshape(-1).clone(), {
        "conditional_calls": calls, "seconds": time.perf_counter() - t0,
        "diverged": diverged, "n_t_mean": sum(n_hist) / len(n_hist),
        "n_t_min": min(n_hist), "n_t_max": max(n_hist), "steps": len(n_hist)}


def evaluate(x_hat, params, info):
    """``x_hat`` is a vector of length dim(X); abs_err is its distance to x*."""
    x_hat = torch.as_tensor(x_hat, dtype=torch.float64).reshape(-1)
    x_star = params["x_star"].reshape(-1).to(torch.float64)
    bad = (info["diverged"] or not torch.isfinite(x_hat).all()
           or float(x_hat.abs().max()) > 50.0)
    if bad:
        return {"x_hat": x_hat.tolist(), "L2": float("inf"),
                "abs_err": float("inf"), "diverged": True,
                **{k: info[k] for k in ("conditional_calls", "seconds")}}
    l2sq = float(oracle.population_l2_squared(x_hat, params))
    return {"x_hat": x_hat.tolist(), "L2_squared": l2sq, "L2": l2sq ** 0.5,
            "abs_err": float((x_hat - x_star).norm()), "diverged": False,
            **{k: info[k] for k in ("conditional_calls", "seconds")}}


def summarise(runs, restarts):
    import statistics as st
    fin = [r for r in runs if not r["diverged"]]
    L = [r["L2"] for r in fin] or [float("nan")]
    return {"L2_mean": st.mean(L), "L2_median": st.median(L),
            "L2_std": st.stdev(L) if len(L) > 1 else float("nan"),
            "success_rate": sum(1 for r in fin if r["abs_err"] < SUCCESS_TOL) / restarts,
            "diverged": len(runs) - len(fin),
            "conditional_calls_mean": (st.mean([r["conditional_calls"] for r in fin])
                                       if fin else float("nan")),
            "seconds_total": sum(r["seconds"] for r in runs)}
