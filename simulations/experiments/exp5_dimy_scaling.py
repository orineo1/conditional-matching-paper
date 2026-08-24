"""Experiment 5 -- momentum benefit vs the dimension of the matched distribution.

QUESTION AS POSED
    Does the useful regime for momentum widen as dim(Y) grows? MMD is estimated
    from n samples OF Y, so as dim(Y) rises those n samples cover the space more
    sparsely and the gradient gets noisier. Hypothesis:

        n*(d) increases with d,   n*(d) = max{ n : LCB95[ Delta(d,n) ] > 0 }
        Delta(d,n) = S_none(d,n) - S_Adam(d,n)

WHAT THIS SCRIPT ACTUALLY MEASURES -- READ BEFORE USING n*(d)
    It does NOT answer the question above. At d >= 8 the no-momentum baseline
    never leaves the neighbourhood of the diffusion prior's endpoint: x_hat is
    ~0.39 on every restart at n = 4 AND at n = 32, so the score is identical
    across an 8x change in sample count. Adam reaches x* = -5 on most restarts.
    Delta therefore measures "escapes a flat region of the landscape" versus
    "does not", not "handles sampling noise better".

    The sticking is NOT finite-sample noise. With n = 2048 (512x the data of
    n = 4) the baseline still escapes 0/8 times and x_hat concentrates MORE
    tightly around 0.36. The plateau is in the population landscape: the MMD
    surface at d = 8 reads 3.295 at x = 0, 3.252 at x = -2, 0.085 at x = -4,
    so there is a long shelf between the prior's basin and the optimum.

    Consequently n*(d) from this script is meaningless and is reported only for
    completeness. The dim(Y) sample-complexity question remains OPEN; answering
    it needs a calibration that guarantees the baseline is a working optimiser
    at every d (e.g. the smallest zeta_d at which no-momentum reaches x* at
    large n), so the comparison is about noise rather than escape.

WHAT IT DOES ESTABLISH
    Adam crosses a plateau that plain gradient guidance cannot, and crosses it
    in the CORRECT direction (landing at -5.12, -5.15, -5.01, -4.98, -4.94),
    on roughly 60-75% of restarts. That is an optimisation result, not a noise
    result. Experiment 5A separates the two mechanisms that could produce it.

WHY A NEW BENCHMARK
    The paper's 2D/5D/10D settings all have dim(Y) = 1 -- they scale dim(X),
    the design variable, not the distribution being matched. Verified from
    params/{2D,5D,10D}_cond_1D_gmm_params.pt, whose mog_means is (2,1,1).
    Experiment 3 covers that axis; this one covers dim(Y).

BENCHMARK
    tfg.dimy_benchmark: X scalar with the same known optimum x* = -5, Y in R^d.
    Per-coordinate conditional variance is exactly 0.12395 for EVERY d (the
    Schur complement is d-independent by construction), coordinates are
    permuted rather than duplicated, and d = 1 reproduces the 2-D benchmark.

CONDITIONAL GENERATOR
    The ANALYTIC conditional P(Y|X=x), sampled by reparameterisation -- not a
    trained model. Deliberate: it removes model error, so any dimension effect
    is attributable to finite-sample MMD noise, which is the hypothesis. This
    differs from Experiments 2-4, which use the trained consistency model.

    python experiments/exp5_dimy_scaling.py --restarts 50
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import paired_stats, penalised_score, save          # noqa: E402
from _models import SEEDS, unconditional_model                    # noqa: E402
from LossFunctions import MMDLoss, RBF                            # noqa: E402
from tfg import oracle                                            # noqa: E402
from tfg.adam_guidance import AdamGuidance                        # noqa: E402
from tfg.dimy_benchmark import as_params                          # noqa: E402


def conditional_samples(params, x, n, gen):
    """n reparameterised draws from the exact P(Y|X=x); differentiable in x."""
    cm, cc, w = oracle.conditional_params(
        x.double(), params["mu_list"], params["Sigma_list"], params["alpha"])
    L = torch.linalg.cholesky(cc if cc.dim() == 2 else cc[0])
    idx = torch.multinomial(w.detach(), n, replacement=True, generator=gen)
    eps = torch.randn(n, cm.shape[1], dtype=cm.dtype, generator=gen)
    return cm[idx] + eps @ L.T


def target_samples(params, size, gen):
    tm, tw = params["target_means"], params["target_weights"]
    tc = params["target_variances"][0]
    L = torch.linalg.cholesky(tc)
    idx = torch.multinomial(tw, size, replacement=True, generator=gen)
    return tm[idx] + torch.randn(size, tm.shape[1], dtype=tm.dtype, generator=gen) @ L.T


def probe_gradient_scale(params, model_uncond, S_G, bw, n=64, probes=12, seed=0):
    """Median |dL/dx| over probe steps -- the scale the guidance must overcome.

    The MMD kernel bandwidth grows with dim(Y) (squared distances scale with d),
    so the raw gradient magnitude falls by orders of magnitude as d rises. Left
    uncorrected the DDIM prior dominates and the guidance does nothing, which
    shows up as a score that does not respond to n at all.
    """
    T = model_uncond.diffusion_steps
    mmd = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")
    mags = []
    x = torch.zeros(1, 1, dtype=torch.float32)
    for i, t in enumerate(range(T - 1, 0, -max(1, (T - 1) // probes))):
        x = x.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = model_uncond.sample_ddim_step(
            x, t, condition_x=None, device="cpu", eta=0.0)
        gen = torch.Generator().manual_seed(seed * 1000 + i)
        y = conditional_samples(params, pred_x0.reshape(-1), n, gen)
        g, = torch.autograd.grad(mmd(y.float(), S_G.float()), x, allow_unused=True)
        mags.append(abs(float(g.reshape(-1)[0])) if g is not None else 0.0)
        x = x_prev.detach()
    mags = sorted(m for m in mags if m > 0)
    return mags[len(mags) // 2] if mags else 1.0


def run(params, model_uncond, S_G, bw, n, temporal, restart, adam_rho=0.4,
        zeta=1.0, beta1=0.9, beta2=0.995, delta=1e-8,
        step_clip="none", step_tau=1.0, x_init=0.0):
    """``step_clip`` is the noise-level trust region of Experiment 8, applied
    here to the un-normalised update so that ``zeta`` can be raised far enough
    to leave a local minimum without the step diverging:

        step_clip="noise":  ||upd|| <= step_tau * sqrt(1 - alphabar_t)

    Semantics match ``tfg.engine.GeneralizedTFG._step_clip`` -- magnitude only,
    direction untouched. ``step_clip="none"`` reproduces the original loop
    exactly, so every earlier Experiment 5/5A/5B result is unaffected.

    ``x_init`` sets the start of the reverse trajectory. ``0.0`` is the original
    behaviour and is kept as the default so earlier results reproduce, but it is
    WRONG as a protocol: reverse diffusion starts from x_T ~ N(0, I), and fixing
    x_T = 0 makes every restart share one basin, so the "restart" axis carries no
    exploration at all. Pass ``x_init="randn"`` to draw x_T per restart, which is
    what makes a success rate mean basin-of-attraction rate.
    """
    T = model_uncond.diffusion_steps
    mmd = MMDLoss(kernel=RBF(bandwidth=bw, device="cpu"), device="cpu")
    adam = (AdamGuidance(beta1=beta1, beta2=beta2, delta=delta, rho=adam_rho,
                         inv_sqrt_alpha=False) if temporal == "adam" else None)
    if isinstance(x_init, str):
        if x_init != "randn":
            raise ValueError(f"unknown x_init {x_init!r}")
        g0 = torch.Generator().manual_seed(0x5EED0000 ^ int(restart))
        x = torch.randn(1, 1, generator=g0, dtype=torch.float32)
    else:
        x = torch.full((1, 1), float(x_init), dtype=torch.float32)
    calls, diverged = 0, False
    t0 = time.perf_counter()
    for t in range(T - 1, 0, -1):
        x = x.detach().clone().requires_grad_(True)
        x_prev, pred_x0 = model_uncond.sample_ddim_step(
            x, t, condition_x=None, device="cpu", eta=0.0)
        gen = torch.Generator().manual_seed(abs(hash((restart, t, n))) % (2**31))
        y = conditional_samples(params, pred_x0.reshape(-1), n, gen)
        calls += n
        loss = mmd(y.float(), S_G.float())
        g, = torch.autograd.grad(loss, x, allow_unused=True)
        g = torch.zeros_like(x) if g is None else g
        upd = zeta * (g.detach() if adam is None else adam.step(g))
        if step_clip != "none":
            with torch.no_grad():
                ab_t = model_uncond.baralphas[t].to(upd.device)
                if step_clip == "noise":
                    ref = step_tau * (1 - ab_t).sqrt()
                elif step_clip == "ddim":
                    ref = step_tau * (x_prev.detach() - x.detach()).norm()
                else:
                    raise ValueError(f"unknown step_clip {step_clip!r}")
                upd = upd * torch.clamp(ref / (upd.norm() + 1e-12), max=1.0)
        with torch.no_grad():
            x = x_prev.detach() - upd
        if not torch.isfinite(x).all() or float(x.abs().max()) > 50.0:
            diverged = True
            break
    return float(x.detach().reshape(-1)[0]), {
        "conditional_calls": calls, "seconds": time.perf_counter() - t0,
        "diverged": diverged}


def evaluate(x_hat, params, info):
    if info["diverged"] or not math.isfinite(x_hat) or abs(x_hat) > 50.0:
        return {"L2": float("inf"), "abs_err": float("inf"), "diverged": True,
                **{k: info[k] for k in ("conditional_calls", "seconds")}}
    l2sq = float(oracle.population_l2_squared(
        torch.tensor([x_hat], dtype=torch.float64), params))
    return {"x_hat": x_hat, "L2_squared": l2sq, "L2": l2sq ** 0.5,
            "abs_err": abs(x_hat - float(params["x_star"])), "diverged": False,
            **{k: info[k] for k in ("conditional_calls", "seconds")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-grid", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-grid", type=int, nargs="*", default=[4, 8, 16, 32])
    ap.add_argument("--restarts", type=int, default=50)
    ap.add_argument("--adam-rho", type=float, default=0.4)
    ap.add_argument("--target-size", type=int, default=250)
    ap.add_argument("--calib-c", type=float, default=0.05,
                    help="global guidance-magnitude constant; zeta_d = C/median|g|_d")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows, n_star = [], {}
    for d in a.d_grid:
        params = as_params(d)
        gen = torch.Generator().manual_seed(987654)
        S_G = target_samples(params, a.target_size, gen)
        d2 = torch.cdist(S_G, S_G, p=2) ** 2
        m = S_G.shape[0]
        bw = float(d2.sum() / (m ** 2 - m))          # frozen per dimension
        mu = unconditional_model(params, seed=SEEDS[0], tag=f"_dimy{d}")
        # Per-dimension guidance calibration, done ONCE on the no-momentum
        # baseline and frozen across every n and both methods at that d:
        #     zeta_d = C / median|dL/dx|_d
        # so the applied update has the same magnitude at every dimension. C is
        # a single global constant fixed at d = 1 (--calib-c); nothing is tuned
        # per method, and the calibration never sees a final L2.
        scale = probe_gradient_scale(params, mu, S_G, bw)
        zeta = a.calib_c / max(scale, 1e-30)
        print(f"d={d}  bandwidth={bw:.4f}  median|g|={scale:.3e}  zeta={zeta:.3e}")
        n_star[d] = None
        for n in a.n_grid:
            cells = {}
            for temporal in ("none", "adam"):
                runs = []
                for r in range(a.restarts):
                    x, info = run(params, mu, S_G, bw, n, temporal, r,
                                  a.adam_rho, zeta=zeta)
                    runs.append(evaluate(x, params, info))
                cells[temporal] = [min(q["L2"], 2.0) if not q["diverged"] else 2.0
                                   for q in runs]
                cells[temporal + "_score"] = penalised_score(runs)
            delta = paired_stats(cells["none"], cells["adam"])
            supported = delta["ci95"][0] > 0
            if supported:
                n_star[d] = n
            rows.append({"d": d, "n": n, "bandwidth": bw, "zeta": zeta,
                         "grad_scale": scale,
                         "score_none": cells["none_score"],
                         "score_adam": cells["adam_score"],
                         "delta": delta, "supported": supported})
            print(f"   n={n:<4} none={cells['none_score']:.4f} "
                  f"adam={cells['adam_score']:.4f} "
                  f"Delta={delta['mean_diff']:+.4f} "
                  f"CI[{delta['ci95'][0]:+.4f},{delta['ci95'][1]:+.4f}] "
                  f"p={delta['perm_p']:.4f} {'SUPPORTED' if supported else ''}",
                  flush=True)
        print(f"   n*(d={d}) = {n_star[d]}")

    print("\nn*(d):", n_star)
    save(f"exp5_dimy_scaling{a.tag}",
         {"config": vars(a), "rows": rows, "n_star": n_star})


if __name__ == "__main__":
    main()
