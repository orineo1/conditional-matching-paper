"""
exp_identifiability.py — Round 2: can guidance IDENTIFY which design point x
produced a target distribution, when two design points share the conditional
MEAN but differ in SHAPE? Pre-registered in HYPOTHESIS.md ("Round 2").

Fully analytic — no trained models:

  Joint GMM over (x, y), diagonal covariances:
    A = N((X_BI, +C), diag(SIGX^2, SIGY_BI^2))   weight 0.25
    B = N((X_BI, -C), diag(SIGX^2, SIGY_BI^2))   weight 0.25
    C = N((X_UNI, 0), diag(SIGX^2, S_UNI^2))     weight 0.50
  With diagonal covariances p(y|x) = sum_k w_k(x) N(mu_yk, sig_yk^2),
  w_k(x) ∝ alpha_k N(x; m_k, SIGX^2). By A/B symmetry E[y|x] == 0 for EVERY x:
  the conditional mean is uninformative about x by construction.
  p(y|X_BI) is bimodal at ±C; p(y|X_UNI) is concentrated N(0, S_UNI^2).
  Run with --verify to print the derivation check (weights, analytic vs
  empirical mean/std at both design points).

  Oracle conditional sampler (documented): default --sampler implicit — exact
  implicit-reparameterisation pathwise gradient (1-D inverse-CDF trick,
  dy/dx = -(dF/dx)/p(y|x); forward = exact hard mixture samples). Needed
  because ALL x-dependence of p(y|x) is in the mixture weights: the repo's
  dist_utils.generate_mog_samples (optimize_LGD's _reference_loss_term oracle)
  blocks weight-gradients entirely (hard multinomial), and the ST-Gumbel
  alternative (--sampler st, kept for reference) was measured SIGN-BIASED near
  the basin boundary here (HYPOTHESIS.md Round-2 sampler note).

  Prior diffusion over x, also analytic: p(x) = sum_k alpha_k N(m_k, SIGX^2),
  cosine schedule replicated from simulations/src/Diffusion.py (s=0.008),
  E[x0|x_t] in closed form (per-component Gaussian posterior + posterior
  weights), driving the same DDIM + LGD guidance step structure as
  Optimization.optimize_LGD. x_T ~ randn (not zeros).

  Step convention: correction Delta_t = step_scale * grad with
  step_scale = 1/sqrt(alpha_t) if --inv_sqrt_alpha (TFG line-9 convention,
  Ori's protocol) else zeta; optionally capped at
  ||Delta_t|| <= step_cap_tau * sqrt(1 - alphabar_t) (--step_cap_tau > 0).
  Default = cap tau=1 + inv_sqrt (the verified-correct combination).

Arms: mean | mmd | mmd_uniform | mmd_witness (backsel via Ori's
simulations/src/witness_utils.apply_backsel, k of n). The mean arm matches the
EMPIRICAL target mean (≈0 for both targets — that is the point).

Per-restart readout: x_hat, |x_hat - X_BI|, |x_hat - X_UNI|, landed-within-0.5
flags, fresh-sample cross-MMD to BOTH targets, gen std, frac within 2*(mode
scale) of the optimized target's nearest mode.

Smoke:  python exp_identifiability.py --smoke
Full:   see submit_identifiability.sh (array over (target, arm/convention) cells)
"""

import os
import sys
import json
import math
import time
import types
import argparse
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "simulations", "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if importlib.util.find_spec("ot") is None:   # POT unused by LossFunctions; see round 1
    sys.modules["ot"] = types.ModuleType("ot")

import torch
import torch.nn.functional as F

from LossFunctions import MMDLoss, RBF
from witness_utils import apply_backsel

# ── Construction constants (pre-registered; HYPOTHESIS.md Round 2) ────────────
X_BI, X_UNI = -2.0, 2.0
SIGX = 0.7
C_SEP = 2.0            # bimodal modes at ±C_SEP
SIGY_BI = 0.25         # bimodal component std
S_UNI = 0.25           # concentrated unimodal std
M_X = torch.tensor([X_BI, X_BI, X_UNI])          # component x-locations
MU_Y = torch.tensor([C_SEP, -C_SEP, 0.0])        # component y-means
SIG_Y = torch.tensor([SIGY_BI, SIGY_BI, S_UNI])  # component y-stds
LOG_ALPHA = torch.log(torch.tensor([0.25, 0.25, 0.50]))

T_STEPS = 100
TAU_GUMBEL = 0.5

# Cosine schedule — replicated from simulations/src/Diffusion.py (s=0.008)
_s = 0.008
_ts = torch.arange(0, T_STEPS, dtype=torch.float32)
_sched = torch.cos((_ts / T_STEPS + _s) / (1 + _s) * torch.pi / 2) ** 2
BARALPHAS = _sched / _sched[0]
BETAS = 1 - BARALPHAS / torch.cat([BARALPHAS[0:1], BARALPHAS[:-1]])
ALPHAS = 1 - BETAS

def set_uni_std(s):
    """Round 3 hook: re-parameterise component C's y-std (the unimodal target's
    concentration). Mutates the module target tensors in place; bimodal blobs
    and all x-structure unchanged, so E[y|x] stays identically 0."""
    global S_UNI
    S_UNI = float(s)
    SIG_Y[2] = float(s)
    TARGET_MODE_SCALE["uni"] = float(s)


TARGET_X = {"bi": X_BI, "uni": X_UNI}
TARGET_MODES = {"bi": [-C_SEP, C_SEP], "uni": [0.0]}
TARGET_MODE_SCALE = {"bi": SIGY_BI, "uni": S_UNI}


# ── Analytic conditional p(y|x) ───────────────────────────────────────────────

def cond_logits(x):
    """Log (unnormalised) conditional weights w_k(x); differentiable in x.
    x: scalar tensor (any shape with one element)."""
    return LOG_ALPHA - (x.reshape(()) - M_X) ** 2 / (2 * SIGX ** 2)


def sample_cond_st(x, n, tau=TAU_GUMBEL):
    """Differentiable oracle sampler (ST-Gumbel over weights; see docstring)."""
    logits = cond_logits(x).unsqueeze(0).expand(n, -1)
    g = F.gumbel_softmax(logits, tau=tau, hard=True)           # [n, K]
    comp = MU_Y.unsqueeze(0) + SIG_Y.unsqueeze(0) * torch.randn(n, 3)
    return (g * comp).sum(dim=-1, keepdim=True)                # [n, 1]


_SQRT2PI = math.sqrt(2.0 * math.pi)


def sample_cond_diff(x, n):
    """Implicit-reparameterisation oracle sampler (Figurnov et al. 2018 style,
    exact in 1-D): forward pass = exact hard mixture samples y0 ~ p(y|x);
    backward pass = the exact pathwise derivative dy/dx = -(dF/dx)/p(y|x)
    (F = conditional CDF), attached via the surrogate
        y = y0 - (F(x, y0) - stop_grad[F(x, y0)]) / p(y0|x).
    Unbiased, unlike the ST-Gumbel sampler below, whose gradient was measured
    to be SIGN-BIASED near the basin boundary in this construction (see
    HYPOTHESIS.md Round-2 sampler note) — all x-dependence here is in the
    mixture weights, the ST estimator's worst case."""
    xs = x.reshape(())
    with torch.no_grad():
        probs0 = torch.softmax(cond_logits(xs), dim=0)
        idx = torch.multinomial(probs0, n, replacement=True)
        y0 = MU_Y[idx] + SIG_Y[idx] * torch.randn(n)                    # [n]
    w = torch.softmax(cond_logits(xs), dim=0)                           # [3], diff. in x
    z = (y0.unsqueeze(-1) - MU_Y) / SIG_Y                               # [n, 3]
    Fv = (w * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).sum(-1)      # [n], diff. in x
    pdf = (probs0 * torch.exp(-0.5 * z ** 2) / (_SQRT2PI * SIG_Y)).sum(-1).clamp_min(1e-8)
    y = y0 - (Fv - Fv.detach()) / pdf
    return y.unsqueeze(-1)                                              # [n, 1]


def sample_cond_hard(x, n, generator=None):
    """Exact (non-differentiable) conditional sampler, for targets and eval."""
    with torch.no_grad():
        probs = torch.softmax(cond_logits(torch.as_tensor(float(x))), dim=0)
        idx = torch.multinomial(probs, n, replacement=True, generator=generator)
        eps = torch.randn(n, generator=generator)
        return (MU_Y[idx] + SIG_Y[idx] * eps).unsqueeze(-1)    # [n, 1]


# ── Analytic prior diffusion over x ───────────────────────────────────────────

def pred_x0_analytic(x_t, t):
    """E[x0 | x_t] for the GMM prior p(x)=sum_k alpha_k N(m_k, SIGX^2) under
    x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps. Closed form, differentiable."""
    abar = BARALPHAS[t]
    var_marg = abar * SIGX ** 2 + (1 - abar)
    logw = LOG_ALPHA - (x_t.reshape(()) - torch.sqrt(abar) * M_X) ** 2 / (2 * var_marg)
    w = torch.softmax(logw, dim=0)
    prec = 1 / SIGX ** 2 + abar / (1 - abar)
    mu_post = (M_X / SIGX ** 2 + torch.sqrt(abar) * x_t.reshape(()) / (1 - abar)) / prec
    return (w * mu_post).sum().reshape(1, 1)


def ddim_step(x_t, t):
    """Deterministic DDIM update (eta=0), twin of Diffusion.sample_ddim_step."""
    abar_t = BARALPHAS[t]
    abar_prev = BARALPHAS[t - 1] if t > 0 else torch.tensor(1.0)
    px0 = pred_x0_analytic(x_t, t)
    noise = (x_t - torch.sqrt(abar_t) * px0) / torch.sqrt(1 - abar_t)
    return torch.sqrt(abar_prev) * px0 + torch.sqrt(1 - abar_prev) * noise, px0


# ── Guided loop (all arms; same structure as optimize_LGD / optimize_capped) ──

def optimize_arm(arm, target_samples, args, run_seed, collect_diag=False):
    """collect_diag=True additionally returns per-run backsel diagnostics
    (means over steps): frac_selected_offmode — of the k selected rows, the
    fraction that are off-mode (|y| > 2*S_UNI) at selection time — and
    frac_offmode, the batch's overall off-mode fraction (the uniform arm's
    expected selection rate). Only meaningful for the backsel arms / uni target."""
    backsel_generator = torch.Generator().manual_seed(run_seed)
    diag_sel, diag_all = [], []
    mmd_loss = MMDLoss(kernel=RBF())
    target_mean = target_samples.mean()

    # Round 7: optional init offset, x_T ~ c + N(0,1) (default c=0 = original).
    # NOTE 2026-09-03: the first attempt at this patch silently failed to match
    # its anchor; the round-7 MMD cells ran WITHOUT the offset. Fixed + verified.
    x_t = torch.randn(1, 1) + getattr(args, "init_offset", 0.0)
    for t in range(T_STEPS - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_t_minus_1, pred_x0 = ddim_step(x_t, t)
        r_t = BETAS[t] / torch.sqrt(1 + BETAS[t] ** 2)

        x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
        y = (sample_cond_diff(x0_sample, args.nsamples) if args.sampler == "implicit"
             else sample_cond_st(x0_sample, args.nsamples, tau=args.tau_gumbel))

        if arm == "mean":
            loss_val = (y.mean() - target_mean).pow(2)
        else:
            y_for_loss = y
            if arm in ("mmd_uniform", "mmd_witness"):
                y_for_loss, _info = apply_backsel(
                    y, target_samples, args.backsel_k,
                    rule=("witness" if arm == "mmd_witness" else "uniform"),
                    witness_floor=args.witness_floor, generator=backsel_generator)
                if collect_diag:
                    with torch.no_grad():
                        offmode = y.detach().abs().view(-1) > 2.0 * S_UNI
                        sel = _info["mask"]
                        if sel.any():
                            diag_sel.append(offmode[sel].float().mean().item())
                        diag_all.append(offmode.float().mean().item())
            elif arm != "mmd":
                raise ValueError(f"unknown arm {arm!r}")
            loss_val = mmd_loss(y_for_loss, target_samples)

        # same -logsumexp aggregation as optimize_LGD (num_x_t = 1)
        log_mean_exp_loss = -torch.logsumexp(torch.stack([-loss_val]), dim=0)
        grad = torch.autograd.grad(log_mean_exp_loss, x_t)[0]

        with torch.no_grad():
            if torch.isnan(grad).any():
                grad = torch.zeros_like(grad)
            step_scale = (1.0 / torch.sqrt(ALPHAS[t])) if args.inv_sqrt_alpha else args.zeta
            delta = step_scale * grad
            if args.step_cap_tau and args.step_cap_tau > 0:
                cap = args.step_cap_tau * torch.sqrt(1.0 - BARALPHAS[t])
                dn = delta.norm()
                if dn > cap:
                    delta = delta * (cap / dn)
            x_t = x_t_minus_1.detach().clone() - delta

    x_hat = x_t.detach().reshape(()).item()
    if collect_diag:
        diag = {
            "frac_selected_offmode": (sum(diag_sel) / len(diag_sel)) if diag_sel else float("nan"),
            "frac_offmode": (sum(diag_all) / len(diag_all)) if diag_all else float("nan"),
        }
        return x_hat, diag
    return x_hat


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(x_hat, target_name, args, eval_seed):
    gen = torch.Generator().manual_seed(eval_seed)
    y_eval = sample_cond_hard(x_hat, args.n_eval, generator=gen)
    tgt_bi = sample_cond_hard(X_BI, args.n_target, generator=gen)
    tgt_uni = sample_cond_hard(X_UNI, args.n_target, generator=gen)
    mmd = MMDLoss(kernel=RBF())

    yv = y_eval.view(-1)
    modes = torch.tensor(TARGET_MODES[target_name])
    d = (yv.view(-1, 1) - modes.view(1, -1)).abs().min(dim=1).values
    return {
        "x_hat": x_hat,
        "dist_bi": abs(x_hat - X_BI), "dist_uni": abs(x_hat - X_UNI),
        "land_bi": int(abs(x_hat - X_BI) <= 0.5),
        "land_uni": int(abs(x_hat - X_UNI) <= 0.5),
        "mmd_to_bi": mmd(y_eval, tgt_bi).item(),
        "mmd_to_uni": mmd(y_eval, tgt_uni).item(),
        "abs_mean_err": abs(yv.mean().item()),
        "gen_std": yv.std().item(),
        "frac_within_2s": (d <= 2.0 * TARGET_MODE_SCALE[target_name]).float().mean().item(),
    }


# ── Construction verification (--verify) ──────────────────────────────────────

def verify_construction():
    print("Construction check (HYPOTHESIS.md Round 2):")
    for name, x in (("x_bi", X_BI), ("x_uni", X_UNI)):
        w = torch.softmax(cond_logits(torch.tensor(x)), dim=0)
        mean_an = (w * MU_Y).sum().item()
        var_an = (w * (SIG_Y ** 2 + MU_Y ** 2)).sum().item() - mean_an ** 2
        ys = sample_cond_hard(x, 100_000, generator=torch.Generator().manual_seed(0)).view(-1)
        print(f"  {name}={x:+.1f}: w={w.tolist()}  E[y|x]={mean_an:+.6f} (emp {ys.mean():+.4f})"
              f"  std={math.sqrt(var_an):.4f} (emp {ys.std():.4f})")
    # E[y|x] == 0 for arbitrary x (A/B symmetry)
    for x in (-3.0, -1.0, 0.0, 1.7):
        w = torch.softmax(cond_logits(torch.tensor(x)), dim=0)
        assert abs((w * MU_Y).sum().item()) < 1e-6, x
    print("  E[y|x] = 0 verified for x in {-3, -1, 0, 1.7} (A/B symmetry) — the "
          "conditional mean is uninformative about x everywhere.")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_ARMS = ["mean", "mmd", "mmd_uniform", "mmd_witness"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", choices=["bi", "uni"], default=["bi", "uni"])
    p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=ALL_ARMS)
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--backsel_k", type=int, default=8)
    p.add_argument("--witness_floor", type=float, default=0.3)
    p.add_argument("--n_target", type=int, default=250)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--step_cap_tau", type=float, default=1.0,
                   help="||Delta_t|| <= tau*sqrt(1-alphabar_t); <=0 disables the cap")
    p.add_argument("--inv_sqrt_alpha", action=argparse.BooleanOptionalAction, default=True,
                   help="step_scale = 1/sqrt(alpha_t) (TFG line-9 / Ori's protocol) "
                        "instead of the constant zeta. Default on; "
                        "--no-inv_sqrt_alpha for the zeta convention.")
    p.add_argument("--sampler", choices=["implicit", "st"], default="implicit",
                   help="Differentiable oracle sampler: 'implicit' (exact "
                        "implicit-reparameterisation pathwise gradient; default) "
                        "or 'st' (ST-Gumbel; kept for reference — measured "
                        "sign-biased near the basin boundary here).")
    p.add_argument("--tau_gumbel", type=float, default=TAU_GUMBEL,
                   help="only used with --sampler st")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--tag", default="")
    p.add_argument("--smoke", action="store_true",
                   help="8 restarts, both targets, arms mean + mmd")
    p.add_argument("--verify", action="store_true",
                   help="print the construction check and exit")
    return p.parse_args()


def main():
    args = parse_args()
    verify_construction()
    if args.verify:
        return
    if args.smoke:
        args.n_restarts = min(args.n_restarts, 8)
        args.arms = [a for a in args.arms if a in ("mean", "mmd")] or ["mean", "mmd"]
        if not args.tag:
            args.tag = "smoke"

    conv = (f"cap{args.step_cap_tau:g}" if args.step_cap_tau and args.step_cap_tau > 0 else "nocap") \
           + ("_inv" if args.inv_sqrt_alpha else "_zeta")
    print(f"Step convention: {conv}")

    rows = []
    for target_name in args.targets:
        for arm in args.arms:
            for i in range(args.n_restarts):
                run_seed = args.seed + i
                torch.manual_seed(run_seed)
                tgt_gen = torch.Generator().manual_seed(run_seed)   # paired target draw
                target_samples = sample_cond_hard(
                    TARGET_X[target_name], args.n_target, generator=tgt_gen)
                t0 = time.time()
                x_hat = optimize_arm(arm, target_samples, args, run_seed)
                elapsed = time.time() - t0
                m = evaluate(x_hat, target_name, args,
                             eval_seed=args.seed + 200_000 + i)
                rows.append({"target": target_name, "arm": arm, "convention": conv,
                             "restart": i, "run_seed": run_seed, "time_s": elapsed, **m})
                print(f"[{target_name} | {arm} | {conv} | {i + 1}/{args.n_restarts}] "
                      f"x_hat={x_hat:+.3f} land(bi,uni)=({m['land_bi']},{m['land_uni']}) "
                      f"MMD(bi)={m['mmd_to_bi']:.4f} MMD(uni)={m['mmd_to_uni']:.4f} "
                      f"std={m['gen_std']:.3f} ({elapsed:.1f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"identifiability_seed{args.seed}_{conv}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "experiment": "identifiability", "convention": conv,
                "constants": {"X_BI": X_BI, "X_UNI": X_UNI, "SIGX": SIGX, "C_SEP": C_SEP,
                               "SIGY_BI": SIGY_BI, "S_UNI": S_UNI, "T_STEPS": T_STEPS},
                "targets": args.targets, "arms": args.arms,
                "n_restarts": args.n_restarts, "nsamples": args.nsamples,
                "backsel_k": args.backsel_k, "witness_floor": args.witness_floor,
                "n_target": args.n_target, "n_eval": args.n_eval,
                "zeta": args.zeta, "step_cap_tau": args.step_cap_tau,
                "inv_sqrt_alpha": args.inv_sqrt_alpha, "sampler": args.sampler,
                "tau_gumbel": args.tau_gumbel,
                "seed": args.seed,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"[Results] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
