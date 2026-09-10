"""
exp_pointwise.py — Round 6: pointwise scalar-target inverse design, the
practitioner baseline. Pre-registered in HYPOTHESIS.md ("Round 6"; competing
predictions P-a vs P-b — either outcome supports the paper claim; we record
which one holds).

Per diffusion step: ONE conditional sample y_1 ~ p(y|x0_hat) (implicit-
reparameterisation sampler, differentiable) and loss (y_1 − a)^2 for a SCALAR
target a (point_sq_a0: a=0; point_sq_a2: a=+2), or |y_1 − a| (point_abs_a2).
No target sample set, no MMD, no back-selection. Corrected convention: step cap
tau=1 + 1/sqrt(alpha_t), x_T ~ randn; no extra grad clip (the cap bounds the
step). Construction and evaluation reused from exp_identifiability (s = 0.25
unimodal component; E[y|x] ≡ 0 everywhere — verified at startup).

The mmd reference arms are NOT rerun: build_pointwise_report.py reuses the
round-2 identifiability cell JSONs (cap1_inv, mmd arm, bi and uni targets).

Smoke:  python exp_pointwise.py --smoke      (a=0 and a=2 squared, both seeds, 8 restarts)
Full:   see submit_pointwise.sh (3-cell array over arms; both seeds per cell)
"""

import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import exp_identifiability as E

ARMS = {
    "point_sq_a0":  ("sq", 0.0),
    "point_sq_a2":  ("sq", 2.0),
    "point_abs_a2": ("abs", 2.0),
}


def optimize_point(arm, args, run_seed):
    """Mirror of exp_identifiability.optimize_arm with the pointwise loss and
    n=1 (the -logsumexp aggregation is the identity for a single term and is
    dropped)."""
    kind, a = ARMS[arm]
    # Round 7: optional init offset, x_T ~ c + N(0,1) (default c=0 = original)
    x_t = torch.randn(1, 1) + getattr(args, "init_offset", 0.0)
    for t in range(E.T_STEPS - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_t_minus_1, pred_x0 = E.ddim_step(x_t, t)
        r_t = E.BETAS[t] / torch.sqrt(1 + E.BETAS[t] ** 2)

        x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
        y1 = (E.sample_cond_diff(x0_sample, 1) if args.sampler == "implicit"
              else E.sample_cond_st(x0_sample, 1, tau=args.tau_gumbel)).reshape(())
        loss = (y1 - a) ** 2 if kind == "sq" else (y1 - a).abs()
        grad = torch.autograd.grad(loss, x_t)[0]

        with torch.no_grad():
            if torch.isnan(grad).any():
                grad = torch.zeros_like(grad)
            step_scale = (1.0 / torch.sqrt(E.ALPHAS[t])) if args.inv_sqrt_alpha else args.zeta
            delta = step_scale * grad
            if args.step_cap_tau and args.step_cap_tau > 0:
                cap = args.step_cap_tau * torch.sqrt(1.0 - E.BARALPHAS[t])
                dn = delta.norm()
                if dn > cap:
                    delta = delta * (cap / dn)
            x_t = x_t_minus_1.detach().clone() - delta

    return x_t.detach().reshape(()).item()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1042])
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--n_target", type=int, default=250)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--step_cap_tau", type=float, default=1.0)
    p.add_argument("--inv_sqrt_alpha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sampler", choices=["implicit", "st"], default="implicit")
    p.add_argument("--tau_gumbel", type=float, default=E.TAU_GUMBEL)
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--tag", default="")
    p.add_argument("--smoke", action="store_true",
                   help="arms point_sq_a0 + point_sq_a2, both seeds, 8 restarts")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_restarts = min(args.n_restarts, 8)
        if "--arms" not in sys.argv:
            args.arms = ["point_sq_a0", "point_sq_a2"]
        if not args.tag:
            args.tag = "smoke"

    # construction sanity (round-2 defaults; E[y|x] == 0 everywhere)
    assert abs(E.S_UNI - 0.25) < 1e-9, "expected the round-2 default s=0.25"
    for x in (-3.0, -1.0, 0.0, 1.7):
        w = torch.softmax(E.cond_logits(torch.tensor(x)), dim=0)
        assert abs((w * E.MU_Y).sum().item()) < 1e-6

    conv = (f"cap{args.step_cap_tau:g}" if args.step_cap_tau and args.step_cap_tau > 0 else "nocap") \
           + ("_inv" if args.inv_sqrt_alpha else "_zeta")
    print(f"Round 6 pointwise | conv={conv} | arms={args.arms} | seeds={args.seeds}")

    rows = []
    for seed in args.seeds:
        for arm in args.arms:
            for i in range(args.n_restarts):
                run_seed = seed + i
                torch.manual_seed(run_seed)          # paired init/noise across arms
                t0 = time.time()
                x_hat = optimize_point(arm, args, run_seed)
                elapsed = time.time() - t0
                # cross-MMD to both round-2 target sets; the frac_within_2s /
                # mode fields are keyed to the "uni" target and are not the
                # round-6 readout (x-space stats are) — documented here.
                m = E.evaluate(x_hat, "uni", args, eval_seed=seed + 200_000 + i)
                rows.append({"target": "scalar", "arm": arm, "a": ARMS[arm][1],
                             "loss_kind": ARMS[arm][0], "convention": conv,
                             "seed": seed, "restart": i, "run_seed": run_seed,
                             "time_s": elapsed, **m})
                if i < 2 or (i + 1) == args.n_restarts:
                    print(f"[{arm} | seed {seed} | {i + 1}/{args.n_restarts}] "
                          f"x_hat={x_hat:+.3f} land(bi,uni)=({m['land_bi']},{m['land_uni']}) "
                          f"MMD(bi)={m['mmd_to_bi']:.4f} MMD(uni)={m['mmd_to_uni']:.4f} "
                          f"({elapsed:.2f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"pointwise_{conv}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "experiment": "pointwise", "convention": conv,
                "arms": args.arms, "arm_defs": {k: {"loss": v[0], "a": v[1]} for k, v in ARMS.items()},
                "seeds": args.seeds, "n_restarts": args.n_restarts,
                "n_target": args.n_target, "n_eval": args.n_eval,
                "zeta": args.zeta, "step_cap_tau": args.step_cap_tau,
                "inv_sqrt_alpha": args.inv_sqrt_alpha, "sampler": args.sampler,
                "s_uni": E.S_UNI,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"[Results] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
