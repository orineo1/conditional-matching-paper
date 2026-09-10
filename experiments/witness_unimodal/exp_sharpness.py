"""
exp_sharpness.py — Round 4: sharpened witness selection p ∝ |w|^β (floor 0) +
deterministic top-k + shuffled-scores control, with the selection diagnostics
round 3 lacked. Pre-registered in HYPOTHESIS.md ("Round 4"; predictions S1-S4).

Reuses exp_identifiability's analytic machinery (construction, set_uni_std,
implicit sampler, analytic prior DDIM, evaluate) and witness_utils'
compute_witness_scores; the selection itself is round-4-specific:

  - Stochastic arms (uniform, b1/b2/b4, b4_shuffled): k=8 indices drawn WITH
    replacement from p; unbiased IPW gradient weights g_i = counts_i/(k p_i)
    applied by value-preserving gradient scaling (row value untouched, so the
    loss VALUE still uses all n rows; E[gradient] = full-batch gradient).
  - witness_topk: deterministic top-k by |w| — BIASED (no IPW correction exists
    for deterministic selection); included as the practical limit.
  - witness_b4_shuffled: β=4 probabilities randomly permuted across the batch
    (same histogram/floor/IPW/RNG usage, zero sample-identity information).

Per-run diagnostics (means over steps): normalized selection entropy
−Σ p ln p / ln n (top-k: ln k/ln n by convention; uniform ≡ 1), witness-score
dispersion max|w|/median|w| and std(w), frac_selected_offmode vs base rate.

Smoke:  python exp_sharpness.py --smoke     (s=0.5, uniform + b4, both seeds, 8 restarts)
Full:   see submit_sharpness.sh (18-cell array over s x arm; both seeds per cell)
"""

import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import exp_identifiability as E
from witness_utils import compute_witness_scores
from LossFunctions import MMDLoss, RBF

S_LIST = [0.10, 0.25, 0.50]
ARM_SPECS = {
    "mmd_uniform":         dict(rule="uniform"),
    "witness_b1_f0":       dict(rule="witness", beta=1.0),
    "witness_b2_f0":       dict(rule="witness", beta=2.0),
    "witness_b4_f0":       dict(rule="witness", beta=4.0),
    "witness_topk":        dict(rule="topk"),
    "witness_b4_shuffled": dict(rule="witness", beta=4.0, shuffled=True),
    # Round 5: WITHOUT-replacement variants (HYPOTHESIS.md "Round 5").
    # uniform_wor uses the EXACT HT weights n/k (true inclusion prob k/n);
    # witness_b*_wor use g_i = 1/pi_i with the approximation
    # pi_i ~= 1-(1-p_i)^k -- exact for i.i.d. (with-replacement) draws,
    # slightly OVERWEIGHTING large-p_i rows under successive without-
    # replacement sampling (their true pi is larger than the approximation);
    # bias grows with beta. k >= n is special-cased to the exact full gradient.
    "uniform_wor":     dict(rule="uniform", wor=True),
    "witness_b1_wor":  dict(rule="witness", beta=1.0, wor=True),
    "witness_b2_wor":  dict(rule="witness", beta=2.0, wor=True),
    "witness_b4_wor":  dict(rule="witness", beta=4.0, wor=True),
}


def apply_sharp_backsel(y, target_samples, k, spec, generator):
    """Round-4 selection. Returns (batch, diag) where batch has the value of y
    on every row and per-row gradient scale g_i (0 for unselected rows)."""
    n = y.shape[0]
    diag = {}
    if spec["rule"] != "uniform":
        scores = compute_witness_scores(y, target_samples)
        a = scores.abs().double().clamp_min(1e-12)
        diag["score_disp"] = (a.max() / a.median().clamp_min(1e-12)).item()
        diag["score_std"] = scores.std().item()
    else:
        scores = None

    if spec["rule"] == "topk":
        idx = scores.abs().topk(k).indices
        gw = torch.zeros(n)
        gw[idx] = 1.0
        sel_mask = gw > 0
        diag["sel_entropy"] = math.log(k) / math.log(n)   # convention (see docstring)
    else:
        if spec["rule"] == "uniform":
            p = torch.full((n,), 1.0 / n, dtype=torch.float64)
        else:
            p = a ** spec["beta"]
            p = p / p.sum()
            if spec.get("shuffled"):
                p = p[torch.randperm(n, generator=generator)]
        if spec.get("wor"):
            # Round 5: k distinct rows, Horvitz-Thompson weights (see ARM_SPECS note)
            if k >= n:
                gw = torch.ones(n)                          # exact: full gradient
                sel_mask = torch.ones(n, dtype=torch.bool)
            elif spec["rule"] == "uniform":
                idx = torch.multinomial(p, k, replacement=False, generator=generator)
                gw = torch.zeros(n)
                gw[idx] = n / k                              # exact HT: pi_i = k/n
                sel_mask = gw > 0
            else:
                idx = torch.multinomial(p, k, replacement=False, generator=generator)
                pi = (1.0 - (1.0 - p) ** k).clamp(1e-12, 1.0)   # approx (see note)
                gw = torch.zeros(n)
                gw[idx] = (1.0 / pi[idx]).float()
                sel_mask = gw > 0
        else:
            idx = torch.multinomial(p, k, replacement=True, generator=generator)
            counts = torch.bincount(idx, minlength=n).double()
            gw = (counts / (k * p)).float()                # IPW: E[gw_i] = 1
            sel_mask = counts > 0
        diag["sel_entropy"] = (-(p * p.clamp_min(1e-300).log()).sum() / math.log(n)).item()

    batch = y.detach() + gw.view(-1, 1) * (y - y.detach())
    return batch, sel_mask, diag


def optimize_sharp(arm, target_samples, args, run_seed):
    """Mirror of exp_identifiability.optimize_arm with round-4 selection."""
    spec = ARM_SPECS[arm]
    generator = torch.Generator().manual_seed(run_seed)
    mmd_loss = MMDLoss(kernel=RBF())
    acc = defaultdict(list)

    x_t = torch.randn(1, 1)
    for t in range(E.T_STEPS - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)
        x_t_minus_1, pred_x0 = E.ddim_step(x_t, t)
        r_t = E.BETAS[t] / torch.sqrt(1 + E.BETAS[t] ** 2)

        x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
        y = (E.sample_cond_diff(x0_sample, args.nsamples) if args.sampler == "implicit"
             else E.sample_cond_st(x0_sample, args.nsamples, tau=args.tau_gumbel))

        y_batch, sel_mask, diag = apply_sharp_backsel(
            y, target_samples, args.backsel_k, spec, generator)
        with torch.no_grad():
            offmode = y.detach().abs().view(-1) > 2.0 * E.S_UNI
            if sel_mask.any():
                acc["frac_selected_offmode"].append(offmode[sel_mask].float().mean().item())
            acc["frac_offmode"].append(offmode.float().mean().item())
            for kk, vv in diag.items():
                acc[kk].append(vv)

        loss_val = mmd_loss(y_batch, target_samples)
        log_mean_exp_loss = -torch.logsumexp(torch.stack([-loss_val]), dim=0)
        grad = torch.autograd.grad(log_mean_exp_loss, x_t)[0]

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

    diag_means = {kk: sum(vv) / len(vv) for kk, vv in acc.items() if vv}
    return x_t.detach().reshape(()).item(), diag_means


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s_list", type=float, nargs="+", default=S_LIST)
    p.add_argument("--arms", nargs="+", choices=list(ARM_SPECS), default=list(ARM_SPECS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1042])
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--backsel_k", type=int, default=8)
    p.add_argument("--n_target", type=int, default=250)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--step_cap_tau", type=float, default=1.0)
    p.add_argument("--inv_sqrt_alpha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sampler", choices=["implicit", "st"], default="implicit")
    p.add_argument("--tau_gumbel", type=float, default=E.TAU_GUMBEL)
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--tag", default="")
    p.add_argument("--out_prefix", default="sharpness",
                   help="output filename prefix (round 5 uses 'replacement' so its "
                        "JSONs don't mix into the round-4 fig5 builder's glob)")
    p.add_argument("--selftest", action="store_true",
                   help="run the k=n exactness test + Monte-Carlo pi-approximation "
                        "check for the without-replacement arms, then exit")
    p.add_argument("--smoke", action="store_true",
                   help="s=0.5, arms mmd_uniform + witness_b4_f0, both seeds, 8 restarts")
    return p.parse_args()


def selftest():
    """(1) k=n exactness: every wor arm must reproduce the full-batch gradient
    exactly. (2) MC check of pi_i ~= 1-(1-p_i)^k at k=8 under successive
    without-replacement sampling (reports the approximation error; informative,
    not an assertion — the bias is documented in HYPOTHESIS.md Round 5)."""
    torch.manual_seed(0)
    y = torch.randn(32, 1, requires_grad=True)
    tgt = torch.randn(250, 1)
    for arm in ("uniform_wor", "witness_b1_wor", "witness_b4_wor"):
        batch, _, _ = apply_sharp_backsel(y, tgt, 32, ARM_SPECS[arm],
                                          torch.Generator().manual_seed(0))
        g = torch.autograd.grad((batch ** 2).sum(), y, retain_graph=True)[0]
        gf = torch.autograd.grad((y ** 2).sum(), y, retain_graph=True)[0]
        assert torch.allclose(g, gf), arm
        print(f"[selftest] k=n exactness OK for {arm}")
    p = torch.rand(32).double() ** 4
    p = p / p.sum()
    k, trials = 8, 20000
    freq = torch.zeros(32)
    gen = torch.Generator().manual_seed(1)
    for _ in range(trials):
        freq[torch.multinomial(p, k, replacement=False, generator=gen)] += 1
    emp = freq / trials
    approx = 1.0 - (1.0 - p) ** k
    err = (emp - approx.float()).abs()
    print(f"[selftest] pi approximation at k=8: max|emp-approx|={err.max():.4f} "
          f"(at p_i={p[err.argmax()]:.3f}), mean={err.mean():.4f} — "
          f"documented small bias, largest for large p_i")


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if args.smoke:
        args.n_restarts = min(args.n_restarts, 8)
        args.s_list = [0.50]
        if "--arms" not in sys.argv:                     # keep explicitly-passed arms
            args.arms = ["mmd_uniform", "witness_b4_f0"]
        if not args.tag:
            args.tag = "smoke"

    conv = (f"cap{args.step_cap_tau:g}" if args.step_cap_tau and args.step_cap_tau > 0 else "nocap") \
           + ("_inv" if args.inv_sqrt_alpha else "_zeta")
    print(f"Round 4 sharpness | conv={conv} | s_list={args.s_list} | arms={args.arms} | seeds={args.seeds}")

    rows = []
    for s in args.s_list:
        E.set_uni_std(s)
        for x in (-3.0, -1.0, 0.0, 1.7):   # E[y|x] == 0 must survive every s
            w = torch.softmax(E.cond_logits(torch.tensor(x)), dim=0)
            assert abs((w * E.MU_Y).sum().item()) < 1e-6
        for seed in args.seeds:
            for arm in args.arms:
                for i in range(args.n_restarts):
                    run_seed = seed + i
                    torch.manual_seed(run_seed)
                    tgt_gen = torch.Generator().manual_seed(run_seed)   # paired across arms
                    target_samples = E.sample_cond_hard(E.X_UNI, args.n_target, generator=tgt_gen)
                    t0 = time.time()
                    x_hat, diag = optimize_sharp(arm, target_samples, args, run_seed)
                    elapsed = time.time() - t0
                    m = E.evaluate(x_hat, "uni", args, eval_seed=seed + 200_000 + i)
                    rows.append({"s": s, "target": "uni", "arm": arm, "convention": conv,
                                 "seed": seed, "restart": i, "run_seed": run_seed,
                                 "time_s": elapsed, **m, **diag})
                    if i < 2 or (i + 1) == args.n_restarts:
                        print(f"[s={s} | seed {seed} | {arm} | {i + 1}/{args.n_restarts}] "
                              f"x_hat={x_hat:+.3f} dist_uni={m['dist_uni']:.3f} "
                              f"H={diag.get('sel_entropy', float('nan')):.3f} "
                              f"sel_off={diag.get('frac_selected_offmode', float('nan')):.3f}"
                              f"/base={diag.get('frac_offmode', float('nan')):.3f} "
                              f"({elapsed:.2f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"{args.out_prefix}_{conv}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "experiment": "sharpness", "convention": conv,
                "s_list": args.s_list, "arms": args.arms, "seeds": args.seeds,
                "arm_specs": {k: {kk: vv for kk, vv in v.items()} for k, v in ARM_SPECS.items()},
                "n_restarts": args.n_restarts, "nsamples": args.nsamples,
                "backsel_k": args.backsel_k, "n_target": args.n_target,
                "n_eval": args.n_eval, "zeta": args.zeta,
                "step_cap_tau": args.step_cap_tau, "inv_sqrt_alpha": args.inv_sqrt_alpha,
                "sampler": args.sampler,
                "estimator": "with-replacement k-draw + IPW grad weights for all "
                             "stochastic arms (topk deterministic, biased, no IPW)",
            },
            "rows": rows,
        }, f, indent=2)
    print(f"[Results] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
