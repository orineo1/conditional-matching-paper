"""
exp_concentration.py — Round 3: does the witness advantage grow with target
concentration? Pre-registered in HYPOTHESIS.md ("Round 3"; predictions C1-C3).

Thin driver over exp_identifiability's machinery: same analytic construction,
corrected convention (step cap tau=1 + 1/sqrt(alpha_t), x_T ~ randn, implicit
sampler), with the unimodal component's y-std swept via
exp_identifiability.set_uni_std(s), s in {0.05, 0.1, 0.25, 0.5, 1.0}. Bimodal
blobs fixed at ±2 / 0.25; x-structure unchanged; E[y|x] ≡ 0 re-verified per s.
Target is always the concentrated unimodal one (x_uni). Arms: mmd (reference),
mmd_uniform, mmd_witness (k=8/32), paired restarts across arms and seeds.

MMD bandwidth policy (fixed rule, same as rounds 1-2, stated in HYPOTHESIS.md):
adaptive multi-bandwidth RBF recomputed per call on the stacked batch — never
tuned per s.

Backsel arms also record witness-selection diagnostics: frac_selected_offmode
(of the k selected rows, the fraction with |y| > 2s at selection time) vs
frac_offmode (the batch base rate = the uniform arm's expectation) — C2.

Smoke:  python exp_concentration.py --smoke      (s=0.05, both seeds, 8 restarts)
Full:   see submit_concentration.sh (15-cell array over s x arm; both seeds per cell)
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

S_LIST = [0.05, 0.10, 0.25, 0.50, 1.00]
ARMS = ["mmd", "mmd_uniform", "mmd_witness"]


def verify_per_s(s):
    """E[y|x] must stay identically 0 at this s (A/B symmetry is s-independent)."""
    for x in (-3.0, -1.0, 0.0, 1.7, 2.0):
        w = torch.softmax(E.cond_logits(torch.tensor(x)), dim=0)
        m = (w * E.MU_Y).sum().item()
        assert abs(m) < 1e-6, (s, x, m)
    print(f"  [verify s={s}] E[y|x]=0 holds; SIG_Y={E.SIG_Y.tolist()}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s_list", type=float, nargs="+", default=S_LIST)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1042])
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--backsel_k", type=int, default=8)
    p.add_argument("--witness_floor", type=float, default=0.3)
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
                   help="s=0.05 only, both seeds, 8 restarts, all three arms")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_restarts = min(args.n_restarts, 8)
        args.s_list = [0.05]
        if not args.tag:
            args.tag = "smoke"

    conv = (f"cap{args.step_cap_tau:g}" if args.step_cap_tau and args.step_cap_tau > 0 else "nocap") \
           + ("_inv" if args.inv_sqrt_alpha else "_zeta")
    print(f"Round 3 concentration sweep | conv={conv} | s_list={args.s_list} | seeds={args.seeds}")

    rows = []
    for s in args.s_list:
        E.set_uni_std(s)
        verify_per_s(s)
        for seed in args.seeds:
            for arm in args.arms:
                collect = arm in ("mmd_uniform", "mmd_witness")
                for i in range(args.n_restarts):
                    run_seed = seed + i
                    torch.manual_seed(run_seed)
                    tgt_gen = torch.Generator().manual_seed(run_seed)   # paired across arms
                    target_samples = E.sample_cond_hard(E.X_UNI, args.n_target, generator=tgt_gen)
                    t0 = time.time()
                    out = E.optimize_arm(arm, target_samples, args, run_seed, collect_diag=collect)
                    x_hat, diag = out if collect else (out, {})
                    elapsed = time.time() - t0
                    m = E.evaluate(x_hat, "uni", args, eval_seed=seed + 200_000 + i)
                    rows.append({"s": s, "target": "uni", "arm": arm, "convention": conv,
                                 "seed": seed, "restart": i, "run_seed": run_seed,
                                 "time_s": elapsed, **m, **diag})
                    if i < 2 or (i + 1) == args.n_restarts:
                        d = (f" sel_off={diag.get('frac_selected_offmode', float('nan')):.3f}"
                             f"/base={diag.get('frac_offmode', float('nan')):.3f}") if diag else ""
                        print(f"[s={s} | seed {seed} | {arm} | {i + 1}/{args.n_restarts}] "
                              f"x_hat={x_hat:+.3f} dist_uni={m['dist_uni']:.3f}{d} "
                              f"({elapsed:.2f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"concentration_{conv}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "experiment": "concentration", "convention": conv,
                "s_list": args.s_list, "arms": args.arms, "seeds": args.seeds,
                "n_restarts": args.n_restarts, "nsamples": args.nsamples,
                "backsel_k": args.backsel_k, "witness_floor": args.witness_floor,
                "n_target": args.n_target, "n_eval": args.n_eval,
                "zeta": args.zeta, "step_cap_tau": args.step_cap_tau,
                "inv_sqrt_alpha": args.inv_sqrt_alpha, "sampler": args.sampler,
                "bandwidth_policy": "adaptive multi-bandwidth RBF per call on the "
                                     "stacked batch (rounds 1-2 rule, not tuned per s)",
                "bimodal_fixed": {"modes": [-E.C_SEP, E.C_SEP], "std": E.SIGY_BI},
            },
            "rows": rows,
        }, f, indent=2)
    print(f"[Results] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
