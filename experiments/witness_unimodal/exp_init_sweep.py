"""
exp_init_sweep.py — Round 7: init sweep to separate objective pull from basin
capture. Pre-registered in HYPOTHESIS.md ("Round 7"; predictions (i)-(iii)).

x_T ~ c + N(0, 1), c swept over {-3, -1.5, 0, +1.5, +3}; everything else is the
corrected convention unchanged (cap tau=1 + 1/sqrt(alpha_t), implicit sampler,
sigma_init = 1). Arms: the round-6 pointwise arms (point_sq_a0, point_sq_a2,
point_abs_a2) and the two MMD references RE-RUN at each offset (mmd_bi = mmd
given the bimodal target set, mmd_uni = mmd given the unimodal target set) —
round-2 rows cannot be reused because the init changed.

Smoke:  python exp_init_sweep.py --smoke      (mmd_bi at c=+3 — prediction (iii)'s
                                               hardest case — both seeds, 8 restarts)
Full:   see submit_init_sweep.sh (25-cell array over c x arm; both seeds per cell)
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
import exp_pointwise as P

C_LIST = [-3.0, -1.5, 0.0, 1.5, 3.0]
ARMS = ["point_sq_a0", "point_sq_a2", "point_abs_a2", "mmd_bi", "mmd_uni"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--c_list", type=float, nargs="+", default=C_LIST)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1042])
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=32,
                   help="mmd arms' per-step conditional sample count (round-2 value)")
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
                   help="mmd_bi at c=+3 only, both seeds, 8 restarts")
    return p.parse_args()


def run_one(arm, c, seed, i, args):
    run_seed = seed + i
    torch.manual_seed(run_seed)
    args.init_offset = c          # read via getattr in both optimize loops
    if arm.startswith("mmd_"):
        target_name = arm.split("_", 1)[1]          # "bi" | "uni"
        tgt_gen = torch.Generator().manual_seed(run_seed)   # paired target draw
        target_samples = E.sample_cond_hard(E.TARGET_X[target_name],
                                            args.n_target, generator=tgt_gen)
        x_hat = E.optimize_arm("mmd", target_samples, args, run_seed)
        eval_target = target_name
    else:
        x_hat = P.optimize_point(arm, args, run_seed)
        eval_target = "uni"       # frac/mode fields keyed to uni; x-space is the readout
    m = E.evaluate(x_hat, eval_target, args, eval_seed=seed + 200_000 + i)
    return x_hat, m


def main():
    args = parse_args()
    if args.smoke:
        args.n_restarts = min(args.n_restarts, 8)
        if "--c_list" not in sys.argv:
            args.c_list = [3.0]
        if "--arms" not in sys.argv:
            args.arms = ["mmd_bi"]
        if not args.tag:
            args.tag = "smoke"

    assert abs(E.S_UNI - 0.25) < 1e-9, "expected the round-2 default s=0.25"
    conv = (f"cap{args.step_cap_tau:g}" if args.step_cap_tau and args.step_cap_tau > 0 else "nocap") \
           + ("_inv" if args.inv_sqrt_alpha else "_zeta")
    print(f"Round 7 init sweep | conv={conv} | c_list={args.c_list} | "
          f"arms={args.arms} | seeds={args.seeds}")

    rows = []
    for c in args.c_list:
        for seed in args.seeds:
            for arm in args.arms:
                for i in range(args.n_restarts):
                    t0 = time.time()
                    x_hat, m = run_one(arm, c, seed, i, args)
                    elapsed = time.time() - t0
                    rows.append({"arm": arm, "init_offset": c, "convention": conv,
                                 "seed": seed, "restart": i, "run_seed": seed + i,
                                 "time_s": elapsed, **m})
                    if i < 2 or (i + 1) == args.n_restarts:
                        print(f"[c={c:+g} | seed {seed} | {arm} | {i + 1}/{args.n_restarts}] "
                              f"x_hat={x_hat:+.3f} land(bi,uni)=({m['land_bi']},{m['land_uni']}) "
                              f"({elapsed:.2f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"initsweep_{conv}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "experiment": "init_sweep", "convention": conv,
                "c_list": args.c_list, "arms": args.arms, "seeds": args.seeds,
                "n_restarts": args.n_restarts, "nsamples": args.nsamples,
                "n_target": args.n_target, "n_eval": args.n_eval,
                "zeta": args.zeta, "step_cap_tau": args.step_cap_tau,
                "inv_sqrt_alpha": args.inv_sqrt_alpha, "sampler": args.sampler,
                "init_sigma": 1.0, "s_uni": E.S_UNI,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"[Results] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
