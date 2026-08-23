"""Toy-world validation of rate-ratio (TFG-Flow-style) guidance, mirroring
run_sanity_compare.py so results are comparable with the twisted-SMC runs.

Methods:
  unguided       gamma=0 baseline (the prior CTMC)
  mc             MC rate ratio, K decodes/step (no assumption on f)
  additive       closed-form rate ratio; theta fitted on M design decodes
                 sampled from the PRIOR (oracle calls happen once, up front)
  additive_full  theta = exact additive projection over the ENTIRE prompt
                 space (3750) — the ceiling of the additive assumption

Each method runs R independent chains per seed and keeps the best decoded
canvas by L (same optimization view as the SMC runs). Reports exact
recovery of x_true, best L vs L(x_true) and the brute-force optimum, and
unique loss-oracle evaluations.

Usage: python discrete_x/run_ctmc_compare.py [--seeds 0 1 2 3 4]
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctmc_guidance import (fit_additive_theta, guided_ctmc_sample)
from masked_diffusion import posterior_marginals
from sanity_task import SanityTask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--beta", type=float, default=200.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--M_design", type=int, default=512)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--no_anneal", action="store_true")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--remask_guided", action="store_true")
    ap.add_argument("--outdir", default="output/ctmc_toy")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    task = SanityTask()
    raw_calls = {"n": 0}

    def loss_fn(x):
        raw_calls["n"] += 1
        return task.loss(x)

    marginals_fn = lambda x: posterior_marginals(task, x)
    vocab = task.vocab_sizes
    x_opt, l_opt = task.brute_force_optimum()
    l_true = task.loss(task.x_true)
    print(f"x_true = '{task.decode(task.x_true)}'  L={l_true:.5f}")
    print(f"x_opt  = '{task.decode(x_opt)}'  L={l_opt:.5f} "
          f"(match_true={bool((x_opt == task.x_true).all())})")

    # exact additive projection over the full space (assumption ceiling)
    full_X = np.array(list(itertools.product(*[range(V) for V in vocab])))
    full_L = np.array([task.loss(x) for x in full_X])
    theta_full = fit_additive_theta(vocab, full_X, full_L, args.beta)

    methods = ["unguided", "mc", "additive", "additive_full"]
    results = {m: [] for m in methods}
    for seed in args.seeds:
        for method in methods:
            rng = np.random.default_rng(
                1000 * seed + methods.index(method))
            evals0 = raw_calls["n"]
            t0 = time.perf_counter()
            theta = None
            estimator = method
            if method == "additive":
                # design sampled from the PRIOR (ancestral, unguided chains)
                Xd = [guided_ctmc_sample(marginals_fn, loss_fn, vocab,
                                         rng, n_steps=args.n_steps,
                                         estimator="unguided")
                      for _ in range(args.M_design)]
                Ld = [loss_fn(x) for x in Xd]
                theta = fit_additive_theta(vocab, np.array(Xd), Ld, args.beta)
                estimator = "additive"
            elif method == "additive_full":
                theta = theta_full
                estimator = "additive"
            best_x, best_l = None, np.inf
            for _ in range(args.restarts):
                x = guided_ctmc_sample(
                    marginals_fn, loss_fn, vocab, rng,
                    n_steps=args.n_steps, beta=args.beta,
                    gamma=args.gamma, estimator=estimator, K=args.K,
                    theta=theta, anneal=not args.no_anneal, eta=args.eta,
                    remask_guided=args.remask_guided)
                l = loss_fn(x)
                if l < best_l:
                    best_x, best_l = x, l
            rec = {
                "seed": seed, "method": method,
                "best_decode": task.decode(best_x),
                "L_best": float(best_l),
                "exact_recovery": bool((best_x == task.x_true).all()),
                "hits_optimum": bool((best_x == x_opt).all()),
                "raw_loss_calls": raw_calls["n"] - evals0,
                "wall_s": time.perf_counter() - t0,
            }
            results[method].append(rec)
            print(f"[seed {seed:2d}] {method:14s} L={best_l:.5f} "
                  f"recover={rec['exact_recovery']} opt={rec['hits_optimum']} "
                  f"evals={rec['raw_loss_calls']:5d}  "
                  f"'{rec['best_decode']}'")

    print("\n=== summary ===")
    print(f"L(x_true)={l_true:.5f}  L(x_opt)={l_opt:.5f}")
    for m in methods:
        rr = results[m]
        rec = sum(r["exact_recovery"] for r in rr)
        opt = sum(r["hits_optimum"] for r in rr)
        print(f"{m:14s} recover {rec}/{len(rr)}  optimum {opt}/{len(rr)}  "
              f"L mean {np.mean([r['L_best'] for r in rr]):.5f}  "
              f"raw calls mean {np.mean([r['raw_loss_calls'] for r in rr]):.0f}")

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "L_x_true": l_true,
                   "L_x_opt": float(l_opt),
                   "x_true": task.decode(task.x_true),
                   "x_opt": task.decode(x_opt),
                   "results": results}, f, indent=2)
    print(f"wrote {args.outdir}/metrics.json")


if __name__ == "__main__":
    main()
