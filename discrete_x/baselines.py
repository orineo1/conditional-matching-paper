"""Equal-compute baselines for the discrete-x SMC experiments.

Methods:
  sdedit_best : B independent UNGUIDED denoisings (same prior, same SDEdit
                noising, same reverse kernel incl. top-k — everything except
                guidance), evaluate L on each final decode, keep the best.
                The direct analogue of the paper's "SDEdit Best" baseline.
  random      : B uniform-random canvas fills, evaluate L, keep the best.

Compute parity: the budget B is set at or above the number of UNIQUE image-
loss evaluations the guided SMC runs consumed (recorded as loss_evals_cum in
their metrics.json), and both share the memoized loss, so the baseline gets
at least as many oracle calls as the method.

Run on cluster: see scripts/submit_baselines.sh
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_smc import (MASK, BertMaskedPrior, LLaDAPrior, TurboImageMMDLoss,
                      sample_token)


def unguided_sdedit_batch(prior, x_src, B, remask_frac, T, top_k, rng):
    """B independent unguided reverse chains from the noised source,
    vectorized like the SMC propagation (batched denoiser calls)."""
    Lp = prior.canvas_len
    t0 = max(1, round(remask_frac * T))
    X = np.tile(x_src, (B, 1))
    X[rng.random((B, Lp)) < (t0 / T)] = MASK
    for t in range(t0, 0, -1):
        marg = prior.batch_marginals(X)
        for b in range(B):
            masked = np.where(X[b] == MASK)[0]
            p_un = 1.0 if t <= 1 else 1.0 / t
            for i in masked[rng.random(len(masked)) < p_un]:
                X[b, i] = sample_token(marg[b, i], rng, top_k)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sdedit_best", "random"],
                    required=True)
    ap.add_argument("--prior", choices=["bert", "llada"], default="bert")
    ap.add_argument("--prefix", default="a photo of")
    ap.add_argument("--target_words", nargs="+", required=True)
    ap.add_argument("--source_words", nargs="+", default=None,
                    help="defaults to target_words (noised-truth setup)")
    ap.add_argument("--remask_frac", type=float, default=0.75)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sampler", choices=["turbo", "base"], default="base")
    ap.add_argument("--n_cond", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=64)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.prior == "llada":
        prior = LLaDAPrior(device, args.prefix,
                           canvas_len=len(args.target_words))
    else:
        prior = BertMaskedPrior("bert-base-uncased", device, args.prefix,
                                canvas_len=len(args.target_words))
    x_true = prior.encode_canvas(args.target_words)
    target_text = prior.decode(x_true)
    src_words = args.source_words or args.target_words
    x_src = prior.encode_canvas(src_words)

    if args.sampler == "base":
        kw = dict(turbo_name="stabilityai/stable-diffusion-xl-base-1.0",
                  n_steps=12, cfg=3.0)
    else:
        kw = dict(n_steps=1, cfg=0.0)
    loss_fn = TurboImageMMDLoss(target_text, device, n_cond=args.n_cond,
                                n_target=args.n_target,
                                save_dir=args.outdir, **kw)
    print(f"method={args.method} prior={args.prior} B={args.B} "
          f"L(x_true)={loss_fn(target_text):.5f}", flush=True)
    target_demo = loss_fn.demographics(loss_fn.S_G)
    print(f"target demographics: {target_demo}", flush=True)

    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(10_000 + seed)
        t_start = time.perf_counter()
        evals_before = loss_fn.n_evals
        if args.method == "sdedit_best":
            X = unguided_sdedit_batch(prior, x_src, args.B,
                                      args.remask_frac, args.T,
                                      args.top_k, rng)
        else:
            X = rng.integers(0, prior.A,
                             size=(args.B, prior.canvas_len))
        losses = np.array([loss_fn(prior.decode(x)) for x in X])
        best = int(np.argmin(losses))
        x_best = X[best]
        res = {
            "method": args.method, "prior": args.prior, "seed": seed,
            "B": args.B,
            "x_best_decode": prior.decode(x_best),
            "L_best": float(losses[best]),
            "L_median": float(np.median(losses)),
            "exact_recovery": bool(np.all(x_best == x_true)),
            "demographics_best": loss_fn.eval_demographics(
                prior.decode(x_best)),
            "unique_loss_evals": loss_fn.n_evals - evals_before,
            "wall_clock_s": time.perf_counter() - t_start,
        }
        results.append(res)
        print(f"[{args.method} seed {seed}] L_best={res['L_best']:.5f} "
              f"(median {res['L_median']:.3f}) exact={res['exact_recovery']} "
              f"unique_evals={res['unique_loss_evals']} "
              f"t={res['wall_clock_s']:.0f}s\n"
              f"  -> '{res['x_best_decode']}'\n"
              f"  demo: {res['demographics_best']} vs target {target_demo}",
              flush=True)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "target_text": target_text,
                   "L_x_true": loss_fn(target_text),
                   "target_demographics": target_demo,
                   "runs": results}, f, indent=2)
    print(f"wrote {args.outdir}/metrics.json")


if __name__ == "__main__":
    main()
