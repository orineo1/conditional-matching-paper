"""Image-scale rate-ratio (TFG-Flow / Nisonoff-style) guidance — the
CTMC alternative to real_smc.py's twisted SMC on the same tasks.

Single chain per restart, no particles/resampling/weights. Guidance
enters by tilting the destination law of each unmasking coordinate with
the f-weighted posterior decodes (estimator=mc; f = exp(-beta L), only
the scalar loss oracle is used — no CLIP-text surrogate, no gradients),
or in closed form under the additive relaxing assumption
(estimator=additive; per-slot theta fitted on prior decodes by a one-pass
ANOVA estimate — full ridge is infeasible at vocab 18k x 8 slots).

Toy validation (run_ctmc_compare.py): mc with gamma=10, K=128, 16
restarts recovers x_true 5/5 seeds — parity with twisted SMC at ~9x
fewer raw oracle calls. additive plateaus: the MMD objective couples
coordinates, so its additive projection is a weak surrogate (and higher
gamma makes it worse). Expect mc to be the working config here.

Example (headline noised-truth recovery):
  python discrete_x/real_ctmc.py --task recover --estimator mc \
      --gamma 10 --K 16 --restarts 4 --seeds 0 1 2 --sampler base \
      --outdir output/ctmc_recover_mc_g10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctmc_guidance import logsumexp, mc_guided_law
from real_smc import (MASK, BertMaskedPrior, LLaDAPrior, TurboImageMMDLoss,
                      sample_token)


def fit_theta_anova(canvas_len, A, X_design, losses, beta, shrink=5.0):
    """One-pass ANOVA additive fit of log f = -beta*L: theta_d(v) is the
    shrunk mean residual (vs global mean) over design rows with x^d = v."""
    y = -beta * np.asarray(losses, dtype=float)
    y = y - y.mean()
    theta = np.zeros((canvas_len, A))
    X = np.asarray(X_design)
    for d in range(canvas_len):
        vals, inv = np.unique(X[:, d], return_inverse=True)
        sums = np.bincount(inv, weights=y)
        cnts = np.bincount(inv).astype(float)
        theta[d, vals] = (sums / cnts) * (cnts / (cnts + shrink))
    return theta


def ctmc_chain(prior, loss_fn, rng, n_steps, beta, gamma, estimator, K,
               theta, x_src, remask_frac, anneal, top_k, eta=0.0,
               remask_guided=False, log_rec=None):
    """eta: DFM detailed-balance stochasticity (Campbell et al. 2024) —
    unmasked coordinates remask at rate eta, unmask rate grows to
    (1 + eta*s)/(1-s); same prior marginal flow, but the chain can
    revise committed tokens.
    remask_guided: selective revision (mc only) — same total churn as
    uniform eta, allocated by softmax of the commit regret
    Delta_d = L_commit(d) - L_maskavg(d) cached free at commit time."""
    Lp = prior.canvas_len
    x = np.asarray(x_src, dtype=int).copy()
    x[rng.random(Lp) < remask_frac] = MASK
    s0 = 1.0 - remask_frac
    L_commit = np.zeros(Lp)
    L_maskavg = np.zeros(Lp)
    grid = np.linspace(s0, 1.0, n_steps + 1)
    for j in range(n_steps):
        s, s_next = grid[j], grid[j + 1]
        masked = np.where(x == MASK)[0]
        last = j == n_steps - 1
        if len(masked) == 0 and (eta == 0 or last):
            break
        ds = s_next - s
        rate_un = (1.0 + eta * s) / max(1.0 - s, 1e-12)
        p_un = 1.0 if last else min(1.0, ds * rate_un)
        to_un = masked[rng.random(len(masked)) < p_un]
        to_remask = np.array([], dtype=int)
        if eta > 0 and not last:
            unmasked = np.where(x != MASK)[0]
            if remask_guided and len(unmasked) > 1:
                delta = L_commit[unmasked] - L_maskavg[unmasked]
                tau = max(float(np.std(delta)), 1e-4)
                w = np.exp((delta - delta.max()) / tau)
                w /= w.sum()
                p_rm = np.minimum(1.0, len(unmasked) * eta * ds * w)
            else:
                p_rm = np.full(len(unmasked), min(1.0, eta * ds))
            to_remask = unmasked[rng.random(len(unmasked)) < p_rm]
        if len(to_un) == 0 and len(to_remask) == 0:
            continue
        marg = prior.batch_marginals(x[None])[0]
        b_s = beta * s_next if anneal else beta
        if len(to_un) == 0:
            pass  # remask-only step: no destination choice, no oracle
        elif estimator == "mc":
            Xd = np.tile(x, (K, 1))
            for d in masked:
                for k in range(K):
                    Xd[k, d] = sample_token(marg[d], rng, top_k)
            losses_k = np.array(
                [loss_fn(prior.decode(Xd[k])) for k in range(K)])
            logf = -b_s * losses_k
            for d in to_un:
                q = mc_guided_law(marg[d], Xd[:, d], logf, gamma)
                x[d] = rng.choice(len(q), p=q)
                sel = Xd[:, d] == x[d]
                L_maskavg[d] = losses_k.mean()
                L_commit[d] = (losses_k[sel].mean() if sel.any()
                               else L_maskavg[d])
        elif estimator == "additive":
            scale = s_next if anneal else 1.0
            for d in to_un:
                logq = np.log(marg[d] + 1e-300) + gamma * scale * theta[d]
                if top_k:
                    keep = np.argpartition(logq, -top_k)[-top_k:]
                    mask_out = np.full_like(logq, -np.inf)
                    mask_out[keep] = logq[keep]
                    logq = mask_out
                q = np.exp(logq - logq.max())
                x[d] = rng.choice(len(q), p=q / q.sum())
        else:  # unguided
            for d in to_un:
                x[d] = sample_token(marg[d], rng, top_k)
        if len(to_remask):
            x[to_remask] = MASK
        if log_rec is not None:
            log_rec.append({"s": float(s_next),
                            "n_masked": int((x == MASK).sum()),
                            "decode": prior.decode(x)})
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["recover", "edit"], default="recover")
    ap.add_argument("--estimator",
                    choices=["mc", "additive", "unguided"], default="mc")
    ap.add_argument("--prior", choices=["bert", "llada"], default="bert")
    ap.add_argument("--prefix", default="a photo of")
    ap.add_argument("--target_words", nargs="+",
                    default="an old man or young woman smiling".split())
    ap.add_argument("--edit_source_words", nargs="+",
                    default="one person sitting quietly inside the office"
                    .split())
    ap.add_argument("--remask_frac", type=float, default=0.75)
    ap.add_argument("--n_steps", type=int, default=16)
    ap.add_argument("--beta", type=float, default=200.0)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--M_design", type=int, default=256)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--no_anneal", action="store_true")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--remask_guided", action="store_true")
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
    x_src = (x_true if args.task == "recover"
             else prior.encode_canvas(args.edit_source_words))

    if args.sampler == "base":
        kw = dict(turbo_name="stabilityai/stable-diffusion-xl-base-1.0",
                  n_steps=12, cfg=3.0)
    else:
        kw = dict(n_steps=1, cfg=0.0)
    loss_fn = TurboImageMMDLoss(target_text, device, n_cond=args.n_cond,
                                n_target=args.n_target,
                                save_dir=args.outdir, **kw)
    l_true = loss_fn(target_text)
    print(f"task={args.task} estimator={args.estimator} gamma={args.gamma} "
          f"K={args.K} R={args.restarts} L(x_true)={l_true:.5f}", flush=True)
    target_demo = loss_fn.demographics(loss_fn.S_G)
    print(f"target demographics: {target_demo}", flush=True)

    theta = None
    theta_evals = 0
    if args.estimator == "additive":
        rng_th = np.random.default_rng(777)
        evals0 = loss_fn.n_evals
        Xd, Ld = [], []
        for _ in range(args.M_design):
            xd = ctmc_chain(prior, loss_fn, rng_th, args.n_steps, args.beta,
                            0.0, "unguided", 0, None, x_src,
                            args.remask_frac, False, args.top_k)
            Xd.append(xd)
            Ld.append(loss_fn(prior.decode(xd)))
        theta = fit_theta_anova(prior.canvas_len, prior.A, Xd, Ld, args.beta)
        theta_evals = loss_fn.n_evals - evals0
        print(f"additive theta fitted on {args.M_design} design decodes "
              f"({theta_evals} unique evals)", flush=True)

    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(10_000 + seed)
        t0 = time.perf_counter()
        evals0 = loss_fn.n_evals
        best_x, best_l, all_l = None, np.inf, []
        for r in range(args.restarts):
            x = ctmc_chain(prior, loss_fn, rng, args.n_steps, args.beta,
                           args.gamma, args.estimator, args.K, theta,
                           x_src, args.remask_frac, not args.no_anneal,
                           args.top_k, eta=args.eta,
                           remask_guided=args.remask_guided)
            l = loss_fn(prior.decode(x))
            all_l.append(float(l))
            print(f"  [seed {seed} chain {r}] L={l:.5f} "
                  f"'{prior.decode(x)}'", flush=True)
            if l < best_l:
                best_x, best_l = x, l
        res = {
            "seed": seed, "estimator": args.estimator,
            "gamma": args.gamma, "K": args.K,
            "x_best_decode": prior.decode(best_x),
            "L_best": float(best_l), "L_chains": all_l,
            "exact_recovery": bool(np.all(best_x == x_true)),
            "demographics_best": loss_fn.eval_demographics(
                prior.decode(best_x)),
            "unique_loss_evals": loss_fn.n_evals - evals0,
            "wall_clock_s": time.perf_counter() - t0,
        }
        results.append(res)
        print(f"[seed {seed}] BEST L={best_l:.5f} exact={res['exact_recovery']} "
              f"unique_evals={res['unique_loss_evals']} "
              f"t={res['wall_clock_s']:.0f}s\n"
              f"  -> '{res['x_best_decode']}'\n"
              f"  demo: {res['demographics_best']} vs {target_demo}",
              flush=True)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "target_text": target_text,
                   "L_x_true": l_true, "target_demographics": target_demo,
                   "theta_design_evals": theta_evals,
                   "runs": results}, f, indent=2)
    print(f"wrote {args.outdir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
