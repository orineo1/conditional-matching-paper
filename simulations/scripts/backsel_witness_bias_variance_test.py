#!/usr/bin/env python
"""
Unbiased bias/variance/MSE test for the backsel selection rules (All /
Uniform / Witness), at a handful of real trajectory states -- the properly-
rescaled counterpart to backsel_state_gradient_variance.py's raw
normalized-variance comparison.

Why a separate script: backsel_state_gradient_variance.py's 'uniform' and
'witness' gradients are NOT unbiased estimators of the full-batch gradient
-- apply_backsel keeps only k rows differentiable (others detached, but
still counted in the loss VALUE), so the mean over redraws systematically
UNDERSHOOTS the full gradient by roughly a factor k/n, and a rule that
"wins" on variance there can just be an artifact of that shrinkage (smaller
vectors have smaller variance almost for free). This script removes that
confound: every selected row's gradient contribution is rescaled *before*
summing (via a zero-cost custom-autograd gradient-scale, so the forward
value seen by the loss is untouched) so that in expectation each rule's
estimator equals the true per-sample-summed gradient:

    Uniform: sample k of n rows uniformly without replacement, scale each
             kept row's gradient by n/k          (exact Horvitz-Thompson
             correction for simple random sampling without replacement).
    Witness: sample k of n rows with replacement, probability p_i propto
             the witness-floor-mixed |scores| distribution
             select_backsel_mask already uses; scale kept row i's gradient
             by 1/(n * p_i) each time it's drawn (importance-sampling
             correction) and let autograd sum the (possibly repeated) rows.
    All:     no subsampling; deterministic baseline, scale = 1.

Every rule is then compared to TWO references at each frozen state:
    true      -- the population/closed-form gradient (grad_variance_utils.
                 true_reference_gradient: MMD against the EXACT analytic
                 conditional GMM, not the network's approximation of it).
    fullbatch -- the empirical gradient using every row of target_samples,
                 no subsampling (answers "how much do we lose vs using
                 everything", separately from "how much do we lose vs
                 truth").

Reports, per state x per rule x per reference: bias = ||mean_hat - ref||,
variance (trace of the empirical covariance across redraws), MSE (checked
against bias^2 + variance), and cosine similarity (mean +/- std across
redraws) -- not just the vector norm, since a direction that's off by a
large angle is a bad descent step even when its magnitude matches.

Usage:
    python backsel_witness_bias_variance_test.py --experiment 5D_cond_1D
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from gmm_experiment_setup import EXPERIMENT_CONFIGS, load_or_generate_gmm_params, load_or_train_models
import experiment_utils
import dist_utils
from LossFunctions import MMDLoss, RBF
from dist_utils import generate_mog_samples_not_differentiable
from witness_utils import compute_witness_scores, select_backsel_mask
from grad_variance_utils import true_reference_gradient
from backsel_state_gradient_variance import capture_states


# ------------------------------------------------------------------------
# Zero-cost gradient rescale: forward value is untouched (so the loss still
# sees the real sample), only the backward-pass gradient through this row
# is multiplied by `scale`.
# ------------------------------------------------------------------------
class _ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (scale,) = ctx.saved_tensors
        return grad_out * scale.view(-1, *([1] * (grad_out.dim() - 1))), None


def scale_grad(x, scale):
    return _ScaleGrad.apply(x, scale.to(x.dtype).to(x.device))


def rescaled_batch(samples, mask, weights):
    """Every row keeps its original VALUE (so the loss sees the same batch
    as the 'all' rule); only the backward-pass gradient through row i is
    scaled by weights[i] if selected, or zeroed out entirely otherwise."""
    scales = torch.where(mask, weights, torch.zeros_like(weights))
    return scale_grad(samples, scales)


def unbiased_gradient(x0_sample, model_cond, CM, mog_means, mog_variances, weights_gmm,
                       nsamples, backsel_k, rule, witness_floor, device, mmd_loss, generator):
    """One redraw: fresh target_samples + mog_samples + selection, one
    backward pass, returns the RESCALED gradient d(loss)/d(x0_sample) --
    an unbiased estimator of the full-batch gradient for 'uniform'/'witness',
    and exactly the full-batch gradient for 'all'."""
    x_leaf = x0_sample.clone().detach().to(device).requires_grad_(True)
    condition = x_leaf.view(1, -1).repeat(nsamples, 1)
    target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
    if not CM:
        target_samples = target_samples[:, model_cond.condition_on:]
    mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights_gmm)

    n = target_samples.shape[0]
    if rule == "all":
        batch = target_samples
    else:
        scores = compute_witness_scores(target_samples, mog_samples) if rule == "witness" else torch.zeros(n)
        mask, counts, probs = select_backsel_mask(
            scores, backsel_k, rule=rule, witness_floor=witness_floor,
            generator=generator, replacement=(rule == "witness"),
        )
        mask = mask.to(device)
        if rule == "uniform":
            # exact Horvitz-Thompson correction for SRSWOR of size k from n
            row_weight = torch.full((n,), n / backsel_k, dtype=torch.float32)
        else:  # witness, sampled WITH replacement: importance-sampling correction,
            # counts[i]>1 means row i was drawn more than once -- each draw
            # contributes its own 1/(n*p_i) term, so weight by counts, not just mask.
            probs = probs.clamp_min(1e-12)
            row_weight = counts.to(torch.float32) / (n * probs)
        batch = rescaled_batch(target_samples, mask, row_weight.to(device))

    loss = mmd_loss(batch, mog_samples)
    grad = torch.autograd.grad(loss, x_leaf)[0]
    return grad.detach().cpu().numpy().copy()


def stats_vs_ref(hats, ref):
    """hats: [n_redraws, dim]; ref: [dim]. Returns bias/var/mse/cosine stats."""
    mean_hat = hats.mean(axis=0)
    bias = float(np.linalg.norm(mean_hat - ref))
    var = float(np.mean(np.sum((hats - mean_hat) ** 2, axis=1)))
    mse = float(np.mean(np.sum((hats - ref) ** 2, axis=1)))
    denom = (np.linalg.norm(hats, axis=1) * (np.linalg.norm(ref) + 1e-12)) + 1e-12
    cos = (hats @ ref) / denom
    return {
        "bias": bias,
        "variance": var,
        "mse": mse,
        "mse_check": bias ** 2 + var,  # should equal `mse` up to redraw noise
        "cos_mean": float(cos.mean()),
        "cos_std": float(cos.std()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD"])
    p.add_argument("--state_seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--step_fracs", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--nsamples", type=int, default=250)
    p.add_argument("--k_frac", type=float, default=0.2, help="backsel_k / nsamples for both rules.")
    p.add_argument("--witness_floor", type=float, default=0.3)
    p.add_argument("--n_redraws", type=int, default=200)
    p.add_argument("--grad_ref_n", type=int, default=2000, help="sample size for the TRUE/population reference gradient.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--redraw_seed_offset", type=int, default=1000)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--base_dir", default=None)
    args = p.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    base_dir = args.base_dir or os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
    params_dir = os.path.join(base_dir, "params")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.experiment)
    results_dir = os.path.join(base_dir, "results", args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BiasVar] experiment={args.experiment} methods={args.methods} "
          f"state_seeds={args.state_seeds} step_fracs={args.step_fracs} "
          f"k_frac={args.k_frac} n_redraws={args.n_redraws} device={device}")

    experiment_utils.set_global_seed(args.seed)
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = load_or_generate_gmm_params(
        cfg, params_dir, results_dir, args.experiment, args.seed,
    )
    model_uncond, model_cond, model_cm = load_or_train_models(
        cfg, mu_list, Sigma_list, alpha, checkpoint_dir, args.experiment,
        args.seed, device, args.force_retrain,
    )
    model_uncond.to(device).eval()
    model_cond.to(device).eval()
    if model_cm is not None:
        model_cm.to(device).eval()

    method_models = {"LGD": (model_cond, False), "LGD-CM": (model_cm, True)}
    mmd_loss = MMDLoss(kernel=RBF())
    rules = ["all", "uniform", "witness"]

    for method in args.methods:
        cond_model, CM_flag = method_models[method]
        backsel_k = max(1, round(args.k_frac * args.nsamples))

        states = []
        for seed in args.state_seeds:
            captured = capture_states(model_uncond, seed, args.step_fracs, device)
            for t_val, x0_sample in captured.items():
                states.append({"state_seed": seed, "t": t_val, "x0_sample": x0_sample})
        print(f"[BiasVar] {method}: {len(states)} states, backsel_k={backsel_k}/{args.nsamples}")

        RULE_SEED_OFFSET = {"all": 0, "uniform": 500, "witness": 1000}
        header = f"{'state':>5} {'rule':>8} {'ref':>10} {'bias':>9} {'var':>9} {'MSE':>9} {'cos_mean':>9} {'cos_std':>8}"
        print(header)
        print("-" * len(header))

        state_results = []
        for si, state in enumerate(states):
            base_seed = args.seed * args.redraw_seed_offset + si
            x0 = state["x0_sample"]

            target_ref_samples = generate_mog_samples_not_differentiable(
                args.grad_ref_n, mog_means, mog_variances, weights
            ).to(device)
            grad_true, grad_true_norm = true_reference_gradient(
                dist_utils, mmd_loss, mu_list, Sigma_list, alpha,
                x0, target_ref_samples, args.grad_ref_n, device,
            )

            entry = {"state_index": si, "state_seed": state["state_seed"], "t": state["t"], "rules": {}}
            for rule in rules:
                rng_seed = base_seed + RULE_SEED_OFFSET[rule]
                grads = []
                for r in range(args.n_redraws):
                    experiment_utils.set_run_seed(rng_seed, r)
                    generator = torch.Generator().manual_seed(rng_seed * 100_000 + r)
                    g = unbiased_gradient(
                        x0, cond_model, CM_flag, mog_means, mog_variances, weights,
                        args.nsamples, backsel_k, rule, args.witness_floor, device, mmd_loss, generator,
                    )
                    grads.append(g)
                grads = np.stack(grads, axis=0)
                entry["rules"][rule] = {"grads": grads}

            grad_full = entry["rules"]["all"]["grads"].mean(axis=0)  # 'all' rule IS the full-batch gradient
            for rule in rules:
                grads = entry["rules"][rule]["grads"]
                vs_true = stats_vs_ref(grads, grad_true)
                vs_full = stats_vs_ref(grads, grad_full)
                entry["rules"][rule] = {
                    "mean_grad": grads.mean(axis=0).tolist(),
                    "vs_true": vs_true,
                    "vs_fullbatch": vs_full,
                }
                for ref_name, s in (("true", vs_true), ("fullbatch", vs_full)):
                    print(f"{si:>5} {rule:>8} {ref_name:>10} "
                          f"{s['bias']:>9.4f} {s['variance']:>9.4f} {s['mse']:>9.4f} "
                          f"{s['cos_mean']:>9.4f} {s['cos_std']:>8.4f}")
            entry["grad_true_norm"] = grad_true_norm
            state_results.append(entry)

        out = {
            "experiment": args.experiment,
            "method": method,
            "seed": args.seed,
            "nsamples": args.nsamples,
            "k_frac": args.k_frac,
            "backsel_k": backsel_k,
            "witness_floor": args.witness_floor,
            "n_redraws": args.n_redraws,
            "grad_ref_n": args.grad_ref_n,
            "state_seeds": args.state_seeds,
            "step_fracs": args.step_fracs,
            "states": state_results,
        }
        out_path = os.path.join(
            results_dir,
            f"{args.experiment}_backsel_bias_variance_{method}_n{args.nsamples}_"
            f"kfrac{args.k_frac:g}_seed{args.seed}.json",
        )
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[BiasVar] {method}: saved {out_path}")


if __name__ == "__main__":
    main()
