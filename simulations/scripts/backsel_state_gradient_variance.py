#!/usr/bin/env python
"""
Direct test of whether witness-function backsel selection reduces gradient
variance relative to uniform selection, at realistic points along a guidance
trajectory -- as opposed to run_backsel_witness_sweep.py's end-to-end metric
(L2/MMD after a full run), this isolates ONE guidance step at a time.

Design (mirrors gradient_variance_vs_unroll_depth.py's "freeze state, redraw
the stochastic pipeline many times" pattern, but compares backsel rules
instead of unroll depth):
  - Pick a handful of representative states: a few diffusion steps (early/
    mid/late in the denoising trajectory -- difficulty differs across steps)
    x a few trajectory seeds (to land in different regions of the target
    distribution). Each (seed, step) pair is one state.
  - States come from UNGUIDED (zeta=0) trajectories of model_uncond, so the
    state a rule is evaluated at never depends on which rule produced it --
    this is what a real guidance loop's frozen point looks like independent
    of the choice under test.
  - At each state independently: freeze x0_sample (and t) completely, then
    redraw --n_redraws times, for 'uniform', 'witness', AND 'full' (no
    subsampling -- the baseline both rules should be compared against):
    fresh target_samples from model_cond, fresh mog_samples, a fresh backsel
    subsample (skipped for 'full'), and the resulting gradient
    d(MMD loss)/d(x0_sample).
  - Report, per state and per rule: mean_grad, Var(grad) (trace of the
    empirical covariance across the redraws), and Var(grad)/||mean_grad||^2.
  - Average those normalized-variance numbers across all states too, but the
    per-state numbers are the primary output, not the average alone -- a
    single mean can hide a rule that only wins at some states.

Also reports, per state and per rule, BIAS against a TRUE population
reference gradient -- drawn from the exact analytic conditional GMM given
the frozen x0_sample (known mu_list/Sigma_list/alpha, no network forward,
same construction as Optimization.py's diag_steps reference; see
--grad_ref_n), independent of which selection rule is under test:
bias_vs_ref = ||mean_grad[rule] - mean_grad_ref||. This is the answer to
"which rule's gradient is actually closer to the truth", as opposed to
normalized_variance, which only measures redraw-to-redraw spread around a
rule's OWN mean and says nothing about whether that mean is biased.

--normalize_by_k_frac rescales the raw subsampled gradient by 1/k_frac
(k_frac = backsel_k/nsamples) for uniform/witness before both the variance
and bias_vs_ref stats are computed -- a no-op for 'full' and for 'ref'
(neither is subsampled). This matters because an unnormalized subsampled
gradient is systematically SMALLER in magnitude than the full-n gradient
(MMDLoss averages, not sums), so shrinkage toward zero can mechanically
produce a smaller bias_vs_ref even with no real selection-quality signal.
Run this script once with --normalize_by_k_frac omitted and once with it
set (same --experiment/--state_seeds/--step_fracs/--nsamples/--k_frac/
--seed, so the frozen states and redraw streams are identical) to tell
apart a real Witness-vs-Uniform gap from that shrinkage artifact:
  - Witness only beats Uniform on bias_vs_ref in the UNNORMALIZED run
    -> shrinkage artifact (Or's hypothesis), not real selection quality.
  - Witness still beats Uniform after normalizing
    -> the witness score is actually finding lower-bias-contributing
       samples, not just producing a different effective shrinkage.

Usage:
    python backsel_state_gradient_variance.py --experiment 5D_cond_1D
    python backsel_state_gradient_variance.py --experiment 5D_cond_1D --normalize_by_k_frac
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
from dist_utils import generate_mog_samples_not_differentiable, generate_mog_samples
from witness_utils import apply_backsel


def capture_states(model_uncond, seed, step_fracs, device):
    """
    Run one UNGUIDED (zeta=0) DDIM trajectory of model_uncond, and capture
    (t, x0_sample) at the diffusion steps closest to the requested fractions
    of the trajectory (0.0 = earliest/noisiest step just below
    diffusion_steps-1, 1.0 = latest/cleanest step just above 0). Matches
    exactly the x0_sample = pred_x0 + r_t * randn expression optimize_LGD
    uses, so captured states look like real guidance-loop states.
    """
    experiment_utils.set_run_seed(seed, 0)
    T = model_uncond.diffusion_steps
    all_steps = list(range(T - 1, 0, -1))  # descending, matches optimize_LGD's pbar
    target_steps = {all_steps[int(round(f * (len(all_steps) - 1)))] for f in step_fracs}

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=False).unsqueeze(0)
    captured = {}
    for t in all_steps:
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
        if t in target_steps:
            current_var = model_uncond.betas[t].to(device)
            r_t = current_var / torch.sqrt(1 + current_var ** 2)
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            captured[t] = x0_sample.detach().clone()
        x_t = x_t_minus_1.detach().clone()
    return captured  # {t: x0_sample}


def grad_stats_for_rule(x0_sample, model_cond, CM, mog_means, mog_variances, weights,
                         nsamples, backsel_k, rule, witness_floor, n_redraws, device,
                         base_seed, mmd_loss, normalize_by_k_frac=False):
    """
    Redraw the sampling + backsel pipeline n_redraws times at this ONE frozen
    x0_sample, for one rule ('uniform' | 'witness' | 'full' -- 'full' skips
    backsel entirely, using every row of target_samples, as the no-
    subsampling baseline both rules should be compared against). Returns
    (grads [n_redraws, condition_on], mean_grad, variance_trace,
    normalized_variance).

    normalize_by_k_frac: rescale each redraw's raw gradient by 1/k_frac
    (k_frac = backsel_k/nsamples) before it's used for either variance or
    mean_grad -- mirrors Optimization.py's Horvitz-Thompson-style rescaling,
    applied at the same point (right after the raw gradient, before any
    downstream comparison). No-op for rule='full' (backsel_k is not used).
    """
    k_frac = min(int(backsel_k), nsamples) / nsamples if backsel_k is not None else 1.0
    grads = []
    for r in range(n_redraws):
        experiment_utils.set_run_seed(base_seed, r)
        generator = torch.Generator().manual_seed(base_seed * 100_000 + r)

        x_leaf = x0_sample.clone().detach().to(device).requires_grad_(True)
        condition = x_leaf.view(1, -1).repeat(nsamples, 1)
        target_samples, _, _ = model_cond.sample(nsamples=nsamples, condition_x=condition, device=device)
        if not CM:
            target_samples = target_samples[:, model_cond.condition_on:]
        mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

        if rule == "full":
            batch = target_samples
        else:
            batch, _ = apply_backsel(
                target_samples, mog_samples, backsel_k, rule=rule,
                witness_floor=witness_floor, generator=generator, replacement=False,
            )
        loss = mmd_loss(batch, mog_samples)
        loss.backward()
        grad = x_leaf.grad.detach()
        if normalize_by_k_frac and rule != "full":
            grad = grad / max(k_frac, 1e-12)
        grads.append(grad.cpu().numpy().copy())

    grads = np.stack(grads, axis=0)
    mean_grad = grads.mean(axis=0)
    centered = grads - mean_grad
    variance_trace = float(np.mean(np.sum(centered ** 2, axis=1)))
    mean_grad_norm_sq = float(np.sum(mean_grad ** 2))
    normalized_variance = variance_trace / (mean_grad_norm_sq + 1e-12)

    return {
        "mean_grad": mean_grad.tolist(),
        "mean_grad_norm": float(np.sqrt(mean_grad_norm_sq)),
        "variance_trace": variance_trace,
        "normalized_variance": normalized_variance,
        "grads": grads.tolist(),
    }


def reference_grad_for_state(x0_sample, mu_list, Sigma_list, alpha, mog_means, mog_variances,
                             weights, nsamples, grad_ref_n, n_redraws, device, base_seed, mmd_loss):
    """
    TRUE population reference gradient at this ONE frozen x0_sample: redraw
    grad_ref_n samples from the EXACT analytic conditional GMM given
    x0_sample (known mu_list/Sigma_list/alpha, not model_cond's learned
    approximation -- no network forward, cheap even at large grad_ref_n) and
    a fresh target-side mog_samples draw, n_redraws times, same sign
    convention as grad_stats_for_rule (loss = +MMD, loss.backward()) so the
    two are directly comparable. This is independent of rule/backsel_k by
    construction -- computed once per state, not once per rule.
    """
    grads = []
    for r in range(n_redraws):
        experiment_utils.set_run_seed(base_seed, r)

        x_leaf = x0_sample.clone().detach().to(device).requires_grad_(True)
        condi_mu, condi_sigma = dist_utils.compute_conditionals(mu_list, Sigma_list, x_leaf.view(-1))
        condi_mu = condi_mu.squeeze(-1)
        condi_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_leaf.view(-1))
        ref_samples = generate_mog_samples(grad_ref_n, condi_mu, condi_sigma, condi_alpha, device=device)
        mog_samples = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

        loss = mmd_loss(ref_samples, mog_samples)
        loss.backward()
        grads.append(x_leaf.grad.detach().cpu().numpy().copy())

    grads = np.stack(grads, axis=0)
    mean_grad = grads.mean(axis=0)
    mean_grad_norm_sq = float(np.sum(mean_grad ** 2))
    return {"mean_grad": mean_grad.tolist(), "mean_grad_norm": float(np.sqrt(mean_grad_norm_sq))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    p.add_argument("--methods", nargs="+", choices=["LGD", "LGD-CM"], default=["LGD"])
    p.add_argument("--state_seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Trajectory seeds to capture states from (2-3 recommended).")
    p.add_argument("--step_fracs", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                   help="Fractions along the denoising trajectory to capture a state at "
                        "(0.0=earliest/noisiest, 1.0=latest/cleanest). 3 values = early/mid/late.")
    p.add_argument("--nsamples", type=int, default=250)
    p.add_argument("--k_frac", type=float, default=0.2, help="backsel_k / nsamples for both rules.")
    p.add_argument("--witness_floor", type=float, default=0.3)
    p.add_argument("--n_redraws", type=int, default=200,
                   help="Independent redraws of the sampling+backsel pipeline per state per rule.")
    p.add_argument("--seed", type=int, default=42, help="Global seed / checkpoint selector.")
    p.add_argument("--redraw_seed_offset", type=int, default=1000,
                   help="Per-state redraw seeds are base_seed = seed*offset + state_index, "
                        "keeping every state's redraw stream independent.")
    p.add_argument("--normalize_by_k_frac", action="store_true",
                   help="Rescale uniform/witness's raw subsampled gradient by 1/k_frac before "
                        "computing variance/bias_vs_ref stats (no-op for full/ref). Run once "
                        "with this OFF and once ON (same experiment/state_seeds/step_fracs/"
                        "nsamples/k_frac/seed) to separate a real Witness-vs-Uniform bias_vs_ref "
                        "gap from a gradient-shrinkage artifact -- see this script's docstring.")
    p.add_argument("--grad_ref_n", type=int, default=2000,
                   help="Sample size for the TRUE/population reference gradient (drawn from the "
                        "exact analytic conditional GMM, no network forward -- can be large "
                        "without real cost).")
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
    print(f"[StateVar] experiment={args.experiment} methods={args.methods} "
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

    for method in args.methods:
        cond_model, CM_flag = method_models[method]
        nsamples_ref = args.nsamples if not CM_flag else args.nsamples
        backsel_k = max(1, round(args.k_frac * nsamples_ref))

        states = []
        for si, seed in enumerate(args.state_seeds):
            captured = capture_states(model_uncond, seed, args.step_fracs, device)
            for t_val, x0_sample in captured.items():
                states.append({"state_seed": seed, "t": t_val, "x0_sample": x0_sample})

        print(f"[StateVar] {method}: captured {len(states)} states "
              f"({len(args.state_seeds)} seeds x {len(set(round(f, 6) for f in args.step_fracs))} step_fracs)")

        RULE_SEED_OFFSET = {"uniform": 0, "witness": 500, "full": 1000, "ref": 1500}
        state_results = []
        per_rule_normalized_variances = {"uniform": [], "witness": [], "full": []}
        per_rule_bias_vs_ref = {"uniform": [], "witness": [], "full": []}
        for si, state in enumerate(states):
            entry = {"state_index": si, "state_seed": state["state_seed"], "t": state["t"],
                     "x0_sample": state["x0_sample"].tolist(), "rules": {}}
            base_seed = args.seed * args.redraw_seed_offset + si

            ref_stats = reference_grad_for_state(
                state["x0_sample"], mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                args.nsamples, args.grad_ref_n, args.n_redraws, device,
                base_seed + RULE_SEED_OFFSET["ref"], mmd_loss,
            )
            entry["ref"] = ref_stats
            ref_mean_grad = np.array(ref_stats["mean_grad"])

            for rule in ("uniform", "witness", "full"):
                stats = grad_stats_for_rule(
                    state["x0_sample"], cond_model, CM_flag, mog_means, mog_variances, weights,
                    args.nsamples, backsel_k, rule, args.witness_floor, args.n_redraws, device,
                    base_seed + RULE_SEED_OFFSET[rule], mmd_loss,
                    normalize_by_k_frac=args.normalize_by_k_frac,
                )
                bias_vs_ref = float(np.linalg.norm(np.array(stats["mean_grad"]) - ref_mean_grad))
                stats["bias_vs_ref"] = bias_vs_ref
                stats["bias_vs_ref_normalized"] = bias_vs_ref / (ref_stats["mean_grad_norm"] + 1e-12)
                entry["rules"][rule] = stats
                per_rule_normalized_variances[rule].append(stats["normalized_variance"])
                per_rule_bias_vs_ref[rule].append(bias_vs_ref)
            state_results.append(entry)
            print(f"  [{method}] state {si} (seed={state['state_seed']} t={state['t']}) | "
                  f"norm_var uniform={entry['rules']['uniform']['normalized_variance']:.6e} "
                  f"witness={entry['rules']['witness']['normalized_variance']:.6e} "
                  f"full={entry['rules']['full']['normalized_variance']:.6e} || "
                  f"bias_vs_ref uniform={entry['rules']['uniform']['bias_vs_ref']:.6e} "
                  f"witness={entry['rules']['witness']['bias_vs_ref']:.6e} "
                  f"full={entry['rules']['full']['bias_vs_ref']:.6e}")

        averaged = {
            rule: {
                "mean_normalized_variance": float(np.mean(per_rule_normalized_variances[rule])),
                "std_normalized_variance": float(np.std(per_rule_normalized_variances[rule])),
                "per_state_normalized_variance": per_rule_normalized_variances[rule],
                "mean_bias_vs_ref": float(np.mean(per_rule_bias_vs_ref[rule])),
                "std_bias_vs_ref": float(np.std(per_rule_bias_vs_ref[rule])),
                "per_state_bias_vs_ref": per_rule_bias_vs_ref[rule],
            }
            for rule in ("uniform", "witness", "full")
        }
        witness_vs_uniform_better_count = sum(
            1 for s in state_results
            if s["rules"]["witness"]["normalized_variance"] < s["rules"]["uniform"]["normalized_variance"]
        )
        witness_vs_full_better_count = sum(
            1 for s in state_results
            if s["rules"]["witness"]["normalized_variance"] < s["rules"]["full"]["normalized_variance"]
        )
        uniform_vs_full_better_count = sum(
            1 for s in state_results
            if s["rules"]["uniform"]["normalized_variance"] < s["rules"]["full"]["normalized_variance"]
        )
        witness_vs_uniform_bias_better_count = sum(
            1 for s in state_results
            if s["rules"]["witness"]["bias_vs_ref"] < s["rules"]["uniform"]["bias_vs_ref"]
        )
        witness_vs_full_bias_better_count = sum(
            1 for s in state_results
            if s["rules"]["witness"]["bias_vs_ref"] < s["rules"]["full"]["bias_vs_ref"]
        )
        uniform_vs_full_bias_better_count = sum(
            1 for s in state_results
            if s["rules"]["uniform"]["bias_vs_ref"] < s["rules"]["full"]["bias_vs_ref"]
        )

        out = {
            "experiment": args.experiment,
            "method": method,
            "seed": args.seed,
            "nsamples": args.nsamples,
            "k_frac": args.k_frac,
            "backsel_k": backsel_k,
            "witness_floor": args.witness_floor,
            "n_redraws": args.n_redraws,
            "state_seeds": args.state_seeds,
            "step_fracs": args.step_fracs,
            "normalize_by_k_frac": args.normalize_by_k_frac,
            "grad_ref_n": args.grad_ref_n,
            "states": state_results,
            "averaged": averaged,
            "witness_vs_uniform_better_count": witness_vs_uniform_better_count,
            "witness_vs_full_better_count": witness_vs_full_better_count,
            "uniform_vs_full_better_count": uniform_vs_full_better_count,
            "witness_vs_uniform_bias_better_count": witness_vs_uniform_bias_better_count,
            "witness_vs_full_bias_better_count": witness_vs_full_bias_better_count,
            "uniform_vs_full_bias_better_count": uniform_vs_full_bias_better_count,
            "n_states": len(state_results),
        }

        # nsamples, k_frac, and norm{on,off} are included so that runs differing in any of them
        # (e.g. comparing nsamples=250 vs nsamples=500, or the un-normalized vs. normalized
        # gradient-shrinkage comparison) land in separate files instead of overwriting each other.
        norm_tag = "normON" if args.normalize_by_k_frac else "normOFF"
        out_path = os.path.join(
            results_dir,
            f"{args.experiment}_backsel_state_variance_{method}_n{args.nsamples}_"
            f"kfrac{args.k_frac:g}_{norm_tag}_seed{args.seed}.json",
        )
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[StateVar] {method}: saved {out_path}")
        print(f"[StateVar] {method}: mean normalized variance -- "
              f"uniform={averaged['uniform']['mean_normalized_variance']:.6e} "
              f"witness={averaged['witness']['mean_normalized_variance']:.6e} "
              f"full={averaged['full']['mean_normalized_variance']:.6e} "
              f"(witness < uniform at {witness_vs_uniform_better_count}/{len(state_results)} states; "
              f"witness < full at {witness_vs_full_better_count}/{len(state_results)}; "
              f"uniform < full at {uniform_vs_full_better_count}/{len(state_results)})")
        print(f"[StateVar] {method}: mean bias_vs_ref (normalize_by_k_frac={args.normalize_by_k_frac}) -- "
              f"uniform={averaged['uniform']['mean_bias_vs_ref']:.6e} "
              f"witness={averaged['witness']['mean_bias_vs_ref']:.6e} "
              f"full={averaged['full']['mean_bias_vs_ref']:.6e} "
              f"(witness < uniform at {witness_vs_uniform_bias_better_count}/{len(state_results)} states; "
              f"witness < full at {witness_vs_full_bias_better_count}/{len(state_results)}; "
              f"uniform < full at {uniform_vs_full_bias_better_count}/{len(state_results)})")


if __name__ == "__main__":
    main()
