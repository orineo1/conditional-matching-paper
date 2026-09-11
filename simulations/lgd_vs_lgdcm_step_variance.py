#!/usr/bin/env python
"""
lgd_vs_lgdcm_step_variance.py — per-step gradient variance/accuracy of LGD's
inner sampler (K-step DDIM unroll through the conditional diffusion model)
vs. LGD-CM's inner sampler (the consistency model's own ~14-step multistep
sampling procedure -- see ConsistencyModeliCT.sample: far fewer, coarser
jump-and-refine steps than LGD's fine-grained DDIM unroll, not a single
network call), along real
optimization trajectories rather than a single hand-picked point.

Uses the "freeze a state, redraw the sampler's noise many times" methodology
from src/grad_variance_utils.py (normalized_variance and
dist_to_ref_normalized metrics, true/population reference-gradient
machinery) across EVERY step of one or more full trajectories, comparing the
two inner samplers directly rather than at a single hand-picked point.

Design:
  - States are captured from --n_trajectories independent UNGUIDED (zeta=0)
    trajectories of model_uncond, one state per outer diffusion step (every
    --step_stride-th step) -- unguided so the states themselves don't already
    depend on which inner sampler produced them (see
    backsel_state_gradient_variance.py on other branches for the same
    convention). Each trajectory uses a different seed.
  - At each captured (trajectory, step, x0_sample): draw --n_redraws times,
    independently for each of the two inner samplers:
      LGD:    a --k_lgd-step DDIM unroll through model_cond (default: the
              full schedule, matching what real LGD guidance actually runs
              each outer step).
      LGD-CM: one draw via the consistency model's own multistep sampling
              (ConsistencyModeliCT.sample -- ~14 network calls at default
              `ts`, each a coarse jump-and-refine step, not one forward pass).
    against the SAME fixed target-sample set (drawn once per state from the
    exact analytic conditional GMM at that state's x -- not model_cond's own
    approximation), and compute the MMD loss + its gradient w.r.t. x.
  - Per (trajectory, step, method): normalized_variance = Var(grad)/||mean_grad||^2
    across the redraws, and dist_to_ref_normalized = ||mean_grad - grad_ref|| /
    ||grad_ref||, where grad_ref is the TRUE population gradient at that state
    (closed-form, no network forward -- see src/grad_variance_utils.py).
  - Averaged (mean + std + SEM) across --n_trajectories at each step index,
    separately for LGD and LGD-CM -- this is the "run e.g. 10 trajectories,
    report the mean of each metric per step" output; per-trajectory raw rows
    are also kept in the output JSON so nothing is thrown away.

Cost warning: this is expensive by construction -- n_trajectories x
n_steps_sampled x n_redraws x 2 methods forward+backward passes, and LGD's
K-step unroll (default: the full schedule) is itself K network calls per
redraw. Use --smoke first, and raise --step_stride / lower --n_redraws /
--n_trajectories before scaling up.

Usage:
    python lgd_vs_lgdcm_step_variance.py --experiment_name 10D_cond_1D --smoke
    python lgd_vs_lgdcm_step_variance.py --experiment_name 10D_cond_1D \
        --n_trajectories 10 --step_stride 10 --n_redraws 30
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "src"))

import grad_variance_utils

ARCH = {
    "2D_cond_1D":  dict(nblocks=3, nunits=128, diffusion_steps=100, condition_on=1),
    "5D_cond_1D":  dict(nblocks=6, nunits=512, diffusion_steps=100, condition_on=4),
    "10D_cond_1D": dict(nblocks=8, nunits=512, diffusion_steps=100, condition_on=9),
}


def capture_trajectory_states(model_uncond, seed, step_stride, condition_on, device, experiment_utils):
    """One UNGUIDED (zeta=0) DDIM trajectory of model_uncond, capturing
    x0_sample = pred_x0 + r_t * randn (matches optimize_LGD's own expression,
    so captured states look like real guidance-loop states) at every
    step_stride-th outer step, always including the last (t=1). Returns a
    list of (step_index, t, x0_sample) in trajectory order (noisy -> clean).
    """
    experiment_utils.set_run_seed(seed, 0)
    T = model_uncond.diffusion_steps
    all_steps = list(range(T - 1, 0, -1))  # descending, matches optimize_LGD's pbar

    x_t = torch.zeros(condition_on, device=device, requires_grad=False).unsqueeze(0)
    captured = []
    for step_idx, t in enumerate(all_steps):
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(x_t, t, condition_x=None, device=device, eta=0.0)
        if step_idx % step_stride == 0 or step_idx == len(all_steps) - 1:
            current_var = model_uncond.betas[t].to(device)
            r_t = current_var / torch.sqrt(1 + current_var ** 2)
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            captured.append((step_idx, t, x0_sample.detach().clone()))
        x_t = x_t_minus_1.detach().clone()
    return captured


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment_name", type=str, default="10D_cond_1D", choices=list(ARCH.keys()))
    p.add_argument("--n_trajectories", type=int, default=10,
                   help="Independent unguided trajectories (different seeds) to average over.")
    p.add_argument("--step_stride", type=int, default=10,
                   help="Capture (and analyze) every step_stride-th outer diffusion step "
                        "instead of every single one -- controls cost. 1 = every step.")
    p.add_argument("--n_redraws", type=int, default=30,
                   help="Independent redraws of each inner sampler per (trajectory, step).")
    p.add_argument("--nsamples", type=int, default=250,
                   help="Batch size for the inner MMD estimator (both methods).")
    p.add_argument("--n_target", type=int, default=None,
                   help="Fixed target-sample count per state (defaults to --nsamples).")
    p.add_argument("--k_lgd", type=int, default=None,
                   help="Unroll depth for LGD's inner DDIM sampler (default: the experiment's "
                        "full diffusion_steps, matching what real LGD guidance actually runs "
                        "each outer step).")
    p.add_argument("--grad_ref_n", type=int, default=2000,
                   help="Sample size for the TRUE/population reference gradient at each state "
                        "(closed-form, no network forward -- cheap even at this size).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; also selects which pretrained checkpoints to load. "
                        "Trajectory i uses seed + i.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for a quick sanity check: 2 trajectories, step_stride=40, "
                        "5 redraws, nsamples=32.")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_trajectories = 2
        args.step_stride = 40
        args.n_redraws = 5
        args.nsamples = 32

    ARCH_CFG = ARCH[args.experiment_name]
    n_target = args.n_target or args.nsamples
    k_lgd = args.k_lgd or ARCH_CFG["diffusion_steps"]
    condition_on = ARCH_CFG["condition_on"]

    BASE_DIR = _HERE
    PARAMS_DIR = os.path.join(BASE_DIR, "params")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", args.experiment_name)
    RESULTS_DIR = args.output_dir or os.path.join(BASE_DIR, "results", args.experiment_name, "lgd_vs_lgdcm_step_variance")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import experiment_utils
    import dist_utils
    import Diffusion
    from ConsistencyModels import ConsistencyModeliCT
    from LossFunctions import MMDLoss, RBF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StepVar] experiment={args.experiment_name} n_trajectories={args.n_trajectories} "
          f"step_stride={args.step_stride} n_redraws={args.n_redraws} k_lgd={k_lgd} device={device}")

    experiment_utils.set_global_seed(args.seed)

    loaded = experiment_utils.load_gmm_params(PARAMS_DIR, args.experiment_name)
    if loaded is None:
        raise FileNotFoundError(f"No GMM params found under {PARAMS_DIR} for '{args.experiment_name}'.")
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
    mu_list = [mu.float() for mu in mu_list]
    Sigma_list = [cov.float() for cov in Sigma_list]
    alpha = alpha.float()
    nfeatures_full = mu_list[0].shape[0]

    # ── models: uncond (state capture), cond diffusion (LGD), CM (LGD-CM) ───
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=condition_on, nblocks=ARCH_CFG["nblocks"], nunits=ARCH_CFG["nunits"],
        condition=False, diffusion_steps=ARCH_CFG["diffusion_steps"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_uncond, "Diffusion_uncond", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load/download the pretrained Diffusion_uncond checkpoint.")
    model_uncond.to(device).eval()

    model_cond = Diffusion.DiffusionModel(
        nfeatures=nfeatures_full, nblocks=ARCH_CFG["nblocks"], nunits=ARCH_CFG["nunits"],
        condition=True, condition_on=condition_on, diffusion_steps=ARCH_CFG["diffusion_steps"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_cond, "Diffusion_cond", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load/download the pretrained Diffusion_cond checkpoint.")
    model_cond.to(device).eval()

    model_cm = ConsistencyModeliCT(
        nfeatures=nfeatures_full - condition_on, condition_on=condition_on,
        nunits=ARCH_CFG["nunits"], depth=ARCH_CFG["nblocks"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_cm, "CM", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load/download the pretrained CM checkpoint.")
    model_cm.to(device).eval()

    mmd_loss = MMDLoss(kernel=RBF())

    # ── main loop: trajectories x captured steps x {LGD, LGD-CM} ────────────
    raw_rows = []  # one row per (trajectory, step, method)
    for traj_i in range(args.n_trajectories):
        traj_seed = args.seed + traj_i
        states = capture_trajectory_states(
            model_uncond, traj_seed, args.step_stride, condition_on, device, experiment_utils
        )
        print(f"[StepVar] trajectory {traj_i} (seed={traj_seed}): {len(states)} states captured")

        for step_idx, t, x0_sample in states:
            x_fixed = x0_sample.view(-1)

            # Fixed target samples for this state: drawn once from the exact analytic
            # conditional GMM at x_fixed (ground truth, not model_cond's approximation).
            target_samples = grad_variance_utils.analytic_target_samples(
                dist_utils, mu_list, Sigma_list, alpha, x_fixed, n_target, device
            )

            # TRUE/population reference gradient at this state (closed-form, independent
            # of which inner sampler is being evaluated).
            grad_ref, grad_ref_norm = grad_variance_utils.true_reference_gradient(
                dist_utils, mmd_loss, mu_list, Sigma_list, alpha, x_fixed, target_samples,
                args.grad_ref_n, device,
            )

            def lgd_sampler(x_leaf, _k=k_lgd):
                cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)
                _, y, _ = grad_variance_utils.ddim_sample_kstep(model_cond, args.nsamples, cond, _k, device)
                return y

            def lgdcm_sampler(x_leaf):
                cond = x_leaf.view(1, -1).repeat(args.nsamples, 1)
                y, _, _ = model_cm.sample(nsamples=args.nsamples, condition_x=cond, device=device)
                return y

            for method_name, sampler_fn, seed_offset in (
                ("LGD", lgd_sampler, 0),
                ("LGD-CM", lgdcm_sampler, 500),
            ):
                mean_grad, stats, _ = grad_variance_utils.redraw_grad_stats(
                    sampler_fn, x_fixed, target_samples, args.n_redraws,
                    traj_seed * 100_000 + step_idx + seed_offset,
                    mmd_loss, experiment_utils, device,
                )
                dist_to_ref, dist_to_ref_normalized = grad_variance_utils.dist_to_ref_stats(
                    mean_grad, grad_ref, grad_ref_norm
                )
                raw_rows.append({
                    "trajectory": traj_i, "step_index": step_idx, "t": t, "method": method_name,
                    **stats,
                    "dist_to_ref": dist_to_ref, "dist_to_ref_normalized": dist_to_ref_normalized,
                    "grad_ref_norm": grad_ref_norm,
                })
            print(f"  [traj {traj_i}] step {step_idx:>3} (t={t:>3}) | "
                  f"LGD nv={raw_rows[-2]['normalized_variance']:.4e} "
                  f"var/dim={raw_rows[-2]['variance_trace_per_dim']:.4e} "
                  f"dist={raw_rows[-2]['dist_to_ref_normalized']:.4f} | "
                  f"LGD-CM nv={raw_rows[-1]['normalized_variance']:.4e} "
                  f"var/dim={raw_rows[-1]['variance_trace_per_dim']:.4e} "
                  f"dist={raw_rows[-1]['dist_to_ref_normalized']:.4f}",
                  flush=True)

    # ── aggregate: mean/std/sem per (step_index, method) across trajectories ─
    step_indices = sorted(set(r["step_index"] for r in raw_rows))
    methods = ["LGD", "LGD-CM"]
    METRICS = ["normalized_variance", "dist_to_ref_normalized",
               "variance_trace", "variance_trace_per_dim", "mean_grad_norm"]
    aggregated = {m: {metric: [] for metric in METRICS} for m in methods}
    aggregated["step_index"] = step_indices
    aggregated["t"] = [next(r["t"] for r in raw_rows if r["step_index"] == s) for s in step_indices]

    for s in step_indices:
        for m in methods:
            vals_by_metric = {
                metric: [r[metric] for r in raw_rows if r["step_index"] == s and r["method"] == m]
                for metric in METRICS
            }
            for metric in METRICS:
                vals = np.array(vals_by_metric[metric])
                aggregated[m][metric].append({
                    "mean": float(vals.mean()), "std": float(vals.std()),
                    "sem": float(vals.std() / max(1, np.sqrt(len(vals) - 1)) if len(vals) > 1 else 0.0),
                    "n": len(vals),
                })

    out = {
        "experiment": args.experiment_name,
        "seed": args.seed,
        "n_trajectories": args.n_trajectories,
        "step_stride": args.step_stride,
        "n_redraws": args.n_redraws,
        "nsamples": args.nsamples,
        "n_target": n_target,
        "k_lgd": k_lgd,
        "grad_ref_n": args.grad_ref_n,
        "raw_rows": raw_rows,
        "aggregated": aggregated,
    }
    out_path = os.path.join(RESULTS_DIR, f"{args.experiment_name}_lgd_vs_lgdcm_step_variance_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[StepVar] Saved results to {out_path}")


if __name__ == "__main__":
    main()
