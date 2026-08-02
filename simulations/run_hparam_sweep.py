#!/usr/bin/env python
"""
Hyperparameter sensitivity sweep for MLGD-F on the synthetic GMM experiments.

Runs the MLGD-F guidance loop (Optimization.optimize_LGD with CM=True) for a
single (nsamples, num_x_t) configuration -- the two Monte Carlo hyperparameters
that the paper sets differently per experiment scale (nsamples=250 / num_x_t=3
for the synthetic GMMs, nsamples=600-1500 / num_x_t=3-10 for MNIST) -- and
saves L2-GMM distance, L2 distance to x*, guidance loss, and wall-clock time
statistics to a JSON file. Intended to be launched many times in parallel
(e.g. as a SLURM job array) by run_hparam_sweep.sh, one job per grid point.

Reuses the pretrained checkpoints from the corresponding Exp_<name>.ipynb
notebook (downloaded automatically from HuggingFace on first use) and the
canonical GMM parameters already saved under simulations/params/ -- no
retraining happens here.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment_name", type=str, default="2D_cond_1D",
                    choices=["2D_cond_1D", "5D_cond_1D", "10D_cond_1D"])
    p.add_argument("--nsamples", type=int, default=250,
                    help="Monte Carlo samples for the MMD guidance loss (NSAMPLES_IN_OPTIM_FOR_MMD in the notebooks)")
    p.add_argument("--num_x_t", type=int, default=3,
                    help="Number of resampled x0 candidates averaged (logsumexp) per guidance step (NUM_X_T_LGD_CM)")
    p.add_argument("--n_attempts", type=int, default=25,
                    help="Number of independent optimization restarts (N_ATTEMP_OPTIM in the notebooks)")
    p.add_argument("--seed", type=int, default=42,
                    help="Global seed; also selects which pretrained checkpoint file to load")
    p.add_argument("--sweep_tag", type=str, default="hparam_sweep",
                    help="Subfolder under results/<experiment_name>/ used to group sweep outputs")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Override the results directory entirely")
    args = p.parse_args()

    # Architecture must match whichever notebook trained/downloaded the checkpoints for that experiment.
    ARCH = {
        "2D_cond_1D":  dict(nblocks_cm=3, nunits_cm=128, diffusion_steps=100, condition_on=1),
        "5D_cond_1D":  dict(nblocks_cm=6, nunits_cm=512, diffusion_steps=100, condition_on=4),
        "10D_cond_1D": dict(nblocks_cm=8, nunits_cm=512, diffusion_steps=100, condition_on=9),
    }[args.experiment_name]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARAMS_DIR = os.path.join(BASE_DIR, "params")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", args.experiment_name)
    RESULTS_DIR = args.output_dir or os.path.join(BASE_DIR, "results", args.experiment_name, args.sweep_tag)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    import experiment_utils
    import dist_utils
    import Diffusion
    import Optimization
    from ConsistencyModels import ConsistencyModeliCT

    env_info = experiment_utils.get_environment_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sweep] experiment={args.experiment_name} nsamples={args.nsamples} "
          f"num_x_t={args.num_x_t} n_attempts={args.n_attempts} seed={args.seed} device={device}")

    experiment_utils.set_global_seed(args.seed)

    loaded = experiment_utils.load_gmm_params(PARAMS_DIR, args.experiment_name)
    if loaded is None:
        raise FileNotFoundError(
            f"No GMM params found under {PARAMS_DIR} for '{args.experiment_name}'. "
            f"Run notebooks/Exp_{args.experiment_name}.ipynb once first so the canonical "
            f"GMM parameters and pretrained checkpoints exist."
        )
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
    mu_list = [mu.float() for mu in mu_list]
    Sigma_list = [cov.float() for cov in Sigma_list]
    alpha = alpha.float()

    condition_on = ARCH["condition_on"]
    nfeatures_y = mu_list[0].shape[0] - condition_on

    cm_model = ConsistencyModeliCT(
        nfeatures=nfeatures_y, condition_on=condition_on,
        nunits=ARCH["nunits_cm"], depth=ARCH["nblocks_cm"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        cm_model, "CM", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load or download the pretrained consistency model checkpoint.")
    cm_model.to(device)

    model_uncond = Diffusion.DiffusionModel(
        nfeatures=condition_on, nblocks=ARCH["nblocks_cm"], nunits=ARCH["nunits_cm"],
        condition=False, diffusion_steps=ARCH["diffusion_steps"],
    )
    if not experiment_utils.load_checkpoint_with_hf_fallback(
        model_uncond, "Diffusion_uncond", CHECKPOINT_DIR, args.experiment_name, args.seed, device
    ):
        raise RuntimeError("Could not load or download the pretrained unconditional diffusion model checkpoint.")
    model_uncond.to(device)

    l2_gmm_list, l2_x_list, times, final_losses, x_preds = [], [], [], [], []

    for i in range(args.n_attempts):
        run_seed = experiment_utils.set_run_seed(args.seed, i)
        start = time.time()
        best_x_t, _, final_loss = Optimization.optimize_LGD(
            model_uncond, cm_model, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha,
            nsamples=args.nsamples, loss="MMD", device=device,
            CM=True, FLAG=False, num_x_t=args.num_x_t,
        )
        elapsed = time.time() - start

        x_pred_t = best_x_t.float().view(-1).cpu()
        mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x_pred_t)
        w_pred = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_pred_t)
        l2_gmm = dist_utils.gmm_l2_distance(mu_pred, Sigma_pred, w_pred, mog_means, mog_variances, weights)
        l2_x = (x_pred_t - x_star.float().cpu()).pow(2).sum().sqrt().item()

        l2_gmm_list.append(l2_gmm)
        l2_x_list.append(l2_x)
        times.append(elapsed)
        final_losses.append(final_loss.item())
        x_preds.append(x_pred_t.tolist())

        print(f"  [{i + 1}/{args.n_attempts}] seed={run_seed} L2_GMM={l2_gmm:.6f} "
              f"L2_x*={l2_x:.6f} final_MMD={final_loss.item():.6f} time={elapsed:.2f}s")

    results = {
        "experiment": args.experiment_name,
        "method": "MLGD-F",
        "hparams": {
            "nsamples": args.nsamples, "num_x_t": args.num_x_t,
            "n_attempts": args.n_attempts, "seed": args.seed,
        },
        "environment": env_info,
        "l2_gmm": l2_gmm_list,
        "l2_x": l2_x_list,
        "final_loss": final_losses,
        "times": times,
        "x_pred": x_preds,
        "summary": {
            "l2_gmm_mean": float(np.mean(l2_gmm_list)), "l2_gmm_std": float(np.std(l2_gmm_list)),
            "l2_x_mean": float(np.mean(l2_x_list)), "l2_x_std": float(np.std(l2_x_list)),
            "final_loss_mean": float(np.mean(final_losses)), "final_loss_std": float(np.std(final_losses)),
            "time_mean": float(np.mean(times)), "time_std": float(np.std(times)),
        },
    }

    out_path = os.path.join(
        RESULTS_DIR,
        f"{args.experiment_name}_ns{args.nsamples}_xt{args.num_x_t}_seed{args.seed}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Sweep] Saved results to {out_path}")


if __name__ == "__main__":
    main()
