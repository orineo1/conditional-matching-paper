#!/usr/bin/env python
"""
Train and save, purely locally, the two checkpoints run_hparam_sweep.py needs:
  - the consistency model (CM) for P(Y|X=x)
  - the unconditional diffusion model for P(X=x)

Mirrors the training cells in notebooks/Exp_<experiment_name>.ipynb (with
FORCE_RETRAIN=True), but as a headless script for cluster batch jobs -- no
Jupyter, and no HuggingFace involved at any point. Saves to the exact same
path run_hparam_sweep.py / the notebooks look for, so once this finishes,
every other script picks the checkpoints up locally.

Usage:
    python train_checkpoints.py --experiment_name 2D_cond_1D
"""
import os
import sys
import argparse
from functools import partial

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment_name", type=str, default="2D_cond_1D",
                    choices=["2D_cond_1D", "5D_cond_1D", "10D_cond_1D"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Must match the architecture / training config in the corresponding notebook.
    ARCH = {
        "2D_cond_1D":  dict(nblocks=3, nunits=128, diffusion_steps=100, condition_on=1,
                             nepochs=20_000, batch_size=1_024),
        "5D_cond_1D":  dict(nblocks=6, nunits=512, diffusion_steps=100, condition_on=4,
                             nepochs=40_000, batch_size=4_096),
        "10D_cond_1D": dict(nblocks=8, nunits=512, diffusion_steps=100, condition_on=9,
                             nepochs=40_000, batch_size=4_096),
    }[args.experiment_name]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARAMS_DIR = os.path.join(BASE_DIR, "params")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", args.experiment_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    import experiment_utils
    import dist_utils
    import Diffusion
    from ConsistencyModels import ConsistencyModeliCT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] experiment={args.experiment_name} seed={args.seed} device={device}")

    experiment_utils.set_global_seed(args.seed)

    loaded = experiment_utils.load_gmm_params(PARAMS_DIR, args.experiment_name)
    if loaded is None:
        raise FileNotFoundError(
            f"No GMM params found under {PARAMS_DIR} for '{args.experiment_name}'. "
            f"These are committed in the repo -- make sure you're on the right branch/commit."
        )
    mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
    mu_list = [mu.float() for mu in mu_list]
    Sigma_list = [cov.float() for cov in Sigma_list]
    alpha = alpha.float()

    condition_on = ARCH["condition_on"]
    nfeatures_y = mu_list[0].shape[0] - condition_on

    # ── Consistency model P(Y|X=x) ──────────────────────────────────────────
    print("[Train] Training consistency model (CM)...")
    experiment_utils.set_global_seed(args.seed)
    data_generator_cm = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
    )
    cm_model = ConsistencyModeliCT(
        nfeatures=nfeatures_y, condition_on=condition_on,
        nunits=ARCH["nunits"], depth=ARCH["nblocks"],
    )
    cm_model.train_model(
        X=None, nepochs=ARCH["nepochs"], batch_size=ARCH["batch_size"],
        device=device, condition=condition_on,
        data_generator=data_generator_cm, use_improved_training=True,
    )
    experiment_utils.save_model_checkpoint(cm_model, "CM", CHECKPOINT_DIR, args.experiment_name, args.seed)

    # ── Unconditional diffusion P(X=x) ──────────────────────────────────────
    print("[Train] Training unconditional diffusion model...")
    experiment_utils.set_global_seed(args.seed)
    data_generator_diff_uncond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
        kernel_func=lambda X: X[:, :condition_on],
    )
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=condition_on, nblocks=ARCH["nblocks"], nunits=ARCH["nunits"],
        condition=False, diffusion_steps=ARCH["diffusion_steps"],
    )
    model_uncond.train_model(
        None, data_generator=data_generator_diff_uncond,
        nepochs=ARCH["nepochs"], batch_size=ARCH["batch_size"],
        condition_on=condition_on, device=device,
    )
    experiment_utils.save_model_checkpoint(
        model_uncond, "Diffusion_uncond", CHECKPOINT_DIR, args.experiment_name, args.seed
    )

    print(f"[Train] Done. Checkpoints saved under {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
