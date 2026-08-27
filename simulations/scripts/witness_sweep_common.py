"""
Shared setup code for run_backsel_witness_sweep.py: per-experiment architecture
configs, GMM parameter loading/generation, and model loading/training.

This mirrors the model-loading boilerplate used by the reuse_frac/momentum grid
scripts elsewhere in the repo (same functions, same on-disk formats via
experiment_utils/dist_utils) but is kept local to this branch/analysis since it
carries no reuse_frac/momentum-specific logic -- purely "get a trained
model_uncond/model_cond/model_cm for one of the toy GMM experiments."
"""

import sys
import os
from functools import partial

import torch

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Diffusion
import dist_utils
import experiment_utils
from ConsistencyModels import ConsistencyModeliCT


EXPERIMENT_CONFIGS = {
    "2D_cond_1D": dict(
        NBLOCKS=3, NUNITS=128, NBLOCKS_CM=3, NUNITS_CM=128,
        NEPOCHS=20_000, BATCH_SIZE=1_024, NEPOCHS_CM=20_000, BATCH_SIZE_CM=1_024,
        DIFFUSION_STEPS=100, CONDITION_ON=1,
    ),
    "5D_cond_1D": dict(
        NBLOCKS=6, NUNITS=512, NBLOCKS_CM=6, NUNITS_CM=512,
        NEPOCHS=40_000, BATCH_SIZE=4_096, NEPOCHS_CM=40_000, BATCH_SIZE_CM=4_096,
        DIFFUSION_STEPS=100, CONDITION_ON=4,
    ),
    "10D_cond_1D": dict(
        NBLOCKS=8, NUNITS=512, NBLOCKS_CM=8, NUNITS_CM=512,
        NEPOCHS=40_000, BATCH_SIZE=4_096, NEPOCHS_CM=40_000, BATCH_SIZE_CM=4_096,
        DIFFUSION_STEPS=100, CONDITION_ON=9,
    ),
}


def load_or_generate_gmm_params(cfg, params_dir, results_dir, experiment_name, global_seed):
    loaded = experiment_utils.load_gmm_params(params_dir, experiment_name)
    if loaded is None:
        loaded = experiment_utils.load_gmm_params(results_dir, experiment_name)

    if loaded is not None:
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = loaded
        mu_list = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha = alpha.float()
        print(f"[GMM] Loaded existing parameters for {experiment_name}")
        return mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star

    print(f"[GMM] No saved parameters found for {experiment_name}, generating fresh...")
    experiment_utils.set_global_seed(global_seed)

    if experiment_name == "2D_cond_1D":
        mu_list = [
            torch.tensor([-5, 5], dtype=torch.float64), torch.tensor([-5, -5], dtype=torch.float64),
            torch.tensor([5, 3], dtype=torch.float64), torch.tensor([5, -1], dtype=torch.float64),
            torch.tensor([0, -3], dtype=torch.float64), torch.tensor([-2, 4], dtype=torch.float64),
            torch.tensor([-2, -3], dtype=torch.float64), torch.tensor([1, 2], dtype=torch.float64),
            torch.tensor([-8, 1], dtype=torch.float64), torch.tensor([7, 5], dtype=torch.float64),
            torch.tensor([0, -5], dtype=torch.float64),
        ]
        Sigma_list = [torch.tensor([[0.5000, 0.1950], [0.1950, 0.2000]], dtype=torch.float64)] * len(mu_list)
        alpha = torch.tensor([1 / len(mu_list)] * len(mu_list), dtype=torch.float64)
        mu_list = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha = alpha.float()
        x_star = torch.tensor([-5])
        mu_temp, Sigma_temp = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
        temp_alpha = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
        mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
            mu_temp, Sigma_temp, temp_alpha, threshold=0.01
        )
    else:
        dim_data = cfg["CONDITION_ON"] + 1
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = \
            dist_utils.get_param_mog_with_target(
                dim_data=dim_data, num_components=4, device="cpu",
                conditional_modes=2, distanceOrScale="Distance",
            )
        mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
            mog_means, mog_variances, weights, threshold=0.001
        )
        mu_list = [mu.float() for mu in mu_list]
        Sigma_list = [cov.float() for cov in Sigma_list]
        alpha = alpha.float()

    experiment_utils.save_gmm_params(
        mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star,
        params_dir, experiment_name,
    )
    return mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star


def load_or_train_models(cfg, mu_list, Sigma_list, alpha, checkpoint_dir, experiment_name,
                          global_seed, device, force_retrain):
    condition_on = cfg["CONDITION_ON"]

    data_generator_cm = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
    )
    model_cm = ConsistencyModeliCT(
        nfeatures=(condition_on + 1) - condition_on, condition_on=condition_on,
        nunits=cfg["NUNITS_CM"], depth=cfg["NBLOCKS_CM"],
    )
    loaded_cm = experiment_utils.load_checkpoint_with_hf_fallback(
        model_cm, "CM", checkpoint_dir, experiment_name, global_seed, device
    ) if not force_retrain else False
    if not loaded_cm:
        experiment_utils.set_global_seed(global_seed)
        model_cm.train_model(
            X=None, nepochs=cfg["NEPOCHS_CM"], batch_size=cfg["BATCH_SIZE_CM"],
            device=device, condition=condition_on,
            data_generator=data_generator_cm, use_improved_training=True,
        )
        experiment_utils.save_model_checkpoint(model_cm, "CM", checkpoint_dir, experiment_name, global_seed)

    data_generator_diff_cond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha, kernel_func=None,
    )
    model_cond = Diffusion.DiffusionModel(
        nfeatures=(condition_on + 1), nblocks=cfg["NBLOCKS"], nunits=cfg["NUNITS"],
        condition=True, condition_on=condition_on, diffusion_steps=cfg["DIFFUSION_STEPS"],
    )
    loaded_cond = experiment_utils.load_checkpoint_with_hf_fallback(
        model_cond, "Diffusion_cond", checkpoint_dir, experiment_name, global_seed, device
    ) if not force_retrain else False
    if not loaded_cond:
        experiment_utils.set_global_seed(global_seed)
        model_cond.train_model(
            None, data_generator=data_generator_diff_cond,
            nepochs=cfg["NEPOCHS"], batch_size=cfg["BATCH_SIZE"], condition_on=condition_on,
        )
        experiment_utils.save_model_checkpoint(model_cond, "Diffusion_cond", checkpoint_dir, experiment_name, global_seed)

    data_generator_diff_uncond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
        kernel_func=lambda X: X[:, :condition_on],
    )
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=condition_on, nblocks=cfg["NBLOCKS"], nunits=cfg["NUNITS"],
        condition=False, diffusion_steps=cfg["DIFFUSION_STEPS"],
    )
    loaded_uncond = experiment_utils.load_checkpoint_with_hf_fallback(
        model_uncond, "Diffusion_uncond", checkpoint_dir, experiment_name, global_seed, device
    ) if not force_retrain else False
    if not loaded_uncond:
        experiment_utils.set_global_seed(global_seed)
        model_uncond.train_model(
            None, data_generator=data_generator_diff_uncond,
            nepochs=cfg["NEPOCHS"], batch_size=cfg["BATCH_SIZE"], condition_on=condition_on,
        )
        experiment_utils.save_model_checkpoint(model_uncond, "Diffusion_uncond", checkpoint_dir, experiment_name, global_seed)

    return model_uncond, model_cond, model_cm
