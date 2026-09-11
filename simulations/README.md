# Simulations

Synthetic experiments for conditional flow/diffusion matching on Gaussian Mixture Models (GMMs).

## Overview

We test conditional generation methods on low-dimensional GMMs where the true conditional distribution is analytically tractable. Experiments compare learned conditionals against the closed-form ground truth using L2-GMM distance and MMD metrics.

Three settings are evaluated:

| Experiment | Data dim | Condition dim |
|---|---|---|
| 2D_cond_1D | 2 | 1 |
| 5D_cond_1D | 5 | 1 |
| 10D_cond_1D | 10 | 1 |

## Structure

```
simulations/
  src/
    dist_utils.py          # GMM sampling, conditioning, and L2/MMD metrics
    Diffusion.py           # DDIM diffusion model with classifier-free guidance
    ConsistencyModels.py   # Improved Consistency Training (iCT, Song et al. 2023)
    LossFunctions.py       # Loss functions (MMD, etc.)
    Optimization.py        # Optimization utilities
    NN_utils.py            # Generic MLP and time embedding building blocks
    experiment_utils.py    # Shared experiment runners
  notebooks/
    Exp_2D_cond_1D.ipynb
    Exp_5D_cond_1D.ipynb
    Exp_10D_cond_1D.ipynb
    toy_example_with_beta_sweep.ipynb
    Exp_2D_infeasible_targets.ipynb
  params/                  # Saved GMM parameters (.pt files)
  results/                 # JSON result files per experiment
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

Open the corresponding notebook in `notebooks/` and run all cells. Each notebook:
1. Loads or generates GMM parameters (saved to `params/`)
2. Trains a diffusion or consistency model conditioned on `x_star`
3. Evaluates the learned conditional against the true GMM conditional
4. Saves metrics to `results/`

### Pre-trained Weights

For the three main experiment notebooks (`Exp_2D_cond_1D`, `Exp_5D_cond_1D`, `Exp_10D_cond_1D`), pre-trained model checkpoints are available and will be **downloaded automatically** from HuggingFace the first time each notebook is run — no manual setup required. This applies as long as the default configuration (seed, architecture hyperparameters) is left unchanged.

The `toy_example_with_beta_sweep.ipynb` notebook does not have pre-trained weights and will train from scratch.

To force retraining from scratch for any notebook, set `FORCE_RETRAIN = True` in the configuration cell.

## Metrics

- **L2-GMM distance**: closed-form L2 distance between two GMMs
- **MMD**: kernel-based Maximum Mean Discrepancy between generated and true samples

## LGD vs. LGD-CM per-step gradient variance/accuracy

`lgd_vs_lgdcm_step_variance.py` directly tests the claim that unrolling
deeper diffusion chains injects more noise into the guidance gradient, by
comparing LGD's inner sampler (a `--k_lgd`-step DDIM unroll through the
pretrained `Diffusion_cond` checkpoint) against LGD-CM's inner sampler (the
pretrained consistency model's own ~14-step multistep sampling procedure) at
every step of one or more real optimization trajectories, not just a single
hand-picked point.

States are captured from `--n_trajectories` independent UNGUIDED (zeta=0)
reference trajectories of `model_uncond`, one per outer diffusion step
(every `--step_stride`-th step) -- unguided so the states themselves don't
already depend on which inner sampler produced them. Each trajectory uses a
different seed.

At each captured state, both samplers are redrawn `--n_redraws` times against
the same fixed target-sample set (drawn from the exact analytic conditional
GMM, not either model's approximation), and per (trajectory, step, method) it
reports `normalized_variance` (`Var(grad)/||mean_grad||^2`),
`variance_trace_per_dim` (raw per-coordinate variance — the fair
cross-dimension comparison), and `dist_to_ref_normalized` (distance from the
mean gradient to the TRUE/population reference gradient, computed
closed-form with no network forward) — averaged (mean ± SEM) across
trajectories at each step. Results (per-trajectory raw rows and the
aggregated means) are saved to a single JSON file; the script does not
produce any plots.

```bash
python lgd_vs_lgdcm_step_variance.py --experiment_name 10D_cond_1D --smoke

python lgd_vs_lgdcm_step_variance.py --experiment_name 10D_cond_1D \
    --n_trajectories 10 --step_stride 10 --n_redraws 30

# or on a SLURM cluster:
export ENV_PATH=/path/to/your/env
export REPO_ROOT=/path/to/conditional-matching-paper
export EXPERIMENT_NAME=10D_cond_1D
sbatch simulations/submit_lgd_vs_lgdcm_step_variance.sh
```

Requires the `Diffusion_uncond`, `Diffusion_cond`, and `CM` checkpoints for
the chosen experiment — either let them download once via the HuggingFace
fallback, or train them locally via `notebooks/Exp_<experiment_name>.ipynb`.
