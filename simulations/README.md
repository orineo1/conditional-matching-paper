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

## Gradient variance vs. unroll depth

`gradient_variance_vs_unroll_depth.py` directly tests the claim that unrolling
deeper diffusion chains injects more noise into the guidance gradient. It
isolates the inner conditional sampler only (`model_cond`, i.e. the pretrained
`Diffusion_cond` checkpoint used by the LGD baseline) at a single fixed
conditioning point `x` (defaults to the experiment's `x_star`) against a
single fixed set of target samples, and for each unroll depth `K` in
`{10, 25, 40, 60, 80, 100}` runs the inner MMD-guidance gradient estimator 100
times, redrawing only the sampler's internal noise each time. It reports
`Var(grad)` (trace of the empirical covariance across the 100 draws)
normalized by `||mean(grad)||^2` for each `K` — if that quantity rises with
`K`, deeper unrolling injects more gradient noise, independent of whether a
given `K` is a more or less accurate sampler (accuracy is never measured
here; only the estimator's spread around its own mean).

```bash
python gradient_variance_vs_unroll_depth.py --experiment_name 2D_cond_1D

# or on a SLURM cluster:
export ENV_PATH=/path/to/your/env
export REPO_ROOT=/path/to/conditional-matching-paper
sbatch simulations/submit_gradient_variance.sh
```

Requires the `Diffusion_cond` checkpoint for the chosen experiment (not
needed by the hyperparameter sweep above, which only uses `CM` and
`Diffusion_uncond`) — either let it download once via the HuggingFace
fallback, or train it locally via `notebooks/Exp_<experiment_name>.ipynb`.
