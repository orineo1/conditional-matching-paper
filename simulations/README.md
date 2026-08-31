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
`Diffusion_cond` checkpoint used by the LGD baseline) at one or more
conditioning points `x`, each against its own fixed set of target samples,
and for each unroll depth `K` in `{10, 25, 40, 60, 80, 100}` runs the inner
MMD-guidance gradient estimator 100 times per `x`, redrawing only the
sampler's internal noise each time. It reports `Var(grad)` (trace of the
empirical covariance across the 100 draws) normalized by `||mean(grad)||^2`
for each `(x, K)` — if that quantity rises with `K`, deeper unrolling injects
more gradient noise, independent of whether a given `K` is a more or less
accurate sampler.

It also computes the TRUE/population reference gradient at each `x` (the
exact analytic conditional GMM, differentiated w.r.t. `x`, scored against
the same fixed target samples via `--grad_ref_n` closed-form samples — no
network forward, so this is cheap even at `grad_ref_n >> nsamples`), and
reports each `K`'s `dist_to_ref`/`dist_to_ref_normalized`: the distance from
that `K`'s mean gradient to the true one. This answers a different question
than `normalized_variance` does — whether the estimator's mean is actually
converging toward the true gradient as `K` grows, or has instead plateaued
near zero (or some other wrong value): a low-variance estimator can still be
consistently wrong, and `normalized_variance` alone can't tell you which.

Conditioning points can be chosen two ways:
- `--n_random_conds N` (recommended, and the sbatch default) — draw `N`
  points at random from the GMM's own marginal over `x`, instead of
  hand-picking them. This is the only choice that stays comparable across
  2D/5D/10D: a fixed point like `x=0` isn't the same "difficulty" in each.
- `--x_conds` (repeatable) — fixed points instead, e.g. `--x_conds -5
  --x_conds 0 --x_conds 5` for a 1D-conditioning experiment.

```bash
python gradient_variance_vs_unroll_depth.py --experiment_name 2D_cond_1D \
    --n_random_conds 5

# or fixed points instead:
python gradient_variance_vs_unroll_depth.py --experiment_name 2D_cond_1D \
    --x_conds -5 --x_conds 0 --x_conds 5

# same command works for 5D/10D (condition_on is 4/9 there, so --x_conds
# would need that many values per point -- --n_random_conds needs no changes):
python gradient_variance_vs_unroll_depth.py --experiment_name 5D_cond_1D --n_random_conds 5
python gradient_variance_vs_unroll_depth.py --experiment_name 10D_cond_1D --n_random_conds 5

# or on a SLURM cluster:
export ENV_PATH=/path/to/your/env
export REPO_ROOT=/path/to/conditional-matching-paper
export EXPERIMENT_NAME=5D_cond_1D   # or 10D_cond_1D
sbatch simulations/submit_gradient_variance.sh   # N_RANDOM_CONDS=5 by default
```

Requires the `Diffusion_cond` checkpoint for the chosen experiment (not
needed by the hyperparameter sweep above, which only uses `CM` and
`Diffusion_uncond`) — either let it download once via the HuggingFace
fallback, or train it locally via `notebooks/Exp_<experiment_name>.ipynb`.
