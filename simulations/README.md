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

## Hyperparameter sensitivity sweep (MLGD-F)

`run_hparam_sweep.py` isolates the MLGD-F guidance loop (`Optimization.optimize_LGD`
with `CM=True`) and sweeps `nsamples` (Monte Carlo samples used to estimate the
MMD guidance loss) and `num_x_t` (number of resampled `x0` candidates averaged
per guidance step) around their paper defaults for a given experiment. It reuses
the canonical GMM parameters in `params/` and the pretrained checkpoints
(auto-downloaded from HuggingFace) — no retraining.

```bash
# single grid point
python run_hparam_sweep.py --experiment_name 2D_cond_1D --nsamples 500 --num_x_t 3

# full sweep on a SLURM cluster (array job, one grid point per task)
export ENV_PATH=/path/to/your/env
export REPO_ROOT=/path/to/conditional-matching-paper
export HF_TOKEN=hf_...
sbatch --partition=your_partition simulations/submit_hparam_sweep.sh

# aggregate results into a CSV + figure once all array tasks finish
python aggregate_hparam_sweep.py --experiment_name 2D_cond_1D
```
