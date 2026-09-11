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

## Backsel gradient-variance diagnostic (uniform vs. witness, frozen states)

`scripts/backsel_state_gradient_variance.py` isolates ONE guidance step at a
time to test whether witness-function backsel selection actually reduces
gradient variance vs. uniform selection -- independent of
`run_backsel_witness_sweep.py`'s end-to-end L2/MMD grid (which only sees the
combined effect of many steps).

- A handful of representative states are captured first: a few diffusion
  steps (`--step_fracs`, e.g. early/mid/late in the denoising trajectory,
  since difficulty differs across steps) x a few trajectory seeds
  (`--state_seeds`, to land in different regions of the target
  distribution). States come from UNGUIDED (zeta=0) trajectories, so which
  rule is under test never influences the states it's evaluated at.
- At each state independently: freeze it completely (fixed `x0_sample`,
  fixed `t`), then redraw the full sampling + backsel pipeline
  `--n_redraws` times (200+ recommended) for both `uniform` and `witness`.
- Reports, per state AND per rule: `mean_grad`, `variance_trace` (trace of
  the empirical gradient covariance across redraws), and
  `normalized_variance` (`variance_trace / ||mean_grad||^2`) -- plus the
  average of `normalized_variance` across all states. The per-state numbers
  are saved in full (not just the average), since a single mean can hide a
  rule that only wins at some states.

```bash
python scripts/backsel_state_gradient_variance.py --experiment 5D_cond_1D

# or on a SLURM cluster (no N_RUNS sweep -- finishes in minutes, not hours):
export ENV_PATH=/path/to/your/env
export REPO_ROOT=/path/to/conditional-matching-paper
sbatch simulations/scripts/run_backsel_state_variance.sh
```

Output: `results/<experiment>/<experiment>_backsel_state_variance_<method>_n<nsamples>_kfrac<k_frac>_seed<seed>.json`
(nsamples and k_frac are in the filename so runs that only differ in either
don't overwrite each other),
containing every state's captured `x0_sample`/`t`, both rules' raw per-redraw
gradients and summary stats, and the cross-state `averaged` block.
