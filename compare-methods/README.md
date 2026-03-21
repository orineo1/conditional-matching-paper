# compare-methods — D-Flow / LGD / LGD-CM comparison pipeline

Branch: `compare-methods` of `conditional-matching-paper`

Trains toy models on a MoG distribution and runs a head-to-head comparison of three conditional optimization algorithms:
- **LGD** (Latent Gradient Descent with Diffusion)
- **LGD-CM** (LGD with Consistency Model)
- **D-Flow** (Flow Matching optimization)

## Structure

```
compare_methods/
  dist_utils.py      — MoG distribution utilities (sampling, conditionals, SWD)
  train_models.py    — Train Diffusion + CM + FM, save checkpoints
  run_compare.py     — Load checkpoints, run all methods, log results
  submit_train.sh    — SLURM job for training
  submit_compare.sh  — SLURM job for comparison
```

## Usage

### Step 1 — Train models

```bash
# Locally (quick smoke-test):
python compare_methods/train_models.py \
    --dim 2 --output_dir /tmp/test_models \
    --nepochs_diff 500 --nepochs_cm 200 --nepochs_fm 200

# On cluster:
sbatch compare_methods/submit_train.sh 2   # 2D
sbatch compare_methods/submit_train.sh 10  # 10D
```

### Step 2 — Run comparison

```bash
# Locally:
python compare_methods/run_compare.py \
    --models_dir /tmp/test_models \
    --output_dir /tmp/test_compare \
    --n_attempts 3 --nsamples_mmd 50 --no_wandb

# On cluster (pass the models dir from step 1):
sbatch compare_methods/submit_compare.sh compare_methods/output/models_2d_<JOB_ID>
```

## Key arguments

### train_models.py
| Arg | Default | Description |
|-----|---------|-------------|
| `--dim` | 2 | Joint distribution dimensionality (2 or 10) |
| `--condition_on` | 1 | Number of x (conditioning) dimensions |
| `--nepochs_diff` | 20000 | Epochs for Diffusion models |
| `--nepochs_cm` | 7500 | Epochs for Consistency Model |
| `--nepochs_fm` | 10000 | Epochs for Flow Matching |
| `--skip_diff / --skip_cm / --skip_fm` | — | Skip individual models |

### run_compare.py
| Arg | Default | Description |
|-----|---------|-------------|
| `--models_dir` | required | Path to train_models.py output |
| `--n_attempts` | 25 | Independent runs per method |
| `--nsamples_mmd` | 250 | MMD samples during optimization |
| `--x_star` | `-5.0` | Target conditioning value x* |
| `--skip_lgd / --skip_lgdcm / --skip_dflow` | — | Skip individual methods |
| `--no_wandb` | — | Disable wandb logging |

## Outputs

```
output/compare_<JOB_ID>/
  results.json              — Per-attempt SWD, L1, runtime for all methods
  summary.json              — Mean ± std across attempts
  comparison_boxplot.png    — Side-by-side box plots (SWD, normalized SWD, L1)
```

## Dependencies

Same conda env as the main DPS pipeline (`scribble_env`), plus:
```bash
pip install flow_matching POT
```

## wandb

Logs to team `conditional-matching`, project `compare-methods`.
