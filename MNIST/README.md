# MNIST Rotation Experiment

This directory contains the code for the MNIST rotation experiment from the paper.
The goal is to find a digit image $x^*$ such that its rotation angle distribution
$\mathcal{P}(Y \mid X = x^*)$ matches a user-specified target $\mathcal{G}(Y)$
(unimodal, bimodal, or uniform), using MLGDF.

## Directory Structure

```
MNIST/
├── run_mlgdf.py            # Main inference script
├── run_mlgdf.sh            # SLURM job script for inference
├── visualize.py            # Visualization & evaluation script
├── visualize.sh            # SLURM job script for visualization
├── requirements.txt        # Python dependencies
├── src/
│   ├── cond_model.py       # Conditional iCT model (CircularAngleConsistencyModel)
│   ├── uncond_model.py     # Unconditional DDPM UNet wrapper
│   ├── classifier.py       # Noise-robust MNIST digit classifier
│   └── dataset.py          # AugmentedMNISTDataset for conditional model training
├── train/
│   ├── train_conditional.py    # Train the conditional iCT model
│   ├── train_conditional.sh    # SLURM script for conditional training
│   ├── train_uncond.py         # Train the unconditional DDPM
│   └── train_unconditional.sh  # SLURM script for unconditional training
├── checkpoints/            # Saved classifier weights (auto-created)
├── results/                # Output .pkl files per run (auto-created)
└── logs/                   # SLURM logs (auto-created)
```

## Pretrained Models

Pretrained unconditional and conditional model checkpoints are available on
Hugging Face at: `anon-submission-cdm/cdm-inverse-design`

The inference script downloads them automatically via `huggingface_hub`.

## Setup

```bash
pip install -r requirements.txt
```

Set your credentials as environment variables :

```bash
export HF_TOKEN=hf_...          # HuggingFace token (required to download checkpoints)
export WANDB_API_KEY=...         # W&B key (optional; skip with --wandb_mode disabled)
export REPO_ROOT=/path/to/repo   # Root of this repository
export ENV_PATH=/path/to/env     # Your Python environment (for SLURM scripts)
```

## Running Experiments

### Quick smoke test (2 seeds, ~1 minute)

```bash
cd MNIST/
python run_mlgdf.py --experiment unimodal --smoke_test --wandb_mode disabled
```

### Full runs (15 seeds each)

**Unimodal target** (digits upright, near 0°):
```bash
python run_mlgdf.py \
    --experiment unimodal \
    --unimodal_var 515 \
    --num_inference_steps 130 \
    --step_size_mode double \
    --num_x_t 3 \
    --nsamples 1500 \
    --wandb_mode disabled
```

**Bimodal target** (digits valid at 0° and 180°):
```bash
python run_mlgdf.py \
    --experiment bimodal \
    --bimodal_var 252 \
    --num_inference_steps 125 \
    --step_size_mode original \
    --num_x_t 10 \
    --nsamples 1500 \
    --wandb_mode disabled
```

**Uniform target** (digits valid at any angle):
```bash
python run_mlgdf.py \
    --experiment uniform \
    --num_inference_steps 290 \
    --step_size_mode original \
    --num_x_t 3 \
    --nsamples 600 \
    --wandb_mode disabled
```

> **Note:** `--clamp` is off by default.

### Via SLURM

Edit `run_mlgdf.sh` to set your partition, then:

```bash
export REPO_ROOT=/path/to/repo
export ENV_PATH=/path/to/env
export HF_TOKEN=hf_...
sbatch run_mlgdf.sh
```

Monitor:
```bash
squeue -u $USER
tail -f logs/MLGDF_<JOBID>.log
```

### Train classifier only

The classifier is trained automatically if missing. To trigger explicitly:

```bash
python run_mlgdf.py --train_classifier_only
```

## Visualization

After running an experiment, visualize results with:

```bash
python visualize.py --results_dir results/unimodal_run/<run_name>/
```

The script finds all `.pkl` files in `--results_dir` recursively, skips any
missing ones, and saves all plots alongside the results.

| Argument | Default | Description |
|---|---|---|
| `--results_dir` | *(required)* | Directory containing `.pkl` result files |
| `--ckpt_dir` | `checkpoints/` | Directory with model checkpoints |
| `--plots_dir` | `<results_dir>/plots/` | Where to save plots |
| `--top_k` | `5` | Number of top images/distributions to plot |
| `--dpi` | `150` | Plot resolution |
| `--no_titles` | off | Also save copies of plots without titles |

### Via SLURM

```bash
export REPO_ROOT=/path/to/repo
export ENV_PATH=/path/to/env
export HF_TOKEN=hf_...
sbatch visualize.sh results/unimodal_run/<run_name>/
```

Optional: pass `top_k` and `dpi` as extra arguments:
```bash
sbatch visualize.sh results/unimodal_run/<run_name>/ 10 200
```

## W&B Logging

Results are logged to Weights & Biases by default. To disable:

```bash
python run_mlgdf.py --experiment unimodal --wandb_mode disabled
```

## Output

Each run saves a `.pkl` file to `results/<experiment>_run/<run_name>/`.

| Key | Description |
|---|---|
| `results` | List of 15 generated images (28×28 numpy arrays) |
| `loss_log` | SWD loss per seed |
| `seed_log` | Seed indices |
| `time_log` | Wall-clock time per seed |
| `x_range` / `target_pdf` | Target distribution for plotting |

## Reproducing Paper Results

| Experiment | `--num_inference_steps` | `--step_size_mode` | `--num_x_t` | `--nsamples` | variance |
|---|---|---|---|---|---|
| Unimodal | 130 | `double` | 3 | 1500 | 515 |
| Bimodal | 125 | `original` | 10 | 1500 | 252 |
| Uniform | 290 | `original` | 3 | 600 | — |

All runs use 15 seeds. Top-5 results by SWD loss are reported. `--clamp` is off by default.
