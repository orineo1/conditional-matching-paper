# MNIST Rotation Experiment

This directory contains the code for the MNIST rotation experiment from the paper.
The goal is to find a digit image $x^*$ such that its rotation angle distribution
$\mathcal{P}(Y \mid X = x^*)$ matches a user-specified target $\mathcal{G}(Y)$
(unimodal, bimodal, or uniform), using MLGDF.

## Directory Structure

```
MNIST/
├── MNIST_MLGDF.py              # Main inference script (run this)
├── MNIST_MLGDF.sh              # SLURM job script for the inference run
├── MLGDF_visualization.py      # Visualization & evaluation script
├── MLGDF_visualization.sh      # SLURM job script for visualization
├── requirements.txt            # Python dependencies
├── src/
│   ├── cond_model.py           # Conditional iCT model (CircularAngleConsistencyModel)
│   ├── uncond_model.py         # Unconditional DDPM UNet wrapper
│   ├── classifier.py           # Noise-robust MNIST digit classifier
│   └── dataset.py              # AugmentedMNISTDataset for conditional model training
├── train/
│   ├── train_conditional.py    # Train the conditional iCT model
│   ├── train_conditional.sh    # SLURM script for conditional training
│   ├── train_uncond.py         # Train the unconditional DDPM
│   └── train_unconditional.sh  # SLURM script for unconditional training
├── checkpoints/                # Saved classifier weights (auto-created)
├── results/                    # Output .pkl files per run (auto-created)
└── logs/                       # SLURM logs (auto-created)
```

## Pretrained Models

Pretrained unconditional and conditional model checkpoints are available on
Hugging Face at: `anon-submission-cdm/cdm-inverse-design`

The inference script downloads them automatically via `huggingface_hub`.

## Setup

```bash
pip install -r requirements.txt
```

Set your credentials as environment variables (never hardcode them):

```bash
export HF_TOKEN=hf_...          # HuggingFace token (needed to download checkpoints)
export WANDB_API_KEY=...         # W&B key (optional; skip with --wandb_mode disabled)
export REPO_ROOT=/path/to/repo   # Root of this repository
export ENV_PATH=/path/to/venv    # Your Python environment (for SLURM scripts)
```

## Running Experiments

### Quick smoke test (2 seeds, runs in ~1 minute)

```bash
cd MNIST/
python MNIST_MLGDF.py \
    --experiment unimodal \
    --smoke_test \
    --wandb_mode disabled
```

### Full runs (15 seeds each)

**Unimodal target** (digits upright, near 0°):
```bash
python MNIST_MLGDF.py \
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
python MNIST_MLGDF.py \
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
python MNIST_MLGDF.py \
    --experiment uniform \
    --num_inference_steps 290 \
    --step_size_mode original \
    --num_x_t 3 \
    --nsamples 600 \
    --wandb_mode disabled
```

> **Note:** `--clamp` is off by default. The paper results were obtained without clamping.

### Via SLURM

Edit `MNIST_MLGDF.sh` to set your partition name, then:

```bash
export REPO_ROOT=/path/to/repo
export ENV_PATH=/path/to/your/env
export HF_TOKEN=hf_...
sbatch MNIST_MLGDF.sh
```

Monitor:
```bash
squeue -u $USER
tail -f logs/MLGDF_<JOBID>.log
```

### Train classifier only

If the classifier checkpoint is missing, it is trained automatically.
You can also trigger this explicitly:

```bash
python MNIST_MLGDF.py --train_classifier_only
```

## Visualization

After running an experiment, visualize results with:

```bash
python MLGDF_visualization.py \
    --results_dir results/unimodal_run/unimodal_var515_st130_ssdouble_xt3_ns1500_cl0/
```

The script automatically finds all `.pkl` files in `--results_dir` (recursively),
skips any that are missing, and saves plots alongside the results.

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--results_dir` | *(required)* | Directory containing `.pkl` result files |
| `--ckpt_dir` | `checkpoints_and_results/` | Directory with model checkpoints |
| `--plots_dir` | `<results_dir>/plots/` | Where to save plots |
| `--top_k` | `5` | Number of top images/distributions to plot |
| `--dpi` | `150` | Plot resolution |
| `--no_titles` | off | Also save copies of plots without titles |

### Via SLURM

Edit `MLGDF_visualization.sh` to set `RESULTS_DIR` and your partition, then:

```bash
export HF_TOKEN=hf_...
sbatch MLGDF_visualization.sh
```

## W&B Logging

Results are logged to Weights & Biases by default.
To disable:

```bash
python MNIST_MLGDF.py --experiment unimodal --wandb_mode disabled
```

Or set `WANDB_MODE=disabled` in your environment.

## Output

Each run saves a `.pkl` file to `results/<experiment>_run/<run_name>/`.
The pickle contains:

| Key | Description |
|---|---|
| `results` | List of 15 generated images (28×28 numpy arrays) |
| `loss_log` | SWD loss per seed |
| `seed_log` | Seed indices |
| `time_log` | Wall-clock time per seed |
| `x_range` / `target_pdf` | Target distribution for plotting |

## Reproducing Paper Results

The exact hyperparameters used in the paper are:

| Experiment | `--num_inference_steps` | `--step_size_mode` | `--num_x_t` | `--nsamples` | variance |
|---|---|---|---|---|---|
| Unimodal   | 130 | `double`   | 3  | 1500 | 515 |
| Bimodal    | 125 | `original` | 10 | 1500 | 252 |
| Uniform    | 290 | `original` | 3  | 600  | —   |

All runs use 15 seeds. Top-5 results by SWD loss are reported.
`--clamp` is **off** by default.
