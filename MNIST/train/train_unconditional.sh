#!/bin/bash
#SBATCH --job-name=train-unconditional
#SBATCH --output=logs/unconditional_%j.log
#SBATCH --error=logs/unconditional_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/checkpoints"
mkdir -p "$SCRIPT_DIR/plots"

source "${ENV_PATH}/bin/activate"

echo "Starting unconditional training on $(hostname) | Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"

cd "$SCRIPT_DIR"

python train_uncond.py \
    --epochs     100  \
    --batch_size 256  \
    --lr         1e-3 \
    --ckpt_dir   checkpoints \
    --plots_dir  plots

echo "Unconditional training complete."
