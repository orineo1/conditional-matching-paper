#!/bin/bash
#SBATCH --job-name=train-conditional
#SBATCH --output=logs/conditional_%j.log
#SBATCH --error=logs/conditional_%j.err
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

echo "Starting conditional training on $(hostname) | Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"

cd "$SCRIPT_DIR"

python train_conditional.py \
    --clf_epochs 15     \
    --threshold  0.9999 \
    --batch_size 256    \
    --epochs     500    \
    --ckpt_dir   checkpoints \
    --plots_dir  plots

echo "Conditional training complete."
