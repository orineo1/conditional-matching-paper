#!/bin/bash
#SBATCH --job-name=train-unconditional
#SBATCH --output=logs/unconditional_%j.log
#SBATCH --error=logs/unconditional_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# --- Conda ---
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

echo "Starting unconditional training on $(hostname) | Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
mkdir -p logs checkpoints

python train_uncond.py \
    --epochs     500  \
    --batch_size 256  \
    --lr         1e-3 \
    --ckpt_dir   checkpoints

echo "Unconditional training complete."
