#!/bin/bash
#SBATCH --job-name=train-unconditional
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/train/logs/unconditional_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/train/logs/unconditional_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

mkdir -p /sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/train/logs
mkdir -p /sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/train/checkpoints

# --- Conda ---
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

cd /sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/train

echo "Starting unconditional training on $(hostname) | Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"

python train_uncond.py \
    --epochs     500  \
    --batch_size 256  \
    --lr         1e-3 \
    --ckpt_dir   checkpoints \
    --plots_dir  plots

echo "Unconditional training complete."