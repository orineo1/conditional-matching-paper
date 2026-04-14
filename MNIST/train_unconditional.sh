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
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting unconditional training on $(hostname) | Job: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
mkdir -p logs checkpoints

python train_unet.py \
    --epochs     500  \
    --batch_size 256  \
    --lr         1e-3 \
    --ckpt_dir   checkpoints

echo "Unconditional training complete."
