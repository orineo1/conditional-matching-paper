#!/bin/bash
#SBATCH --job-name=scribble-lora
#SBATCH --output=scribble_tune/output/train_%j.log
#SBATCH --error=scribble_tune/output/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=catfish

# Conda setup (no module load — doesn't work on this cluster)
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting LoRA training on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Create output directory
mkdir -p scribble_tune/output

# Launch training with accelerate
accelerate launch \
    --mixed_precision fp16 \
    scribble_tune/train_lora.py \
    --config scribble_tune/config.yaml

echo "Training complete."
