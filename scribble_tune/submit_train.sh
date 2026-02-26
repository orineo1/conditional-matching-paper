#!/bin/bash
#SBATCH --job-name=scribble-lora
#SBATCH --output=scribble_tune/output/train_%j.log
#SBATCH --error=scribble_tune/output/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Activate environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Install deps if needed
# pip install -r scribble_tune/requirements.txt

echo "Starting LoRA training on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

# Create output directory
mkdir -p scribble_tune/output

# Launch training with accelerate
accelerate launch \
    --mixed_precision fp16 \
    scribble_tune/train_lora.py \
    --config scribble_tune/config.yaml

echo "Training complete."
