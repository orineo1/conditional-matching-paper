#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --output=scribble_tune/output/setup_%j.log
#SBATCH --error=scribble_tune/output/setup_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=catfish

# Conda setup (no module load — doesn't work on this cluster)
export PATH="/sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/miniconda3/bin:$PATH"
source /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/miniconda3/etc/profile.d/conda.sh

ENV_DIR="/sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env"

echo "Creating conda environment at $ENV_DIR"

conda create -y -p "$ENV_DIR" python=3.10
conda activate "$ENV_DIR"

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Diffusers stack
pip install diffusers transformers accelerate peft

# Training utilities
pip install tensorboard pyyaml

# QuickDraw data
pip install quickdraw

# Image processing
pip install Pillow

echo "Environment setup complete at $ENV_DIR"
echo "Installed packages:"
pip list
