#!/bin/bash
#SBATCH --job-name=laion-download
#SBATCH --output=finetune_hed/output/download_%j.log
#SBATCH --error=finetune_hed/output/download_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting LAION download on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Install img2dataset if not present
pip install -q img2dataset

# Create output directory
mkdir -p finetune_hed/output
mkdir -p finetune_hed/data

python finetune_hed/download_laion.py \
    --data_dir ./finetune_hed/data \
    --num_samples 200000 \
    --min_aesthetic 5.0 \
    --image_size 512

echo "Download complete."
