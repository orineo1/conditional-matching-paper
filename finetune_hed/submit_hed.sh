#!/bin/bash
#SBATCH --job-name=hed-compute
#SBATCH --output=finetune_hed/output/hed_%j.log
#SBATCH --error=finetune_hed/output/hed_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting HED edge map computation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

pip install -q controlnet_aux

python finetune_hed/compute_hed.py \
    --images_dir ./finetune_hed/data/images_raw \
    --output_dir ./finetune_hed/data/hed_512 \
    --resolution 512

echo "HED computation complete."
