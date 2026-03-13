#!/bin/bash
#SBATCH --job-name=hed-validate
#SBATCH --output=finetune_hed/output/validate_%j.log
#SBATCH --error=finetune_hed/output/validate_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting HED validation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p finetune_hed/output/validation

python finetune_hed/validate_hed.py \
    --config finetune_hed/config.yaml \
    --checkpoint finetune_hed/output/final/merged \
    --output_dir finetune_hed/output/validation

echo "Validation complete."
