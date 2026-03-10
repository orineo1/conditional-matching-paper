#!/bin/bash
#SBATCH --job-name=scribble-compare-gen
#SBATCH --output=finetune_scribble/output/compare_gen_%j.log
#SBATCH --error=finetune_scribble/output/compare_gen_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Generation comparison on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python finetune_scribble/compare_generation.py \
    --checkpoint finetune_scribble/output/checkpoint-1000/merged \
    --prompt "male face" \
    --num_seeds 10

echo "Done."
