#!/bin/bash
#SBATCH --job-name=validate-lora
#SBATCH --output=scribble_tune/output/validate_%j.log
#SBATCH --error=scribble_tune/output/validate_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=catfish

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

echo "Running LoRA validation on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"

python scribble_tune/validate.py \
    --config scribble_tune/config.yaml \
    --checkpoint scribble_tune/output/checkpoint-50000 \
    --wandb_project conditional-flow \
    --wandb_entity conditional-matching

echo "Validation complete."
