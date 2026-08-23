#!/bin/bash
#SBATCH --job-name=cdm-baselines
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/baselines_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/baselines_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

TGT="an old man or young woman smiling"

# 1. SDEdit-best, BERT prior (main baseline for the headline table)
python discrete_x/baselines.py --method sdedit_best --prior bert \
  --target_words $TGT --sampler base --B 256 \
  --outdir output/baselines_${SLURM_JOB_ID}_sdedit_bert

# 2. SDEdit-best, LLaDA prior
python discrete_x/baselines.py --method sdedit_best --prior llada \
  --target_words $TGT --sampler base --B 256 \
  --outdir output/baselines_${SLURM_JOB_ID}_sdedit_llada

# 3. Random prompt search
python discrete_x/baselines.py --method random --prior bert \
  --target_words $TGT --sampler base --B 256 \
  --outdir output/baselines_${SLURM_JOB_ID}_random

echo "Baselines complete."
