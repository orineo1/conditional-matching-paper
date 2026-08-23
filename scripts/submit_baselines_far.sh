#!/bin/bash
#SBATCH --job-name=cdm-baselines-far
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

# far-source editing baseline: unguided SDEdit-best from the office source
python discrete_x/baselines.py --method sdedit_best --prior bert \
  --target_words an old man or young woman smiling \
  --source_words one person sitting quietly inside the office \
  --sampler base --B 256 \
  --outdir output/baselines_${SLURM_JOB_ID}_sdedit_far_bert

echo "Far-source baseline complete."
