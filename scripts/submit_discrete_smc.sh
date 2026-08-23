#!/bin/bash
#SBATCH --job-name=discrete-smc
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/discrete_smc_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/discrete_smc_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting discrete-x SMC (image-mode MMD) on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

# Image-mode loss: L(prompt) = MMD^2 in CLIP image space between n_cond
# SDXL-Turbo generations of the prompt (fixed CRN latents) and a target set
# from the hidden ground-truth prompt. Both x0_hat estimators, seed 0.
python discrete_x/real_smc.py \
    --loss image \
    --prefix "a photo of a" \
    --target_words person standing in the rain \
    --source_words man walking down the street \
    --remask_frac 0.75 \
    --estimators mode sampled \
    --seeds 0 \
    --n_particles 128 \
    --T 16 \
    --beta 200 \
    --beta_anneal \
    --top_k 50 \
    --n_dec 4 \
    --n_cond 8 \
    --n_target 64 \
    --outdir output/discrete_smc_${SLURM_JOB_ID}

echo "Discrete-x SMC complete."
