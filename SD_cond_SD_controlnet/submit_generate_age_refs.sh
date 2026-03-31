#!/bin/bash
#SBATCH --job-name=age-refs
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/age_refs_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/age_refs_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon
#SBATCH --array=0-1

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting age reference generation on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
mkdir -p /sci/labs/orzuk/shaulytolk/hf_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q matplotlib scikit-learn

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

GENDERS=("woman" "man")
GENDER=${GENDERS[$SLURM_ARRAY_TASK_ID]}

echo "Generating reference images for gender: $GENDER"

python SD_cond_SD_controlnet/generate_age_references.py \
    --gender "$GENDER" \
    --n_samples 100 \
    --output_dir reference_images \
    --age_mean 50.0 \
    --age_std 100.0 \
    --age_min 1 \
    --age_max 99 \
    --seed 42 \
    --model_id "stabilityai/sdxl-turbo" \
    --n_steps 4 \
    --guidance_scale 0.0

echo "Age reference generation complete for $GENDER."
