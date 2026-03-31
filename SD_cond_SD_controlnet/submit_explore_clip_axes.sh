#!/bin/bash
#SBATCH --job-name=clip-axes
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/clip_axes_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/clip_axes_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting CLIP axis exploration on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
mkdir -p /sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q matplotlib scikit-learn

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

python SD_cond_SD_controlnet/explore_clip_axes.py \
    --n_per_pole 30 \
    --output_dir SD_cond_SD_controlnet/output/clip_axes_${SLURM_JOB_ID} \
    --model_id "stabilityai/sdxl-turbo" \
    --n_steps 4 \
    --guidance_scale 0.0 \
    --seed 42

echo "CLIP axis exploration complete."
