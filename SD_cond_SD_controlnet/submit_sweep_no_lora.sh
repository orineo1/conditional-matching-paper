#!/bin/bash
#SBATCH --job-name=sweep-no-lora
#SBATCH --output=SD_cond_SD_controlnet/output/sweep_no_lora_%j.log
#SBATCH --error=SD_cond_SD_controlnet/output/sweep_no_lora_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting sweep NO LORA on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q matplotlib
mkdir -p SD_cond_SD_controlnet/output

python SD_cond_SD_controlnet/sweep_guidance.py \
    --no_lora_only \
    --output_dir SD_cond_SD_controlnet/output/sweep_no_lora_${SLURM_JOB_ID} \
    --seed 42 \
    --n_steps 30 \
    --zetas "0,0.2,1.0,5.0" \
    --strength 0.5 \
    --guidance_scale 7.5 \
    --n_targets 20 \
    --num_variations 20 \
    --edge_method hed_scribble \
    --sprinter_every 3 \
    --sprinter_count 5

echo "Sweep (no LoRA) complete."
