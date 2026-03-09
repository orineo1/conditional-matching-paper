#!/bin/bash
#SBATCH --job-name=sweep-lora-scale
#SBATCH --output=SD_cond_SD_controlnet/output/sweep_lora_scale_%j.log
#SBATCH --error=SD_cond_SD_controlnet/output/sweep_lora_scale_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting LoRA Scale Sweep on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q matplotlib
mkdir -p SD_cond_SD_controlnet/output

python SD_cond_SD_controlnet/sweep_lora_scale.py \
    --lora_path scribble_tune/output/checkpoint-50000 \
    --output_dir SD_cond_SD_controlnet/output/sweep_lora_scale_${SLURM_JOB_ID} \
    --seed 42 \
    --n_steps 30 \
    --lora_scales "0.0,0.25,0.5,0.75,1.0" \
    --strength 0.5 \
    --guidance_scale 7.5 \
    --sprinter_every 3 \
    --sprinter_count 5 \
    --edge_method hed_scribble

echo "LoRA scale sweep complete."
