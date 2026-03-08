#!/bin/bash
#SBATCH --job-name=sweep-lora-cfg
#SBATCH --output=SD_cond_SD_controlnet/output/sweep_%j.log
#SBATCH --error=SD_cond_SD_controlnet/output/sweep_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup (no module load — doesn't work on this cluster)
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting LoRA+CFG sweep on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install plotting deps if missing
pip install -q matplotlib

# Create output directory
mkdir -p SD_cond_SD_controlnet/output

python SD_cond_SD_controlnet/sweep_lora_cfg.py \
    --lora_path scribble_tune/output/checkpoint-50000 \
    --output_dir SD_cond_SD_controlnet/output/sweep_${SLURM_JOB_ID} \
    --seed 42 \
    --n_steps 30 \
    --guidance_scales "3.0,5.0,7.5,12.0"

echo "Sweep complete."
