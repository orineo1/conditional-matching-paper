#!/bin/bash
#SBATCH --job-name=dps-debug-grad
#SBATCH --output=SD_cond_SD_controlnet/output/debug_gradient_%j.log
#SBATCH --error=SD_cond_SD_controlnet/output/debug_gradient_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup (no module load — doesn't work on this cluster)
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting DPS gradient debug on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install deps if missing
pip install -q matplotlib scikit-learn

# Create output directory
mkdir -p SD_cond_SD_controlnet/output

python SD_cond_SD_controlnet/debug_gradient.py \
    --lora_path scribble_tune/output/checkpoint-50000 \
    --output_dir SD_cond_SD_controlnet/output/debug_gradient_${SLURM_JOB_ID} \
    --n_steps 10 \
    --strength 0.5 \
    --num_variations 20 \
    --n_targets 20 \
    --base_zeta 0.2 \
    --guidance_scale 7.5 \
    --controlnet_scale 0.5 \
    --edge_method hed_scribble \
    --detailed_step 5 \
    --zeta_multipliers "1,10,100,1000" \
    --seed 42

echo "Debug gradient analysis complete."
