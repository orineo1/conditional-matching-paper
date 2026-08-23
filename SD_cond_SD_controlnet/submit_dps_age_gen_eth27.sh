#!/bin/bash
#SBATCH --job-name=dps-eth27-age
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_eth27_age_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_eth27_age_%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting DPS eth27 age-diversity on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
mkdir -p /sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q matplotlib scikit-learn controlnet_aux

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper
mkdir -p SD_cond_SD_controlnet/output scripts/assets

# Extract HED scribble from the cheekbones scribble (already an edge map — use as-is)
# The scribble_cheekbones.png is already a scribble; copy to assets for reference
cp scripts/assets/scribble_cheekbones.png scripts/assets/eth27_cheekbones.png 2>/dev/null || true

python SD_cond_SD_controlnet/run_dps_synthetic_targets.py \
    --output_dir SD_cond_SD_controlnet/output/dps_eth27_age_${SLURM_JOB_ID} \
    --target_mode age_gender_combined \
    --reference_images_dir reference_images \
    --scribble_path scripts/assets/scribble_cheekbones.png \
    --n_targets 100 \
    --n_steps 500 \
    --start_step 300 \
    --num_variations 25 \
    --n_eval 100 \
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.8 \
    --sprinter_steps 4 \
    --sprinter_guidance_scale 1.5 \
    --sprinter_variation_prompt "a superrealistic professional photograph of a person of any ethnicity" \
    --sprinter_eval_prompt "a superrealistic professional photograph of a person of any ethnicity" \
    --architect_model_id "stabilityai/sdxl-turbo" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --loss_fn mmd \
    --loss_scale 1.0 \
    --bandwidth_scale 1.0 \
    --kernel_alpha 2.0 \
    --seed 8700

echo "DPS eth27 age-diversity complete."
