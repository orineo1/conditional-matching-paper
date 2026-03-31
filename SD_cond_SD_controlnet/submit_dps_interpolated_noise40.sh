#!/bin/bash
#SBATCH --job-name=dps-interp
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_interp_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_interp_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export WANDB_API_KEY=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

echo "Starting DPS interpolated-target pipeline on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
mkdir -p /sci/labs/orzuk/shaulytolk/hf_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q matplotlib scikit-learn controlnet_aux

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper
mkdir -p SD_cond_SD_controlnet/output

python SD_cond_SD_controlnet/run_dps_synthetic_targets.py \
    --output_dir SD_cond_SD_controlnet/output/dps_interpolated_n40_${SLURM_JOB_ID} \
    --target_mode interpolated \
    --anchor_a_path manly_man.png \
    --anchor_b_path feminine_woman.png \
    --scribble_path scripts/assets/zeta5_final_guided.png \
    --n_targets 100 \
    --n_steps 250 \
    --start_step 150 \
    --num_variations 100 \
    --n_eval 100 \
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --sprinter_variation_prompt "a superrealistic professional photograph of" \
    --sprinter_eval_prompt "a superrealistic professional photograph of" \
    --architect_model_id "stabilityai/sdxl-turbo" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --loss_fn mmd \
    --loss_scale 1.0 \
    --bandwidth_scale 1.0 \
    --kernel_alpha 1.0 \
    --seed 1

echo "DPS interpolated-target pipeline complete."
