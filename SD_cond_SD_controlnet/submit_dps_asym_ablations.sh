#!/bin/bash
#SBATCH --job-name=dps-slerp-abl
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_slerp_abl_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_slerp_abl_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon
#SBATCH --array=0-1

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export WANDB_API_KEY=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

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q matplotlib scikit-learn controlnet_aux

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper
mkdir -p SD_cond_SD_controlnet/output

# Ablation variants:
#   0: n_steps=250, start_step=125
#   1: n_steps=500, start_step=250

ZETA=5.0
SEED=1

case $SLURM_ARRAY_TASK_ID in
    0) N_STEPS=250; START=125; SUFFIX="steps250" ;;
    1) N_STEPS=500; START=250; SUFFIX="steps500" ;;
esac

echo "Ablation $SLURM_ARRAY_TASK_ID: $SUFFIX (zeta=$ZETA, n_steps=$N_STEPS, start=$START, seed=$SEED)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python SD_cond_SD_controlnet/run_dps_synthetic_targets.py \
    --output_dir SD_cond_SD_controlnet/output/dps_slerp_${SUFFIX}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --wandb_project "conditional-flow" \
    --anchor_a_path SD_cond_SD_controlnet/assets/anchor_man.png \
    --anchor_b_path SD_cond_SD_controlnet/assets/anchor_woman.png \
    --scribble_path SD_cond_SD_controlnet/assets/scribble.png \
    --target_mode slerp \
    --n_targets 100 \
    --n_steps $N_STEPS \
    --start_step $START \
    --num_variations 100 \
    --base_zeta $ZETA \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --n_eval 100 \
    --sprinter_variation_prompt "a superrealistic professional photograph of" \
    --sprinter_eval_prompt "a superrealistic professional photograph of" \
    --architect_model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --loss_fn mmd \
    --loss_scale 1.0 \
    --bandwidth_scale 1.0 \
    --kernel_alpha 1.0 \
    --seed $SEED

echo "Slerp ablation $SUFFIX complete."