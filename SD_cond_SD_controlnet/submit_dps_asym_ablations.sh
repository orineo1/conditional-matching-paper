#!/bin/bash
#SBATCH --job-name=dps-asym-abl
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_asym_abl_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/dps_asym_abl_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon
#SBATCH --array=0-2

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
#   0: zeta=2 (weaker guidance)
#   1: start_step=150 (less noising, strength=0.4)
#   2: seed=42 (different random seed)

ZETA=5.0
START=125
SEED=1
SUFFIX=""

case $SLURM_ARRAY_TASK_ID in
    0) ZETA=2.0;  SUFFIX="zeta2" ;;
    1) START=150;  SUFFIX="lessnoise" ;;
    2) SEED=42;    SUFFIX="seed42" ;;
esac

echo "Ablation $SLURM_ARRAY_TASK_ID: $SUFFIX (zeta=$ZETA, start=$START, seed=$SEED)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python SD_cond_SD_controlnet/run_dps.py \
    --output_dir SD_cond_SD_controlnet/output/dps_asym_${SUFFIX}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --wandb_project "conditional-flow" \
    --n_steps 250 \
    --start_step $START \
    --num_variations 100 \
    --n_targets 100 \
    --male_ratio 0.3 \
    --base_zeta $ZETA \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --n_eval 100 \
    --sprinter_variation_prompt "a superrealistic professional photograph of" \
    --sprinter_target_man_prompt "a superrealistic portrait photograph of a man, studio lighting" \
    --sprinter_target_woman_prompt "a superrealistic portrait photograph of a woman, studio lighting" \
    --sprinter_eval_prompt "a superrealistic professional photograph of" \
    --architect_model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --loss_fn mmd \
    --loss_scale 1.0 \
    --bandwidth_scale 1.0 \
    --kernel_alpha 1.0 \
    --seed $SEED

echo "Ablation $SUFFIX complete."
