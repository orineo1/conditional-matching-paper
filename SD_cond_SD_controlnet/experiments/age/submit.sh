#!/bin/bash
#SBATCH --job-name=mlgdf-gender
#SBATCH --output=mlgdf_gender_%j.log
#SBATCH --error=mlgdf_gender_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=<YOUR_PARTITION>

# ── 1. Environment ────────────────────────────────────────────────────────────
source <YOUR_ENV_PATH>/bin/activate

export HF_HOME="<YOUR_CACHE_DIR>/hf_cache"
export MPLCONFIGDIR="<YOUR_CACHE_DIR>/.matplotlib"
export XDG_CACHE_HOME="<YOUR_CACHE_DIR>/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 2. Verify environment ─────────────────────────────────────────────────────
echo "=== JOB ${SLURM_JOB_ID} on $(hostname) ==="
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ── 3. Run ────────────────────────────────────────────────────────────────────
cd <YOUR_REPO_DIR>
OUTPUT_DIR="output/mlgdf_gender_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

python run_gender.py \
    --output_dir        "$OUTPUT_DIR" \
    --wandb_project     "mlgdf-gender" \
    \
    --n_steps           30 \
    --start_step        15 \
    \
    --base_zeta         5.0 \
    --guidance_scale    0.0 \
    --controlnet_scale  0.5 \
    --loss_fn           mmd \
    --loss_scale        1.0 \
    --bandwidth_scale   1.0 \
    --kernel_alpha      1.0 \
    \
    --num_variations    6 \
    --n_targets         100 \
    --n_eval            10 \
    \
    --sprinter_variation_prompt  "a superrealistic professional photograph of" \
    --sprinter_eval_prompt       "a superrealistic professional photograph of" \
    \
    --groups \
        "Woman:a superrealistic portrait photograph of a woman, studio lighting:50" \
        "Man:a superrealistic portrait photograph of a man, studio lighting:50" \
    \
    --architect_model_id  "stabilityai/sdxl-turbo" \
    --sprinter_model_id   "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --seed 1

echo "Done. Outputs in $OUTPUT_DIR"
