#!/bin/bash
#SBATCH --job-name=mlgdf-age
#SBATCH --output=mlgdf_age_%j.log
#SBATCH --error=mlgdf_age_%j.err
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
OUTPUT_DIR="output/mlgdf_age_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

python run_age.py \
    --output_dir        "$OUTPUT_DIR" \
    --wandb_project     "mlgdf-age" \
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
    --n_eval            10 \
    \
    --age_min           10 \
    --age_max           80 \
    --age_step          1 \
    --age_gender        man \
    \
    --sprinter_variation_prompt  "a superrealistic professional photograph of" \
    --sprinter_eval_prompt       "a superrealistic professional photograph of" \
    --seed 1

echo "Done. Outputs in $OUTPUT_DIR"
