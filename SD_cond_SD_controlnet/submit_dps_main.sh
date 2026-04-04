#!/bin/bash
#SBATCH --job-name=dps-main
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/dps_main_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/dps_main_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment & Path Setup ──────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env"
export PATH="$ENV_PATH/bin:$PATH"
PYTHON="$ENV_PATH/bin/python"

# ── 2. Redirect Caches to Lab (Fixes "No space left on device") ──────────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"

# Ensure these directories exist
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification (Prints to your .log file) ───────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
echo "Python Executable: $($PYTHON -c 'import sys; print(sys.executable)')"
$PYTHON -c "import torch; print(f'GPU Check: {torch.cuda.is_available()}')"
echo "==================================="

# ── 4. Runtime Configs ───────────────────────────────────────────────────────
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to project
cd /sci/labs/orzuk/ori_m/conditional-matching-paper
mkdir -p SD_cond_SD_controlnet/output

# ── 5. Run the Pipeline ──────────────────────────────────────────────────────
$PYTHON SD_cond_SD_controlnet/run_dps.py \
    --output_dir SD_cond_SD_controlnet/output/dps_main_${SLURM_JOB_ID} \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --n_targets 6 \
    --base_zeta 2.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --n_eval 6 \
    --sprinter_variation_prompt "a superrealistic professional photograph of" \
    --sprinter_target_man_prompt "a superrealistic portrait photograph of a man, studio lighting" \
    --sprinter_target_woman_prompt "a superrealistic portrait photograph of a woman, studio lighting" \
    --sprinter_eval_prompt "a superrealistic professional photograph of" \
    --architect_model_id "stabilityai/sdxl-turbo" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --loss_fn swd \
    --loss_scale 100.0 \
    --bandwidth_scale 0.3 \
    --kernel_alpha 1.0 \
    --seed 1

# ── 6. Sync to GDrive ────────────────────────────────────────────────────────
OUTPUT_DIR=SD_cond_SD_controlnet/output/dps_main_${SLURM_JOB_ID}
echo "Syncing $OUTPUT_DIR to Google Drive..."
rclone copy "$OUTPUT_DIR" "gdrive:conditional-matching/runs/dps_main_${SLURM_JOB_ID}" \
    --tpslimit 10 --cache-rps 50 --transfers 4
echo "✅ Job Complete."