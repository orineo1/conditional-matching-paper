#!/bin/bash
#SBATCH --job-name=mlgd-f
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/mlgd_f_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/mlgd_f_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment ────────────────────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ── 2. Redirect caches to lab storage ─────────────────────────────────────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="/sci/labs/orzuk/ori_m/conditional-matching-paper/mlgd_f"
cd "$REPO"

OUTPUT_DIR="output/mlgd_f_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_mlgd_f.py \
    --output_dir "$OUTPUT_DIR" \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --n_targets 20 \
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --n_eval 6 \
    --loss_fn mmd \
    --loss_scale 1.0 \
    --bandwidth_scale 1.0 \
    --kernel_alpha 1.0 \
    --architect_model_id "stabilityai/sdxl-turbo" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --seed 1

# ── 6. Offline analysis ───────────────────────────────────────────────────────
echo "Running offline analysis..."
python src/analysis.py \
    --run_dir "$OUTPUT_DIR" \
    --plots_dir "$OUTPUT_DIR/plots"
echo "✅ Analysis complete."

# ── 7. Sync to GDrive ─────────────────────────────────────────────────────────
echo "Syncing to Google Drive..."
rclone copy "$OUTPUT_DIR" \
    "gdrive:conditional-matching/runs/mlgd_f_${SLURM_JOB_ID}" \
    --tpslimit 10 --cache-rps 50 --transfers 4
echo "✅ Done."
