#!/bin/bash
#SBATCH --job-name=mlgd-f-age
#SBATCH --output=mlgd_f_age_%j.log
#SBATCH --error=mlgd_f_age_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch submit_mlgd_f_age.sh
source "$ENV_PATH/bin/activate"

# ── 2. Caches — redirect to lab storage to avoid home quota issues ────────────
# Uncomment and set LAB_ROOT to a writable directory on your cluster:
# export LAB_ROOT="/path/to/your/lab/storage"
# export HF_HOME="$LAB_ROOT/hf_cache"
# export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
# export XDG_CACHE_HOME="$LAB_ROOT/.cache"
# mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="YOUR_REPO_PATH_HERE"
cd "$REPO"

OUTPUT_DIR="output/mlgd_f_age_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_mlgd_f.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "MLGDF-EXP" \
    --mode age \
    --age_min 10 \
    --age_max 80 \
    --age_step 1 \
    --n_per_age 0 \
    --age_gender man \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --loss_fn mmd \
    --seed 1

# ── 6. Offline analysis (run manually when needed) ────────────────────────────
# python src/analysis.py --run_dir "$OUTPUT_DIR" --plots_dir "$OUTPUT_DIR/plots"

# ── 7. (Optional) Sync outputs ────────────────────────────────────────────────
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/mlgd_f_age_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
