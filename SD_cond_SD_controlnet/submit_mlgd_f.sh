#!/bin/bash
#SBATCH --job-name=mlgd-f
#SBATCH --output=mlgd_f_%j.log
#SBATCH --error=mlgd_f_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch submit_mlgd_f.sh
source "$ENV_PATH/bin/activate"

# ── 2. Caches (optional — redirect if your home quota is limited) ─────────────
# export HF_HOME=/path/to/hf_cache
# export MPLCONFIGDIR=/path/to/matplotlib_cache
# export XDG_CACHE_HOME=/path/to/cache

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Repo root — adjust to wherever you cloned the repo
REPO="$(cd "$(dirname "$0")" && pwd)"
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

# ── 7. (Optional) Sync outputs ────────────────────────────────────────────────
# Uncomment and adjust if you want to sync results to remote storage:
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/mlgd_f_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
