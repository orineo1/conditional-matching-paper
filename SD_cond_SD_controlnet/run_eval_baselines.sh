#!/bin/bash
#SBATCH --job-name=eval-baselines
#SBATCH --output=eval_baselines_%j.log
#SBATCH --error=eval_baselines_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Usage:
#   export ENV_PATH=/path/to/your/env
#   sbatch slurm/run_eval_baselines.sh SkewedTarget 241
#   sbatch slurm/run_eval_baselines.sh BalancedTarget 241
#   sbatch slurm/run_eval_baselines.sh GenderInterpolation 241
#   sbatch slurm/run_eval_baselines.sh AgeInterpolation 177

EXPERIMENT=${1:-SkewedTarget}
MLGD_F_MINUTES=${2:-241}

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH before submitting:
#   export ENV_PATH=/path/to/your/env
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
echo "Experiment    : $EXPERIMENT"
echo "MLGD-F budget : ${MLGD_F_MINUTES} min"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="YOUR_REPO_PATH_HERE"
cd "$REPO"

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/eval_baselines.py \
    --experiment      "$EXPERIMENT" \
    --lgd_cm_minutes  "$MLGD_F_MINUTES" \
    --repo_path       "$REPO" \
    --wandb_project   "eval-baselines"

# To sync outputs to a remote (e.g. GDrive via rclone), add:
#   --gdrive_root "remote:your-bucket/baselines"

echo "✅ eval_baselines done."
