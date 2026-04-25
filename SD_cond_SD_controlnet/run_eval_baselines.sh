#!/bin/bash
#SBATCH --job-name=eval-baselines
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/eval_baselines_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/eval_baselines_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Usage:
#   sbatch run_eval_baselines.sh SkewedTarget 241
#   sbatch run_eval_baselines.sh BalancedTarget 241
#   sbatch run_eval_baselines.sh GenderInterpolation 241
#   sbatch run_eval_baselines.sh AgeInterpolation 177

EXPERIMENT=${1:-SkewedTarget}
LGD_CM_MINUTES=${2:-241}

# ── 1. Environment ────────────────────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ── 2. Caches ─────────────────────────────────────────────────────────────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
echo "Experiment : $EXPERIMENT"
echo "LGD-CM min : $LGD_CM_MINUTES"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# ── 5. Run ────────────────────────────────────────────────────────────────────
python SD_cond_SD_controlnet/eval_baselines.py \
    --experiment "$EXPERIMENT" \
    --lgd_cm_minutes "$LGD_CM_MINUTES" \
    --repo_path "/sci/labs/orzuk/ori_m/conditional-matching-paper/SD_cond_SD_controlnet" \
    --gdrive_root "gdrive:conditional-matching/runs"

echo "✅ eval_baselines done."
