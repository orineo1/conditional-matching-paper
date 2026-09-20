#!/bin/bash
#SBATCH --job-name=pgd
#SBATCH --output=pgd_%j.log
#SBATCH --error=pgd_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# Usage:
#   export ENV_PATH=/path/to/your/env
#   sbatch pgd_submit.sh <EXPERIMENT> <TARGET_MINUTES> [SEED] [LOSS_FN]
#
#   EXPERIMENT     one of GenderTarget100, GenderTarget1, SkewedTarget,
#                  BalancedTarget, GenderInterpolation, AgeInterpolation
#                  (see EXPERIMENT_PRESETS in scripts/run_pgd.py)
#   TARGET_MINUTES wall-clock budget in minutes -- set this to the matching
#                  MLGD-F run's measured runtime so both methods get the
#                  same time budget (see mlgd_f_<jobid>.log, or
#                  experiments/<Experiment>/baselines/baselines_meta.json)
#   SEED           optional -- overrides the experiment preset's default seed
#   LOSS_FN        optional -- "mmd" (default) or "l2"
#
# Examples:
#   sbatch pgd_submit.sh GenderTarget100 241
#   sbatch pgd_submit.sh GenderTarget1   241 1    l2
#   sbatch pgd_submit.sh SkewedTarget    241
#   sbatch pgd_submit.sh BalancedTarget  241            # vs. existing MLGD-F BalancedTarget run
#   sbatch pgd_submit.sh GenderInterpolation 241
#   sbatch pgd_submit.sh AgeInterpolation    177

EXPERIMENT=${1:?experiment name required, e.g. GenderTarget100}
TARGET_MINUTES=${2:?target_minutes required -- the matching MLGD-F runtime}
SEED=${3:-}
LOSS_FN=${4:-mmd}

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
source "$ENV_PATH/bin/activate"

# ── 2. Caches (optional — redirect if your home quota is limited) ─────────────
# export HF_HOME=/path/to/hf_cache
# export MPLCONFIGDIR=/path/to/matplotlib_cache
# export XDG_CACHE_HOME=/path/to/cache

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
echo "Experiment      : $EXPERIMENT"
echo "Target minutes  : $TARGET_MINUTES"
echo "Seed override   : ${SEED:-<preset default>}"
echo "Loss fn         : $LOSS_FN"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Path to THIS subdirectory (SD_cond_SD_controlnet/), not the outer git repo root —
# scripts/run_pgd.py below is resolved relative to this path.
REPO="YOUR_REPO_PATH_HERE/SD_cond_SD_controlnet"
cd "$REPO"

OUTPUT_DIR="$REPO/output/pgd_${EXPERIMENT}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Optimization-step (pixel-space gradient descent) vs. projection-step
# (Adam search onto the VAE decoder's range) split within the fixed
# target_minutes budget -- tune these to compare splits at equal total cost.
OPT_STEPS=3
PROJ_ADAM_STEPS=200

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_pgd.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "pgd-${EXPERIMENT,,}" \
    --experiment "$EXPERIMENT" \
    --target_minutes "$TARGET_MINUTES" \
    ${SEED:+--seed "$SEED"} \
    --loss_fn "$LOSS_FN" \
    --opt_steps "$OPT_STEPS" \
    --opt_lr 0.05 \
    --proj_adam_steps "$PROJ_ADAM_STEPS" \
    --proj_lr 0.03 \
    --num_variations 6

# ── 6. (Optional) Sync outputs ────────────────────────────────────────────────
# Uncomment and adjust if you want to sync results to remote storage:
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/pgd_${EXPERIMENT}_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
