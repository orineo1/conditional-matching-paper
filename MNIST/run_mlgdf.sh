#!/bin/bash
#SBATCH --job-name=mnist-MLGDF
#SBATCH --output=logs/MLGDF_%j.log
#SBATCH --error=logs/MLGDF_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR RUN HERE
# ══════════════════════════════════════════════════════════════════════════════

# Which experiment?  unimodal | bimodal | uniform
EXPERIMENT="unimodal"

# ── Unimodal settings (ignored for bimodal/uniform) ──────────────────────────
UNIMODAL_VAR=515

# ── Bimodal settings (ignored for unimodal/uniform) ──────────────────────────
BIMODAL_VAR=252

# ── Shared hyperparameters ────────────────────────────────────────────────────
NUM_INFERENCE_STEPS=130
STEP_SIZE_MODE="double"   # original | half | double | tripleLinear | doubleLinear | no_linear | dps
NUM_X_T=3
NSAMPLES=1500
CLAMP=false                # true | false

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_ENTITY=""           # your W&B username or team, or leave blank for default
WANDB_MODE="online"       # online | offline | disabled

# ── Misc ──────────────────────────────────────────────────────────────────────
SMOKE_TEST=false          # true = 2 seeds only, for quick debug

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch run_mlgdf.sh
source "$ENV_PATH/bin/activate"

# ══════════════════════════════════════════════════════════════════════════════
# 2. Caches  (optional — avoids re-downloading HF models every run)
# ══════════════════════════════════════════════════════════════════════════════
# Uncomment and set LAB_ROOT to a writable directory on your cluster:
# export LAB_ROOT="/path/to/your/lab/storage"
# export HF_HOME="$LAB_ROOT/hf_cache"
# export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
# mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Secrets
# ══════════════════════════════════════════════════════════════════════════════
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname $(dirname "$SCRIPT_DIR"))}"

mkdir -p "$REPO_ROOT/MNIST/logs"
mkdir -p "$REPO_ROOT/MNIST/checkpoints"
mkdir -p "$REPO_ROOT/MNIST/results/${EXPERIMENT}_run"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT           : $REPO_ROOT"
echo "    experiment          : $EXPERIMENT"
echo "    num_inference_steps : $NUM_INFERENCE_STEPS"
echo "    step_size_mode      : $STEP_SIZE_MODE"
echo "    num_x_t             : $NUM_X_T"
echo "    nsamples            : $NSAMPLES"
echo "    clamp               : $CLAMP"
[ "$EXPERIMENT" = "unimodal" ] && echo "    unimodal_var        : $UNIMODAL_VAR"
[ "$EXPERIMENT" = "bimodal"  ] && echo "    bimodal_var         : $BIMODAL_VAR"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 6. Train classifier if missing
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/MNIST"

CLF="$REPO_ROOT/MNIST/checkpoints/robust_classifier.pth"
if [ ! -f "$CLF" ]; then
    echo "No classifier found — training..."
    python run_mlgdf.py --train_classifier_only
fi

# ══════════════════════════════════════════════════════════════════════════════
# 7. Build and run python command
# ══════════════════════════════════════════════════════════════════════════════
CMD="python run_mlgdf.py \
    --experiment            $EXPERIMENT \
    --num_inference_steps   $NUM_INFERENCE_STEPS \
    --step_size_mode        $STEP_SIZE_MODE \
    --num_x_t               $NUM_X_T \
    --nsamples              $NSAMPLES \
    --unimodal_var          $UNIMODAL_VAR \
    --bimodal_var           $BIMODAL_VAR \
    --wandb_entity          \"$WANDB_ENTITY\" \
    --wandb_mode            $WANDB_MODE"

[ "$CLAMP"      = "true" ] && CMD="$CMD --clamp"
[ "$SMOKE_TEST" = "true" ] && CMD="$CMD --smoke_test"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
