#!/bin/bash
#SBATCH --job-name=mnist-lgd
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/lgd_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/lgd_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

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
STEP_SIZE_MODE="double"          # original | half | double | tripleLinear | doubleLinear | no_linear | dps
NUM_X_T=3
NSAMPLES=1500
CLAMP=false                       # true | false

# ── Misc ──────────────────────────────────────────────────────────────────────
SMOKE_TEST=false                 # true = 2 seeds only, for quick debug
WANDB_ENTITY=""

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ══════════════════════════════════════════════════════════════════════════════
# 2. Caches
# ══════════════════════════════════════════════════════════════════════════════
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Keys
# ══════════════════════════════════════════════════════════════════════════════
export REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
export HF_TOKEN="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"
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Directories
# ══════════════════════════════════════════════════════════════════════════════
mkdir -p "$REPO_ROOT/MNIST/logs"
mkdir -p "$REPO_ROOT/MNIST/checkpoints"
mkdir -p "$REPO_ROOT/MNIST/results/${EXPERIMENT}_run"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
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
    python MNIST_MLGDF.py --train_classifier_only
fi

# ══════════════════════════════════════════════════════════════════════════════
# 7. Build python command
# ══════════════════════════════════════════════════════════════════════════════
CMD="python MNIST_MLGDF.py \
    --experiment $EXPERIMENT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --step_size_mode $STEP_SIZE_MODE \
    --num_x_t $NUM_X_T \
    --nsamples $NSAMPLES \
    --unimodal_var $UNIMODAL_VAR \
    --bimodal_var $BIMODAL_VAR \
    --wandb_entity \"$WANDB_ENTITY\""

# add optional flags
[ "$CLAMP"      = "true" ] && CMD="$CMD --clamp"
[ "$SMOKE_TEST" = "true" ] && CMD="$CMD --smoke_test"

# ══════════════════════════════════════════════════════════════════════════════
# 8. Run
# ══════════════════════════════════════════════════════════════════════════════
echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
