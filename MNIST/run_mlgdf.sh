#!/bin/bash
#SBATCH --job-name=mnist-MLGDF
#SBATCH --output=logs/MLGDF_%j.log
#SBATCH --error=logs/MLGDF_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

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

# ── TFG Mean Guidance (Algorithm 1, Line 8) ───────────────────────────────────
TFG_N_ITER=0               # 0 = plain LGD (no mean guidance)
TFG_MU=0.0                 # mean guidance step size

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_ENTITY=""           # your W&B username or team, or leave blank for default
WANDB_MODE="online"       # online | offline | disabled

# ── Misc ──────────────────────────────────────────────────────────────────────
SMOKE_TEST=false          # true = 2 seeds only, for quick debug

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
# Activate your conda/venv environment:
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting, e.g.:
                                    #   export ENV_PATH=/path/to/your/env

# ══════════════════════════════════════════════════════════════════════════════
# 2. Caches  (optional — avoids re-downloading HF models every run)
# ══════════════════════════════════════════════════════════════════════════════
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Secrets  — READ FROM ENVIRONMENT, never hardcode here
#    Set these in your ~/.bashrc or pass via `sbatch --export`:
#        export HF_TOKEN=hf_...
#        export WANDB_API_KEY=...
# ══════════════════════════════════════════════════════════════════════════════
: "${HF_TOKEN:?HF_TOKEN is not set. Export it before submitting.}"
export HF_TOKEN
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # optional; wandb also reads ~/.netrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
# REPO_ROOT should point to the root of this repository.
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"

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
echo "    tfg_n_iter          : $TFG_N_ITER"
echo "    tfg_mu              : $TFG_MU"
[ "$EXPERIMENT" = "unimodal" ] && echo "    unimodal_var        : $UNIMODAL_VAR"
[ "$EXPERIMENT" = "bimodal"  ] && echo "    bimodal_var         : $BIMODAL_VAR"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 6. Train classifier if missing
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/MNIST"
export PYTHONPATH="$REPO_ROOT/MNIST/src:$PYTHONPATH"

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
    --tfg_n_iter            $TFG_N_ITER \
    --tfg_mu                $TFG_MU \
    --wandb_entity          \"$WANDB_ENTITY\" \
    --wandb_mode            $WANDB_MODE"

[ "$CLAMP"      = "true" ] && CMD="$CMD --clamp"
[ "$SMOKE_TEST" = "true" ] && CMD="$CMD --smoke_test"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
