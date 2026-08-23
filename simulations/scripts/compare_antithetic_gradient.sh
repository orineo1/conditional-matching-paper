#!/bin/bash
#SBATCH --job-name=antithetic-gradient
#SBATCH --output=antithetic_gradient_%j.log
#SBATCH --error=antithetic_gradient_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR RUN HERE
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT="2D_cond_1D"
T=""                    # diffusion timestep to freeze x_t at; empty = mid-trajectory
NSAMPLES=250
N_TRIALS=200
N_REF=5000
SEED=42
FORCE_RETRAIN=true

# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════
source "${ENV_PATH}/bin/activate"   # export ENV_PATH=/path/to/your/env before submitting

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
mkdir -p "$REPO_ROOT/simulations/checkpoints/${EXPERIMENT}"
mkdir -p "$REPO_ROOT/simulations/results/${EXPERIMENT}"

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    experiment : $EXPERIMENT"
echo "    nsamples   : $NSAMPLES"
echo "    n_trials   : $N_TRIALS"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

CMD="python compare_antithetic_gradient.py \
    --experiment $EXPERIMENT \
    --nsamples   $NSAMPLES \
    --n_trials   $N_TRIALS \
    --n_ref      $N_REF \
    --seed       $SEED"

[ -n "$T" ] && CMD="$CMD --t $T"
[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
