#!/bin/bash
#SBATCH --job-name=reuse-adamdps-grid
#SBATCH --output=reuse_adamdps_%j.log
#SBATCH --error=reuse_adamdps_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR RUN HERE
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT="5D_cond_1D"      # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
NUM_X_T=3
N_RUNS=25
NSAMPLES=250
SEED=42
METHODS="LGD LGD-CM"
REUSE_FRACS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
ADAMDPS="0 1"                 # 0=off, 1=on (beta1=0.9, beta2=0.999)
FORCE_RETRAIN=true
SMOKE_TEST=false              # true = 2 runs / tiny grid, for quick debug

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
echo "    experiment  : $EXPERIMENT"
echo "    reuse_fracs : $REUSE_FRACS"
echo "    adamdps     : $ADAMDPS"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

if [ "$SMOKE_TEST" = "true" ]; then
    N_RUNS=2
    REUSE_FRACS="0.0 0.5"
fi

CMD="python run_reuse_adamdps_grid.py \
    --experiment  $EXPERIMENT \
    --num_x_t     $NUM_X_T \
    --n_runs      $N_RUNS \
    --nsamples    $NSAMPLES \
    --seed        $SEED \
    --methods     $METHODS \
    --reuse_fracs $REUSE_FRACS \
    --adamdps     $ADAMDPS"

[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
