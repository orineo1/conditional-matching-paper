#!/bin/bash
#SBATCH --job-name=select-control3-normalized-10D
#SBATCH --output=select_control3_normalized_10D_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=select_control3_normalized_10D_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition, or override at
                                      #     submit time: sbatch --partition=<name> sbatch_select_control3_normalized_10D.sh

# ══════════════════════════════════════════════════════════════════════════════
# Control 3: Uniform, n=250, k_frac=0.5, gradient rescaled by n/nsel
# (--normalize_by_k_frac; removes the implicit gradient-magnitude shrinkage,
# tests whether Uniform's edge over the n=250,k_frac=1.0 baseline survives it)
# Before submitting: export REPO_ROOT=/path/to/conditional-matching-paper
#                     export ENV_PATH=/path/to/your/venv
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT="10D_cond_1D"

# ── Shared hyperparameters ────────────────────────────────────────────────────
NUM_X_T=1
N_RUNS=25
SEED=42
METHODS="${METHODS:-LGD-CM}"
NSAMPLES_LIST="250"
K_FRACS="0.5"
RULES="uniform"
WITNESS_FLOOR=0.3
BACKSEL_REPLACEMENT=false
NORMALIZE_BY_K_FRAC=true
USE_INV_SQRT_ALPHA_SCALE=false
RUN_TAG="control3_normalized"
ZETA=1.0
FORCE_RETRAIN=false

DIAG_STEPS=""
GRAD_REF_N=2000
ALPHA_LIST=""

SMOKE_TEST=false

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting

# ══════════════════════════════════════════════════════════════════════════════
# 2. Caches
# ══════════════════════════════════════════════════════════════════════════════
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
mkdir -p "$REPO_ROOT/simulations/checkpoints/${EXPERIMENT}"
mkdir -p "$REPO_ROOT/simulations/results/${EXPERIMENT}"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT            : $REPO_ROOT"
echo "    experiment           : $EXPERIMENT"
echo "    num_x_t              : $NUM_X_T"
echo "    n_runs               : $N_RUNS"
echo "    methods              : $METHODS"
echo "    nsamples_list        : $NSAMPLES_LIST"
echo "    k_fracs              : $K_FRACS"
echo "    rules                : $RULES"
echo "    witness_floor        : $WITNESS_FLOOR"
echo "    backsel_replacement  : $BACKSEL_REPLACEMENT"
echo "    normalize_by_k_frac  : $NORMALIZE_BY_K_FRAC"
echo "    use_inv_sqrt_alpha   : $USE_INV_SQRT_ALPHA_SCALE"
echo "    zeta                 : $ZETA"
echo "    run_tag              : $RUN_TAG"
echo "    force_retrain        : $FORCE_RETRAIN"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Build and run python command
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

if [ "$SMOKE_TEST" = "true" ]; then
    N_RUNS=2
    NSAMPLES_LIST="20 50"
    K_FRACS="0.5 1.0"
fi

CMD="python run_backsel_witness_sweep.py \
    --experiment          $EXPERIMENT \
    --num_x_t             $NUM_X_T \
    --n_runs              $N_RUNS \
    --seed                $SEED \
    --methods              $METHODS \
    --nsamples_list        $NSAMPLES_LIST \
    --k_fracs               $K_FRACS \
    --rules                 $RULES \
    --witness_floor        $WITNESS_FLOOR \
    --zeta                 $ZETA \
    --run_tag              $RUN_TAG"

[ "$BACKSEL_REPLACEMENT" = "true" ] && CMD="$CMD --backsel_replacement"
[ "$NORMALIZE_BY_K_FRAC" = "true" ] && CMD="$CMD --normalize_by_k_frac"
[ "$USE_INV_SQRT_ALPHA_SCALE" = "true" ] && CMD="$CMD --use_inv_sqrt_alpha_scale"
[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"
[ -n "$DIAG_STEPS" ] && CMD="$CMD --diag_steps $DIAG_STEPS --grad_ref_n $GRAD_REF_N"
[ -n "$ALPHA_LIST" ] && CMD="$CMD --alpha_list $ALPHA_LIST"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
