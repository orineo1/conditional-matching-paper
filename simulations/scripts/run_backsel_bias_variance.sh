#!/bin/bash
#SBATCH --job-name=witness-bias-var
#SBATCH --output=bias_variance_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=bias_variance_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# Runs backsel_witness_bias_variance_test.py -- the properly-rescaled
# bias/variance/MSE + cosine-similarity test of All vs Uniform vs Witness
# gradient selection, against both the true (population/closed-form) and
# full-batch (no subsampling) reference gradients, at a handful of real
# trajectory states redrawn N_REDRAWS times each. This is the unbiased
# counterpart to run_backsel_state_variance.sh's raw normalized-variance
# comparison -- see the script docstring for why that distinction matters.
# ══════════════════════════════════════════════════════════════════════════════

# All of these can be overridden without editing the file, e.g.:
#   sbatch --partition=catfish --export=ALL,EXPERIMENT=10D_cond_1D scripts/run_backsel_bias_variance.sh
# (sbatch itself only accepts its own flags -- #SBATCH/scheduler options -- so a
# job-specific setting like EXPERIMENT has to travel in via --export, not as a
# made-up "--EXPERIMENT=..." flag on the sbatch command line. The leading "ALL,"
# is required -- without it, --export REPLACES the whole environment instead of
# adding to it, and this script's already-exported ENV_PATH/REPO_ROOT would be lost.)

EXPERIMENT="${EXPERIMENT:-5D_cond_1D}"          # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
METHODS="${METHODS:-LGD-CM}"                    # any of: LGD LGD-CM
STATE_SEEDS="${STATE_SEEDS:-1 2 3}"             # trajectory seeds to capture states from (2-3 recommended)
STEP_FRACS="${STEP_FRACS:-0.1 0.5 0.9}"         # early/mid/late positions along the denoising trajectory
                                                 # (0.0=earliest/noisiest, 1.0=latest/cleanest)
NSAMPLES="${NSAMPLES:-250}"
K_FRAC="${K_FRAC:-0.2}"                         # backsel_k / nsamples, held fixed for this diagnostic
WITNESS_FLOOR="${WITNESS_FLOOR:-0.3}"
N_REDRAWS="${N_REDRAWS:-200}"                   # independent redraws per state per rule
GRAD_REF_N="${GRAD_REF_N:-2000}"                # sample size for the TRUE/population reference gradient
SEED="${SEED:-42}"
FORCE_RETRAIN="${FORCE_RETRAIN:-false}"

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
echo "    REPO_ROOT     : $REPO_ROOT"
echo "    experiment    : $EXPERIMENT"
echo "    methods       : $METHODS"
echo "    state_seeds   : $STATE_SEEDS"
echo "    step_fracs    : $STEP_FRACS"
echo "    k_frac        : $K_FRAC"
echo "    n_redraws     : $N_REDRAWS"
echo "    grad_ref_n    : $GRAD_REF_N"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Run
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

CMD="python backsel_witness_bias_variance_test.py \
    --experiment      $EXPERIMENT \
    --methods          $METHODS \
    --state_seeds       $STATE_SEEDS \
    --step_fracs         $STEP_FRACS \
    --nsamples            $NSAMPLES \
    --k_frac               $K_FRAC \
    --witness_floor          $WITNESS_FLOOR \
    --n_redraws               $N_REDRAWS \
    --grad_ref_n               $GRAD_REF_N \
    --seed                      $SEED"

[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
