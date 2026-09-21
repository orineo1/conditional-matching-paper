#!/bin/bash
#SBATCH --job-name=select-statevar-normON-10D
#SBATCH --output=select_statevar_normON_10D_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=select_statevar_normON_10D_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# Runs ONLY backsel_state_gradient_variance.py -- the frozen-state gradient
# variance diagnostic (uniform vs witness, at a handful of real trajectory
# states, redrawn 200+ times each). No N_RUNS sweep, no k_frac/rule grid: a
# few dozen forward/backward passes per state instead of run_backsel_
# witness_sweep.sh's full end-to-end grid, so this finishes in minutes.
# ══════════════════════════════════════════════════════════════════════════════

# Which experiment?  2D_cond_1D | 5D_cond_1D | 10D_cond_1D
EXPERIMENT="10D_cond_1D"

METHODS="LGD-CM"                  # any of: LGD LGD-CM
STATE_SEEDS="1 2 3"            # trajectory seeds to capture states from (2-3 recommended)
STEP_FRACS="0.1 0.5 0.9"       # early/mid/late positions along the denoising trajectory
                                # (0.0=earliest/noisiest, 1.0=latest/cleanest)
NSAMPLES=250
K_FRAC=0.5                     # backsel_k / nsamples, held fixed for this diagnostic
WITNESS_FLOOR=0.3
N_REDRAWS=200                  # independent redraws per state per rule
SEED=42
NORMALIZE_BY_K_FRAC=true      # true = rescale uniform/witness's raw gradient by 1/k_frac
                                # before variance/bias_vs_ref stats (no-op for full/ref).
                                # Run once false, once true (same everything else) to separate
                                # a real witness-vs-uniform bias_vs_ref gap from a gradient-
                                # shrinkage artifact -- see backsel_state_gradient_variance.py.
GRAD_REF_N=2000                # sample size for the TRUE/population reference gradient
FORCE_RETRAIN=false

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
echo "    normalize_by_k_frac : $NORMALIZE_BY_K_FRAC"
echo "    grad_ref_n    : $GRAD_REF_N"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Run
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

CMD="python backsel_state_gradient_variance.py \
    --experiment      $EXPERIMENT \
    --methods          $METHODS \
    --state_seeds       $STATE_SEEDS \
    --step_fracs         $STEP_FRACS \
    --nsamples            $NSAMPLES \
    --k_frac               $K_FRAC \
    --witness_floor          $WITNESS_FLOOR \
    --n_redraws               $N_REDRAWS \
    --seed                     $SEED \
    --grad_ref_n                $GRAD_REF_N"

[ "$NORMALIZE_BY_K_FRAC" = "true" ] && CMD="$CMD --normalize_by_k_frac"
[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
