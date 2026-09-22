#!/bin/bash
#SBATCH --job-name=witness-vs-uniform-downstream
#SBATCH --output=paired_downstream_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=paired_downstream_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# Runs witness_vs_uniform_paired_downstream_test.py -- the matched-compute,
# paired head-to-head of Witness vs. Uniform backsel selection at ONE fixed
# k_frac (not a sweep), with the exact per-row HT/importance rescaling wired
# into the actual guidance step (backsel_ht_rescale=True) and every restart
# using identical randomness (x_T, target draws, sampler noise) for both
# rules, so only the selection rule differs. Reports paired differences
# (mean +/- bootstrap 95% CI, Wilcoxon signed-rank p-value, win-count) on
# final MMD loss, L2 to the target GMM, and L2 to x*, for both "All restarts"
# and "Top-10" subsets. See the script docstring for the full design and why
# it differs from run_backsel_witness_sweep.py's (unpaired, swept) grid.
#
# All of these can be overridden without editing the file, e.g.:
#   sbatch --partition=catfish --export=ALL,EXPERIMENT=10D_cond_1D,N_RESTARTS=30 \
#       scripts/run_witness_vs_uniform_downstream.sh
# (sbatch itself only accepts its own flags -- so a job-specific setting like
# EXPERIMENT has to travel in via --export, not as a made-up "--EXPERIMENT=..."
# flag on the sbatch command line. The leading "ALL," is required -- without
# it, --export REPLACES the whole environment instead of adding to it, and
# this script's already-exported ENV_PATH/REPO_ROOT would be lost.)
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT="${EXPERIMENT:-5D_cond_1D}"          # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
METHODS="${METHODS:-LGD-CM}"                    # any of: LGD LGD-CM
NUM_X_T="${NUM_X_T:-3}"
NSAMPLES="${NSAMPLES:-250}"
K_FRAC="${K_FRAC:-0.5}"                         # FIXED nsel/ncond -- not swept, per the task spec
WITNESS_FLOOR="${WITNESS_FLOOR:-0.3}"
N_RESTARTS="${N_RESTARTS:-25}"                  # R, paired restarts (>=25 recommended)
TOP10_RANK_BY="${TOP10_RANK_BY:-min}"           # min | max | mean | witness | uniform
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
echo "    nsamples      : $NSAMPLES"
echo "    k_frac        : $K_FRAC"
echo "    n_restarts    : $N_RESTARTS"
echo "    top10_rank_by : $TOP10_RANK_BY"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Run
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

CMD="python witness_vs_uniform_paired_downstream_test.py \
    --experiment      $EXPERIMENT \
    --methods          $METHODS \
    --num_x_t           $NUM_X_T \
    --nsamples            $NSAMPLES \
    --k_frac               $K_FRAC \
    --witness_floor          $WITNESS_FLOOR \
    --n_restarts               $N_RESTARTS \
    --top10_rank_by              $TOP10_RANK_BY \
    --seed                        $SEED"

[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
