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
# paired head-to-head of backsel selection "arms" (Uniform, and one or more
# Witness temperatures) at ONE fixed k_frac (not a sweep), with every restart
# using identical randomness (x_T, target draws, sampler noise) across ALL
# arms, so only the selection rule/temperature/rescale-mode differs. Reports
# paired differences (mean +/- bootstrap 95% CI, Wilcoxon signed-rank
# p-value, win-count) on final MMD loss, L2 to the target GMM, and L2 to x*,
# for every pair of arms, for both "All restarts" and "Top-10" subsets. See
# the script docstring for the full design and why it differs from
# run_backsel_witness_sweep.py's (unpaired, swept) grid.
#
# All of these can be overridden without editing the file, e.g.:
#   sbatch --partition=catfish --export=ALL,EXPERIMENT=10D_cond_1D,WITNESS_TEMPERATURES="1.0 0.3" \
#       scripts/run_witness_vs_uniform_downstream.sh
# (sbatch itself only accepts its own flags -- so a job-specific setting like
# EXPERIMENT has to travel in via --export, not as a made-up "--EXPERIMENT=..."
# flag on the sbatch command line. The leading "ALL," is required -- without
# it, --export REPLACES the whole environment instead of adding to it, and
# this script's already-exported ENV_PATH/REPO_ROOT would be lost. Multi-value
# settings like WITNESS_TEMPERATURES/RESCALE_MODES must be one quoted,
# space-separated string, e.g. WITNESS_TEMPERATURES="1.0 0.3".)
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT="${EXPERIMENT:-5D_cond_1D}"          # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
METHODS="${METHODS:-LGD-CM}"                    # any of: LGD LGD-CM
NUM_X_T="${NUM_X_T:-3}"
NSAMPLES="${NSAMPLES:-250}"
K_FRAC="${K_FRAC:-0.5}"                         # FIXED nsel/ncond -- not swept, per the task spec
WITNESS_FLOOR="${WITNESS_FLOOR:-0.3}"
# One Witness arm per value (space-separated), labeled witness_T<value>. Include 1.0 for
# "no temperature" (the original plain-|scores| weighting) alongside any sharpened value(s),
# e.g. "1.0 0.3", to get a full Uniform / no-temperature-Witness / temperature-Witness comparison.
WITNESS_TEMPERATURES="${WITNESS_TEMPERATURES:-1.0}"
# Which gradient-rescale correction(s) to run (space-separated): 'ht' (exact per-row correction,
# unbiased for either rule -- the default), 'raw' (no rescaling at all -- the original/production
# apply_backsel behavior), 'kfrac' (flat post-hoc 1/k_frac correction, exact only for uniform).
# Pass e.g. "ht raw" to see how much the correction itself matters, not just the selection rule.
RESCALE_MODES="${RESCALE_MODES:-ht}"
N_RESTARTS="${N_RESTARTS:-25}"                  # R, paired restarts (>=25 recommended)
TOP10_METRIC="${TOP10_METRIC:-l2_gmm}"          # final_loss | l2_gmm | l2_x
TOP10_RANK_BY="${TOP10_RANK_BY:-min}"           # min | max | mean | a | b
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
    --witness_temperatures     $WITNESS_TEMPERATURES \
    --rescale_modes               $RESCALE_MODES \
    --n_restarts                    $N_RESTARTS \
    --top10_metric                    $TOP10_METRIC \
    --top10_rank_by                     $TOP10_RANK_BY \
    --seed                                $SEED"

[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
