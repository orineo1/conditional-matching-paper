#!/bin/bash
#SBATCH --job-name=witness-backsel-sweep
#SBATCH --output=witness_sweep_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=witness_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# End-to-end L2/MMD grid: nsamples x k_frac x rule (uniform/witness) vs. the
# full (no-subsampling) baseline. See run_backsel_witness_sweep.py's docstring
# for full details. For gradient-variance diagnostics instead, use the
# separate run_backsel_state_variance.sh.
# ══════════════════════════════════════════════════════════════════════════════

# Which experiment?  2D_cond_1D | 5D_cond_1D | 10D_cond_1D
EXPERIMENT="5D_cond_1D"

# ── Shared hyperparameters ────────────────────────────────────────────────────
NUM_X_T=1                     # fixed, not swept
N_RUNS=25
SEED=42
METHODS="${METHODS:-LGD LGD-CM}"  # any of: LGD LGD-CM; overridable via env (see
                                   # "SPLITTING INTO TWO JOBS" below)
NSAMPLES_LIST="50 100 250 500" # the "n" axis
K_FRACS="0.1 0.2 0.5 1.0"      # the "proportion" axis (backsel_k / nsamples);
                                # 1.0 (full baseline) is always included
RULES="uniform witness"        # any of: uniform witness
WITNESS_FLOOR=0.3              # defensive-mixture floor for rule=witness (0.3-0.5)
WITNESS_TEMPERATURE=1.0        # |score|^(1/T) before the floor blend; T>1 flattens
                                # toward uniform, T<1 sharpens toward top-|score| rows
BACKSEL_REPLACEMENT=false      # true = sample backsel_k indices with replacement
NORMALIZE_BY_K_FRAC=false      # true = rescale grad by 1/k_frac (magnitude-normalize
                                # across k_frac); no-op at k_frac=1.0
USE_INV_SQRT_ALPHA_SCALE=false # true = scale grad by 1/sqrt(alpha_t) instead of zeta
FORCE_RETRAIN=false            # true = always retrain, overwriting saved checkpoints

# ── Diagnostics (see run_backsel_witness_sweep.py's docstring for detail) ──────
DIAG_STEPS="99 75 50 25 1"     # t-values to log grad_norm_error(_vs_ref) at;
                                # empty = disabled. RAW per-seed values only, not
                                # a variance analysis (use run_backsel_state_
                                # variance.sh for that)
GRAD_REF_N=2000                # sample size for the true/population reference
                                # gradient used by grad_norm_error_vs_ref
ALPHA_LIST=""                  # sweep witness_floor over these values; empty =
                                # just WITNESS_FLOOR

# ── Misc ──────────────────────────────────────────────────────────────────────
SMOKE_TEST=false               # true = 2 runs / tiny grid only, for quick debug

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
echo "    witness_temperature  : $WITNESS_TEMPERATURE"
echo "    backsel_replacement  : $BACKSEL_REPLACEMENT"
echo "    normalize_by_k_frac  : $NORMALIZE_BY_K_FRAC"
echo "    use_inv_sqrt_alpha   : $USE_INV_SQRT_ALPHA_SCALE"
echo "    diag_steps           : ${DIAG_STEPS:-(disabled)}"
echo "    grad_ref_n           : $GRAD_REF_N"
echo "    alpha_list           : ${ALPHA_LIST:-(just witness_floor)}"
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
    --witness_temperature   $WITNESS_TEMPERATURE"

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

# ══════════════════════════════════════════════════════════════════════════════
# SPLITTING INTO TWO JOBS (LGD on one machine, LGD-CM on another)
# ══════════════════════════════════════════════════════════════════════════════
# Safe -- outputs are per-method already -- but load_or_train_models() always
# loads/trains all three models regardless of --methods, so two jobs starting
# simultaneously with no cached checkpoints would race. Fix: run one small
# warm-up job first, then the two real jobs:
#
#   export REPO_ROOT=... ENV_PATH=...
#   METHODS="LGD LGD-CM" sbatch --job-name=warmup \
#       --export=ALL,SMOKE_TEST=true run_backsel_witness_sweep.sh
#   # wait for it to finish, then:
#   METHODS="LGD"    sbatch --job-name=witness-LGD    run_backsel_witness_sweep.sh
#   METHODS="LGD-CM" sbatch --job-name=witness-LGD-CM run_backsel_witness_sweep.sh
