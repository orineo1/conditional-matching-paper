#!/bin/bash
#SBATCH --job-name=cdm-hparam-sweep
#SBATCH --output=logs/hparam_sweep_%A_%a.log
#SBATCH --error=logs/hparam_sweep_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition
#SBATCH --array=0-12                 # <-- 7 nsamples points + 6 num_x_t points, see grid below

# ══════════════════════════════════════════════════════════════════════════════
# MLGD-F hyperparameter sensitivity sweep (nsamples, num_x_t) on the synthetic
# GMM experiments. Reviewer question: "How sensitive are results to the choice
# of [nsamples] and [num_x_t]? Is there a principled way to set these, or are
# they tuned per task?"
#
# Runs two 1D ablations around the paper's default (nsamples=250, num_x_t=3
# for 2D_cond_1D):
#   - vary nsamples,  num_x_t fixed at its default
#   - vary num_x_t,   nsamples fixed at its default
# Each grid point launches run_hparam_sweep.py, which itself performs
# N_ATTEMP_OPTIM=25 independent restarts and reports mean/std, matching how
# the paper's own tables are built.
#
# Submit with:
#   export ENV_PATH=/path/to/your/env
#   export REPO_ROOT=/path/to/conditional-matching-paper
#   export HF_TOKEN=hf_...
#   sbatch --partition=your_partition simulations/submit_hparam_sweep.sh
# ══════════════════════════════════════════════════════════════════════════════

# ── Which experiment (2D_cond_1D | 5D_cond_1D | 10D_cond_1D) ──────────────────
EXPERIMENT_NAME="2D_cond_1D"

# ── Defaults for the fixed hyperparameter in each 1D sweep ────────────────────
DEFAULT_NSAMPLES=250
DEFAULT_NUM_X_T=3

# ── Grid points ─────────────────────────────────────────────────────────────
# Array indices 0-6  -> nsamples sweep (num_x_t held at DEFAULT_NUM_X_T)
# Array indices 7-12 -> num_x_t sweep  (nsamples held at DEFAULT_NSAMPLES)
NSAMPLES_GRID=(25 50 100 250 500 1000 2000)
NUM_X_T_GRID=(1 2 3 5 10 20)

N_ATTEMPTS=25
SEED=42

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

export HF_TOKEN="${HF_TOKEN:-}"   # optional: only needed if the HF checkpoint repo requires auth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT/simulations"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"
mkdir -p logs

# ══════════════════════════════════════════════════════════════════════════════
# 2. Resolve this array task to a (nsamples, num_x_t) grid point
# ══════════════════════════════════════════════════════════════════════════════
IDX=${SLURM_ARRAY_TASK_ID:-0}
NS_COUNT=${#NSAMPLES_GRID[@]}

if [ "$IDX" -lt "$NS_COUNT" ]; then
    NSAMPLES=${NSAMPLES_GRID[$IDX]}
    NUM_X_T=$DEFAULT_NUM_X_T
    SWEEP_AXIS="nsamples"
else
    XT_IDX=$((IDX - NS_COUNT))
    NSAMPLES=$DEFAULT_NSAMPLES
    NUM_X_T=${NUM_X_T_GRID[$XT_IDX]}
    SWEEP_AXIS="num_x_t"
fi

echo "=== JOB ${SLURM_ARRAY_JOB_ID:-manual}_${IDX} ON $(hostname) ==="
echo "    experiment   : $EXPERIMENT_NAME"
echo "    sweep_axis   : $SWEEP_AXIS"
echo "    nsamples     : $NSAMPLES"
echo "    num_x_t      : $NUM_X_T"
echo "    n_attempts   : $N_ATTEMPTS"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Run
# ══════════════════════════════════════════════════════════════════════════════
python run_hparam_sweep.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --nsamples        "$NSAMPLES" \
    --num_x_t         "$NUM_X_T" \
    --n_attempts      "$N_ATTEMPTS" \
    --seed            "$SEED" \
    --sweep_tag       "hparam_sweep"

EXIT_CODE=$?
echo "=== JOB ${SLURM_ARRAY_JOB_ID:-manual}_${IDX} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
