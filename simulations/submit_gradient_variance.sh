#!/bin/bash
#SBATCH --job-name=cdm-grad-var
#SBATCH --output=logs/gradvar_%j.log
#SBATCH --error=logs/gradvar_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=catfish   # override with sbatch --partition=

# ══════════════════════════════════════════════════════════════════════════════
# Reviewer Q1: does unrolling deeper diffusion chains inject more noise into
# the guidance gradient? Isolates the inner conditional sampler (model_cond)
# at a fixed x, fixed target samples, varying only the sampler's internal
# noise across 100 trials per K, for K = 10,25,40,60,80,100 DDIM steps.
#
# Submit with:
#   export ENV_PATH=/path/to/your/conda/or/venv/env   # dir containing bin/python
#   export REPO_ROOT=/path/to/conditional-matching-paper
#   export EXPERIMENT_NAME=10D_cond_1D                # optional, defaults to 2D_cond_1D
#   export X_CONDS="-5|0|5"                           # optional, sweep several x points
#                                                      # instead of just the saved x_star
#                                                      # (pipe-separated; each may itself be
#                                                      # space-separated for multi-dim x, e.g.
#                                                      # X_CONDS="1.0 2.0|3.0 4.0")
#   sbatch simulations/submit_gradient_variance.sh
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME="${EXPERIMENT_NAME:-2D_cond_1D}"   # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
K_VALUES="10,25,40,60,80,100"
N_TRIALS=100
NSAMPLES=250
SEED=42
X_CONDS="${X_CONDS:--2|0|2}"                       # empty = just the saved x_star

export ENV_PATH="${ENV_PATH:?ENV_PATH is not set. Export it before submitting (dir containing bin/python).}"
PYTHON="$ENV_PATH/bin/python"

export HF_TOKEN="${HF_TOKEN:-}"   # optional: only needed if Diffusion_cond isn't cached locally yet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT/simulations"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"
mkdir -p logs

# X_CONDS="-5|0|5" -> XCOND_ARGS=(--x_conds -5 --x_conds 0 --x_conds 5)
XCOND_ARGS=()
if [ -n "$X_CONDS" ]; then
    IFS='|' read -ra XCOND_POINTS <<< "$X_CONDS"
    for point in "${XCOND_POINTS[@]}"; do
        XCOND_ARGS+=(--x_conds $point)
    done
fi

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    experiment : $EXPERIMENT_NAME"
echo "    K_values   : $K_VALUES"
echo "    n_trials   : $N_TRIALS"
echo "    x_conds    : ${X_CONDS:-(experiment's saved x_star)}"
"$PYTHON" -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

"$PYTHON" gradient_variance_vs_unroll_depth.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --k_values         "$K_VALUES" \
    --n_trials          "$N_TRIALS" \
    --nsamples          "$NSAMPLES" \
    --seed              "$SEED" \
    "${XCOND_ARGS[@]}"

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
