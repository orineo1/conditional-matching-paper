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
#   sbatch simulations/submit_gradient_variance.sh
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME="${EXPERIMENT_NAME:-2D_cond_1D}"   # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
K_VALUES="10,25,40,60,80,100"
N_TRIALS=100
NSAMPLES=250
SEED=42

export ENV_PATH="${ENV_PATH:?ENV_PATH is not set. Export it before submitting (dir containing bin/python).}"
PYTHON="$ENV_PATH/bin/python"

export HF_TOKEN="${HF_TOKEN:-}"   # optional: only needed if Diffusion_cond isn't cached locally yet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT/simulations"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"
mkdir -p logs

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    experiment : $EXPERIMENT_NAME"
echo "    K_values   : $K_VALUES"
echo "    n_trials   : $N_TRIALS"
"$PYTHON" -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

"$PYTHON" gradient_variance_vs_unroll_depth.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --k_values         "$K_VALUES" \
    --n_trials          "$N_TRIALS" \
    --nsamples          "$NSAMPLES" \
    --seed              "$SEED"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    "$PYTHON" plot_gradient_variance.py --experiment_name "$EXPERIMENT_NAME" --seed "$SEED"
fi

echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
