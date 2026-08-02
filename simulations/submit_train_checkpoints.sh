#!/bin/bash
#SBATCH --job-name=cdm-train-ckpt
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=catfish   # override with sbatch --partition=

# ══════════════════════════════════════════════════════════════════════════════
# One-time, purely-local training of the CM + unconditional-diffusion
# checkpoints that run_hparam_sweep.py needs. No HuggingFace, no network
# access required. Run this once per experiment before submitting the sweep.
#
# Submit with:
#   export ENV_PATH=/path/to/your/conda/or/venv/env   # dir containing bin/python
#   export REPO_ROOT=/path/to/conditional-matching-paper
#   sbatch simulations/submit_train_checkpoints.sh
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME="2D_cond_1D"
SEED=42

export ENV_PATH="${ENV_PATH:?ENV_PATH is not set. Export it before submitting (dir containing bin/python).}"
PYTHON="$ENV_PATH/bin/python"

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT/simulations"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"
mkdir -p logs checkpoints/"$EXPERIMENT_NAME"

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    experiment : $EXPERIMENT_NAME"
"$PYTHON" -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

"$PYTHON" train_checkpoints.py --experiment_name "$EXPERIMENT_NAME" --seed "$SEED"

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
