#!/bin/bash
#SBATCH --job-name=hybrid_sweep
#SBATCH --output=logs/hybrid_sweep_%A_%a.out
#SBATCH --error=logs/hybrid_sweep_%A_%a.err
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# One array task per experiment: 0 -> 2D_cond_1D, 1 -> 5D_cond_1D, 2 -> 10D_cond_1D.
# Adjust --array above if you only want a subset (e.g. --array=1-2 to skip 2D).
#
# Usage:
#   mkdir -p logs
#   sbatch run_hybrid_sweep.sbatch
#
# Override the sweep grid / methods / run count via env vars, e.g.:
#   N_RUNS=10 METHODS="LGD" sbatch run_hybrid_sweep.sbatch

set -euo pipefail

EXPERIMENTS=(2D_cond_1D 5D_cond_1D 10D_cond_1D)
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

NUM_X_T=${NUM_X_T:-3}
N_RUNS=${N_RUNS:-25}
SEED=${SEED:-42}
METHODS=${METHODS:-"LGD LGD-CM"}
REUSE_FRACS=${REUSE_FRACS:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"}

echo "Running hybrid sweep: experiment=${EXPERIMENT} num_x_t=${NUM_X_T} n_runs=${N_RUNS} methods=${METHODS}"

cd "$(dirname "$0")"

# Activate your environment here if needed, e.g.:
# module load cuda/12.1
# source /path/to/venv/bin/activate

python run_hybrid_sweep.py \
    --experiment "${EXPERIMENT}" \
    --num_x_t "${NUM_X_T}" \
    --n_runs "${N_RUNS}" \
    --seed "${SEED}" \
    --methods ${METHODS} \
    --reuse_fracs ${REUSE_FRACS}
