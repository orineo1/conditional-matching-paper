#!/bin/bash
#SBATCH --job-name=witness-point
#SBATCH --array=0-2
#SBATCH --output=witness_point_%A_%a.log
#SBATCH --error=witness_point_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=glacier          # CPU-only; fully analytic

# Round 6 (pointwise scalar-target baseline) — pattern of submit_replacement.sh.
# Pre-registration: HYPOTHESIS.md "Round 6" (P-a vs P-b, both stated).
# The mmd reference arms are NOT rerun — round-2 identifiability cell JSONs
# are reused by the report builder.
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # torch + POT + scipy
#   sbatch experiments/witness_unimodal/submit_pointwise.sh
#
# 3 cells = one per arm; BOTH seeds (42, 1042) inside each cell,
# 40 restarts per seed.

N_RESTARTS=40
SEEDS="42 1042"
N_TARGET=250
N_EVAL=256

PY="${ENV_PATH}/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"; mkdir -p "$MPLCONFIGDIR"
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/experiments/witness_unimodal/results"

A=$(echo "point_sq_a0 point_sq_a2 point_abs_a2" | cut -d" " -f$((SLURM_ARRAY_TASK_ID + 1)))
TAG="cell_${A}"
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] cell: arm=$A seeds='$SEEDS' ==="

CMD=""$PY" experiments/witness_unimodal/exp_pointwise.py \
    --arms          $A \
    --seeds         $SEEDS \
    --n_restarts    $N_RESTARTS \
    --n_target      $N_TARGET \
    --n_eval        $N_EVAL \
    --step_cap_tau  1.0 --inv_sqrt_alpha \
    --tag           $TAG"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
"$PY" experiments/witness_unimodal/build_pointwise_report.py
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
