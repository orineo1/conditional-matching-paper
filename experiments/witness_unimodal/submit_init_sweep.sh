#!/bin/bash
#SBATCH --job-name=witness-init
#SBATCH --array=0-24
#SBATCH --output=witness_init_%A_%a.log
#SBATCH --error=witness_init_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=glacier          # CPU-only; fully analytic

# Round 7 (init sweep) — pattern of submit_pointwise.sh.
# Pre-registration: HYPOTHESIS.md "Round 7" (predictions (i)-(iii)).
# The MMD reference arms ARE re-run here (init changed; round-2 rows unusable).
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # torch + POT + scipy
#   sbatch experiments/witness_unimodal/submit_init_sweep.sh
#
# 25 cells = 5 offsets x 5 arms; BOTH seeds (42, 1042) inside each cell,
# 40 restarts per seed -> 80 per (arm, c).

N_RESTARTS=40
SEEDS="42 1042"
NSAMPLES=32
N_TARGET=250
N_EVAL=256

PY="${ENV_PATH}/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"; mkdir -p "$MPLCONFIGDIR"
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/experiments/witness_unimodal/results"

C=$(echo "-3.0 -1.5 0.0 1.5 3.0" | cut -d" " -f$((SLURM_ARRAY_TASK_ID / 5 + 1)))
A=$(echo "point_sq_a0 point_sq_a2 point_abs_a2 mmd_bi mmd_uni" \
    | cut -d" " -f$((SLURM_ARRAY_TASK_ID % 5 + 1)))
TAG="cell_c${C}_${A}"
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] cell: c=$C arm=$A seeds='$SEEDS' ==="

CMD=""$PY" experiments/witness_unimodal/exp_init_sweep.py \
    --c_list        $C \
    --arms          $A \
    --seeds         $SEEDS \
    --n_restarts    $N_RESTARTS \
    --nsamples      $NSAMPLES \
    --n_target      $N_TARGET \
    --n_eval        $N_EVAL \
    --step_cap_tau  1.0 --inv_sqrt_alpha \
    --tag           $TAG"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
"$PY" experiments/witness_unimodal/build_init_sweep_report.py
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
