#!/bin/bash
#SBATCH --job-name=witness-conc
#SBATCH --array=0-14
#SBATCH --output=witness_conc_%A_%a.log
#SBATCH --error=witness_conc_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=glacier          # CPU-only; fully analytic

# Round 3 (concentration sweep) array — pattern of submit_identifiability.sh.
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # torch + POT + scipy
#   sbatch experiments/witness_unimodal/submit_concentration.sh
#
# 15 cells = 5 s-values x 3 arms; BOTH seeds (42, 1042) run inside each cell,
# 40 restarts per seed -> 80 pooled pairs per s. Pre-registration: HYPOTHESIS.md
# "Round 3" (C1-C3). Do not submit until that section has been reviewed.

N_RESTARTS=40
SEEDS="42 1042"
NSAMPLES=32
BACKSEL_K=8
WITNESS_FLOOR=0.3
N_TARGET=250
N_EVAL=256

PY="${ENV_PATH}/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"; mkdir -p "$MPLCONFIGDIR"
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/experiments/witness_unimodal/results"

S=$(echo "0.05 0.10 0.25 0.50 1.00" | cut -d" " -f$((SLURM_ARRAY_TASK_ID / 3 + 1)))
A=$(echo "mmd mmd_uniform mmd_witness" | cut -d" " -f$((SLURM_ARRAY_TASK_ID % 3 + 1)))
TAG="cell_s${S}_${A}"
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] cell: s=$S arm=$A seeds='$SEEDS' ==="

CMD=""$PY" experiments/witness_unimodal/exp_concentration.py \
    --s_list        $S \
    --arms          $A \
    --seeds         $SEEDS \
    --n_restarts    $N_RESTARTS \
    --nsamples      $NSAMPLES \
    --backsel_k     $BACKSEL_K \
    --witness_floor $WITNESS_FLOOR \
    --n_target      $N_TARGET \
    --n_eval        $N_EVAL \
    --step_cap_tau  1.0 --inv_sqrt_alpha \
    --tag           $TAG"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
"$PY" experiments/witness_unimodal/build_concentration_report.py
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
