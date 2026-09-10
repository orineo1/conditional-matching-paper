#!/bin/bash
#SBATCH --job-name=witness-repl
#SBATCH --array=0-7
#SBATCH --output=witness_repl_%A_%a.log
#SBATCH --error=witness_repl_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=glacier          # CPU-only; fully analytic

# Round 5 (replacement mode) — pattern of submit_sharpness.sh.
# Pre-registration: HYPOTHESIS.md "Round 5" (R5). The WITH-replacement side is
# NOT rerun — round-4 result JSONs are reused (same seeds/restarts).
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # torch + POT + scipy
#   sbatch experiments/witness_unimodal/submit_replacement.sh
#
# 8 cells = 2 s-values x 4 without-replacement arms; BOTH seeds inside each
# cell, 40 restarts per seed -> 80 pooled pairs per (s, arm).

N_RESTARTS=40
SEEDS="42 1042"
NSAMPLES=32
BACKSEL_K=8
N_TARGET=250
N_EVAL=256

PY="${ENV_PATH}/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"; mkdir -p "$MPLCONFIGDIR"
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/experiments/witness_unimodal/results"

S=$(echo "0.25 0.50" | cut -d" " -f$((SLURM_ARRAY_TASK_ID / 4 + 1)))
A=$(echo "uniform_wor witness_b1_wor witness_b2_wor witness_b4_wor" \
    | cut -d" " -f$((SLURM_ARRAY_TASK_ID % 4 + 1)))
TAG="cell_s${S}_${A}"
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] cell: s=$S arm=$A seeds='$SEEDS' ==="

# selftest first (cheap): k=n exactness + pi-approximation MC check
"$PY" experiments/witness_unimodal/exp_sharpness.py --selftest || exit 1

CMD=""$PY" experiments/witness_unimodal/exp_sharpness.py \
    --s_list        $S \
    --arms          $A \
    --seeds         $SEEDS \
    --n_restarts    $N_RESTARTS \
    --nsamples      $NSAMPLES \
    --backsel_k     $BACKSEL_K \
    --n_target      $N_TARGET \
    --n_eval        $N_EVAL \
    --step_cap_tau  1.0 --inv_sqrt_alpha \
    --out_prefix    replacement \
    --tag           $TAG"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
"$PY" experiments/witness_unimodal/build_replacement_report.py
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
