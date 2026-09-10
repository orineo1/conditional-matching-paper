#!/bin/bash
#SBATCH --job-name=witness-sharp
#SBATCH --array=0-17
#SBATCH --output=witness_sharp_%A_%a.log
#SBATCH --error=witness_sharp_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=glacier          # CPU-only; fully analytic

# Round 4 (selection sharpening + shuffled control) — pattern of
# submit_concentration.sh. Pre-registration: HYPOTHESIS.md "Round 4" (S1-S4).
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # torch + POT + scipy
#   sbatch experiments/witness_unimodal/submit_sharpness.sh
#
# 18 cells = 3 s-values x 6 arms; BOTH seeds (42, 1042) inside each cell,
# 40 restarts per seed -> 80 pooled pairs per (s, arm).

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

S=$(echo "0.10 0.25 0.50" | cut -d" " -f$((SLURM_ARRAY_TASK_ID / 6 + 1)))
A=$(echo "mmd_uniform witness_b1_f0 witness_b2_f0 witness_b4_f0 witness_topk witness_b4_shuffled" \
    | cut -d" " -f$((SLURM_ARRAY_TASK_ID % 6 + 1)))
TAG="cell_s${S}_${A}"
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] cell: s=$S arm=$A seeds='$SEEDS' ==="

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
    --tag           $TAG"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
"$PY" experiments/witness_unimodal/build_sharpness_report.py
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
