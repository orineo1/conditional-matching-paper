#!/bin/bash
#SBATCH --job-name=witness-ident
#SBATCH --array=0-11
#SBATCH --output=witness_ident_%A_%a.log
#SBATCH --error=witness_ident_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=glacier          # CPU-only; fully analytic, no GPU/no training

# Round 2 (identifiability) array — pattern of submit_array.sh. No pip installs.
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # torch + POT (no HF needed: analytic)
#   sbatch experiments/witness_unimodal/submit_identifiability.sh
#
# 12 cells = 2 targets x [mean, mmd, mmd_uniform, mmd_witness under cap+inv_sqrt
#            (the verified-correct convention)  +  mmd under cap-only and
#            inv_sqrt-only (the step-convention factor, mmd arm only)].

N_RESTARTS=40
NSAMPLES=32
BACKSEL_K=8
WITNESS_FLOOR=0.3
N_TARGET=250
N_EVAL=256
SEED=${SEED:-42}

PY="${ENV_PATH}/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"; mkdir -p "$MPLCONFIGDIR"
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/experiments/witness_unimodal/results"

T=$(echo "bi uni" | cut -d" " -f$((SLURM_ARRAY_TASK_ID / 6 + 1)))
case $((SLURM_ARRAY_TASK_ID % 6)) in
  0) A="mean";        CONV="--step_cap_tau 1.0 --inv_sqrt_alpha" ;;
  1) A="mmd";         CONV="--step_cap_tau 1.0 --inv_sqrt_alpha" ;;
  2) A="mmd_uniform"; CONV="--step_cap_tau 1.0 --inv_sqrt_alpha" ;;
  3) A="mmd_witness"; CONV="--step_cap_tau 1.0 --inv_sqrt_alpha" ;;
  4) A="mmd";         CONV="--step_cap_tau 1.0 --no-inv_sqrt_alpha" ;;   # cap only
  5) A="mmd";         CONV="--step_cap_tau 0 --inv_sqrt_alpha" ;;        # inv_sqrt only
esac
TAG="cell_${T}_${A}"
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] cell: target=$T arm=$A conv='$CONV' ==="

CMD=""$PY" experiments/witness_unimodal/exp_identifiability.py \
    --targets       $T \
    --arms          $A \
    --n_restarts    $N_RESTARTS \
    --nsamples      $NSAMPLES \
    --backsel_k     $BACKSEL_K \
    --witness_floor $WITNESS_FLOOR \
    --n_target      $N_TARGET \
    --n_eval        $N_EVAL \
    --seed          $SEED \
    --tag           $TAG \
    $CONV"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?
"$PY" experiments/witness_unimodal/build_identifiability_report.py
echo "=== JOB ${SLURM_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
