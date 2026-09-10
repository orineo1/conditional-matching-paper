#!/bin/bash
#SBATCH --job-name=witness-cell
#SBATCH --array=0-19
#SBATCH --output=witness_unimodal_%j.log
#SBATCH --error=witness_unimodal_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=glacier          # CPU-only partition; this toy needs no GPU

# Follows the pattern of simulations/scripts/run_backsel_witness_sweep.sh:
# no pip installs, env via ENV_PATH, repo via REPO_ROOT. Do NOT submit until
# HYPOTHESIS.md has been reviewed.
#
#   export REPO_ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper
#   export ENV_PATH=/path/to/your/env      # env must have torch, POT, huggingface_hub
#   sbatch experiments/witness_unimodal/submit.sh

# ── Configure the run ─────────────────────────────────────────────────────────
N_RESTARTS=40
NSAMPLES=32
BACKSEL_K=8
NUM_X_T=1
N_EVAL=256
WITNESS_FLOOR=0.3
ZETA=1.0
MEAN_GRAD_CLIP=1.0        # mean arm only (deviation 3); ignored when STEP_CAP_TAU is set
STEP_CAP_TAU=1.0          # round 2: ||Delta_t|| <= tau*sqrt(1-alphabar_t) on ALL arms (empty = off)
SEED=42
TARGETS=""                # empty = all five (bimodal_c1 bimodal_c2 unimodal_s010/025/050)
ARMS=""                   # empty = all four (mean mmd mmd_uniform mmd_witness)
TAG="full"
SMOKE_TEST=false          # true = 8 restarts, 2 targets, for a quick sanity pass

# ── Environment ───────────────────────────────────────────────────────────────
# conda-style env: use its python directly (no bin/activate in conda envs)
PY="${ENV_PATH}/bin/python"

export HF_HOME="${HF_HOME:-/sci/labs/orzuk/shaulytolk/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
export HF_TOKEN="${HF_TOKEN:-}"

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/simulations/checkpoints/2D_cond_1D" \
         "$REPO_ROOT/experiments/witness_unimodal/results"

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT   : $REPO_ROOT"
echo "    n_restarts  : $N_RESTARTS | n=$NSAMPLES k=$BACKSEL_K num_x_t=$NUM_X_T"
echo "    targets     : ${TARGETS:-(all)} | arms: ${ARMS:-(all)} | tag: $TAG"
echo "============================================"

# --- per-array-task cell selection ---
T=$(echo "bimodal_c1 bimodal_c2 unimodal_s010 unimodal_s025 unimodal_s050" | cut -d" " -f$((SLURM_ARRAY_TASK_ID / 4 + 1)))
A=$(echo "mean mmd mmd_uniform mmd_witness" | cut -d" " -f$((SLURM_ARRAY_TASK_ID % 4 + 1)))
TARGETS="$T"; ARMS="$A"; TAG="cell_${T}_${A}${STEP_CAP_TAU:+_cap1}"
echo "cell: target=$T arm=$A"

CMD=""$PY" experiments/witness_unimodal/exp_witness_unimodal.py \
    --n_restarts   $N_RESTARTS \
    --nsamples     $NSAMPLES \
    --backsel_k    $BACKSEL_K \
    --num_x_t      $NUM_X_T \
    --n_eval       $N_EVAL \
    --witness_floor $WITNESS_FLOOR \
    --zeta         $ZETA \
    --mean_grad_clip $MEAN_GRAD_CLIP \
    --seed         $SEED \
    --tag          $TAG"

[ -n "$STEP_CAP_TAU" ]      && CMD="$CMD --step_cap_tau $STEP_CAP_TAU"
[ -n "$TARGETS" ]           && CMD="$CMD --targets $TARGETS"
[ -n "$ARMS" ]              && CMD="$CMD --arms $ARMS"
[ "$SMOKE_TEST" = "true" ]  && CMD="$CMD --smoke"

echo "Running: $CMD"
eval $CMD
EXIT_CODE=$?

# Build/refresh the report from everything in results/
"$PY" experiments/witness_unimodal/build_report.py

echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
