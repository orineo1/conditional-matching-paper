#!/bin/bash
#SBATCH --job-name=time-dps-step
#SBATCH --output=time_dps_step_%j.log
#SBATCH --error=time_dps_step_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ══════════════════════════════════════════════════════════════════════════════
# Quick timing check for ONE MLGD-F guidance step (Architect U-Net forward ->
# VAE decode -> Sprinter candidate sampling -> backsel -> backward pass), with
# a naive projection to the full run's wall-clock time given --n_steps. No
# file saving, no wandb, no outer diffusion loop -- just the per-step cost.
#
# Submit with:
#   export ENV_PATH=/path/to/your/env
#   export NUM_VARIATIONS=100
#   export BACKSEL_K=50
#   export BACKSEL_RULE=witness
#   export N_STEPS=250
#   sbatch time_dps_step.sh
#
# Or run directly (no SLURM) on a machine with a free GPU:
#   ENV_PATH=/path/to/your/env NUM_VARIATIONS=100 BACKSEL_K=50 \
#       BACKSEL_RULE=witness N_STEPS=250 bash time_dps_step.sh
# ══════════════════════════════════════════════════════════════════════════════

NUM_VARIATIONS="${NUM_VARIATIONS:-6}"
BACKSEL_K="${BACKSEL_K:-}"                 # empty = None (backprop all num_variations)
BACKSEL_RULE="${BACKSEL_RULE:-uniform}"    # uniform | witness
WITNESS_FLOOR="${WITNESS_FLOOR:-0.3}"
WITNESS_TEMPERATURE="${WITNESS_TEMPERATURE:-1.0}"
LOSS_FN="${LOSS_FN:-mmd}"
N_WARMUP="${N_WARMUP:-1}"
N_REPEATS="${N_REPEATS:-5}"
N_STEPS="${N_STEPS:-}"                     # empty = skip the total-time projection
SEED="${SEED:-0}"

export ENV_PATH="${ENV_PATH:?ENV_PATH is not set. Export it before submitting (dir containing bin/python).}"
PYTHON="$ENV_PATH/bin/python"

# Path to THIS subdirectory (SD_cond_SD_controlnet/) -- scripts/time_dps_step.py
# is resolved relative to here.
REPO="${REPO:?REPO is not set. Export it to the SD_cond_SD_controlnet directory before submitting.}"
cd "$REPO"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKSEL_ARGS=()
[ -n "$BACKSEL_K" ] && BACKSEL_ARGS=(--backsel_k "$BACKSEL_K")

NSTEPS_ARGS=()
[ -n "$N_STEPS" ] && NSTEPS_ARGS=(--n_steps "$N_STEPS")

echo "=== JOB ${SLURM_JOB_ID:-local} ON $(hostname) ==="
echo "    num_variations : $NUM_VARIATIONS"
echo "    backsel_k      : ${BACKSEL_K:-(None, all)}"
echo "    backsel_rule   : $BACKSEL_RULE"
echo "    loss_fn        : $LOSS_FN"
echo "    n_steps        : ${N_STEPS:-(no projection)}"
"$PYTHON" -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

"$PYTHON" scripts/time_dps_step.py \
    --num_variations "$NUM_VARIATIONS" \
    --backsel_rule "$BACKSEL_RULE" \
    --witness_floor "$WITNESS_FLOOR" \
    --witness_temperature "$WITNESS_TEMPERATURE" \
    --loss_fn "$LOSS_FN" \
    --n_warmup "$N_WARMUP" \
    --n_repeats "$N_REPEATS" \
    --seed "$SEED" \
    "${BACKSEL_ARGS[@]}" \
    "${NSTEPS_ARGS[@]}"

echo "=== JOB ${SLURM_JOB_ID:-local} FINISHED (exit $?) ==="
