#!/bin/bash
#SBATCH --job-name=cdms-sweep
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon
#SBATCH --array=0-13   # 7 grad_clamp × 2 diffusion_steps = 14 configs

REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
LOG_DIR="$REPO_ROOT/simulations/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$REPO_ROOT/simulations/results/2D_cond_1D/gridsearch"

LOGFILE="$LOG_DIR/sweep_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "START  : $(date)"
echo "Job    : $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID)"
echo "Host   : $(hostname)"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

source "$ENV_PATH/bin/activate" || { echo "ERROR: failed to activate env"; exit 1; }

export REPO_ROOT="$REPO_ROOT"
export WANDB_API_KEY="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"

SCRIPT="$REPO_ROOT/simulations/cdms_gridsearch_simple.py"
cd "$REPO_ROOT/simulations"

python "$SCRIPT" \
    --config_id    "$SLURM_ARRAY_TASK_ID" \
    --wandb_project "cdms-gridsearch"

EXIT_CODE=$?
echo "END: $(date) | EXIT: $EXIT_CODE"
exit $EXIT_CODE
