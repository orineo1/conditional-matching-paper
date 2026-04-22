#!/bin/bash
#SBATCH --job-name=mnist-gs
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon
#SBATCH --array=0-55

REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
LOG_DIR="$REPO_ROOT/MNIST/logs/gridsearch"

mkdir -p "$LOG_DIR"
mkdir -p "$REPO_ROOT/MNIST/results/gridsearch"

LOGFILE="$LOG_DIR/gs_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "START  : $(date)"
echo "Job    : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Host   : $(hostname)"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

source "$ENV_PATH/bin/activate" || { echo "ERROR activating env"; exit 1; }

export REPO_ROOT="$REPO_ROOT"
export WANDB_API_KEY="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"
export HF_TOKEN="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"

cd "$REPO_ROOT/MNIST"

python mnist_gridsearch.py \
    --config_id    "$SLURM_ARRAY_TASK_ID" \
    --wandb_project "mnist-gridsearch"

EXIT_CODE=$?
echo "END: $(date) | EXIT: $EXIT_CODE"
exit $EXIT_CODE