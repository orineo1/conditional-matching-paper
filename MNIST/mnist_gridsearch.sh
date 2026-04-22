#!/bin/bash
#SBATCH --job-name=mnist-varsweep
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon
#SBATCH --array=0-19

REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
LOG_DIR="$REPO_ROOT/MNIST/logs/variance_sweep"
SCRIPT="$REPO_ROOT/MNIST/mnist_variance_sweep.py"

mkdir -p "$LOG_DIR"
mkdir -p "$REPO_ROOT/MNIST/checkpoints"
mkdir -p "$REPO_ROOT/MNIST/results/variance_sweep"

LOGFILE="$LOG_DIR/sweep_${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "START  : $(date)"
echo "Job    : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Host   : $(hostname)"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

source "$ENV_PATH/bin/activate" || { echo "ERROR activating env"; exit 1; }

export REPO_ROOT="$REPO_ROOT"
export HF_TOKEN="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"
export WANDB_API_KEY="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"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

cd "$REPO_ROOT/MNIST"

# Train classifier once using task 0 as a gate, all others wait
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python "$SCRIPT" --train_classifier_only
else
    # Wait until checkpoint exists (max 30 min)
    CLF="$REPO_ROOT/MNIST/checkpoints/robust_classifier.pth"
    WAIT=0
    until [ -f "$CLF" ] || [ $WAIT -ge 1800 ]; do
        sleep 30
        WAIT=$((WAIT + 30))
        echo "Waiting for classifier... ${WAIT}s"
    done
    [ -f "$CLF" ] || { echo "ERROR: classifier never appeared"; exit 1; }
fi

python "$SCRIPT" \
    --config_id    "$SLURM_ARRAY_TASK_ID" \
    --wandb_entity ""

EXIT_CODE=$?
echo "END: $(date) | EXIT: $EXIT_CODE"
exit $EXIT_CODE