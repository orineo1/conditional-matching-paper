#!/bin/bash
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/uncond_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/uncond_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── Environment ───────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

export WANDB_API_KEY=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
export REPO_PATH="${REPO_PATH:-/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST}"

echo "Job:  $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Args: $@"

mkdir -p "$REPO_PATH/logs"

cd "$REPO_PATH"
python train_mnist_unconditional.py "$@"