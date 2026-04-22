#!/bin/bash
#SBATCH --job-name=train-clf
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=salmon

REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"

source "$ENV_PATH/bin/activate" || { echo "ERROR activating env"; exit 1; }

export REPO_ROOT="$REPO_ROOT"
export HF_TOKEN="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"
export WANDB_API_KEY="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"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

cd "$REPO_ROOT/MNIST"
python mnist_variance_sweep.py --train_classifier_only

echo "Classifier trained and saved."