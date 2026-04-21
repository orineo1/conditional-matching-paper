#!/bin/bash
#SBATCH --job-name=cdms-sweep
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/simulations/logs/cdms_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/simulations/logs/cdms_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon
#SBATCH --array=0-63   # 4 grad_clamp × 2 noise_scale × 2 num_x_t × 2 nunits × 2 depth = 64 configs

REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"

mkdir -p "$REPO_ROOT/simulations/logs"
mkdir -p "$REPO_ROOT/simulations/results/2D_cond_1D/gridsearch"

source "$ENV_PATH/bin/activate"

export REPO_ROOT="$REPO_ROOT"
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"   # or set in ~/.bashrc

cd "$REPO_ROOT/simulations"

echo "=========================================="
echo "Job:    $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Host:   $(hostname)"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "=========================================="

python "$REPO_ROOT/simulations/cdms_gridsearch.py" \
    --config_id     "$SLURM_ARRAY_TASK_ID"         \
    --wandb_project "cdms-sweep"                    \
    --wandb_entity  "YOUR_WANDB_ENTITY_HERE"

echo "Done — task $SLURM_ARRAY_TASK_ID"
