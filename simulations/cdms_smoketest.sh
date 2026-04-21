#!/bin/bash
#SBATCH --job-name=cdms-smoketest
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/simulations/logs/smoketest_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/simulations/logs/smoketest_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=salmon

REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"

mkdir -p "$REPO_ROOT/simulations/logs"
mkdir -p "$REPO_ROOT/simulations/results/2D_cond_1D/gridsearch"

source "$ENV_PATH/bin/activate"

export REPO_ROOT="$REPO_ROOT"
export WANDB_API_KEY="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"

cd "$REPO_ROOT/simulations"

echo "=========================================="
echo "Job: $SLURM_JOB_ID | Host: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# config_id=0 is: grad_clamp=0.25, noise_scale=0.0, num_x_t=3, nunits=128, depth=3
# i.e. the simplest/cheapest config — good for smoke testing
python "$REPO_ROOT/simulations/cdms_gridsearch.py" \
    --config_id     0               \
    --wandb_project "cdms-smoketest" \
    --wandb_entity  "YOUR_WANDB_ENTITY_HERE" \
    --smoke_test    # ← triggers n_cdms_samples=5, nepochs_cm=500, zetas=[0,1,4]

echo "Smoketest done."
