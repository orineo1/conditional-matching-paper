#!/bin/bash
#SBATCH --job-name=uniform_steps
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/uniform_steps_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/uniform_steps_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon
#SBATCH --array=0-23

# ── 1. Environment ─────────────────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ── 2. Caches ──────────────────────────────────────────────────────────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. W&B + CUDA ──────────────────────────────────────────────────────────
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 4. Verification ────────────────────────────────────────────────────────
echo "=== JOB ${SLURM_JOB_ID} ARRAY ${SLURM_ARRAY_TASK_ID} ON $(hostname) ==="
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 5. Run ─────────────────────────────────────────────────────────────────
# Ensure the log directory exists
mkdir -p /sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs

# Corrected Directory: The .py file is in MNIST, not MNIST/src
cd /sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST

# Execute the sweep script
python mnist_uniform_steps_sweep.py --config_id ${SLURM_ARRAY_TASK_ID}

echo "=== ARRAY TASK ${SLURM_ARRAY_TASK_ID} FINISHED ==="