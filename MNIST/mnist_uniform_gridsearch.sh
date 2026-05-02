#!/bin/bash
#SBATCH --job-name=unif-grid
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/uniform_grid_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/uniform_grid_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon
#SBATCH --array=0-53   # 4 nsamples × 5 steps × 5 ss_modes × 5 num_x_t = 500

# ── 1. Environment ─────────────────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ── 2. Caches ──────────────────────────────────────────────────────────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Keys ────────────────────────────────────────────────────────────────
export REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
export HF_TOKEN="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"
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ── 4. Directories ─────────────────────────────────────────────────────────
mkdir -p "$REPO_ROOT/MNIST/logs"
mkdir -p "$REPO_ROOT/MNIST/checkpoints"
mkdir -p "$REPO_ROOT/MNIST/results/uniform_gridsearch"

# ── 5. Info ────────────────────────────────────────────────────────────────
echo "=== JOB ${SLURM_JOB_ID} ARRAY ${SLURM_ARRAY_TASK_ID} ON $(hostname) ==="
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ── 6. Train classifier once (task 0 gates the rest) ──────────────────────
cd "$REPO_ROOT/MNIST"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python mnist_uniform_gridsearch.py --train_classifier_only
fi

# All tasks (including 0) wait until the checkpoint exists
CLF="$REPO_ROOT/MNIST/checkpoints/robust_classifier.pth"
WAIT=0
until [ -f "$CLF" ] || [ $WAIT -ge 1800 ]; do
    sleep 30
    WAIT=$((WAIT + 30))
    echo "Waiting for classifier... ${WAIT}s"
done
[ -f "$CLF" ] || { echo "ERROR: classifier never appeared"; exit 1; }

# ── 7. Run config ──────────────────────────────────────────────────────────
python mnist_uniform_gridsearch.py \
    --config_id "$SLURM_ARRAY_TASK_ID" \
    --wandb_entity ""

EXIT_CODE=$?
echo "=== ARRAY TASK ${SLURM_ARRAY_TASK_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
