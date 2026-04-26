#!/bin/bash
#SBATCH --job-name=eps-g-experiment
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/eps_g_experiment_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/eps_g_experiment_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=goldfish
#SBATCH --nodelist=goldfish-01

# ── 1. Environment ────────────────────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ── 2. Caches ─────────────────────────────────────────────────────────────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# ── 5. Run ────────────────────────────────────────────────────────────────────
python eps_g_experiment.py

echo "✅ eps_g_experiment done."
