#!/bin/bash
#SBATCH --job-name=simulations
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/sim_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/sim_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Usage:
#   sbatch submit_sim.sh Exp_2D_cond_1D
#   sbatch submit_sim.sh Exp_5D_cond_1D
#   sbatch submit_sim.sh Exp_10D_cond_1D
#   sbatch submit_sim.sh toy_example_with_beta_sweep

NOTEBOOK=${1:-Exp_2D_cond_1D}

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
echo "Notebook: $NOTEBOOK"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Install any missing simulation deps ────────────────────────────────────
pip install flow_matching POT huggingface_hub -q

# ── 5. Run the notebook ───────────────────────────────────────────────────────
REPO="/sci/labs/orzuk/ori_m/conditional-matching-paper"
NB_IN="$REPO/simulations/notebooks/${NOTEBOOK}.ipynb"
NB_OUT="$REPO/simulations/notebooks/${NOTEBOOK}_executed_${SLURM_JOB_ID}.ipynb"

cd "$REPO/simulations/notebooks"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=14400 \
    --output "$NB_OUT" \
    "$NB_IN"

echo "✅ Notebook executed. Output: $NB_OUT"
