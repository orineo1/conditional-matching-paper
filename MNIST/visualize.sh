#!/bin/bash
#SBATCH --job-name=mnist-viz
#SBATCH --output=logs/viz_%j.log
#SBATCH --error=logs/viz_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# USAGE
#   sbatch visualize.sh <results_dir> [top_k] [dpi]
#
#   results_dir  : directory containing .pkl result file(s)  (required)
#   top_k        : number of top images to show              (default: 5)
#   dpi          : plot resolution                           (default: 150)
#
# Example:
#   sbatch visualize.sh results/unimodal_run/unimodal_var515_st130_ssdouble_xt3_ns1500_cl0
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR="${1:?Usage: sbatch visualize.sh <results_dir> [top_k] [dpi]}"
TOP_K="${2:-5}"
DPI="${3:-150}"

# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting

# ══════════════════════════════════════════════════════════════════════════════
# Secrets  — READ FROM ENVIRONMENT, never hardcode here
#    export HF_TOKEN=hf_...
# ══════════════════════════════════════════════════════════════════════════════
: "${HF_TOKEN:?HF_TOKEN is not set. Export it before submitting.}"
export HF_TOKEN

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# ══════════════════════════════════════════════════════════════════════════════
# Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"

mkdir -p "$REPO_ROOT/MNIST/logs"

# ══════════════════════════════════════════════════════════════════════════════
# Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT   : $REPO_ROOT"
echo "    RESULTS_DIR : $RESULTS_DIR"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# Run visualization
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/MNIST"

python visualize.py \
    --results_dir "$RESULTS_DIR" \
    --top_k       $TOP_K \
    --dpi         $DPI

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
