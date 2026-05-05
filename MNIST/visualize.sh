#!/bin/bash
#SBATCH --job-name=mnist-viz
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/viz_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/logs/viz_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE HERE
# ══════════════════════════════════════════════════════════════════════════════

# Directory containing your .pkl result file(s)
RESULTS_DIR="/sci/labs/orzuk/ori_m/conditional-matching-paper/MNIST/results/unimodal_run/unimodal_var515_st130_ssdouble_xt3_ns1500_cl0"

# Top-k images/distributions to plot
TOP_K=5

# Plot DPI
DPI=150

# Also save plots without titles? (true/false)
NO_TITLES=false

# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

export REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
: "${HF_TOKEN:?HF_TOKEN is not set. Run: export HF_TOKEN=hf_...}"
export HF_TOKEN

export HF_HOME="$HOME/.cache/huggingface"
export MPLCONFIGDIR="$HOME/.config/matplotlib"
export PYTHONPATH="$REPO_ROOT/MNIST/src:$REPO_ROOT/MNIST:$PYTHONPATH"

mkdir -p "$REPO_ROOT/MNIST/logs"

# ══════════════════════════════════════════════════════════════════════════════
# Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    RESULTS_DIR : $RESULTS_DIR"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# Run visualization
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/MNIST"

NO_TITLES_FLAG=""
[ "$NO_TITLES" = "true" ] && NO_TITLES_FLAG="--no_titles"

python visualize.py \
    --results_dir "$RESULTS_DIR" \
    --top_k       $TOP_K \
    --dpi         $DPI \
    $NO_TITLES_FLAG

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
