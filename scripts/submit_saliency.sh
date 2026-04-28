#!/bin/bash
#SBATCH --job-name=gender-saliency
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Path Setup ────────────────────────────────────────────────────
# Everything points to YOUR space to avoid permission errors
BASE_DIR="/sci/labs/orzuk/ori_m/conditional-matching-paper"
OUTPUT_DIR="${BASE_DIR}/output/saliency_${SLURM_JOB_ID}"

# Create output folder before starting
mkdir -p "$OUTPUT_DIR"

# Set SLURM logs to live inside the new output folder
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/output/saliency_%j/job_log.out
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/output/saliency_%j/job_error.err

# ── 2. Environment Setup (YOUR Private Env) ───────────────────────────
# Point to the new environment we just built and verified on salmon-01
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
# Using 'source' for a venv/conda-like activation
source "$ENV_PATH/bin/activate"

# ── 3. Runtime Logging ───────────────────────────────────────────────
# This captures all python prints and errors into a file in your output dir
exec > >(tee -a "${OUTPUT_DIR}/runtime_log.txt") 2>&1

echo "------------------------------------------------"
echo "Job Started:   $(date)"
echo "Host:          $(hostname)"
echo "GPU ID:        $CUDA_VISIBLE_DEVICES"
echo "Job ID:        $SLURM_JOB_ID"
echo "Output Dir:    $OUTPUT_DIR"
echo "Using Env:     $ENV_PATH"
echo "------------------------------------------------"

# ── 4. Execution ─────────────────────────────────────────────────────
export HF_HOME="/sci/labs/orzuk/ori_m/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"

# Default scribble path if none provided as argument
SCRIBBLE_PATH="${1:-scripts/assets/zeta5_input_scribble.png}"

# Run the script
python scripts/gender_saliency.py \
    --scribble_path "$SCRIBBLE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_seeds 20 \
    --sigma 2.5 \
    --device cuda

echo "------------------------------------------------"
echo "Gender saliency complete at: $(date)"