#!/bin/bash
#SBATCH --job-name=gender-saliency
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# 1. Define a consistent output directory using the Job ID
# This creates a folder like: output/saliency_123456
OUTPUT_DIR="/sci/labs/orzuk/shaulytolk/conditional-matching-paper/output/saliency_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# 2. Redirect Logs and Errors to that same folder
# %j will be replaced by the Job ID automatically by SLURM
# Note: These paths must be set at the top, but we can also use "exec" to capture everything
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/output/saliency_%j/job_log.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/output/saliency_%j/job_error.err

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

# 3. Capture all stdout and stderr from this point forward into a log file inside the folder
exec > >(tee -a "${OUTPUT_DIR}/runtime_log.txt") 2>&1

echo "------------------------------------------------"
echo "Job Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "------------------------------------------------"

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

SCRIBBLE_PATH="${1:-scripts/assets/zeta5_input_scribble.png}"

# Use the OUTPUT_DIR variable here so Python saves images/numpy to the same spot as logs
python scripts/gender_saliency.py \
    --scribble_path "$SCRIBBLE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_seeds 20 \
    --sigma 2.5 \
    --device cuda

echo "------------------------------------------------"
echo "Gender saliency complete at: $(date)"