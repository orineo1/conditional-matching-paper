#!/bin/bash
#SBATCH --job-name=gender-saliency
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# 1. CHANGE: Use YOUR directory (ori_m) instead of shaulytolk
BASE_DIR="/sci/labs/orzuk/ori_m/conditional-matching-paper"
OUTPUT_DIR="${BASE_DIR}/output/saliency_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# 2. CHANGE: Log paths to YOUR directory
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/output/saliency_%j/job_log.out
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/output/saliency_%j/job_error.err

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

# NOTE: Ensure this environment exists in your path or use shaulytolk's if it has global read permissions
conda activate ${BASE_DIR}/scribble_env

exec > >(tee -a "${OUTPUT_DIR}/runtime_log.txt") 2>&1

echo "------------------------------------------------"
echo "Job Started: $(date)"
echo "Output Directory: $OUTPUT_DIR"
echo "------------------------------------------------"

# 3. CHANGE: Use YOUR cache
export HF_HOME="/sci/labs/orzuk/ori_m/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"

SCRIBBLE_PATH="${1:-scripts/assets/zeta5_input_scribble.png}"

# Ensure the python script path is correct for your location
python scripts/gender_saliency.py \
    --scribble_path "$SCRIBBLE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_seeds 20 \
    --sigma 2.5 \
    --device cuda

echo "------------------------------------------------"
echo "Gender saliency complete at: $(date)"