#!/bin/bash
#SBATCH --job-name=run-compare-2d
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/ori_m/compare_env
PYTHON=/sci/labs/orzuk/ori_m/compare_env/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

MODELS_DIR=${1:?"Usage: sbatch submit_run_compare_2d.sh <models_dir>"}
LOG_FILE="/sci/labs/orzuk/ori_m/conditional-matching-paper/run_compare_2d_${SLURM_JOB_ID}.log"

echo "models_dir: $MODELS_DIR" | tee "$LOG_FILE"

$PYTHON run_compare_2d.py \
    --models_dir "$MODELS_DIR" \
    --n_attempts 25 \
    --nsamples   250 \
    --num_x_t    3 \
    >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Done. Results in: $MODELS_DIR/run_compare_2d_results.json"
    rm -f "$LOG_FILE"
else
    echo "FAILED (exit $EXIT_CODE) — log at: $LOG_FILE"
    exit $EXIT_CODE
fi