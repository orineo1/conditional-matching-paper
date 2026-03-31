#!/bin/bash
#SBATCH --job-name=compare-run
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# ── Conda setup ───────────────────────────────────────────────────────────────
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/ori_m/compare_env
PYTHON=/sci/labs/orzuk/ori_m/compare_env/bin/python

export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# ── Required arg: path to models directory ────────────────────────────────────
MODELS_DIR=${1:?"Usage: sbatch submit_compare_logged.sh <models_dir>"}
OUTPUT_DIR="compare-methods/output/compare_${SLURM_JOB_ID}"
LOG_FILE="/sci/labs/orzuk/ori_m/conditional-matching-paper/compare_run_${SLURM_JOB_ID}.log"

echo "Starting compare-methods run on $(hostname)" | tee "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Models dir: $MODELS_DIR" | tee -a "$LOG_FILE"
echo "Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE (deleted on success)" | tee -a "$LOG_FILE"

$PYTHON -m pip install -q flow_matching POT


mkdir -p compare-methods/output

# ── Run comparison, capturing all output ──────────────────────────────────────
$PYTHON compare-methods/run_compare.py \
    --models_dir "$MODELS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_attempts 25 \
    --nsamples_mmd 250 \
    --num_x_t 3 \
    --nsamples_swd 10000 \
    --num_projections_swd 500 \
    --x_star -5.0 \
    --wandb_project compare-methods \
    --wandb_entity conditional-matching \
    --seed 42 \
    >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# ── On success: remove log. On failure: keep it. ──────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo "Comparison complete. Results in: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    rm -f "$LOG_FILE"
    echo "SUCCESS — log deleted."
else
    echo "" >> "$LOG_FILE"
    echo "=== FAILED (exit code $EXIT_CODE) ===" >> "$LOG_FILE"
    echo "FAILED (exit code $EXIT_CODE) — log kept at: $LOG_FILE"
    exit $EXIT_CODE
fi