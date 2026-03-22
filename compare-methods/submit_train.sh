#!/bin/bash
#SBATCH --job-name=compare-train
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# ── Conda setup ───────────────────────────────────────────────────────────────
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# ── Log file setup ────────────────────────────────────────────────────────────
DIM=${1:-2}
OUTPUT_DIR="compare-methods/output/models_${DIM}d_${SLURM_JOB_ID}"
LOG_FILE="/sci/labs/orzuk/ori_m/conditional-matching-paper/compare_train_${SLURM_JOB_ID}.log"

echo "Starting compare-methods training on $(hostname)" | tee "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE (deleted on success)" | tee -a "$LOG_FILE"

pip install -q flow_matching POT

# ── Run training, capturing all output ───────────────────────────────────────
python compare-methods/train_models.py \
    --output_dir "$OUTPUT_DIR" \
    --dim "$DIM" \
    --condition_on 1 \
    --nblocks 3 \
    --nunits 128 \
    --diffusion_steps 100 \
    --nepochs_diff 20000 \
    --nepochs_cm 7500 \
    --nepochs_fm 10000 \
    --batch_size_diff 512 \
    --batch_size_cm 1024 \
    --batch_size_fm 1024 \
    --seed 42 \
    >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

mkdir -p compare-methods/output

# ── On success: remove log. On failure: keep it. ──────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete. Models in: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "OUTPUT_DIR=$OUTPUT_DIR" > compare-methods/output/last_train_${SLURM_JOB_ID}.env
    rm -f "$LOG_FILE"
    echo "SUCCESS — log deleted."
else
    echo "" >> "$LOG_FILE"
    echo "=== FAILED (exit code $EXIT_CODE) ===" >> "$LOG_FILE"
    echo "FAILED (exit code $EXIT_CODE) — log kept at: $LOG_FILE"
    exit $EXIT_CODE
fi