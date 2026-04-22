#!/bin/bash
# submit_all.sh — submit all four scribble experiments to SLURM
# Usage:
#   bash submit_all.sh              # full run
#   bash submit_all.sh --sanity     # quick sanity check (n=4) only

SANITY=""
if [ "$1" == "--sanity" ]; then
    SANITY="--sanity_check"
    echo "Running in SANITY CHECK mode"
fi

# ── Paths ──────────────────────────────────────────────────────────────────
LAB_ROOT="/sci/labs/orzuk/ori_m"
REPO="$LAB_ROOT/conditional-matching-paper"
LOG_DIR="$REPO/scribble_logs"
mkdir -p "$LOG_DIR"

for EXP in 50_50 25_75 gender_interp age_interp; do

    if [ "$EXP" == "age_interp" ]; then
        TIME="48:00:00"
    else
        TIME="24:00:00"
    fi

    # Write a proper per-experiment bash script (avoids sh/source issue)
    SCRIPT="$LOG_DIR/run_${EXP}.sh"

    cat > "$SCRIPT" << ENDBATCH
#!/bin/bash
#SBATCH --job-name=scribble_${EXP}
#SBATCH --output=${LOG_DIR}/${EXP}_%j.log
#SBATCH --error=${LOG_DIR}/${EXP}_%j.err
#SBATCH --time=${TIME}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment ─────────────────────────────────────────────────────────
export ENV_PATH="${LAB_ROOT}/dps_env"
source \$ENV_PATH/bin/activate

# ── 2. Caches (avoid home-full errors) ────────────────────────────────────
export HF_HOME="${LAB_ROOT}/hf_cache"
export MPLCONFIGDIR="${LAB_ROOT}/.matplotlib_cache"
export XDG_CACHE_HOME="${LAB_ROOT}/.cache"
mkdir -p "\$HF_HOME" "\$MPLCONFIGDIR" "\$XDG_CACHE_HOME"

# ── 3. W&B + CUDA ──────────────────────────────────────────────────────────
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 4. Verification ────────────────────────────────────────────────────────
echo "=== JOB \${SLURM_JOB_ID} STARTING ON \$(hostname) ==="
echo "Experiment: ${EXP}"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
python -c "import wandb; print(f'W&B: {wandb.__version__}')"
echo "============================================"

# ── 5. Run ─────────────────────────────────────────────────────────────────
cd ${REPO}
python run_experiment.py --exp ${EXP} ${SANITY}

echo "=== JOB ${EXP} FINISHED ==="
ENDBATCH

    chmod +x "$SCRIPT"
    JOB_ID=$(sbatch --parsable "$SCRIPT")
    echo "Submitted ${EXP}  job_id=${JOB_ID}  time=${TIME}"
    echo "  log : ${LOG_DIR}/${EXP}_${JOB_ID}.log"
    echo "  err : ${LOG_DIR}/${EXP}_${JOB_ID}.err"
    echo "  script: $SCRIPT"
done

echo ""
echo "Monitor : squeue -u \$USER"
echo "Watch   : tail -f ${LOG_DIR}/<exp>_<job_id>.log"
echo "W&B     : https://wandb.ai"
