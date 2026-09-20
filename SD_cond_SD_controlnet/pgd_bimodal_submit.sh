#!/bin/bash
#SBATCH --job-name=pgd-bimodal
#SBATCH --output=pgd_bimodal_%j.log
#SBATCH --error=pgd_bimodal_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch pgd_bimodal_submit.sh
source "$ENV_PATH/bin/activate"

# ── 2. Caches (optional — redirect if your home quota is limited) ─────────────
# export HF_HOME=/path/to/hf_cache
# export MPLCONFIGDIR=/path/to/matplotlib_cache
# export XDG_CACHE_HOME=/path/to/cache

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Path to THIS subdirectory (SD_cond_SD_controlnet/), not the outer git repo root —
# scripts/run_pgd.py below is resolved relative to this path.
REPO="YOUR_REPO_PATH_HERE/SD_cond_SD_controlnet"
cd "$REPO"

OUTPUT_DIR="$REPO/output/pgd_bimodal_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# man scribble -> bimodal (50% man / 50% woman) target, matching the existing
# experiments/BalancedTarget MLGD-F run's target split and controlnet_scale.
# PGD-only comparison -- MLGD-F for this target already exists in
# experiments/BalancedTarget/scribble_mlgdd.png, no need to rerun it.
#
# --target_minutes: set to the recorded BalancedTarget MLGD-F runtime
# (see experiments/BalancedTarget/baselines/baselines_meta.json or the
# original wandb run) so both methods get the same time budget.

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_pgd.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "pgd-bimodal" \
    --mode gender \
    --target_minutes 241 \
    --opt_steps 3 \
    --opt_lr 0.05 \
    --proj_adam_steps 200 \
    --proj_lr 0.03 \
    --num_variations 6 \
    --controlnet_scale 0.5 \
    --loss_fn mmd \
    --target_prompts \
        "Man:a superrealistic portrait photograph of a man, studio lighting:50" \
        "Woman:a superrealistic portrait photograph of a woman, studio lighting:50" \
    --seed 5

# ── 6. (Optional) Sync outputs ────────────────────────────────────────────────
# Uncomment and adjust if you want to sync results to remote storage:
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/pgd_bimodal_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
