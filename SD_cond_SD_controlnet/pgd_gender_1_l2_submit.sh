#!/bin/bash
#SBATCH --job-name=pgd-gender-1-l2
#SBATCH --output=pgd_gender_1_l2_%j.log
#SBATCH --error=pgd_gender_1_l2_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch pgd_gender_1_l2_submit.sh
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

OUTPUT_DIR="$REPO/output/pgd_gender_1_l2_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Same as pgd_gender_100_submit.sh, but the target is a SINGLE woman CLIP
# sample instead of 100 -- an ablation on target-distribution size. Same
# --target_minutes budget as the matching MLGD-F run.

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_pgd.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "pgd-gender-l2" \
    --mode gender \
    --target_minutes 241 \
    --opt_steps 3 \
    --opt_lr 0.05 \
    --proj_adam_steps 200 \
    --proj_lr 0.03 \
    --num_variations 6 \
    --controlnet_scale 0.5 \
    --loss_fn l2 \
    --target_prompts \
        "Woman:a superrealistic portrait photograph of a woman, studio lighting:1" \
    --seed 1

# ── 6. (Optional) Sync outputs ────────────────────────────────────────────────
# Uncomment and adjust if you want to sync results to remote storage:
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/pgd_gender_1_l2_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
