#!/bin/bash
#SBATCH --job-name=mlgd-f
#SBATCH --output=mlgd_f_%j.log
#SBATCH --error=mlgd_f_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch submit_mlgd_f.sh
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
# scripts/run_mlgd_f.py below is resolved relative to this path.
REPO="YOUR_REPO_PATH_HERE/SD_cond_SD_controlnet"
cd "$REPO"

OUTPUT_DIR="$REPO/output/mlgd_f_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# ── 5. Run ────────────────────────────────────────────────────────────────────
ADAMDPS=false   # set to true to enable AdamDPS gradient stabilization
EXTRA_ARGS=()
[ "$ADAMDPS" = "true" ] && EXTRA_ARGS+=(--adamdps)

python scripts/run_mlgd_f.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "mlgdf-gender" \
    --mode gender \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --num_variations_schedule constant \
    --num_variations_min 1 \
    --reuse_frac 0.0 \
    "${EXTRA_ARGS[@]}" \
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --loss_fn mmd \
    --target_prompts \
        "Man:a superrealistic portrait photograph of a man, studio lighting:10" \
        "Woman:a superrealistic portrait photograph of a woman, studio lighting:10" \
    --seed 1



# ── 6. Offline analysis (run manually when needed) ────────────────────────────
# python src/analysis.py --run_dir "$OUTPUT_DIR" --plots_dir "$OUTPUT_DIR/plots"

# ── 7. (Optional) Sync outputs ────────────────────────────────────────────────
# Uncomment and adjust if you want to sync results to remote storage:
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/mlgd_f_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
