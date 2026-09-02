#!/bin/bash
#SBATCH --job-name=mlgd-f-age
#SBATCH --output=mlgd_f_age_%j.log
#SBATCH --error=mlgd_f_age_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch submit_mlgd_f_age.sh
source "$ENV_PATH/bin/activate"

# ── 2. Caches — redirect to lab storage to avoid home quota issues ────────────
# Uncomment and set LAB_ROOT to a writable directory on your cluster:
# export LAB_ROOT="/path/to/your/lab/storage"
# export HF_HOME="$LAB_ROOT/hf_cache"
# export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
# export XDG_CACHE_HOME="$LAB_ROOT/.cache"
# mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

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

OUTPUT_DIR="output/mlgd_f_age_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Toggles below (defaults in parens; see run_mlgd_f.py --help for full detail):
#   backsel_k=(None=all) N     -- backprop only N of num_variations fresh
#                                 samples/step (loss still sees all of them)
#   backsel_rule=(uniform) | witness  -- witness scores samples first, backprops
#                                 the highest-|score| N for a lower-variance grad
#   witness_floor=(0.3, 0.3-0.5 recommended) -- only used by rule=witness
#   witness_replacement=(off) -- add --witness_replacement to enable
#   use_adam=(off, uses zeta_i*grad) -- add --use_adam (+ --adam_lr/--adam_beta1/
#                                 --adam_beta2/--adam_eps, default 0.01/0.9/0.999/1e-8)

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_mlgd_f.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "mlgdf-age" \
    --mode age \
    --age_min 10 \
    --age_max 80 \
    --age_step 1 \
    --n_per_age 0 \
    --age_gender man \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --backsel_k 20 \
    --backsel_rule uniform \
    --witness_floor 0.3 \
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --loss_fn mmd \
    --seed 1

# ── 6. Offline analysis (run manually when needed) ────────────────────────────
# python src/analysis.py --run_dir "$OUTPUT_DIR" --plots_dir "$OUTPUT_DIR/plots"

# ── 7. (Optional) Sync outputs ────────────────────────────────────────────────
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/mlgd_f_age_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
