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

# Backprop-subsampling (--backsel_k / --backsel_rule): with e.g. num_variations=100
# and backsel_k=20, the loss still sees 100 fresh Sprinter samples/step but gradient
# only flows through 20 of them each step — cuts backward-pass cost ~5x for the same
# loss fidelity. Leave --backsel_k unset (or >= num_variations) to backprop through
# all of them (original behavior). Two selection rules:
#   uniform (default) — the 20 are picked with no extra cost.
#   witness           — scores all 100 first (one cheap extra no_grad forward pass),
#                        then backprops through the 20 with the largest |MMD witness
#                        score| (samples in the region of biggest mismatch to target)
#                        for a lower-variance gradient at the same k.
# Uncomment one of the lines below to enable it:
# BACKSEL_ARGS="--backsel_k 20 --backsel_rule uniform"
# BACKSEL_ARGS="--backsel_k 20 --backsel_rule witness --witness_floor 0.1"
BACKSEL_ARGS=""

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
    --reuse_frac 0.0 \
    $BACKSEL_ARGS \
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
