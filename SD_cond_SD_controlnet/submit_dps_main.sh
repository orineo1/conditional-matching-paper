#!/bin/bash
#SBATCH --job-name=dps-main
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/dps_main_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/dps_main_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment Setup (YOUR Private Env) ──────────────────────────────────
# Point to the new environment we just built and verified on salmon-01
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

# ── 2. Redirect Caches to Lab (Keep these to avoid "Home Full" error) ────────
export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification ──────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import torch; print(f'GPU Check: {torch.cuda.is_available()}')"
echo "==================================="

# ── 4. Runtime Configs ───────────────────────────────────────────────────────
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# Create output dir early so rclone doesn't fail if the script crashes
OUTPUT_DIR="SD_cond_SD_controlnet/output/dps_main_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# ── 5. Run the Pipeline ──────────────────────────────────────────────────────
# We just use 'python' now because your environment is 'source'-ed
python SD_cond_SD_controlnet/run_dps.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "dps_man_woman_ratio" \
    --n_steps 30 \
    --start_step 15 \
    --num_variations 6 \
    --n_targets 6 \
    --ratio 0.3 \        # ← 30% man / 70% woman
    --base_zeta 5.0 \
    --guidance_scale 0.0 \
    --controlnet_scale 0.5 \
    --n_eval 6 \
    --n_eval_final 50 \
    --sprinter_variation_prompt "a superrealistic professional photograph of" \
    --sprinter_target_man_prompt "a superrealistic portrait photograph of a man, studio lighting" \
    --sprinter_target_woman_prompt "a superrealistic portrait photograph of a woman, studio lighting" \
    --sprinter_eval_prompt "a superrealistic professional photograph of" \
    --architect_model_id "stabilityai/sdxl-turbo" \
    --sprinter_model_id "stabilityai/sdxl-turbo" \
    --controlnet_model_id "xinsir/controlnet-scribble-sdxl-1.0" \
    --loss_fn mmd \
    --loss_scale 1.0 \
    --bandwidth_scale 1.0 \
    --kernel_alpha 1.0 \
    --seed 1
# ── 6. Run offline analysis (PCA, t-SNE, KDE, boxplot, heatmap) ──────────────
echo "Running analysis..."
python SD_cond_SD_controlnet/analysis.py \
    --run_dir "$OUTPUT_DIR" \
    --plots_dir "$OUTPUT_DIR/plots"
echo "✅ Analysis complete."

# ── 7. Sync to GDrive ────────────────────────────────────────────────────────
echo "Syncing $OUTPUT_DIR to Google Drive..."
rclone copy "$OUTPUT_DIR" "gdrive:conditional-matching/runs/dps_main_${SLURM_JOB_ID}" \
    --tpslimit 10 --cache-rps 50 --transfers 4 \
    --exclude "photos_lgd_cm/**" \
    --exclude "photos_regular/**"
echo "✅ Job Process Finished."