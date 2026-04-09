#!/bin/bash
#SBATCH --job-name=dps-repro-check
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/repro_check_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/repro_check_%j.err
#SBATCH --time=1:00:00  # Doubled time since we run twice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── 1. Environment Setup ─────────────────────────────────────────────────────
export ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
source $ENV_PATH/bin/activate

export LAB_ROOT="/sci/labs/orzuk/ori_m"
export HF_HOME="$LAB_ROOT/hf_cache"
export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
export XDG_CACHE_HOME="$LAB_ROOT/.cache"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# ── 2. Define Paths and Shared Config ────────────────────────────────────────
SEED=1
RUN_ID="${SLURM_JOB_ID}"
OUTPUT_A="SD_cond_SD_controlnet/output/repro_A_${RUN_ID}"
OUTPUT_B="SD_cond_SD_controlnet/output/repro_B_${RUN_ID}"

# We use the exact parameters you provided
COMMON_ARGS="--n_steps 30 --start_step 15 --num_variations 6 --n_targets 6 --base_zeta 5.0 --guidance_scale 0.0 --controlnet_scale 0.5 --n_eval 6 --architect_model_id stabilityai/sdxl-turbo --sprinter_model_id stabilityai/sdxl-turbo --controlnet_model_id xinsir/controlnet-scribble-sdxl-1.0 --loss_fn mmd --seed $SEED"

# ── 3. Run Experiment A ──────────────────────────────────────────────────────
echo "=== STARTING RUN A ==="
python SD_cond_SD_controlnet/run_dps.py --output_dir "$OUTPUT_A" $COMMON_ARGS

# ── 4. Run Experiment B ──────────────────────────────────────────────────────
echo "=== STARTING RUN B (SAME SEED: $SEED) ==="
python SD_cond_SD_controlnet/run_dps.py --output_dir "$OUTPUT_B" $COMMON_ARGS

# ── 5. Verification Logic ────────────────────────────────────────────────────
echo "===================================================="
echo "         REPRODUCIBILITY VERIFICATION               "
echo "===================================================="

# A. Compare Final MMD Value from metrics.json
MMD_A=$(python3 -c "import json; print(json.load(open('$OUTPUT_A/metrics.json'))['final_lgd_cm_mmd'])")
MMD_B=$(python3 -c "import json; print(json.load(open('$OUTPUT_B/metrics.json'))['final_lgd_cm_mmd'])")

echo "Final MMD (Run A): $MMD_A"
echo "Final MMD (Run B): $MMD_B"

if [ "$MMD_A" == "$MMD_B" ]; then
    echo "✅ SUCCESS: MMD values match exactly."
else
    echo "❌ FAILURE: MMD values differ!"
fi

# B. Compare Image Hash (Bit-for-bit check)
HASH_A=$(md5sum "$OUTPUT_A/final_scribble_lgd_cm.png" | awk '{print $1}')
HASH_B=$(md5sum "$OUTPUT_B/final_scribble_lgd_cm.png" | awk '{print $1}')

echo "Image Hash (Run A): $HASH_A"
echo "Image Hash (Run B): $HASH_B"

if [ "$HASH_A" == "$HASH_B" ]; then
    echo "✅ SUCCESS: Output images are bit-for-bit identical."
else
    echo "❌ FAILURE: Images differ! Check your torch.Generator implementation."
fi

# ── 6. Sync Results ──────────────────────────────────────────────────────────
echo "Syncing results to GDrive..."
rclone copy "$OUTPUT_A" "gdrive:conditional-matching/runs/repro_test_${RUN_ID}/A" --tpslimit 10
rclone copy "$OUTPUT_B" "gdrive:conditional-matching/runs/repro_test_${RUN_ID}/B" --tpslimit 10

echo "✅ Job Process Finished."