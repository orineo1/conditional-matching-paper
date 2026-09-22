#!/bin/bash
#SBATCH --job-name=mlgd-f
#SBATCH --output=mlgd_f_%j.log
#SBATCH --error=mlgd_f_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu   # <-- change here, or override at submit time:
                          #     sbatch --partition=<name> gender_submit_mlgd_f.sh

# ── 1. Environment ────────────────────────────────────────────────────────────
# Set ENV_PATH to your Python environment before submitting:
#   export ENV_PATH=/path/to/your/env
#   sbatch gender_submit_mlgd_f.sh
source "$ENV_PATH/bin/activate"

# ── Configurable settings (env-overridable; edit the defaults here, or export
# before submitting, e.g. REPO_PATH=/my/path WANDB_PROJECT=my-proj SEED=42 \
#     sbatch gender_submit_mlgd_f.sh) ───────────────────────────────────────────
REPO_PATH="${REPO_PATH:?REPO_PATH is not set. Export it before submitting (path to the outer git repo root, containing SD_cond_SD_controlnet/).}"
WANDB_PROJECT="${WANDB_PROJECT:-mlgdf-gender}"
WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it before submitting.}"
SEED="${SEED:-1}"
N_STEPS="${N_STEPS:-30}"
START_STEP="${START_STEP:-15}"
NUM_VARIATIONS="${NUM_VARIATIONS:-6}"
BACKSEL_K="${BACKSEL_K:-20}"
BACKSEL_RULE="${BACKSEL_RULE:-uniform}"
BASE_ZETA="${BASE_ZETA:-5.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"
# Each "name:prompt:n" triple separated by '|' (prompts themselves may contain
# spaces/commas, just not '|'). Override e.g.:
#   TARGET_PROMPTS='Man:a photo of a man:5|Woman:a photo of a woman:15' sbatch ...
TARGET_PROMPTS="${TARGET_PROMPTS:-Man:a superrealistic portrait photograph of a man, studio lighting:10|Woman:a superrealistic portrait photograph of a woman, studio lighting:10}"
IFS='|' read -r -a TARGET_PROMPTS_ARR <<< "$TARGET_PROMPTS"

# ── 2. Caches (optional — redirect if your home quota is limited) ─────────────
# export HF_HOME=/path/to/hf_cache
# export MPLCONFIGDIR=/path/to/matplotlib_cache
# export XDG_CACHE_HOME=/path/to/cache

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
echo "    REPO_PATH      : $REPO_PATH"
echo "    WANDB_PROJECT  : $WANDB_PROJECT"
echo "    SEED           : $SEED"
echo "    N_STEPS        : $N_STEPS"
echo "    START_STEP     : $START_STEP"
echo "    NUM_VARIATIONS : $NUM_VARIATIONS"
echo "    BACKSEL_K      : $BACKSEL_K"
echo "    BACKSEL_RULE   : $BACKSEL_RULE"
echo "    BASE_ZETA      : $BASE_ZETA"
echo "    GUIDANCE_SCALE : $GUIDANCE_SCALE"
echo "    TARGET_PROMPTS : $TARGET_PROMPTS"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# scripts/run_mlgd_f.py below is resolved relative to this path (THIS
# subdirectory, SD_cond_SD_controlnet/, not the outer git repo root).
REPO="$REPO_PATH/SD_cond_SD_controlnet"
cd "$REPO"

OUTPUT_DIR="$REPO/output/mlgd_f_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Toggles below (defaults in parens; see run_mlgd_f.py --help for full detail):
#   backsel_k=(None=all) N     -- backprop only N of num_variations fresh
#                                 samples/step (loss still sees all of them)
#   backsel_rule=(uniform) | witness  -- witness scores samples first, backprops
#                                 the highest-|score| N for a lower-variance grad
#   witness_floor=(0.3, 0.3-0.5 recommended) -- only used by rule=witness
#   witness_temperature=(1.0) -- |score|^(1/T) before the floor blend; T>1
#                                 flattens toward uniform, T<1 sharpens
#   witness_replacement=(off) -- add --witness_replacement to enable
#   use_adam=(off, uses zeta_i*grad) -- add --use_adam (+ --adam_lr/--adam_beta1/
#                                 --adam_beta2/--adam_eps, default 0.01/0.9/0.999/1e-8)

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/run_mlgd_f.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --mode gender \
    --n_steps "$N_STEPS" \
    --start_step "$START_STEP" \
    --num_variations "$NUM_VARIATIONS" \
    --backsel_k "$BACKSEL_K" \
    --backsel_rule "$BACKSEL_RULE" \
    --witness_floor 0.3 \
    --witness_temperature 1.0 \
    --base_zeta "$BASE_ZETA" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --controlnet_scale 0.5 \
    --loss_fn mmd \
    --target_prompts "${TARGET_PROMPTS_ARR[@]}" \
    --seed "$SEED"



# ── 6. Offline analysis (run manually when needed) ────────────────────────────
# python src/analysis.py --run_dir "$OUTPUT_DIR" --plots_dir "$OUTPUT_DIR/plots"

# ── 7. (Optional) Sync outputs ────────────────────────────────────────────────
# Uncomment and adjust if you want to sync results to remote storage:
# rclone copy "$OUTPUT_DIR" "remote:your-bucket/mlgd_f_${SLURM_JOB_ID}" \
#     --tpslimit 10 --transfers 4
echo "✅ Done."
