#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --output=eval_all_%j.log
#SBATCH --error=eval_all_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu

# N-sample evaluation of all baseline scribbles (run AFTER run_eval_baselines.sh).
#
# Usage:
#   export ENV_PATH=/path/to/your/env
#   sbatch run_eval_all_experiments.sh BalancedTarget 2000     # one experiment, N=2000
#   sbatch run_eval_all_experiments.sh all 2000                # every experiment
#   sbatch run_eval_all_experiments.sh BalancedTarget 100      # quick smoke test
#
# Outputs -> experiments/<Exp>/eval/ : scribbles_all.png, target_*.png,
#   preview_<method>.png, results_table.csv, eval_results.json
# (and experiments/eval_all_results.json when EXPERIMENT=all)

EXPERIMENT=${1:-all}
N_EVAL=${2:-2000}

# ── 1. Environment ────────────────────────────────────────────────────────────
source "$ENV_PATH/bin/activate"

# ── 2. Caches — redirect to lab storage to avoid home quota issues ────────────
# export LAB_ROOT="/path/to/your/lab/storage"
# export HF_HOME="$LAB_ROOT/hf_cache"
# export MPLCONFIGDIR="$LAB_ROOT/.matplotlib_cache"
# export XDG_CACHE_HOME="$LAB_ROOT/.cache"
# mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# ── 3. Verification ───────────────────────────────────────────────────────────
echo "=== JOB STARTING ON $(hostname) ==="
echo "Experiment : $EXPERIMENT"
echo "N_EVAL     : $N_EVAL"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
echo "============================================"

# ── 4. Runtime configs ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="YOUR_REPO_PATH_HERE"
cd "$REPO"

# ── 5. Run ────────────────────────────────────────────────────────────────────
python scripts/eval_all_experiments.py \
    --experiment    "$EXPERIMENT" \
    --n_eval        "$N_EVAL" \
    --repo_path     "$REPO" \
    --wandb_project "eval-all-experiments"

echo "✅ eval_all_experiments done."
