#!/bin/bash
#SBATCH --job-name=cdms-smoketest
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=salmon

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="/sci/labs/orzuk/ori_m/conditional-matching-paper"
ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"
LOG_DIR="$REPO_ROOT/simulations/logs"

# ── Create all dirs explicitly before SLURM tries to write logs ───────────────
mkdir -p "$LOG_DIR"
mkdir -p "$REPO_ROOT/simulations/results/2D_cond_1D/gridsearch"

# ── Redirect ALL stdout+stderr to a single timestamped file we control ────────
# This way you always know exactly where to look, regardless of SLURM settings
LOGFILE="$LOG_DIR/smoketest_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "START  : $(date)"
echo "Job    : $SLURM_JOB_ID"
echo "Host   : $(hostname)"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
echo "LOGFILE: $LOGFILE"
echo "=========================================="

# ── Activate env ──────────────────────────────────────────────────────────────
echo "[1/4] Activating conda env: $ENV_PATH"
source "$ENV_PATH/bin/activate" || { echo "ERROR: failed to activate env"; exit 1; }
echo "      Python: $(which python)  $(python --version)"

# ── Env vars ──────────────────────────────────────────────────────────────────
export REPO_ROOT="$REPO_ROOT"
export WANDB_API_KEY="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"

# ── Verify the script exists ───────────────────────────────────────────────────
SCRIPT="$REPO_ROOT/simulations/cdms_gridsearch.py"
echo "[2/4] Checking script: $SCRIPT"
ls -lh "$SCRIPT" || { echo "ERROR: script not found at $SCRIPT"; exit 1; }

# ── cd into simulations so relative imports work ──────────────────────────────
cd "$REPO_ROOT/simulations"
echo "[3/4] Working dir: $(pwd)"
echo "      .py files : $(ls *.py 2>/dev/null | tr '\n' ' ')"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[4/4] Launching smoketest (config_id=0) ..."
echo "      (grad_clamp=0.25 | noise_scale=0.0 | num_x_t=3 | CM=128u×3d)"

python "$SCRIPT"              \
    --config_id     0         \
    --wandb_project "cdms-smoketest" \
    --smoke_test

EXIT_CODE=$?
echo "=========================================="
echo "END    : $(date)"
echo "EXIT   : $EXIT_CODE"
echo "LOG AT : $LOGFILE"
echo "=========================================="
exit $EXIT_CODE
