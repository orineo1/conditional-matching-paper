#!/bin/bash
#SBATCH --job-name=exp-cond-1d
#SBATCH --output=exp_cond_1d_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=exp_cond_1d_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# Headlessly execute one of the Exp_<N>D_cond_1D.ipynb notebooks end-to-end via
# papermill (LGD + LGD-CM optimization, both starting x_T from real Gaussian
# noise -- see Optimization.optimize_LGD). Submit once per dimension:
#
#   export REPO_ROOT=... ENV_PATH=...
#   EXPERIMENT=2D_cond_1D  sbatch --job-name=exp-2d  run_exp_cond_1d.sh
#   EXPERIMENT=5D_cond_1D  sbatch --job-name=exp-5d  run_exp_cond_1d.sh
#   EXPERIMENT=10D_cond_1D sbatch --job-name=exp-10d run_exp_cond_1d.sh
#
# FORCE_RETRAIN defaults to true here: each notebook's Config cell is tagged
# "parameters", so this script overrides it via `papermill -p` regardless of
# what's checked into the notebook -- always trains the CM/diffusion models
# from scratch instead of loading a cached checkpoint. Pass
# FORCE_RETRAIN=false to reuse existing checkpoints instead.
# ══════════════════════════════════════════════════════════════════════════════

# Which experiment?  2D_cond_1D | 5D_cond_1D | 10D_cond_1D
EXPERIMENT="${EXPERIMENT:-2D_cond_1D}"
FORCE_RETRAIN="${FORCE_RETRAIN:-true}"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting
pip install -q papermill

# ══════════════════════════════════════════════════════════════════════════════
# 2. Caches
# ══════════════════════════════════════════════════════════════════════════════
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
NOTEBOOKS_DIR="$REPO_ROOT/simulations/notebooks"
IN_NB="$NOTEBOOKS_DIR/Exp_${EXPERIMENT}.ipynb"
OUT_DIR="$REPO_ROOT/simulations/results/${EXPERIMENT}"
mkdir -p "$OUT_DIR"
OUT_NB="$OUT_DIR/Exp_${EXPERIMENT}_executed_${SLURM_JOB_ID:-local}.ipynb"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT      : $REPO_ROOT"
echo "    experiment     : $EXPERIMENT"
echo "    force_retrain  : $FORCE_RETRAIN"
echo "    input nb       : $IN_NB"
echo "    output nb      : $OUT_NB"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Run the notebook end-to-end
# ══════════════════════════════════════════════════════════════════════════════
# --cwd so the notebook's own `os.getcwd()`-relative BASE_DIR (../params,
# ../checkpoints, ../results) resolves the same way it does when run
# interactively from simulations/notebooks/. -p FORCE_RETRAIN overrides the
# notebook's own default via its "parameters"-tagged Config cell.
#
# papermill -p uses Python's ast.literal_eval on the value, falling back to
# the raw string on failure -- ast.literal_eval("false") (bash-style
# lowercase) raises and falls back to the STRING "false", which is truthy in
# Python, silently flipping `if not force_retrain: load else train` the
# wrong way. Capitalize to the Python literal ("True"/"False") so
# literal_eval actually parses it as a bool.
FORCE_RETRAIN_PY="$(tr '[:lower:]' '[:upper:]' <<< "${FORCE_RETRAIN:0:1}")${FORCE_RETRAIN:1}"
echo "papermill -p FORCE_RETRAIN $FORCE_RETRAIN_PY"
papermill "$IN_NB" "$OUT_NB" --cwd "$NOTEBOOKS_DIR" --log-output \
    -p FORCE_RETRAIN "$FORCE_RETRAIN_PY"

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
