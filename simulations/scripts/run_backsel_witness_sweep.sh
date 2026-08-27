#!/bin/bash
#SBATCH --job-name=witness-backsel-sweep
#SBATCH --output=witness_sweep_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=witness_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR RUN HERE
# ══════════════════════════════════════════════════════════════════════════════

# Which experiment?  2D_cond_1D | 5D_cond_1D | 10D_cond_1D
EXPERIMENT="5D_cond_1D"

# ── Shared hyperparameters ────────────────────────────────────────────────────
NUM_X_T=3                     # fixed, not swept
N_RUNS=25
SEED=42
METHODS="LGD LGD-CM"           # any of: LGD LGD-CM
NSAMPLES_LIST="50 100 250 500" # the "n" axis
K_FRACS="0.1 0.2 0.5 1.0"      # the "proportion" axis (backsel_k / nsamples); 1.0
                                # is always included automatically as the "full"
                                # (no-subsampling) baseline even if you omit it
RULES="uniform witness"        # any of: uniform witness
WITNESS_FLOOR=0.3              # defensive-mixture floor for rule=witness (0.3-0.5 recommended);
                                # also the "canonical" alpha that feeds the main JSON/plots/summary
                                # regardless of what ALPHA_LIST below sweeps
BACKSEL_REPLACEMENT=false      # true = sample backsel_k indices with replacement
FORCE_RETRAIN=false            # true | false — true always retrains and overwrites the saved checkpoints

# ── Diagnostics: WHY witness sampling wins or loses, not just whether it does ──
# On by default (small fixed per-run overhead: 5 extra timesteps' worth of
# bookkeeping, not a new multiplicative grid dimension) -- writes
# gradient_variance_*.csv (per-seed ||grad_subsampled - grad_full|| + a
# variance_ratio vs. uniform) and witness_diagnostics_*.csv (per-step scenario
# heterogeneity: witness_std, ess_raw -- tells you whether THIS experiment/
# conditioning even produces enough per-sample mismatch for witness sampling to
# have anything to exploit, independent of the final downstream metric).
# Recommended: inspect these BEFORE trusting/expanding the main grid below --
# if ess_raw stays close to nsamples everywhere, that's a real answer (no
# heterogeneity here for any rule to exploit), not a bug.
DIAG_STEPS="99 75 50 25 1"     # empty string = disable (matches the original,
                                # diagnostics-free behavior)

# Alpha sweep (witness_floor) -- cheap, but OFF by default here (single value =
# WITNESS_FLOOR above) to keep the default grid size unchanged. Set e.g.
# ALPHA_LIST="0.0 0.15 0.3 0.5" to sweep it -- multiplies the witness side of
# the grid by len(ALPHA_LIST); do this on a small targeted run first (few
# nsamples/k_fracs), not the full grid, per the diagnostics-first workflow
# above. Writes an extra *_alpha_sweep.csv when more than one value is set.
ALPHA_LIST=""                  # empty string = just WITNESS_FLOOR, no sweep

# NOTE: the grid runs len(NSAMPLES_LIST) x len(K_FRACS) x len(RULES) x len(METHODS)
# points (with the k_frac=1.0 point computed once per (method, nsamples) and
# reused across rules, since there's nothing to select from at k_frac=1.0),
# times len(ALPHA_LIST) for the witness side if you set it above, each with
# N_RUNS seeded optimize_LGD calls. Trim the lists above or raise --time if
# that's too slow for your cluster.

# ── Misc ──────────────────────────────────────────────────────────────────────
SMOKE_TEST=false               # true = 2 runs / tiny grid only, for quick debug

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
# Activate your conda/venv environment:
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting, e.g.:
                                    #   export ENV_PATH=/path/to/your/env

# ══════════════════════════════════════════════════════════════════════════════
# 2. Caches  (optional — avoids re-downloading HF checkpoints every run)
# ══════════════════════════════════════════════════════════════════════════════
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# Optional — only needed if the HuggingFace checkpoint repo requires auth:
export HF_TOKEN="${HF_TOKEN:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
# REPO_ROOT should point to the root of this repository.
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"

mkdir -p "$REPO_ROOT/simulations/checkpoints/${EXPERIMENT}"
mkdir -p "$REPO_ROOT/simulations/results/${EXPERIMENT}"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Info
# ══════════════════════════════════════════════════════════════════════════════
echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT            : $REPO_ROOT"
echo "    experiment           : $EXPERIMENT"
echo "    num_x_t              : $NUM_X_T"
echo "    n_runs               : $N_RUNS"
echo "    methods              : $METHODS"
echo "    nsamples_list        : $NSAMPLES_LIST"
echo "    k_fracs              : $K_FRACS"
echo "    rules                : $RULES"
echo "    witness_floor        : $WITNESS_FLOOR"
echo "    backsel_replacement  : $BACKSEL_REPLACEMENT"
echo "    diag_steps           : ${DIAG_STEPS:-(disabled)}"
echo "    alpha_list           : ${ALPHA_LIST:-(just witness_floor)}"
echo "    force_retrain        : $FORCE_RETRAIN"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Build and run python command
# ══════════════════════════════════════════════════════════════════════════════
cd "$REPO_ROOT/simulations/scripts"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"

if [ "$SMOKE_TEST" = "true" ]; then
    N_RUNS=2
    NSAMPLES_LIST="20 50"
    K_FRACS="0.5 1.0"
fi

CMD="python run_backsel_witness_sweep.py \
    --experiment          $EXPERIMENT \
    --num_x_t             $NUM_X_T \
    --n_runs              $N_RUNS \
    --seed                $SEED \
    --methods              $METHODS \
    --nsamples_list        $NSAMPLES_LIST \
    --k_fracs               $K_FRACS \
    --rules                 $RULES \
    --witness_floor        $WITNESS_FLOOR"

[ "$BACKSEL_REPLACEMENT" = "true" ] && CMD="$CMD --backsel_replacement"
[ "$FORCE_RETRAIN" = "true" ] && CMD="$CMD --force_retrain"
[ -n "$DIAG_STEPS" ] && CMD="$CMD --diag_steps $DIAG_STEPS"
[ -n "$ALPHA_LIST" ] && CMD="$CMD --alpha_list $ALPHA_LIST"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
