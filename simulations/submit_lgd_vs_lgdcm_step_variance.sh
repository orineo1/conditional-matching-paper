#!/bin/bash
#SBATCH --job-name=cdm-lgd-vs-cm-stepvar
#SBATCH --output=logs/stepvar_%j.log
#SBATCH --error=logs/stepvar_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=catfish   # override with sbatch --partition=

# ══════════════════════════════════════════════════════════════════════════════
# Per-step gradient variance/accuracy of LGD's inner sampler (K-step DDIM
# unroll through model_cond) vs. LGD-CM's inner sampler (the consistency
# model's own ~14-step multistep sampling procedure), averaged over
# N_TRAJECTORIES independent optimization trajectories. See
# lgd_vs_lgdcm_step_variance.py's docstring for the full
# design. For the original single-point, sweep-K-only version, use the
# separate submit_gradient_variance.sh instead.
#
# Submit with:
#   export ENV_PATH=/path/to/your/conda/or/venv/env   # dir containing bin/python
#   export REPO_ROOT=/path/to/conditional-matching-paper
#   export EXPERIMENT_NAME=10D_cond_1D                # optional, defaults to 10D_cond_1D
#   sbatch simulations/submit_lgd_vs_lgdcm_step_variance.sh
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME="${EXPERIMENT_NAME:-10D_cond_1D}"   # 2D_cond_1D | 5D_cond_1D | 10D_cond_1D
N_TRAJECTORIES=10          # independent unguided trajectories to average per-step metrics over
STEP_STRIDE=10             # analyze every STEP_STRIDE-th outer diffusion step (1 = every step)
N_REDRAWS=30               # independent redraws of each inner sampler per (trajectory, step)
NSAMPLES=250               # batch size for the inner MMD estimator (both methods)
K_LGD=""                   # unroll depth for LGD's inner sampler; empty = experiment's full
                            # diffusion_steps (matches what real LGD guidance actually runs)
GRAD_REF_N=2000            # sample size for the TRUE/population reference gradient at each
                            # state (closed-form, no network forward -- cheap even here)
SEED=42                    # base seed; trajectory i uses SEED + i
SMOKE_TEST=false           # true = tiny sanity-check run (2 trajectories, coarse stride,
                            # few redraws) instead of the real sweep above

export ENV_PATH="${ENV_PATH:?ENV_PATH is not set. Export it before submitting (dir containing bin/python).}"
PYTHON="$ENV_PATH/bin/python"

export HF_TOKEN="${HF_TOKEN:-}"   # optional: only needed if checkpoints aren't cached locally yet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"
cd "$REPO_ROOT/simulations"
export PYTHONPATH="$REPO_ROOT/simulations/src:$PYTHONPATH"
mkdir -p logs

KLGD_ARGS=()
[ -n "$K_LGD" ] && KLGD_ARGS=(--k_lgd "$K_LGD")

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    experiment      : $EXPERIMENT_NAME"
echo "    n_trajectories  : $N_TRAJECTORIES"
echo "    step_stride     : $STEP_STRIDE"
echo "    n_redraws       : $N_REDRAWS"
echo "    nsamples        : $NSAMPLES"
echo "    k_lgd           : ${K_LGD:-(full diffusion_steps)}"
echo "    grad_ref_n      : $GRAD_REF_N"
echo "    smoke_test      : $SMOKE_TEST"
"$PYTHON" -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "============================================"

if [ "$SMOKE_TEST" = "true" ]; then
    "$PYTHON" lgd_vs_lgdcm_step_variance.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --seed             "$SEED" \
        --smoke \
        --plot
else
    "$PYTHON" lgd_vs_lgdcm_step_variance.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --n_trajectories   "$N_TRAJECTORIES" \
        --step_stride       "$STEP_STRIDE" \
        --n_redraws          "$N_REDRAWS" \
        --nsamples            "$NSAMPLES" \
        --grad_ref_n          "$GRAD_REF_N" \
        --seed                "$SEED" \
        "${KLGD_ARGS[@]}" \
        --plot
fi

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
