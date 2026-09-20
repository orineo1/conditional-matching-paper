#!/bin/bash
# pgd_opt_lr_sweep.sh -- submit the GenderInterpolation/SkewedTarget/BalancedTarget/
# AgeInterpolation PGD runs at three --opt_lr values each, via pgd_submit.sh.
#
# This is a plain bash driver (NOT itself an sbatch script) that just calls
# `sbatch pgd_submit.sh ...` in a loop -- run it directly on the login node:
#
#   export ENV_PATH=/path/to/your/env
#   bash pgd_opt_lr_sweep.sh
#
# opt_lr=0.05 is the current default; 0.5 and 2.0 are included because the
# default's per-pixel step size looked too small in earlier runs (opt_grad_norm
# ~4 spread over a 3x512x512 image gives a per-pixel step on the order of
# 1e-4 at opt_lr=0.05 -- see proj_l2_init staying ~0, i.e. the ambient step
# barely moved w off the VAE's manifold). Adjust OPT_LRS below to taste.
#
# All other hyperparameters (target_minutes/seed/num_variations per
# experiment) match the earlier agreed values; --experiment fills in the rest
# (target prompts, mode, controlnet_scale, canonical source scribble) via
# EXPERIMENT_PRESETS in scripts/run_pgd.py. Every hyperparameter -- including
# opt_lr -- is logged to the run's wandb config (vars(args)).

set -euo pipefail

OPT_LRS=(0.05 0.5 2.0)

# EXPERIMENT TARGET_MINUTES SEED LOSS_FN NUM_VARIATIONS
RUNS=(
    "GenderInterpolation 168 1 mmd 100"
    "SkewedTarget        168 1 mmd 100"
    "BalancedTarget      168 1 mmd 100"
    "AgeInterpolation    204 3 mmd 120"
)

for run in "${RUNS[@]}"; do
    read -r EXPERIMENT TARGET_MINUTES SEED LOSS_FN NUM_VARIATIONS <<< "$run"
    for OPT_LR in "${OPT_LRS[@]}"; do
        echo "Submitting: $EXPERIMENT  target_minutes=$TARGET_MINUTES  seed=$SEED  " \
             "loss_fn=$LOSS_FN  num_variations=$NUM_VARIATIONS  opt_lr=$OPT_LR"
        sbatch pgd_submit.sh "$EXPERIMENT" "$TARGET_MINUTES" "$SEED" "$LOSS_FN" \
            "$NUM_VARIATIONS" "$OPT_LR"
    done
done
