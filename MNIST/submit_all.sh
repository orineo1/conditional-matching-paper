#!/bin/bash
# =============================================================
# submit_all.sh
# Submit all MNIST training runs as separate SLURM jobs.
# Usage: bash submit_all.sh
# =============================================================

REPO_PATH="/sci/labs/orzuk/ori_m/conditional-matching-paper"

# ── Unconditional UNet ────────────────────────────────────────
sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=uncond_baseline_100 \
    slurm/train_uncond.sh \
        --run_name   "uncond_unet_baseline_100ep" \
        --nepochs    100 \
        --batch_size 256 \
        --lr         1e-3 \
        --weight_decay 1e-4 \
        --wandb_project "mnist-unconditional"

sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=uncond_baseline_500 \
    slurm/train_uncond.sh \
        --run_name   "uncond_unet_baseline_500ep" \
        --nepochs    500 \
        --batch_size 256 \
        --lr         1e-3 \
        --weight_decay 1e-4 \
        --wandb_project "mnist-unconditional"

sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=uncond_ema_100 \
    slurm/train_uncond.sh \
        --use_ema \
        --ema_decay  0.9999 \
        --run_name   "uncond_unet_EMA_100ep" \
        --nepochs    100 \
        --batch_size 256 \
        --lr         1e-3 \
        --weight_decay 1e-4 \
        --wandb_project "mnist-unconditional"

sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=uncond_ema_500 \
    slurm/train_uncond.sh \
        --use_ema \
        --ema_decay  0.9999 \
        --run_name   "uncond_unet_EMA_500ep" \
        --nepochs    500 \
        --batch_size 256 \
        --lr         1e-3 \
        --weight_decay 1e-4 \
        --wandb_project "mnist-unconditional"

# ── Conditional Consistency Model ─────────────────────────────
sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=cond_baseline_500 \
    slurm/train_cond.sh \
        --run_name   "cond_cm_baseline_500ep" \
        --nepochs    500 \
        --batch_size 256 \
        --lr         1e-4 \
        --weight_decay 1e-4 \
        --cond_noise   0.05 \
        --pixel_dropout 0.1 \
        --wandb_project "mnist-conditional-cm"

sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=cond_baseline_1000 \
    slurm/train_cond.sh \
        --run_name   "cond_cm_baseline_1000ep" \
        --nepochs    1000 \
        --batch_size 256 \
        --lr         1e-4 \
        --weight_decay 1e-4 \
        --cond_noise   0.05 \
        --pixel_dropout 0.1 \
        --wandb_project "mnist-conditional-cm"

sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=cond_ema_500 \
    slurm/train_cond.sh \
        --use_ema \
        --ema_decay  0.9999 \
        --run_name   "cond_cm_EMA_500ep" \
        --nepochs    500 \
        --batch_size 256 \
        --lr         1e-4 \
        --weight_decay 1e-4 \
        --cond_noise   0.05 \
        --pixel_dropout 0.1 \
        --wandb_project "mnist-conditional-cm"

sbatch --export=ALL,REPO_PATH=$REPO_PATH \
    --job-name=cond_ema_1000 \
    slurm/train_cond.sh \
        --use_ema \
        --ema_decay  0.9999 \
        --run_name   "cond_cm_EMA_1000ep" \
        --nepochs    1000 \
        --batch_size 256 \
        --lr         1e-4 \
        --weight_decay 1e-4 \
        --cond_noise   0.05 \
        --pixel_dropout 0.1 \
        --wandb_project "mnist-conditional-cm"

echo "All jobs submitted. Check with: squeue -u $USER"
