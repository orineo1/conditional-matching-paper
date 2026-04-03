#!/bin/bash
#!/bin/bash
#SBATCH --job-name=dps-main
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/dps_main_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/dps_main_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export WANDB_API_KEY=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

echo "Job ID: $SLURM_JOB_ID"

# ================================================================
# run_mnist_training.sh
# Launches all training runs for WandB comparison.
# Run this on the cluster: bash run_mnist_training.sh
# ================================================================

# Set your WandB API key (or `wandb login` beforehand)
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export WANDB_API_KEY=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

# Set repo path so the training scripts can import your modules
export REPO_PATH="/content/GlobalConditional"   # ← adjust to your cluster path

echo "============================================================"
echo " MNIST Unconditional UNet Training"
echo "============================================================"

# ── Model A (BASELINE): exact thesis reproduction, no EMA ──────
python train_mnist_unconditional.py \
    --run_name   "uncond_unet_baseline" \
    --nepochs    100 \
    --batch_size 256 \
    --lr         1e-3 \
    --weight_decay 1e-4 \
    --wandb_project "mnist-unconditional" &

python train_mnist_unconditional.py \
    --run_name   "uncond_unet_baseline" \
    --nepochs    500 \
    --batch_size 256 \
    --lr         1e-3 \
    --weight_decay 1e-4 \
    --wandb_project "mnist-unconditional" &
# ── Model B: same architecture + EMA (μ=0.9999) ────────────────
python train_mnist_unconditional.py \
    --use_ema \
    --ema_decay  0.9999 \
    --run_name   "uncond_unet_EMA" \
    --nepochs    100 \
    --batch_size 256 \
    --lr         1e-3 \
    --weight_decay 1e-4 \
    --wandb_project "mnist-unconditional" &

python train_mnist_unconditional.py \
    --use_ema \
    --ema_decay  0.9999 \
    --run_name   "uncond_unet_EMA" \
    --nepochs    500 \
    --batch_size 256 \
    --lr         1e-3 \
    --weight_decay 1e-4 \
    --wandb_project "mnist-unconditional" &


wait
echo "Unconditional training done."

echo ""
echo "============================================================"
echo " MNIST Conditional Consistency Model Training"
echo "============================================================"

# ── Model A (BASELINE): exact thesis reproduction, iCT, mu=0 ──
python train_mnist_conditional.py \
    --run_name   "cond_cm_baseline" \
    --nepochs    500 \
    --batch_size 256 \
    --lr         1e-4 \
    --weight_decay 1e-4 \
    --cond_noise   0.05 \
    --pixel_dropout 0.1 \
    --wandb_project "mnist-conditional-cm" &

# ── Model A (BASELINE): exact thesis reproduction, iCT, mu=0 ──
python train_mnist_conditional.py \
    --run_name   "cond_cm_baseline" \
    --nepochs    1000 \
    --batch_size 256 \
    --lr         1e-4 \
    --weight_decay 1e-4 \
    --cond_noise   0.05 \
    --pixel_dropout 0.1 \
    --wandb_project "mnist-conditional-cm" &

# ── Model B: same + EMA target (mu=0.9999) ─────────────────────
python train_mnist_conditional.py \
    --use_ema \
    --ema_decay  0.9999 \
    --run_name   "cond_cm_EMA" \
    --nepochs    500 \
    --batch_size 256 \
    --lr         1e-4 \
    --weight_decay 1e-4 \
    --cond_noise   0.05 \
    --pixel_dropout 0.1 \
    --wandb_project "mnist-conditional-cm" &

python train_mnist_conditional.py \
    --use_ema \
    --ema_decay  0.9999 \
    --run_name   "cond_cm_EMA" \
    --nepochs    1000 \
    --batch_size 256 \
    --lr         1e-4 \
    --weight_decay 1e-4 \
    --cond_noise   0.05 \
    --pixel_dropout 0.1 \
    --wandb_project "mnist-conditional-cm" &

wait
echo "All training done."
