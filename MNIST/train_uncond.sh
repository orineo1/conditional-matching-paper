#!/bin/bash
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/logs/uncond_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/logs/uncond_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# ── Environment ───────────────────────────────────────────────
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export WANDB_API_KEY=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
export REPO_PATH="${REPO_PATH:-/sci/labs/orzuk/ori_m/conditional-matching-paper}"

echo "Job:  $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Args: $@"

mkdir -p /sci/labs/orzuk/ori_m/conditional-matching-paper/logs

cd "$REPO_PATH"
python train_mnist_unconditional.py "$@"
