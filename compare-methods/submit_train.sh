#!/bin/bash
#SBATCH --job-name=compare-train
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/compare_train_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/compare_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# ── Conda setup ───────────────────────────────────────────────────────────────
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting compare-methods training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

# ── Install extra deps ────────────────────────────────────────────────────────
pip install -q flow_matching POT

# ── 2D run ────────────────────────────────────────────────────────────────────
DIM=${1:-2}
OUTPUT_DIR="compare_methods/output/models_${DIM}d_${SLURM_JOB_ID}"

python compare_methods/train_models.py \
    --output_dir "$OUTPUT_DIR" \
    --dim "$DIM" \
    --condition_on 1 \
    --nblocks 3 \
    --nunits 128 \
    --diffusion_steps 100 \
    --nepochs_diff 20000 \
    --nepochs_cm 7500 \
    --nepochs_fm 10000 \
    --batch_size_diff 512 \
    --batch_size_cm 1024 \
    --batch_size_fm 1024 \
    --seed 42

echo "Training complete. Models in: $OUTPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR" > compare_methods/output/last_train_${SLURM_JOB_ID}.env
