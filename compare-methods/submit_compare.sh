#!/bin/bash
#SBATCH --job-name=compare-run
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/compare_run_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/compare_run_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# ── Conda setup ───────────────────────────────────────────────────────────────
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export WANDB_API_KEY=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

echo "Starting compare-methods run on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

pip install -q flow_matching POT

# ── Required arg: path to models directory ────────────────────────────────────
# Pass as first argument, e.g.:
#   sbatch submit_compare.sh compare_methods/output/models_2d_44221193
MODELS_DIR=${1:?"Usage: sbatch submit_compare.sh <models_dir>"}

OUTPUT_DIR="compare_methods/output/compare_${SLURM_JOB_ID}"

python compare_methods/run_compare.py \
    --models_dir "$MODELS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_attempts 25 \
    --nsamples_mmd 250 \
    --num_x_t 3 \
    --nsamples_swd 10000 \
    --num_projections_swd 500 \
    --x_star -5.0 \
    --wandb_project compare-methods \
    --wandb_entity conditional-matching \
    --seed 42

echo "Comparison complete. Results in: $OUTPUT_DIR"
