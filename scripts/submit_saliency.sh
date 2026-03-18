#!/bin/bash
#SBATCH --job-name=gender-saliency
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/saliency_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/saliency_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting gender saliency on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

# ── Set your scribble path here ──
SCRIBBLE_PATH="${1:-scripts/assets/zeta5_input_scribble.png}"

python scripts/gender_saliency.py \
    --scribble_path "$SCRIBBLE_PATH" \
    --output_dir output/saliency_${SLURM_JOB_ID} \
    --n_seeds 20 \
    --sigma 2.5 \
    --device cuda

echo "Gender saliency complete."
