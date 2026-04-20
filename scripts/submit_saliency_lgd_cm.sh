#!/bin/bash
#SBATCH --job-name=saliency-lgd-cm
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/saliency_lgd_cm_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/saliency_lgd_cm_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q numpy --force-reinstall

echo "Starting saliency analysis (lgd_cm) on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

# Step 1: gender saliency map on the SOURCE scribble
python scripts/gender_saliency.py \
    --scribble_path scripts/assets/lgd_cm_source_scribble.png \
    --output_dir output/saliency_lgd_cm_${SLURM_JOB_ID} \
    --n_seeds 20 \
    --sigma 2.5 \

# Step 2: pixel diff between source and optimized scribble
python scripts/scribble_diff.py \
    --guided scripts/assets/lgd_cm_optimized_scribble.png \
    --unguided scripts/assets/lgd_cm_source_scribble.png \
    --output_dir output/scribble_diff_lgd_cm_${SLURM_JOB_ID} \
    --sigma 2.0

echo "Saliency lgd_cm analysis complete."
