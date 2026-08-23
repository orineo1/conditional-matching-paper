#!/bin/bash
#SBATCH --job-name=target-emb
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/target_emb_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/target_emb_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=catfish

# Use ssa_env which has latent_clip_torch + diffusers + transformers
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/thesis_zuk_shaul/ssa_env

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
mkdir -p /sci/labs/orzuk/shaulytolk/hf_cache

echo "Starting target embedding computation on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

python scripts/compute_target_embeddings.py \
    --anchor_a_path manly_man.png \
    --anchor_b_path feminine_woman.png \
    --n_targets 100 \
    --output_dir output/target_embeddings

echo "Done."
