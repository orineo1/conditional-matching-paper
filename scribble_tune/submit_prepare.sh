#!/bin/bash
#SBATCH --job-name=prep-quickdraw
#SBATCH --output=scribble_tune/output/prepare_%j.log
#SBATCH --error=scribble_tune/output/prepare_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=catfish

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

mkdir -p scribble_tune/data/quickdraw_512
mkdir -p scribble_tune/output

echo "Starting QuickDraw data preparation..."
python scribble_tune/prepare_data.py --config scribble_tune/config.yaml

echo "Data preparation complete."
ls -lh scribble_tune/data/quickdraw_512/ | tail -5
wc -l scribble_tune/data/quickdraw_512/metadata.jsonl
