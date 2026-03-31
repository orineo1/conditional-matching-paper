#!/bin/bash
#SBATCH --job-name=validate-pc1
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/validate_pc1_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/validate_pc1_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

python SD_cond_SD_controlnet/validate_pc1.py

echo "validate_pc1 complete."
