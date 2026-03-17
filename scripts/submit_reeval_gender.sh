#!/bin/bash
#SBATCH --job-name=reeval-gender
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/reeval_gender_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/reeval_gender_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

# facenet-pytorch for MTCNN
export PYTHONPATH=/sci/labs/orzuk/shaulytolk/.local_packages:$PYTHONPATH

echo "Starting gender re-eval (fine-tuned weights) on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

# Evaluate all runs (auto-selects fine-tuned weights)
python scripts/reeval_gender.py --base_dir autoresearch_output

echo "Re-evaluation complete."
