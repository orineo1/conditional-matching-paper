#!/bin/bash
#SBATCH --job-name=gender-eval
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/gender_eval_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/gender_eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export WANDB_API_KEY=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

echo "Starting gender evaluation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

pip install "sympy>=1.12,<1.14"

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

python scripts/evaluate_gender_balance.py \
    --image_dir SD_cond_SD_controlnet/output/dps_hed_ft_44245644/ \
    --run_name gender_eval_hed_ft \
    --wandb_project conditional-matching \
    --weights_path /sci/labs/orzuk/shaulytolk/models/fairface/res34_fair_align_multi_7_20190809.pt

echo "Gender evaluation complete."
