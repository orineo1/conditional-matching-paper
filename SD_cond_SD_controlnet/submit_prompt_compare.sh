#!/bin/bash
#SBATCH --job-name=prompt-cmp
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/prompt_compare_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/prompt_compare_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper/SD_cond_SD_controlnet

python run_prompt_compare.py \
    --scribble_path /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scripts/assets/zeta5_final_guided.png \
    --n_images 100 \
    --seed 1 \
    --output_dir output/prompt_compare_${SLURM_JOB_ID}
