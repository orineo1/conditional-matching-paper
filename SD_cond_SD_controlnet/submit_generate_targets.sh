#!/bin/bash
# =============================================================================
# submit_generate_targets.sh
# Stage 1: generate one canonical man + one canonical woman,
#           duplicate 50x each, interpolate VAE latents over 100 steps.
# ~20 min on L40S.
# Run: sbatch submit_generate_targets.sh
# =============================================================================
#SBATCH --job-name=gen-targets
#SBATCH --output=/sci/labs/orzuk/ori_m/conditional-matching-paper/gen_targets_%j.log
#SBATCH --error=/sci/labs/orzuk/ori_m/conditional-matching-paper/gen_targets_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Job $SLURM_JOB_ID on $(hostname)"
export HF_HOME=/sci/labs/orzuk/ori_m/hf_cache
mkdir -p $HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q tqdm

cd /sci/labs/orzuk/ori_m/conditional-matching-paper

python generate_targets.py \
    --output_dir  SD_cond_SD_controlnet/output/interpolation_experiment \
    --n_copies    50 \
    --n_interp    100 \
    --controlnet_scale 0.4 \
    --man_seed    0 \
    --woman_seed  1 \
    --seed        42 \
    --sprinter_model_id    "stabilityai/sdxl-turbo" \
    --architect_model_id   "stabilityai/stable-diffusion-xl-base-1.0" \
    --controlnet_model_id  "xinsir/controlnet-scribble-sdxl-1.0"

echo "generate_targets.py complete."
