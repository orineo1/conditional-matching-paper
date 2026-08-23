#!/bin/bash
#SBATCH --job-name=discrete-smc
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/discrete_smc_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/discrete_smc_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

ARM="${1:-flip}"
echo "Starting discrete-x SMC suite arm=$ARM on $(hostname) GPU: $CUDA_VISIBLE_DEVICES (job $SLURM_JOB_ID)"

export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

COMMON="--loss image --prefix \"a photo of a\" --estimators mode sampled \
  --seeds 0 1 2 --n_particles 128 --T 16 --beta 200 --beta_anneal --top_k 50 \
  --n_dec 4 --n_cond 8 --n_target 64 --remask_frac 0.75"
# mode-only for the newer arms: the estimator question is settled (mode wins
# at image scale at ~1/12 the cost)
COMMON_MODE="--loss image --prefix \"a photo of a\" --estimators mode \
  --seeds 0 1 2 --n_particles 128 --T 16 --beta 200 --beta_anneal --top_k 50 \
  --n_cond 8 --n_target 64 --remask_frac 0.75"

case "$ARM" in
  flip)
    # multi-seed check of the mode-vs-sampled ranking on the rain pair
    eval python discrete_x/real_smc.py $COMMON \
      --target_words person standing in the rain \
      --source_words man walking down the street \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_flip
    ;;
  instances)
    # two additional source->target pairs
    eval python discrete_x/real_smc.py $COMMON \
      --target_words woman dancing on the beach \
      --source_words man sitting in the office \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_beach
    eval python discrete_x/real_smc.py $COMMON \
      --target_words dog running through the snow \
      --source_words cat sleeping on the sofa \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_snow
    ;;
  corrupt)
    # "truly noisy" start: random-token substitution before re-masking
    eval python discrete_x/real_smc.py $COMMON \
      --target_words person standing in the rain \
      --source_words man walking down the street \
      --corrupt_frac 0.4 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_corrupt
    ;;
  tilt)
    # gradient-shortlisted, exactly-scored CLIP-text proposal tilt
    eval python discrete_x/real_smc.py $COMMON \
      --target_words person standing in the rain \
      --source_words man walking down the street \
      --grad_tilt --tilt_scale 1.0 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_tilt
    ;;
  wide)
    # 10-token canvas: room for framing/composition words
    eval python discrete_x/real_smc.py $COMMON_MODE \
      --target_words person standing alone in the rain seen from far behind \
      --source_words man walking with his dog down the sunny city street \
      --grad_tilt --tilt_scale 1.0 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_wide
    ;;
  generic)
    # naturally diverse single-prompt target (mixed genders/ages)
    eval python discrete_x/real_smc.py $COMMON_MODE \
      --target_prompts \"a photo of a person smiling at the camera\" \
      --source_words man sitting in the office \
      --grad_tilt --tilt_scale 1.0 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_generic
    ;;
  mixture)
    # true CDM: 50/50 mixture G, no single generating prompt exists
    eval python discrete_x/real_smc.py $COMMON_MODE \
      --target_prompts \"a photo of an old man smiling\" \"a photo of a young woman smiling\" \
      --source_words person smiling at the camera \
      --grad_tilt --tilt_scale 1.0 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_mixture
    ;;
  mixture_base)
    # mixture with the NON-DISTILLED sampler: does the optimizer now find a
    # straddling prompt instead of collapsing to one mode?
    eval python discrete_x/real_smc.py $COMMON_MODE \
      --target_prompts \"a photo of an old man smiling\" \"a photo of a young woman smiling\" \
      --source_words person smiling at the camera \
      --grad_tilt --tilt_scale 1.0 \
      --sampler base \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_mixture_base
    ;;
  multimodal_recover)
    # verifiable multimodal target: ground truth IS a disjunction prompt,
    # rendered bimodally by the base sampler; note prefix drops trailing 'a'
    eval python discrete_x/real_smc.py --loss image --prefix \"a photo of\" \
      --estimators mode --seeds 0 1 2 --n_particles 128 --T 16 \
      --beta 200 --beta_anneal --top_k 50 --n_cond 8 --n_target 64 \
      --remask_frac 0.75 \
      --target_words an old man or young woman smiling \
      --source_words one person sitting quietly inside the office \
      --grad_tilt --tilt_scale 1.0 \
      --sampler base \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_mmrecover
    ;;
  mmrecover_corr)
    # multimodal recovery + remasking corrector (exact recovery in text mode)
    eval python discrete_x/real_smc.py --loss image --prefix \"a photo of\" \
      --estimators mode --seeds 0 1 2 --n_particles 128 --T 16 \
      --beta 200 --beta_anneal --top_k 50 --n_cond 8 --n_target 64 \
      --remask_frac 0.75 \
      --target_words an old man or young woman smiling \
      --source_words one person sitting quietly inside the office \
      --grad_tilt --tilt_scale 1.0 \
      --sampler base \
      --remask_sigma 0.2 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_mmrecover_corr
    ;;
  mmrecover_noised)
    # recovery proper: start from the ground-truth sentence itself, 75%
    # re-masked (classic denoising setup), corrector on
    eval python discrete_x/real_smc.py --loss image --prefix \"a photo of\" \
      --estimators mode --seeds 0 1 2 --n_particles 128 --T 16 \
      --beta 200 --beta_anneal --top_k 50 --n_cond 8 --n_target 64 \
      --remask_frac 0.75 \
      --target_words an old man or young woman smiling \
      --source_words an old man or young woman smiling \
      --grad_tilt --tilt_scale 1.0 \
      --sampler base \
      --remask_sigma 0.2 \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_mmrecover_noised
    ;;
  mmrecover_noised_llada)
    # noised-truth recovery with the TRAINED masked-diffusion LM as prior
    eval python discrete_x/real_smc.py --loss image --prefix \"a photo of\" \
      --estimators mode --seeds 0 1 2 --n_particles 128 --T 16 \
      --beta 200 --beta_anneal --top_k 50 --n_cond 8 --n_target 64 \
      --remask_frac 0.75 \
      --target_words an old man or young woman smiling \
      --source_words an old man or young woman smiling \
      --grad_tilt --tilt_scale 1.0 \
      --sampler base \
      --remask_sigma 0.2 \
      --prior llada \
      --outdir output/discrete_smc_${SLURM_JOB_ID}_mmrecover_noised_llada
    ;;
  *)
    echo "unknown arm: $ARM"; exit 1
    ;;
esac

echo "Suite arm $ARM complete."
