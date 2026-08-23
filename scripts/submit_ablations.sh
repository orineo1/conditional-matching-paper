#!/bin/bash
#SBATCH --job-name=cdm-ablate
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/ablate_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/ablate_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Ablation grid on the headline noised-truth recovery task:
#   target = source = "an old man or young woman smiling", 75% re-masked,
#   base sampler, mode estimator, N=128 T=16 beta=200, seeds 0-2.
# Full stack (mmrecover_noised: 0.083/0.093/0.087) and no-guidance
# baseline (sdedit_best: 0.091-0.098) are already run; each arm here
# removes exactly ONE component from the full stack.

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

ARM="${1:?usage: sbatch submit_ablations.sh <no_tilt|no_corr|no_anneal|turbo>}"
echo "Ablation arm=$ARM on $(hostname) GPU: $CUDA_VISIBLE_DEVICES (job $SLURM_JOB_ID)"

BASE="--loss image --prefix \"a photo of\" --estimators mode \
  --seeds 0 1 2 --n_particles 128 --T 16 --beta 200 --top_k 50 \
  --n_cond 8 --n_target 64 --remask_frac 0.75 \
  --target_words an old man or young woman smiling \
  --source_words an old man or young woman smiling"

BASE_EDIT="--loss image --prefix \"a photo of\" --estimators mode \
  --seeds 0 1 2 --n_particles 128 --T 16 --beta 200 --top_k 50 \
  --n_cond 8 --n_target 64 --remask_frac 0.75 \
  --target_words an old man or young woman smiling \
  --source_words one person sitting quietly inside the office"

case "$ARM" in
  no_tilt)
    # full stack minus the cross-modal tilted proposals
    eval python discrete_x/real_smc.py $BASE \
      --beta_anneal --sampler base --remask_sigma 0.2 \
      --outdir output/ablate_${SLURM_JOB_ID}_no_tilt
    ;;
  no_corr)
    # full stack minus the remasking corrector
    eval python discrete_x/real_smc.py $BASE \
      --beta_anneal --sampler base --grad_tilt --tilt_scale 1.0 \
      --outdir output/ablate_${SLURM_JOB_ID}_no_corr
    ;;
  no_anneal)
    # full stack minus beta-annealing (constant beta=200)
    eval python discrete_x/real_smc.py $BASE \
      --sampler base --grad_tilt --tilt_scale 1.0 --remask_sigma 0.2 \
      --outdir output/ablate_${SLURM_JOB_ID}_no_anneal
    ;;
  turbo)
    # full stack but distilled 1-step sampler (distillation-collapse check
    # on the recovery task)
    eval python discrete_x/real_smc.py $BASE \
      --beta_anneal --sampler turbo --grad_tilt --tilt_scale 1.0 \
      --remask_sigma 0.2 \
      --outdir output/ablate_${SLURM_JOB_ID}_turbo
    ;;
  edit_no_tilt)
    # EDITING task (far source) minus tilted proposals; A/B vs
    # mmrecover_corr 45397484 (full stack, 0.106) and multimodal_recover
    # 45396857 (no corrector, 0.115)
    eval python discrete_x/real_smc.py $BASE_EDIT \
      --beta_anneal --sampler base --remask_sigma 0.2 \
      --outdir output/ablate_${SLURM_JOB_ID}_edit_no_tilt
    ;;
  edit_no_anneal)
    # EDITING task minus beta-annealing
    eval python discrete_x/real_smc.py $BASE_EDIT \
      --sampler base --grad_tilt --tilt_scale 1.0 --remask_sigma 0.2 \
      --outdir output/ablate_${SLURM_JOB_ID}_edit_no_anneal
    ;;
  *)
    echo "unknown arm: $ARM"; exit 1
    ;;
esac

echo "Ablation arm $ARM complete."
