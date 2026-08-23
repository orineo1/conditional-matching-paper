#!/bin/bash
#SBATCH --job-name=cdm-ctmc
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/ctmc_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/ctmc_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Rate-ratio (TFG-Flow / Nisonoff-style) guidance at image scale — the
# CTMC alternative to twisted SMC, same tasks and budgets. Toy validation:
# mc gamma=10 recovers 5/5 (parity with SMC); additive is the
# assumption-check arm (expected weak on a coupling MMD objective).

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

ARM="${1:?usage: sbatch submit_ctmc.sh <recover_g10|recover_g3|edit_g10|recover_additive>}"
echo "CTMC arm=$ARM on $(hostname) GPU: $CUDA_VISIBLE_DEVICES (job $SLURM_JOB_ID)"

case "$ARM" in
  recover_g10)
    python discrete_x/real_ctmc.py --task recover --estimator mc \
      --gamma 10 --K 16 --restarts 4 --seeds 0 1 2 --sampler base \
      --outdir output/ctmc_${SLURM_JOB_ID}_recover_g10
    ;;
  recover_g3)
    python discrete_x/real_ctmc.py --task recover --estimator mc \
      --gamma 3 --K 16 --restarts 4 --seeds 0 1 2 --sampler base \
      --outdir output/ctmc_${SLURM_JOB_ID}_recover_g3
    ;;
  edit_g10)
    python discrete_x/real_ctmc.py --task edit --estimator mc \
      --gamma 10 --K 16 --restarts 4 --seeds 0 1 2 --sampler base \
      --outdir output/ctmc_${SLURM_JOB_ID}_edit_g10
    ;;
  recover_additive)
    python discrete_x/real_ctmc.py --task recover --estimator additive \
      --gamma 3 --M_design 256 --restarts 4 --seeds 0 1 2 --sampler base \
      --outdir output/ctmc_${SLURM_JOB_ID}_recover_additive
    ;;
  *)
    echo "unknown arm: $ARM"; exit 1
    ;;
esac

echo "CTMC arm $ARM complete."
