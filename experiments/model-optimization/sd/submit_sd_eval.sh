#!/bin/bash
#SBATCH --job-name=sd-perf-eval
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sd_perf_eval_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sd_perf_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Usage: sbatch submit_sd_eval.sh output/sd_perf/<arm>_seed<seed> [more run dirs...]
# Env: EVAL_N (2000), EVAL_BS (8), CLIP_BS (32). Seeds are taken from the run (identical across arms).
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/cdm-perf
[ $# -ge 1 ] || { echo "run dir(s) required"; exit 1; }
for RUN in "$@"; do
  echo "=== eval $RUN on $(hostname) job $SLURM_JOB_ID ==="
  python experiments/model-optimization/sd/eval_final.py --run_dir "$RUN" \
      --eval_n "${EVAL_N:-2000}" --eval_batch_size "${EVAL_BS:-8}" --clip_batch_size "${CLIP_BS:-32}"
  echo "exit: $?"
done
