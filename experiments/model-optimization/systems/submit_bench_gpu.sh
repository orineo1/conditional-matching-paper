#!/bin/bash
#SBATCH --job-name=cdm-sys-bench-gpu
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sys_bench_gpu_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sys_bench_gpu_%j.err
# Agent 5: CPU factors + batched restarts on CUDA (B=1,8,32,128), the only runner that can
# benefit from a GPU (batched rows amortise kernel launches). torch.compile skipped, no MPS.
#   ssh -p 2222 shaulytolk@localhost "bash -lc 'mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs && sbatch /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/model-optimization/systems/submit_bench_gpu.sh'"
set -euo pipefail
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
echo "=== $(hostname) $(date) ==="; python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
OUT=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sys_bench_gpu_rows_${SLURM_JOB_ID}.csv
for SETTING in 2D 10D; do
  python ../experiments/model-optimization/systems/bench.py --setting $SETTING --repeats 7 --skip-mps --skip-compile --cuda \
      --out ${OUT%.csv}_${SETTING}.csv 2>&1 | grep -v Warning
done
echo "=== done $(date) ==="
