#!/bin/bash
#SBATCH --job-name=cdm-sys-bench
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sys_bench_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sys_bench_%j.err
# Agent 5 systems benchmark (CPU, glacier). Full factor scan incl. torch.compile, no MPS.
# Submit from the login node with a login shell:
#   ssh -p 2222 shaulytolk@localhost "bash -lc 'mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs && sbatch /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/model-optimization/systems/submit_bench.sh'"
set -euo pipefail
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
echo "=== $(hostname) $(date) threads=$OMP_NUM_THREADS ==="; python -c "import torch; print(torch.__version__, torch.get_num_threads())"
OUT=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sys_bench_rows_${SLURM_JOB_ID}.csv
for SETTING in 2D 5D 10D; do
  python ../experiments/model-optimization/systems/bench.py --setting $SETTING --repeats 7 --skip-mps \
      --out ${OUT%.csv}_${SETTING}.csv 2>&1 | grep -v Warning
done
python ../experiments/model-optimization/systems/microbatch_mmd.py 2>&1 | grep -v Warning
echo "=== done $(date) ==="
