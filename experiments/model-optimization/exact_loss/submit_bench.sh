#!/bin/bash
#SBATCH --job-name=mmd_bench_cpu
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/mmd_bench_cpu_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/mmd_bench_cpu_%j.err
# Full CPU grid of exact_loss/bench_mmd.py (reference vs exact fast_mmd variants).
# Submit (from the cluster, login shell):  sbatch experiments/model-optimization/exact_loss/submit_bench.sh
# No pip installs (scribble_env rule).  Resumable: re-running reuses bench_raw/cpu_*.json.
set -euo pipefail
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
python ../experiments/model-optimization/exact_loss/bench_mmd.py --grid full --device cpu
echo "wrote experiments/model-optimization/exact_loss/bench_results.csv + bench_summary.md"
