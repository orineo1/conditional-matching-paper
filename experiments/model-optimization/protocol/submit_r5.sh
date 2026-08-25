#!/bin/bash
# Round 5 -- round-5 fair trust-region cells (needs protocol/zeta_star.json from the calibration) (36 array tasks: {2D,5D,10D} x n{4,8,16,32} x {A,B,C}, R=100 at offset 6000).
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-35%36 ../experiments/model-optimization/protocol/submit_r5.sh
#   python ../experiments/model-optimization/protocol/cells_r5.py report   # -> r5_tables.md
#SBATCH --job-name=a4r5
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a4r5_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a4r5_%A_%a.err
set -euo pipefail
CONDA_ROOT=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
python ../experiments/model-optimization/protocol/cells_r5.py run --index "${SLURM_ARRAY_TASK_ID}" --restarts "${RESTARTS:-100}" --offset "${OFFSET:-6000}"
