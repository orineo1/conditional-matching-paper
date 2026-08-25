#!/bin/bash
# Round 5 -- per-arm zeta calibration (6 array tasks: {2D,5D,10D} x {trust,notrust}).
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-5 ../experiments/model-optimization/protocol/submit_calibrate.sh
#   python ../experiments/model-optimization/protocol/calibrate_zeta.py report   # -> zeta_star.json
#SBATCH --job-name=a4calib
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a4calib_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a4calib_%A_%a.err
set -euo pipefail
CONDA_ROOT=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
python ../experiments/model-optimization/protocol/calibrate_zeta.py run --index "${SLURM_ARRAY_TASK_ID}" --restarts "${RESTARTS:-40}"
