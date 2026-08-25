#!/bin/bash
# Agent 6 -- ROUND-5 CONFIRMATORY re-run, corrected protocol: trust@zeta*_t vs notrust@zeta*_n (+2D zeta=8 sensitivity) (offset 7000, 100 restarts),
# one GROUP (baseline + its candidates, same node/process) per array task (CPU, glacier).  Header/conda block copied from
# experiments/model-optimization/estimator/submit_screen.sh.
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-2%3 ../experiments/model-optimization/verification/submit_heldout_r5.sh    # 3 groups = 28 cells
#
# Results: experiments/model-optimization/verification/heldout_runs_r5/<cell>_off7000.json
# Then locally: python ../experiments/model-optimization/verification/analyze_r5.py
#
# Needs on the cluster: simulations/ (incl. artifacts/checkpoints/*.pt and params/),
# experiments/model-optimization/ (estimator/engine_runner.py is imported). No pip installs.
#SBATCH --job-name=a6heldr5
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a6heldr5_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a6heldr5_%A_%a.err

set -euo pipefail
CONDA_ROOT=/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations

RESTARTS=${RESTARTS:-100}
OFFSET=${OFFSET:-7000}
HELD=../experiments/model-optimization/verification/heldout_r5_cells.py
echo "heldout-r5 task ${SLURM_ARRAY_TASK_ID} -> $(python $HELD list --offset $OFFSET 2>/dev/null | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")"
python $HELD run --index "${SLURM_ARRAY_TASK_ID}" --restarts "$RESTARTS" --offset "$OFFSET"
