#!/bin/bash
# Agent 6 -- ROUND-4 held-out: replay_fifo16/cohort16_trust vs trust_noise1 at equal fresh cost (offset 5000, 100 restarts),
# one GROUP (baseline + its candidates, same node/process) per array task (CPU, glacier).  Header/conda block copied from
# experiments/model-optimization/estimator/submit_screen.sh.
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-2%3 ../experiments/model-optimization/verification/submit_heldout_r4.sh    # 3 groups = 21 cells
#
# Results: experiments/model-optimization/verification/heldout_runs/<cell>_off5000.json
# Then locally: python ../experiments/model-optimization/verification/analyze_r4.py
#
# Needs on the cluster: simulations/ (incl. artifacts/checkpoints/*.pt and params/),
# experiments/model-optimization/ (estimator/engine_runner.py is imported). No pip installs.
#SBATCH --job-name=a6heldr4
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a6heldr4_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a6heldr4_%A_%a.err

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
OFFSET=${OFFSET:-5000}
HELD=../experiments/model-optimization/verification/heldout_r4_cells.py
echo "heldout-r4 task ${SLURM_ARRAY_TASK_ID} -> $(python $HELD list --offset $OFFSET 2>/dev/null | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")"
python $HELD run --index "${SLURM_ARRAY_TASK_ID}" --restarts "$RESTARTS" --offset "$OFFSET"
