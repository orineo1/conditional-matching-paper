#!/bin/bash
# Agent M -- experiment M-9 (replay at tiny fresh budgets, all arms + trust),
# pre-registered in hypotheses/agentM.yaml.  One array task per
# (setting x fresh budget) group; the four arms of a group run in ONE process
# (same node, same restart seeds -> clean pairing).  9 groups.
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-8 ../experiments/model-optimization/replay/submit_m9.sh
# after completion:
#   python ../experiments/model-optimization/replay/cells_m9.py report
#
# Needs: simulations/ (artifacts/checkpoints, params) + experiments/model-optimization/
# incl. the updated tfg/{config,replay}.py and estimator/engine_runner.py.
# NO pip installs.  Restarts 0..99 at OFFSET 4000 (fresh, reserved for M-9).
# Existing runs_m9/ arm files are skipped, so re-submitting is safe.
#SBATCH --job-name=aM9replay
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aM9_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aM9_%A_%a.err

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
OFFSET=${OFFSET:-4000}
CELLS=../experiments/model-optimization/replay/cells_m9.py
echo "task ${SLURM_ARRAY_TASK_ID} -> $(python $CELLS list 2>/dev/null | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")"
python $CELLS run --index "${SLURM_ARRAY_TASK_ID}" --restarts "$RESTARTS" --offset "$OFFSET"
