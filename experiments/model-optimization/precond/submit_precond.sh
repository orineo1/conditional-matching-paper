#!/bin/bash
# Agent P -- round-3 preconditioning screening on the cluster, one cell per
# array task (CPU, glacier).
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   N=$(python ../experiments/model-optimization/precond/cells.py list 2>/dev/null | wc -l)   # 90
#   sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/precond/submit_precond.sh
#
# Then, after all tasks finish (locally or on the cluster):
#   python ../experiments/model-optimization/precond/cells.py report
#
# Needs on the cluster: simulations/ (incl. artifacts/checkpoints/*.pt and
# params/) and experiments/model-optimization/ at the same commit as local
# (tfg/precond.py + config/engine/engine_runner hooks). No pip installs.
# Restarts 0..39 (offset 0); offsets >= 1000 are reserved for the verifier.
# Cells already present in precond/runs/ are skipped, so re-submission is safe.
#SBATCH --job-name=aPprecond
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aPprecond_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aPprecond_%A_%a.err

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

RESTARTS=${RESTARTS:-40}
OFFSET=${OFFSET:-0}
CELLS=../experiments/model-optimization/precond/cells.py
echo "task ${SLURM_ARRAY_TASK_ID} -> $(python $CELLS list 2>/dev/null | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")"
python $CELLS run --index "${SLURM_ARRAY_TASK_ID}" --restarts "$RESTARTS" --offset "$OFFSET"
