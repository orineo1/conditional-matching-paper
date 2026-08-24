#!/bin/bash
# Agent 4 -- Stage-1 screening on the cluster, one cell per array task (CPU, glacier).
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   N=$(python ../experiments/model-optimization/estimator/screen.py list 2>/dev/null | wc -l)   # 258 at the time of writing
#   sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/estimator/submit_screen.sh
# Round 2 (scale-free clipping, combinations, Pareto; cells already in runs/ are skipped):
#   N=$(python ../experiments/model-optimization/estimator/screen.py list --round 2 2>/dev/null | wc -l)   # 336
#   ROUND=2 sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/estimator/submit_screen.sh
#
# Then, after all tasks finish (locally or on the cluster):
#   python ../experiments/model-optimization/estimator/screen.py report
#
# Needs on the cluster: simulations/ (incl. artifacts/checkpoints/*.pt and params/),
# experiments/model-optimization/. No pip installs. Restarts 0..39 (offset 0);
# offsets >= 1000 are reserved for the verifier.
#SBATCH --job-name=a4screen
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a4screen_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/a4screen_%A_%a.err

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
ROUND=${ROUND:-1}
SCREEN=../experiments/model-optimization/estimator/screen.py
echo "round $ROUND task ${SLURM_ARRAY_TASK_ID} -> $(python $SCREEN list --round $ROUND 2>/dev/null | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")"
python $SCREEN run --round "$ROUND" --index "${SLURM_ARRAY_TASK_ID}" --restarts "$RESTARTS" --offset "$OFFSET"
