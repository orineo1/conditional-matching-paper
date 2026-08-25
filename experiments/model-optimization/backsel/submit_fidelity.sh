#!/bin/bash
# Agent B, B-R6 stage 1 -- gradient fidelity at n = 128, one array task per setting
# (CPU, glacier).  Pattern copied from estimator/submit_screen.sh.
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-4 ../experiments/model-optimization/backsel/submit_fidelity.sh
#   python ../experiments/model-optimization/backsel/fidelity.py report     # -> fidelity_tables.md/.csv/.png
#
# Needs on the cluster: simulations/ (artifacts/checkpoints incl. uncond_*_dimy{8,16}.pt,
# cm_seed20240401_dx9dy1.pt, uncond_seed20240401_dx9.pt; results/tfg/exp5b_zeta_calibration_v2.json),
# experiments/model-optimization/ with the round-4 tree (tfg/backsel.py).  NO pip installs.
# ~10 trajectories x 11 probes x 20 draws per setting: a few minutes each on 2 threads.
#SBATCH --job-name=aBfidel
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aBfidel_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aBfidel_%A_%a.err

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

SETTINGS=(dimy8 dimy16 nuis8 nuis16 10D)
S=${SETTINGS[$SLURM_ARRAY_TASK_ID]}
echo "task ${SLURM_ARRAY_TASK_ID} -> $S"
python ../experiments/model-optimization/backsel/fidelity.py run --setting "$S" \
    --trajectories "${TRAJ:-10}" --draws "${DRAWS:-20}" --dir fidelity_runs
