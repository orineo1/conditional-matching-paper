#!/bin/bash
# Agent B, B-R6b stage 2 -- end-to-end at n = 128, one array task per setting,
# all arms in ONE process (paired restarts 9000..9099).  CPU, glacier.
#
#   cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
#   sbatch --array=0-3 ../experiments/model-optimization/backsel/submit_stage2.sh
#   python ../experiments/model-optimization/backsel/stage2.py report     # -> stage2_tables.md / stage2_rows.csv
#
# Needs: the round-4 tree (tfg/backsel.py, backsel/fidelity.py, stage2.py), checkpoints
# uncond_*_dimy16.pt, cm_seed20240401_dx9dy1.pt, uncond_seed20240401_dx9.pt,
# results/tfg/exp5b_zeta_calibration_v2.json.  NO pip installs.
#SBATCH --job-name=aBstage2
#SBATCH --partition=glacier
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aBstage2_%A_%a.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/aBstage2_%A_%a.err

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

SETTINGS=(10D dimy16 nuis16 10D_z003)   # 10D_z003 = labelled zeta=0.03125 sensitivity group
S=${SETTINGS[$SLURM_ARRAY_TASK_ID]}
echo "task ${SLURM_ARRAY_TASK_ID} -> $S"
python ../experiments/model-optimization/backsel/stage2.py run --setting "$S" \
    --restarts "${RESTARTS:-100}" --offset "${OFFSET:-9000}" --dir stage2_runs
