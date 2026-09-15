#!/bin/bash
#SBATCH --job-name=pvd-setup
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_setup_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_setup_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Usage: sbatch submit_setup.sh SCENARIO       (SCENARIO in {A, B}; SMOKE=1 for the tiny build)
# Builds the shared cache all arms of that scenario read (byte-identical x0 / y* / G).
SCENARIO=${1:?"SCENARIO required: A or B"}
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

# ---- caches OFF the 5 GB /sci/home quota ------------------------------------
# A full $HOME is what silently killed job 46103120 (wandb could not open its log,
# the run died at step 98/125 and SLURM still reported COMPLETED 0:0).
LAB=/sci/labs/orzuk/shaulytolk
CDM=$LAB/cdm-perf
export HF_HOME=$LAB/hf_cache
export WANDB_DIR=$CDM/wandb
export WANDB_CACHE_DIR=$CDM/wandb
export WANDB_CONFIG_DIR=$CDM/wandb
export WANDB_ARTIFACT_DIR=$CDM/wandb
export MPLCONFIGDIR=$CDM/.cache
export XDG_CACHE_HOME=$CDM/.cache
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$CDM/logs"
# fail fast if the write target is nearly full — never lose a multi-hour run to ENOSPC
FREE_MB=$(df -Pm "$CDM" | awk 'NR==2{print $4}')
if [ "${FREE_MB:-0}" -lt 1024 ]; then
  echo "ABORT: only ${FREE_MB} MB free on $CDM (need >= 1024 MB). Free space before rerunning." >&2
  exit 1
fi
HOME_FREE_MB=$(df -Pm "$HOME" 2>/dev/null | awk 'NR==2{print $4}')
[ "${HOME_FREE_MB:-9999}" -lt 200 ] && echo "WARNING: \$HOME has only ${HOME_FREE_MB} MB free (caches are redirected, but watch it)." >&2
echo "disk: $CDM ${FREE_MB} MB free; \$HOME ${HOME_FREE_MB} MB free"
# -----------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd
# ORACLE_ONLY=1 recomputes ONLY cache/A/oracle.pt from the existing cache (safe to run
# after arms have already used that cache; nothing else is rewritten).
python setup_targets.py --scenario "$SCENARIO" ${SMOKE:+--smoke} ${ORACLE_ONLY:+--oracle_only}
echo "exit: $?"
