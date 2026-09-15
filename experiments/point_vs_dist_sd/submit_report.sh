#!/bin/bash
#SBATCH --job-name=pvd-report
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_report_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_report_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=glacier

# Builds RESULTS.md + figures for the point-vs-dist experiment. Analysis only, no GPU:
# the login node kills it (exit 137) because the 2000x2000 MMD kernels exceed its limits.
P=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env/bin/python

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
cd /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd
$P build_report.py --runs runs --cache cache
echo "exit: $?"
