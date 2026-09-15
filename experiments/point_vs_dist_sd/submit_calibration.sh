#!/bin/bash
#SBATCH --job-name=pvd-calib
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_calib_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_calib_%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# The two fairness calibrations registered in HYPOTHESIS.md "AMENDMENT" (2026-09-10).
# Usage: sbatch submit_calibration.sh point_probe | pgd_lr
WHAT=${1:?"point_probe | pgd_lr"}
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
REPO=/sci/labs/orzuk/shaulytolk/cdm-perf
EXP=$REPO/experiments/point_vs_dist_sd
cd "$EXP"

case "$WHAT" in
  point_probe)
    # arm 2's own useful band: gradient with the arm's n_cond=1, loss readout averaged over 32
    python debug_gradient.py --cache cache/B --out debug/B_point --loss_fn point \
        --n_cond 1 --eval_n_cond 32 --repeats 4 --base_zeta 4.0
    echo "point probe exit: $?"
    ;;
  pgd_lr)
    # eps fixed at the best of the registered sweep (32/255); lr in {eps/100, eps/10, eps/3}
    EPS=32
    for DIV in 100 10 3; do
      LR=$(python -c "print(($EPS/255.0)/$DIV)")
      OUT=$EXP/runs/B/pgd_eps${EPS}_lrdiv${DIV}
      mkdir -p "$OUT"
      python run_pgd_arm.py --cache cache/B --output_dir "$OUT" \
          --eps_255 $EPS --steps 200 --lr "$LR" --seed 42 --cn_scale 0.5
      echo "pgd lr=eps/$DIV ($LR) exit: $?"
    done
    python - <<'PYEOF'
import glob, json, os
# Registered two-stage criterion. Step 0 is the UN-optimised x0 under 1-sample noise, so it
# is excluded from "min"; and 1-sample training values are NOT comparable to the 2000-sample
# fresh L(x0)=0.4218 (that comparison scored a non-descending run as a success once).
print("\n=== PGD lr sweep — stage (i): descent on its OWN objective ===")
for d in sorted(glob.glob("runs/B/pgd_eps32_lrdiv*")):
    m = json.load(open(os.path.join(d, "metrics_partial.json")))
    L = [s["loss"] for s in m["steps"]]
    f10, l10 = sum(L[:10])/10, sum(L[-10:])/10
    print(f"{os.path.basename(d):24s} lr {m['args']['lr']:.5f}  step0 {L[0]:.4f}  "
          f"first10 {f10:.4f}  last10 {l10:.4f}  min(steps>=1) {min(L[1:]):.4f}  "
          f"descended: {'YES' if l10 < f10 else 'NO'}  (ever below step0: {min(L[1:]) < L[0]})")
print("stage (ii): fresh 2000-sample eval, ONLY for an lr passing stage (i).")
PYEOF
    ;;
  *) echo "unknown: $WHAT"; exit 1 ;;
esac
