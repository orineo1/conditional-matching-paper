#!/bin/bash
#SBATCH --job-name=pvd-arm
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_arm_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/pvd_arm_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Usage: sbatch submit_arm.sh ARM SCENARIO        ARM in {pgd, point, mmd}; SCENARIO in {A, B}
#        SMOKE=1 sbatch submit_arm.sh mmd B       (5-6 guided steps / N=8 / eval 64 / pgd 10 steps)
# Requires the scenario cache (submit_setup.sh) to exist. One arm per job. Appendix-E
# config below; mismatches vs the pipeline are documented in HYPOTHESIS.md (n_MC absent;
# ControlNet scale via the new --variation_cn_scale 0.5).
ARM=${1:?"ARM required: pgd | point | mmd"}
SCENARIO=${2:?"SCENARIO required: A | B"}
SEED=${SEED:-1}
# Optional overrides (defaults = Appendix E / the original registration):
#   ZETA      guidance strength           (default 4.0)
#   TRUST_TAU noise-level trust region    (default 0 = off)
#   GATE=1    cheap 20-guided-step gate at the same SDEdit strength (n_steps 40/start 20)
#   TAG       suffix for the output dir   (default: auto from ZETA/TRUST_TAU/GATE)
ZETA=${ZETA:-5.0}
TRUST_TAU=${TRUST_TAU:-0}
#   N_COND    override the arm's conditional samples per step (default: 1 point / 100 mmd)
#   BACKSEL   K>0: differentiate only K of them (loss VALUE stays full-batch) — used by the
#             zeta=0 drift control to afford a 32-sample readout at ~1-sample cost
N_COND=${N_COND:-}
BACKSEL=${BACKSEL:-0}

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
export WANDB_MODE=${WANDB_MODE:-offline}
REPO=/sci/labs/orzuk/shaulytolk/cdm-perf
EXP=$REPO/experiments/point_vs_dist_sd
CACHE=$EXP/cache/$SCENARIO
[ -f "$CACHE/targets_cache.npz" ] || { echo "cache $CACHE missing — run submit_setup.sh $SCENARIO first"; exit 1; }
cd "$REPO"

# ---- SHARED BACKBONE = Ori's working gender configuration (2026-09-13 re-spec) ----
# n_steps 250 / start_step 125 / num_variations 100 / base_zeta 5.0 / gs 0 / cn 0.5 /
# seed 1 / 10 targets per group, witness back-selection k=50 floor 0.0 temperature 0.3.
# Trust region OFF by default (Ori's config has none). The zeta12/tau0.2 work was a
# response to a configuration that lacked witness back-selection; keep it only via the
# explicit ZETA/TRUST_TAU overrides, as a labelled secondary arm.
N_STEPS=250;  START=125;  CN=0.5;  EVAL_N=2000;  EVAL_BS=8
PGD_STEPS=200; PGD_EPS_GRID="32 64 128"
if [ "${SMOKE:-0}" = "1" ]; then
  N_STEPS=10; START=5; EVAL_N=64; PGD_STEPS=10; PGD_EPS_GRID="32"
fi
if [ "${GATE:-0}" = "1" ]; then          # 20 guided steps, same SDEdit strength 0.5
  N_STEPS=40; START=20; EVAL_N=256
fi
N_VAR_MMD=100;  [ "${SMOKE:-0}" = "1" ] && N_VAR_MMD=8
# witness back-selection: on by default for the diffusion arms (Ori's config)
BACKSEL_K=${BACKSEL_K:-50};      [ "${SMOKE:-0}" = "1" ] && BACKSEL_K=4
BACKSEL_RULE=${BACKSEL_RULE:-witness}
W_FLOOR=${W_FLOOR:-0.0}
W_TEMP=${W_TEMP:-0.3}
# effective conditional-sample count for this arm (needed for the tag and the flags)
EFF_NCOND=${N_COND:-$N_VAR_MMD}
[ "$ARM" = "point" ] && [ -n "$N_COND" ] && EFF_NCOND=$N_COND
# back-selection is a no-op at n_cond = 1
# ...and it does not exist at all for pgd (no conditional batch to select from), so that
# arm must not carry a back-selection suffix (the A-scenario pgd dirs were mislabelled
# "_witness50" by an earlier version of this logic; they used no back-selection).
USE_WITNESS=0
[ "$ARM" != "pgd" ] && [ "$BACKSEL" = "0" ] && [ "$BACKSEL_K" != "0" ] \
  && [ "$EFF_NCOND" -gt 1 ] && USE_WITNESS=1
# output-dir suffix — ONE place. Built so a new-backbone run can never overwrite an
# earlier one: every run carries its back-selection descriptor, plus any override.
if [ -z "${TAG:-}" ]; then
  TAG=""
  [ "$ZETA" != "5.0" ] && TAG="${TAG}_zeta${ZETA}"
  [ "$TRUST_TAU" != "0" ] && TAG="${TAG}_tau${TRUST_TAU}"
  [ "${GATE:-0}" = "1" ] && TAG="${TAG}_gate"
  [ -n "$N_COND" ] && TAG="${TAG}_n${N_COND}"
  [ "$BACKSEL" != "0" ] && TAG="${TAG}_bs${BACKSEL}"
  if [ "$USE_WITNESS" = "1" ]; then
    TAG="${TAG}_${BACKSEL_RULE}${BACKSEL_K}"
    [ "$W_FLOOR" != "0.0" ] && TAG="${TAG}f${W_FLOOR}"
    [ "$W_TEMP" != "0.3" ] && TAG="${TAG}t${W_TEMP}"
  elif [ "$BACKSEL" = "0" ] && [ "$ARM" != "pgd" ]; then
    TAG="${TAG}_nobacksel"
  fi
fi

run_eval () {  # $1 = run dir
  python experiments/model-optimization/sd/eval_final.py --run_dir "$1" \
      --eval_n "$EVAL_N" --eval_batch_size "$EVAL_BS" --cn_scale "$CN" \
      --target_cache "$CACHE"
  echo "eval exit: $?"
}

case "$ARM" in
  pgd)
    for EPS in $PGD_EPS_GRID; do
      OUT=$EXP/runs/$SCENARIO/pgd_eps${EPS}${TAG}
      mkdir -p "$OUT"
      python "$EXP/run_pgd_arm.py" --cache "$CACHE" --output_dir "$OUT" \
          --eps_255 "$EPS" --steps "$PGD_STEPS" --seed "$SEED" --cn_scale "$CN"
      echo "pgd eps=$EPS exit: $?"
      run_eval "$OUT"
    done
    ;;
  point|mmd)
    OUT=$EXP/runs/$SCENARIO/${ARM}${TAG}
    mkdir -p "$OUT"
    if [ "$ARM" = "point" ]; then
      # Same backbone as arm 3: 100 conditional samples, witness k=50. The loss is the
      # point loss and the WITNESS is scored against y* (see generation.py). Use
      # N_COND=1 for the original delta_{y*} variant, where back-selection is a no-op.
      EXTRA=(--loss_fn point --point_target_pt "$CACHE/point_target.pt"
             --num_variations "${N_COND:-$N_VAR_MMD}")
    else
      EXTRA=(--loss_fn mmd --num_variations "${N_COND:-$N_VAR_MMD}")
    fi
    # back-selection: BACKSEL overrides (our rules); otherwise the shared witness config
    if [ "$BACKSEL" != "0" ]; then
      EXTRA+=(--backsel "$BACKSEL")
    elif [ "$USE_WITNESS" = "1" ]; then
      EXTRA+=(--backsel_k "$BACKSEL_K" --backsel_rule "$BACKSEL_RULE"
              --witness_floor "$W_FLOOR" --witness_temperature "$W_TEMP")
    fi
    python SD_cond_SD_controlnet/scripts/run_mlgd_f.py \
        --output_dir "$OUT" --wandb_project point_vs_dist_sd --mode gender \
        --target_cache "$CACHE" \
        --n_steps "$N_STEPS" --start_step "$START" \
        --base_zeta "$ZETA" --trust_noise "$TRUST_TAU" --guidance_scale 0.0 \
        --controlnet_scale "$CN" --variation_cn_scale "$CN" \
        --seed "$SEED" --seeded_rng --no_vis --profile \
        --eval_n 64 --eval_batch_size "$EVAL_BS" --eval_n_intermediate 8 \
        "${EXTRA[@]}"
    echo "run exit: $?"
    run_eval "$OUT"      # the authoritative fresh-sample eval + saved embeddings
    python "$EXP/gate_check.py" "$OUT"     # P-C1 trend criterion (HYPOTHESIS.md amendment)
    ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
echo "arm $ARM scenario $SCENARIO done."
