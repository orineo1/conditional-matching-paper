#!/bin/bash
#SBATCH --job-name=sd-perf-dev
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sd_perf_dev_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sd_perf_dev_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Usage:  sbatch submit_sd_dev.sh ARM        (ARM in: baseline novis trust
#           backsel_k<K>_{kcenter,uniform,strat} trust_backsel[_k<K>] trust_backsel_{uniform,strat}[_k<K>])
#   K is parsed from the arm name (default 8) so output dirs never collide; BACKSEL_K env is a fallback.
# Env overrides: SEED (default 1), TRUST_TAU (default 1.0), BACKSEL_K (default 8),
#                N_VAR (default 32), EVAL_N (default 2000).
# Reduced-but-faithful 50/50 gender task (see sd/PIPELINE.md sec 5): SDXL-base
# architect, prompt "", DDIM 100 steps from 50, base_zeta 5, 50+50 targets, N=32.
# All arms: --seeded_rng --profile, same seed, shared target cache
# (run `baseline` FIRST so the cache exists before the other arms start).

ARM=${1:?"ARM required: baseline novis trust backsel_k<K>_kcenter backsel_k<K>_uniform trust_backsel[_k<K>]"}
SEED=${SEED:-1}
TRUST_TAU=${TRUST_TAU:-0.25}
BACKSEL_K=${BACKSEL_K:-8}
if [[ "$ARM" =~ _k([0-9]+) ]]; then BACKSEL_K=${BASH_REMATCH[1]}; fi
N_VAR=${N_VAR:-32}
EVAL_N=${EVAL_N:-2000}

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "sd-perf dev arm=$ARM seed=$SEED on $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-offline}

cd /sci/labs/orzuk/shaulytolk/cdm-perf
OUT=output/sd_perf/${ARM}_seed${SEED}
CACHE=output/sd_perf/targets_seed${SEED}
mkdir -p "$OUT"
if [ "$ARM" != "baseline" ] && [ ! -f "$CACHE/targets_cache.npz" ]; then
  echo "WARNING: target cache $CACHE missing — run the baseline arm first for byte-identical targets."
fi

case "$ARM" in
  baseline)            EXTRA=() ;;
  novis)               EXTRA=(--no_vis --arch_single_batch) ;;
  trust)               EXTRA=(--no_vis --arch_single_batch --trust_noise "$TRUST_TAU") ;;
  backsel_k*_kcenter)  EXTRA=(--no_vis --arch_single_batch --backsel "$BACKSEL_K" --backsel_rule kcenter) ;;
  backsel_k*_uniform)  EXTRA=(--no_vis --arch_single_batch --backsel "$BACKSEL_K" --backsel_rule uniform) ;;
  backsel_k*_soft)     EXTRA=(--no_vis --arch_single_batch --backsel "$BACKSEL_K" --backsel_rule uniform --backsel_weighting soft --backsel_soft_tau_scale "${SOFT_TAU_SCALE:-1.0}") ;;
  backsel_k*_strat_soft) EXTRA=(--no_vis --arch_single_batch --backsel "$BACKSEL_K" --backsel_rule strat --backsel_weighting soft --backsel_soft_tau_scale "${SOFT_TAU_SCALE:-1.0}") ;;
  trust_backsel_soft*) EXTRA=(--no_vis --arch_single_batch --trust_noise "$TRUST_TAU" --backsel "$BACKSEL_K" --backsel_rule uniform --backsel_weighting soft --backsel_soft_tau_scale "${SOFT_TAU_SCALE:-1.0}") ;;
  trust_backsel|trust_backsel_k*) EXTRA=(--no_vis --arch_single_batch --trust_noise "$TRUST_TAU" --backsel "$BACKSEL_K" --backsel_rule kcenter) ;;
  backsel_k*_strat)    EXTRA=(--no_vis --arch_single_batch --backsel "$BACKSEL_K" --backsel_rule strat) ;;
  trust_backsel_uniform*) EXTRA=(--no_vis --arch_single_batch --trust_noise "$TRUST_TAU" --backsel "$BACKSEL_K" --backsel_rule uniform) ;;
  trust_backsel_strat*)   EXTRA=(--no_vis --arch_single_batch --trust_noise "$TRUST_TAU" --backsel "$BACKSEL_K" --backsel_rule strat) ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac

python SD_cond_SD_controlnet/scripts/run_mlgd_f.py \
    --output_dir "$OUT" \
    --wandb_project sd_perf \
    --mode gender \
    --n_steps 100 --start_step 50 \
    --num_variations "$N_VAR" \
    --base_zeta 5.0 --guidance_scale 0.0 --controlnet_scale 0.5 \
    --loss_fn mmd --kernel_alpha 1.0 --bandwidth_scale 1.0 --loss_scale 1.0 \
    --target_prompts \
        "Man:a superrealistic portrait photograph of a man, studio lighting:50" \
        "Woman:a superrealistic portrait photograph of a woman, studio lighting:50" \
    --seed "$SEED" \
    --seeded_rng --profile \
    --eval_n "$EVAL_N" --eval_batch_size 8 --eval_n_intermediate 10 \
    --target_cache "$CACHE" \
    "${EXTRA[@]}"
echo "exit: $?"

python - "$OUT" <<'EOF'
import json, sys, os
d = sys.argv[1]
m = json.load(open(os.path.join(d, "metrics.json")))
print("ARM", d, "final mlgd_f", m["final_mlgd_f_mmd"], "regular", m["final_regular_mmd"],
      "delta", m["mmd_delta"], "opt_time_s", round(m["optimization_time_sec"], 1),
      "eval_n", m["eval_n_final"])
print("profile_summary", json.dumps(m["profile_summary"], indent=1))
caps = [s["correction_norm_raw"] / s["trust_cap_tau1"] for s in m["steps"] if s["trust_cap_tau1"] > 0]
print("corr_norm_raw / cap_tau1  min/median/max:", min(caps), sorted(caps)[len(caps)//2], max(caps))
EOF
echo "dev arm $ARM done."
