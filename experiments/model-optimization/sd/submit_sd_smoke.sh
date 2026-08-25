#!/bin/bash
#SBATCH --job-name=sd-perf-smoke
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sd_perf_smoke_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/sd_perf_smoke_%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

# Smoke test of the opt-in perf flags: 5 guided steps, N=8, eval 64, catfish (L4 22GB).
# Runs two configs back to back: (a) every flag on (backsel kcenter k=2, trust, no_vis,
# single-batch architect, profile); (b) backsel uniform with visualisation ON and the
# CFG double batch (exercises the untouched paths + backsel together).
# NO pip installs (scribble_env is fragile — see memory).

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "sd-perf smoke on $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=sd_perf

export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /sci/labs/orzuk/shaulytolk/cdm-perf
SEED=${SEED:-1}
ROOT=output/sd_perf/smoke_${SLURM_JOB_ID}
mkdir -p "$ROOT"

COMMON=(
  --mode gender
  --n_steps 10 --start_step 5
  --num_variations 8
  --base_zeta 5.0 --guidance_scale 0.0 --controlnet_scale 0.5
  --loss_fn mmd
  --target_prompts
    "Man:a superrealistic portrait photograph of a man, studio lighting:10"
    "Woman:a superrealistic portrait photograph of a woman, studio lighting:10"
  --seed "$SEED"
  --seeded_rng --profile
  --eval_n 64 --eval_batch_size 8 --eval_n_intermediate 4
  --target_cache "$ROOT/targets"
  --wandb_project sd_perf
)

echo "=== (a) all flags: no_vis single_batch trust=1.0 backsel kcenter k=2 vbs=2 ==="
python SD_cond_SD_controlnet/scripts/run_mlgd_f.py "${COMMON[@]}" \
    --output_dir "$ROOT/all_flags" \
    --no_vis --arch_single_batch --trust_noise 1.0 \
    --backsel 2 --backsel_rule kcenter --variation_batch_size 2
echo "exit (a): $?"

echo "=== (b) vis on, CFG batch, backsel uniform k=3, vbs=1 ==="
python SD_cond_SD_controlnet/scripts/run_mlgd_f.py "${COMMON[@]}" \
    --output_dir "$ROOT/vis_backsel_uniform" \
    --backsel 3 --backsel_rule uniform
echo "exit (b): $?"

echo "=== summary ==="
for d in all_flags vis_backsel_uniform; do
  python - "$ROOT/$d" <<'EOF'
import json, sys, os
d = sys.argv[1]
m = json.load(open(os.path.join(d, "metrics.json")))
print(d, "final mlgd_f", m["final_mlgd_f_mmd"], "regular", m["final_regular_mmd"],
      "delta", m["mmd_delta"], "time_s", round(m["optimization_time_sec"], 1))
p = json.load(open(os.path.join(d, "profile.json")))
print("  profile summary:", json.dumps(p["summary"], indent=1))
for s in m["steps"]:
    print("  step", s["step"], "mmd", round(s["mmd_loss"], 5), "corr_raw", round(s["correction_norm_raw"], 3),
          "cap_tau1", round(s["trust_cap_tau1"], 3), "scale", s["trust_scale"],
          "backsel", (s["backsel"] or {}).get("selected"), "regen_err", (s["backsel"] or {}).get("regen_max_abs_err"))
EOF
done
echo "smoke done."
