#!/bin/bash
# LM-on-LM survey distribution matching (discrete_x/lm_survey.py).
# Usage: sbatch scripts/submit_lm_survey.sh <arm>
#   probe   - reachability check: probe framings + brute force over the whole
#             controlled-vocabulary space, all synthetic questions
#   compare - method comparison (unguided / SMC / CTMC) vs the brute optimum
# Follows the submit_saliency.sh pattern: no pip installs, scribble_env as-is.
#SBATCH --job-name=lm-survey
#SBATCH --output=lm_survey_%j.log
#SBATCH --error=lm_survey_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon
#SBATCH --exclude=salmon-07

ARM="${1:-probe}"
RESPONDER="${RESPONDER:-Qwen/Qwen2.5-7B-Instruct}"
ROOT=/sci/labs/orzuk/shaulytolk/conditional-matching-paper

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate "$ROOT/scribble_env"
export HF_HOME=/sci/labs/orzuk/shaulytolk/.cache/huggingface
cd "$ROOT"

echo "arm=$ARM responder=$RESPONDER"

TAG="$(echo "$RESPONDER" | tr '/' '_')"

if [ "$ARM" = "real" ]; then
  # Real Pew targets: reachability + brute-force optimum per question.
  # Run once with an instruct responder and once with its base model to test
  # whether instruction tuning collapses the answer distribution the way
  # distillation collapsed image diversity.
  for Q in pew_conception pew_govsize pew_crim_sent pew_just_error \
           pew_abortion pew_polfut; do
    python discrete_x/lm_survey.py \
      --responder "$RESPONDER" \
      --question "$Q" \
      --methods probe,brute \
      --vocab small --canvas_len 3 \
      --outdir "output/lm_survey_real_${SLURM_JOB_ID}/${TAG}/$Q"
  done
elif [ "$ARM" = "probe" ]; then
  # Is the task well posed? Reachability of each target over the full
  # 8^3 controlled-vocabulary space, plus hand-written framings for contrast.
  for Q in synthetic_split synthetic_polarized synthetic_skewed; do
    python discrete_x/lm_survey.py \
      --responder "$RESPONDER" \
      --question "$Q" \
      --methods probe,brute \
      --vocab small --canvas_len 3 \
      --outdir "output/lm_survey_probe_${SLURM_JOB_ID}/$Q"
  done
elif [ "$ARM" = "ladder" ]; then
  # THE V-LADDER: nested vocabularies V = 4,8,16,32 at canvas_len 3, so the
  # whole space stays brute-forceable (max 32^3 = 32,768) and every method is
  # scored against the EXACT optimum as the alphabet grows. This is the
  # crossover curve: single-chain rate guidance (CTMC) vs population
  # selection (SMC) vs unguided restarts, as a function of |V| alone.
  for Q in pew_conception pew_crim_sent; do
    for V in v4 v8 v16 v32; do
      python discrete_x/lm_survey.py \
        --responder "$RESPONDER" \
        --question "$Q" \
        --methods brute,unguided,smc,ctmc \
        --vocab "$V" --canvas_len 3 \
        --seeds 0 1 2 \
        --B 256 --N 128 --T 16 \
        --gamma 10 --K 16 --restarts 4 --n_steps 16 \
        --outdir "output/lm_survey_ladder_${SLURM_JOB_ID}/${TAG}/${Q}/${V}"
    done
  done
elif [ "$ARM" = "depth" ]; then
  # THE D-LADDER: vocabulary fixed at V=8, canvas length swept. The V-ladder
  # showed no separation between guided and unguided search at 3 slots, where
  # a chain makes only 3 commitments and partial states carry almost no
  # information. This tests the other axis -- commitment depth. Brute force
  # stays exact through D=5 (8^5 = 32,768); D=8 and D=12 compare methods to
  # each other only. Unguided is now budget-matched to the guided arms.
  for Q in pew_conception pew_crim_sent; do
    for D in 3 4 5; do
      python discrete_x/lm_survey.py \
        --responder "$RESPONDER" --question "$Q" \
        --methods brute,smc,ctmc,unguided \
        --vocab v8 --canvas_len "$D" --seeds 0 1 2 \
        --N 128 --T 16 --gamma 10 --K 16 --restarts 4 --n_steps 16 \
        --outdir "output/lm_survey_depth_${SLURM_JOB_ID}/${TAG}/${Q}/d${D}"
    done
    for D in 8 12; do
      python discrete_x/lm_survey.py \
        --responder "$RESPONDER" --question "$Q" \
        --methods smc,ctmc,unguided \
        --vocab v8 --canvas_len "$D" --seeds 0 1 2 \
        --N 128 --T 16 --gamma 10 --K 16 --restarts 4 --n_steps 16 \
        --outdir "output/lm_survey_depth_${SLURM_JOB_ID}/${TAG}/${Q}/d${D}"
    done
  done
elif [ "$ARM" = "compare" ]; then
  # Method comparison against the exact optimum, 3 seeds.
  for Q in synthetic_polarized synthetic_split; do
    python discrete_x/lm_survey.py \
      --responder "$RESPONDER" \
      --question "$Q" \
      --methods brute,unguided,smc,ctmc \
      --vocab small --canvas_len 3 \
      --seeds 0 1 2 \
      --B 256 --N 128 --T 16 \
      --gamma 10 --K 16 --restarts 4 --n_steps 16 \
      --outdir "output/lm_survey_compare_${SLURM_JOB_ID}/$Q"
  done
else
  echo "unknown arm: $ARM"; exit 1
fi
