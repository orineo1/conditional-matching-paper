#!/bin/bash
#SBATCH --job-name=dl-hf-checkpoints
#SBATCH --output=dl_hf_checkpoints_%j.log   # written to wherever you run `sbatch` from
#SBATCH --error=dl_hf_checkpoints_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=YOUR_PARTITION   # <-- change to your cluster partition, or override at
                                      #     submit time: sbatch --partition=<name> sbatch_download_hf_checkpoints.sh

# ══════════════════════════════════════════════════════════════════════════════
# Pre-fetches CM/Diffusion_cond/Diffusion_uncond checkpoints from the
# HuggingFace fallback repo (see experiment_utils.HF_REPO_ID /
# load_checkpoint_with_hf_fallback) straight into simulations/checkpoints/<exp>,
# so later GPU jobs (run_exp_cond_1d.sh, run_backsel_witness_sweep.sh, ...) find
# them locally on the FIRST check and never need network access themselves.
#
# No GPU requested on purpose: this only needs CPU + internet egress. If this
# job's log shows a HuggingFace connection error, that's confirmation the
# PARTITION you submitted to has no internet access -- try a different
# partition (e.g. a CPU/login-adjacent one) if your cluster has one.
#
# Before submitting: export REPO_ROOT=/path/to/conditional-matching-paper
#                     export ENV_PATH=/path/to/your/venv
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS="${EXPERIMENTS:-5D_cond_1D 10D_cond_1D}"
MODELS="CM Diffusion_cond Diffusion_uncond"
SEED="${SEED:-42}"
HF_REPO_ID="anon-submission-cdm/cdm-inverse-design"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════
source "${ENV_PATH}/bin/activate"   # set ENV_PATH before submitting
pip install -q huggingface_hub

# ══════════════════════════════════════════════════════════════════════════════
# 2. Repo root & directories
# ══════════════════════════════════════════════════════════════════════════════
export REPO_ROOT="${REPO_ROOT:?REPO_ROOT is not set. Export it before submitting.}"

echo "=== JOB ${SLURM_JOB_ID} ON $(hostname) ==="
echo "    REPO_ROOT   : $REPO_ROOT"
echo "    experiments : $EXPERIMENTS"
echo "    seed        : $SEED"
echo "    repo_id     : $HF_REPO_ID"
echo "    network check:"
curl -sI --max-time 10 https://huggingface.co | head -1 || echo "    (curl to huggingface.co failed/timed out)"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Download every (experiment, model) checkpoint straight into its local path
# ══════════════════════════════════════════════════════════════════════════════
python3 - "$REPO_ROOT" "$HF_REPO_ID" "$SEED" $EXPERIMENTS -- $MODELS <<'PYEOF'
import os, shutil, sys
from huggingface_hub import hf_hub_download

args = sys.argv[1:]
repo_root, hf_repo_id, seed = args[0], args[1], args[2]
rest = args[3:]
sep = rest.index("--")
experiments, models = rest[:sep], rest[sep + 1:]

ok, failed = [], []
for experiment in experiments:
    checkpoint_dir = os.path.join(repo_root, "simulations", "checkpoints", experiment)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for model in models:
        fname = f"{experiment}_{model}_seed{seed}.pt"
        local_path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(local_path):
            print(f"[skip] already present: {local_path}")
            ok.append(fname)
            continue
        hf_path = f"simulations/checkpoints/{experiment}/{fname}"
        try:
            downloaded = hf_hub_download(repo_id=hf_repo_id, filename=hf_path)
            shutil.copy(downloaded, local_path)
            print(f"[ok] {fname} -> {local_path}")
            ok.append(fname)
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            failed.append(fname)

print(f"\n[Summary] {len(ok)} ok, {len(failed)} failed")
if failed:
    print("Failed:", failed)
    sys.exit(1)
PYEOF

EXIT_CODE=$?
echo "=== JOB ${SLURM_JOB_ID} FINISHED (exit ${EXIT_CODE}) ==="
exit $EXIT_CODE
