#!/bin/bash
#SBATCH --job-name=mmd_bench_gpu
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/mmd_bench_gpu_%j.out
#SBATCH --error=/sci/labs/orzuk/shaulytolk/cdm-perf/logs/mmd_bench_gpu_%j.err
# CUDA (L4) grid of exact_loss/bench_mmd.py: reference vs exact fast_mmd variants,
# float32 + float64, incl. the CLIP-768 / ~100-250 target regime of the SD pipeline.
# Submit:  sbatch experiments/model-optimization/exact_loss/submit_bench_gpu.sh
# Writes bench_raw/cuda_*.json; the summary merges cpu+cuda if both exist:
#   python bench_mmd.py --aggregate --device cpu,cuda
set -euo pipefail
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env
export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p /sci/labs/orzuk/shaulytolk/cdm-perf/logs
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python ../experiments/model-optimization/exact_loss/bench_mmd.py --grid full --device cuda
# merged summary (cpu raw files present only if submit_bench.sh already ran)
python ../experiments/model-optimization/exact_loss/bench_mmd.py --aggregate --device cpu,cuda || true
echo done
