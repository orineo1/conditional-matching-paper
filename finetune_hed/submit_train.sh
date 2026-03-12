#!/bin/bash
#SBATCH --job-name=hed-finetune
#SBATCH --output=finetune_hed/output/train_%j.log
#SBATCH --error=finetune_hed/output/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

echo "Starting HED fine-tune on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Create output directory
mkdir -p finetune_hed/output

# Launch multi-GPU training with FSDP
accelerate launch \
    --use_fsdp \
    --num_processes=4 \
    --mixed_precision fp16 \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_state_dict_type SHARDED_STATE_DICT \
    finetune_hed/train_hed.py \
    --config finetune_hed/config.yaml

echo "Training complete."
