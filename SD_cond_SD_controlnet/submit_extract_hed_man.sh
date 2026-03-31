#!/bin/bash
#SBATCH --job-name=hed-man
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/hed_man_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/hed_man_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=salmon

# Conda setup
export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q controlnet_aux

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

python3 -c "
from controlnet_aux import HEDdetector
from PIL import Image

hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
img = Image.open('manly_man.png').convert('RGB').resize((512, 512))
scribble = hed(img, scribble=True)
scribble.save('scripts/assets/manly_man_hed.png')
print('Saved scripts/assets/manly_man_hed.png')
"

echo "HED extraction complete."
