#!/bin/bash
#SBATCH --job-name=fix-env
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/fix_env_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/fix_env_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=catfish

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

# Switch branch
cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper
git checkout asymmetric-target
echo "Branch: $(git branch --show-current)"

# Restore packages
pip install diffusers==0.36.0 'transformers>=4.36.0' 'huggingface_hub>=0.25.0'

# Verify
echo "=== Versions after fix ==="
python -c "import diffusers; print('diffusers:', diffusers.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)"

echo "Done."
