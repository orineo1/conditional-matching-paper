#!/bin/bash
#SBATCH --job-name=scrib-samples
#SBATCH --output=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/scrib_samples_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/conditional-matching-paper/scrib_samples_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=salmon

export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate /sci/labs/orzuk/shaulytolk/conditional-matching-paper/scribble_env

export HF_HOME=/sci/labs/orzuk/shaulytolk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q controlnet_aux

cd /sci/labs/orzuk/shaulytolk/conditional-matching-paper

python3 -c "
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import os

device = 'cuda'
out_dir = 'output/zeta5_guided_samples'
os.makedirs(out_dir, exist_ok=True)

scribble = Image.open('scripts/assets/zeta5_final_guided.png').convert('RGB').resize((512, 512))
scribble.save(os.path.join(out_dir, 'scribble.png'))

print('Loading ControlNet...', flush=True)
controlnet = ControlNetModel.from_pretrained(
    'xinsir/controlnet-scribble-sdxl-1.0', torch_dtype=torch.float16).to(device)

print('Loading SDXL Turbo + ControlNet pipeline...', flush=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    'stabilityai/sdxl-turbo',
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
).to(device)
pipe.set_progress_bar_config(disable=True)

prompt = 'a superrealistic professional photograph of'
n = 100
print(f'Generating {n} images...', flush=True)
for i in range(n):
    g = torch.Generator(device=device).manual_seed(i)
    img = pipe(
        prompt=prompt,
        image=scribble,
        num_inference_steps=4,
        guidance_scale=0.0,
        controlnet_conditioning_scale=0.5,
        height=512, width=512,
        generator=g,
    ).images[0]
    img.save(os.path.join(out_dir, f'sample_{i:04d}.png'))
    if (i+1) % 10 == 0:
        print(f'  [{i+1}/{n}]', flush=True)

print(f'Done. Saved to {out_dir}')
"

echo "Scribble sample generation complete."
