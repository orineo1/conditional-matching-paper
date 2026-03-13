import torch
import os
import zipfile
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from peft import PeftModel
from huggingface_hub import hf_hub_download



def _resolve_lora_path(lora_path: str) -> str:
    """
    If lora_path starts with 'hf://', download it from HuggingFace Hub.
    Expects format: hf://<repo_id>/<filename>
    If the downloaded file is a zip, extracts it and returns the folder path.
    """
    if not lora_path.startswith("hf://"):
        return lora_path

    without_prefix = lora_path[len("hf://"):]
    parts = without_prefix.split("/")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    if local_path.endswith(".zip"):
        extract_dir = local_path.replace(".zip", "")
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(local_path, 'r') as zf:
                zf.extractall(extract_dir)

        if os.path.exists(os.path.join(extract_dir, "adapter_config.json")):
            return extract_dir

        for sub in os.listdir(extract_dir):
            sub_path = os.path.join(extract_dir, sub)
            if os.path.isdir(sub_path):
                if os.path.exists(os.path.join(sub_path, "adapter_config.json")):
                    return sub_path

        raise FileNotFoundError(f"adapter_config.json not found inside {extract_dir}")

    return local_path

def load_models(device, architect_lora_path=None):
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16
    ).to(device)

    sprinter = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    architect = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to(device)

    if architect_lora_path:
        resolved_path = _resolve_lora_path(architect_lora_path)
        architect.unet = PeftModel.from_pretrained(architect.unet, resolved_path)
        print(f"Loaded architect LoRA from {resolved_path}")

    architect.vae.to(dtype=torch.float32)
    sprinter.vae.to(dtype=torch.float32)

    original_call = StableDiffusionXLControlNetPipeline.__call__
    StableDiffusionXLControlNetPipeline.__call__ = lambda self, *args, **kwargs: (
        original_call.__wrapped__(self, *args, **kwargs)
        if hasattr(original_call, '__wrapped__')
        else original_call(self, *args, **kwargs)
    )

    return architect, sprinter

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad_(False)

def setup_gradient_checkpointing(architect, sprinter):
    architect.unet.enable_gradient_checkpointing()
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    for m in [architect.unet, architect.vae, sprinter.unet, sprinter.controlnet, sprinter.vae]:
        freeze_module(m)