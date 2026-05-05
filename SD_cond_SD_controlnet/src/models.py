import os
import zipfile

import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from peft import PeftModel


def _resolve_lora_path(lora_path: str) -> str:
    """
    Resolve a LoRA path to a local directory.
    Supports:
      - Local filesystem paths (returned unchanged)
      - HuggingFace Hub paths in the form hf://<repo_id>/<filename>
        (downloaded, unzipped if needed, adapter_config.json located)
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
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(extract_dir)

        if os.path.exists(os.path.join(extract_dir, "adapter_config.json")):
            return extract_dir

        for sub in os.listdir(extract_dir):
            sub_path = os.path.join(extract_dir, sub)
            if os.path.isdir(sub_path):
                if os.path.exists(os.path.join(sub_path, "adapter_config.json")):
                    return sub_path

        raise FileNotFoundError(
            f"adapter_config.json not found inside {extract_dir}"
        )

    return local_path


def load_models(
    device,
    architect_lora_path=None,
    architect_unet_path=None,
    controlnet_model_id="xinsir/controlnet-scribble-sdxl-1.0",
    sprinter_model_id="stabilityai/sdxl-turbo",
    architect_model_id="stabilityai/sdxl-turbo",
):
    """
    Load the Architect (SDXL) and Sprinter (SDXL + ControlNet-Scribble) pipelines.

    Optionally loads a fine-tuned architect U-Net or a LoRA adapter.
    Both VAEs are cast to float32 for gradient stability.

    Returns: (architect, sprinter)
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    sprinter = StableDiffusionXLControlNetPipeline.from_pretrained(
        sprinter_model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    architect = StableDiffusionXLPipeline.from_pretrained(
        architect_model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    architect.scheduler = DDIMScheduler.from_config(architect.scheduler.config)

    if architect_unet_path:
        architect.unet = UNet2DConditionModel.from_pretrained(
            architect_unet_path, torch_dtype=torch.float16,
        ).to(device)
        print(f"Loaded fine-tuned architect U-Net from {architect_unet_path}")
    elif architect_lora_path:
        resolved_path = _resolve_lora_path(architect_lora_path)
        architect.unet = PeftModel.from_pretrained(architect.unet, resolved_path)
        print(f"Loaded architect LoRA from {resolved_path}")

    architect.vae.to(dtype=torch.float32)
    sprinter.vae.to(dtype=torch.float32)
    architect.set_progress_bar_config(disable=True)
    sprinter.set_progress_bar_config(disable=True)

    return architect, sprinter


def freeze_module(module):
    """Freeze all parameters of a module."""
    for p in module.parameters():
        p.requires_grad_(False)


def setup_gradient_checkpointing(architect, sprinter):
    """Enable gradient checkpointing and freeze all weights for DPS."""
    architect.unet.enable_gradient_checkpointing()
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    for m in [architect.unet, architect.vae,
              sprinter.unet, sprinter.controlnet, sprinter.vae]:
        freeze_module(m)
