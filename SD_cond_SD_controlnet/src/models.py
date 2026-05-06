"""
models.py — Model loading for the MLGD-F pipeline.

Loads the Architect (SDXL Base) and Sprinter (SDXL Turbo + ControlNet-Scribble)
diffusion pipelines. Optionally loads a LoRA or fully fine-tuned U-Net onto the
Architect. Supports HuggingFace Hub paths via the `hf://` prefix.
"""

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
    If lora_path starts with 'hf://', download from HuggingFace Hub.
    Expects format: hf://<repo_id>/<filename>
    If the downloaded file is a zip, extracts it and returns the folder path.
    Local paths pass through unchanged.
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
    architect_model_id="stabilityai/stable-diffusion-xl-base-1.0",
):
    """
    Load Architect and Sprinter pipelines.

    Args:
        device:               torch device string ('cuda' or 'cpu').
        architect_lora_path:  Optional LoRA path (local or hf://).
        architect_unet_path:  Optional fully fine-tuned U-Net path (takes
                              priority over architect_lora_path).
        controlnet_model_id:  HuggingFace model ID for the ControlNet.
        sprinter_model_id:    HuggingFace model ID for the Sprinter.
        architect_model_id:   HuggingFace model ID for the Architect.

    Returns:
        (architect, sprinter) pipeline tuple.
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

    original_call = StableDiffusionXLControlNetPipeline.__call__
    StableDiffusionXLControlNetPipeline.__call__ = lambda self, *args, **kwargs: (
        original_call.__wrapped__(self, *args, **kwargs)
        if hasattr(original_call, "__wrapped__")
        else original_call(self, *args, **kwargs)
    )

    return architect, sprinter


def freeze_module(module):
    """Freeze all parameters in a module (no gradient updates)."""
    for p in module.parameters():
        p.requires_grad_(False)


def setup_gradient_checkpointing(architect, sprinter):
    """
    Enable gradient checkpointing on both pipelines and freeze all weights.
    Must be called before the MLGD-F loop.
    """
    architect.unet.enable_gradient_checkpointing()
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    for m in [
        architect.unet,
        architect.vae,
        sprinter.unet,
        sprinter.controlnet,
        sprinter.vae,
    ]:
        freeze_module(m)
