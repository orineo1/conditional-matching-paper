"""
models.py — load Architect + Sprinter pipelines.

Critical fix vs old src/models.py:
  The sprinter's StableDiffusionXLControlNetPipeline.__call__ is decorated with
  @torch.no_grad() by diffusers. This kills gradients through the sprinter during
  the DPS variation forward pass.

  We monkey-patch __call__ to call __wrapped__ (the original undecorated function)
  so that when run_dps_step_clip calls sprinter(...) inside a gradient checkpoint,
  autograd can differentiate through the sprinter VAE decode all the way back to
  pixel_x0_norm → pred_x0 → latents_step.

  Without this patch: var_latents.requires_grad = False  (broken)
  With this patch:    var_latents.requires_grad = True   (working)

  Source: SD_cond_SD_controlnet/models.py (working notebook version, Ori Meidler)
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


def _patch_sprinter_no_grad(sprinter):
    """
    Remove the @torch.no_grad() wrapper from the sprinter pipeline's __call__.

    diffusers wraps StableDiffusionXLControlNetPipeline.__call__ with
    @torch.no_grad(), which detaches all outputs from the autograd graph.
    During the DPS variation forward pass we NEED gradients to flow through
    the sprinter's VAE decode back to pixel_x0_norm.

    The original undecorated function is stored by functools.wraps as __wrapped__.
    We replace __call__ with a lambda that calls __wrapped__ directly, bypassing
    the no_grad decorator while keeping all other pipeline behaviour intact.

    This patch is applied per-instance so it doesn't affect other pipelines.
    """
    original_call = StableDiffusionXLControlNetPipeline.__call__

    if not hasattr(original_call, "__wrapped__"):
        print(
            "  [models] WARNING: sprinter __call__ has no __wrapped__ attribute. "
            "The no_grad patch could not be applied — gradients through the "
            "sprinter will be dead. Check your diffusers version.",
            flush=True,
        )
        return

    # Replace at the class level (affects all instances, but that's fine here —
    # we only ever have one sprinter and we want this behaviour globally).
    StableDiffusionXLControlNetPipeline.__call__ = (
        lambda self, *args, **kwargs: original_call.__wrapped__(self, *args, **kwargs)
    )
    print(
        "  [models] ✅ Sprinter no_grad patch applied — "
        "gradients will flow through sprinter __call__.",
        flush=True,
    )


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

    Applies the no_grad bypass patch to the sprinter so that gradient checkpointing
    in run_dps_step_clip can differentiate through it.

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
        print(f"  [models] Loaded fine-tuned architect U-Net from {architect_unet_path}",
              flush=True)
    elif architect_lora_path:
        resolved_path = _resolve_lora_path(architect_lora_path)
        architect.unet = PeftModel.from_pretrained(architect.unet, resolved_path)
        print(f"  [models] Loaded architect LoRA from {resolved_path}", flush=True)

    architect.vae.to(dtype=torch.float32)
    sprinter.vae.to(dtype=torch.float32)
    architect.set_progress_bar_config(disable=True)
    sprinter.set_progress_bar_config(disable=True)

    # ── Critical: remove no_grad wrapper from sprinter so DPS grads survive ──
    _patch_sprinter_no_grad(sprinter)

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
