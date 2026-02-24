import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from transformers import CLIPModel


def load_models(device):
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16
    ).to(device)

    sprinter = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    architect = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    architect.vae.to(dtype=torch.float32)
    sprinter.vae.to(dtype=torch.float32)

    # ── CLIP (ViT-L/14) — frozen, used for MMD guidance ──────────────────────
    # Loaded in float32 so grad flows cleanly through vision encoder
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device).to(torch.float32)

    # Monkey-patch sprinter to allow gradients through __call__
    original_call = StableDiffusionXLControlNetPipeline.__call__
    StableDiffusionXLControlNetPipeline.__call__ = lambda self, *args, **kwargs: (
        original_call.__wrapped__(self, *args, **kwargs)
        if hasattr(original_call, '__wrapped__')
        else original_call(self, *args, **kwargs)
    )

    return architect, sprinter, clip_model   # ← now returns 3 objects


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad_(False)


def setup_gradient_checkpointing(architect, sprinter):
    architect.unet.enable_gradient_checkpointing()
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    # Freeze everything — grad flows through graph, not through weights
    for m in [architect.unet, architect.vae,
              sprinter.unet, sprinter.controlnet, sprinter.vae]:
        freeze_module(m)
    # CLIP weights also frozen — but NOT wrapped in no_grad so graph stays live
    # freeze_module is enough; no_grad would break Option B gradient path
    freeze_module(clip_model)  # called from main after load_models