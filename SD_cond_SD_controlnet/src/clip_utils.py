import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def load_clip_model(device, model_id="openai/clip-vit-large-patch14"):
    """Load and freeze CLIP ViT-L/14. Returns (clip_model, clip_processor)."""
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    return clip_model, clip_processor


def encode_images_clip(pixel_tensor, clip_model, clip_processor):
    """
    Encode a batch of images through CLIP (differentiable).

    Args:
        pixel_tensor: [B, 3, H, W] float in [0, 1]
    Returns:
        [B, 768] L2-normalized embeddings
    """
    resized = F.interpolate(pixel_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=pixel_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=pixel_tensor.device).view(1, 3, 1, 1)
    normalized = (resized - mean) / std

    emb = clip_model.vision_model(pixel_values=normalized).pooler_output
    emb = clip_model.visual_projection(emb)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # [B, 768]
