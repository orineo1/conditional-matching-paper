import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def load_latent_clip_model(device):
    """
    Load and freeze Latent-CLIP model (ViT-B/8, 512-dim output).
    Returns (model, tokenizer).
    """
    import latent_clip

    model_name = "ViT-B-8"
    pretrained = "wendlerc/latent-clip-b-8-512-34b-80k"

    model, _ = latent_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = latent_clip.get_tokenizer(model_name)
    return model, tokenizer


def encode_latents_clip(latent_tensor, model, vae_scaling_factor):
    """
    Encode VAE latents directly through Latent-CLIP (no pixel decode).

    latent_tensor: [B, 4, 64, 64] — channels-first VAE latents (already × scaling_factor)
    model: Latent-CLIP model
    vae_scaling_factor: float — divide latents by this to get raw latent space

    Returns: [B, 512] L2-normalized embeddings (differentiable)
    """
    # Undo VAE scaling
    latents_raw = latent_tensor / vae_scaling_factor

    # Channels-first [B,4,64,64] → channels-last [B,64,64,4]
    latents_cl = latents_raw.permute(0, 2, 3, 1).contiguous()

    emb = model.encode_image(latents_cl.float())
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # [B, 512]


def encode_targets_latent_clip(pil_images, vae, model, device):
    """
    Encode PIL target images through VAE encoder → Latent-CLIP.
    Used for building the target embedding cloud (no grad needed).

    pil_images: list of PIL images
    vae: SDXL VAE (for encoding to latent space)
    model: Latent-CLIP model

    Returns: [N, 512] L2-normalized embeddings
    """
    tensors = [TF.to_tensor(img).unsqueeze(0) for img in pil_images]
    pixel_tensor = torch.cat(tensors, dim=0).to(device).to(torch.float32)
    pixel_tensor = (pixel_tensor * 2.0) - 1.0  # [0,1] → [-1,1]

    scaling_factor = vae.config.scaling_factor

    with torch.no_grad():
        # Encode in batches to avoid OOM
        emb_list = []
        batch_size = 4
        for i in range(0, len(pil_images), batch_size):
            batch = pixel_tensor[i : i + batch_size]
            latent_dist = vae.encode(batch.to(vae.dtype)).latent_dist
            latents = latent_dist.mean * scaling_factor  # [B,4,64,64]
            emb = encode_latents_clip(latents.float(), model, scaling_factor)
            emb_list.append(emb)
        embeddings = torch.cat(emb_list, dim=0)

    return embeddings
