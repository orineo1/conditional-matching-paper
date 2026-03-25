import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def load_latent_clip_model(device):
    """
    Load and freeze Latent-CLIP model (ViT-B/8, 512-dim output).
    Returns (model, tokenizer).
    """
    import latent_clip
    from huggingface_hub import hf_hub_download

    model_name = "Latent-ViT-B-8-512"

    # Download checkpoint from HF hub (cached after first download)
    checkpoint_path = hf_hub_download(
        repo_id="wendlerc/latent-clip-b-8-512-34b-80k",
        filename="checkpoints/epoch_34.pt",
    )

    # Load checkpoint manually with weights_only=False
    # (latent_clip's checkpoint contains numpy scalars, PyTorch 2.6+ rejects by default)
    model = latent_clip.create_model_and_transforms(
        model_name, device=device
    )[0]
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Unwrap checkpoint wrapper (epoch, state_dict, optimizer, etc.)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Strip DDP "module." prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Keep on CPU by default — move to GPU only when encoding
    model.to("cpu")

    tokenizer = latent_clip.get_tokenizer(model_name)
    return model, tokenizer


def encode_latents_clip(latent_tensor, model, vae_scaling_factor):
    """
    Encode VAE latents directly through Latent-CLIP (no pixel decode).

    latent_tensor: [B, 4, 64, 64] — channels-first VAE latents (already × scaling_factor)
    model: Latent-CLIP model (must already be on same device as latent_tensor)
    vae_scaling_factor: float — divide latents by this to get raw latent space

    Returns: [B, 512] L2-normalized embeddings (differentiable)
    """
    # Undo VAE scaling
    latents_raw = latent_tensor / vae_scaling_factor

    # Keep channels-first [B,4,64,64] — conv1 expects 4 input channels
    emb = model.encode_image(latents_raw.float())
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
