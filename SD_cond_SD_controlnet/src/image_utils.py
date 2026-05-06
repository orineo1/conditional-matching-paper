"""
image_utils.py — Image utilities for the MLGD-F pipeline.

Functions:
    sobel_proxy      Differentiable Sobel edge detection on a batch tensor.
    latent_to_pil    VAE decode + post-process to PIL image.
    build_base_image Construct the oval stick-figure base conditioning image.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T


def sobel_proxy(img_tensor, device):
    """
    Differentiable Sobel edge magnitude from an RGB image tensor.

    Args:
        img_tensor: [1, 3, H, W] float tensor.
        device:     torch device.

    Returns:
        [1, 3, H, W] float tensor of normalised edge magnitude,
        replicated across channels for ControlNet compatibility.
    """
    gray = img_tensor.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kx, padding=1)
    grad_y = F.conv2d(gray, ky, padding=1)
    edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-8)
    return edge_mag.repeat(1, 3, 1, 1)


def latent_to_pil(sample, vae, image_processor):
    """
    Decode an Architect latent tensor to a PIL image.

    Args:
        sample:          [1, C, H, W] latent tensor.
        vae:             VAE from the Architect pipeline.
        image_processor: Image processor from the Architect pipeline.

    Returns:
        PIL.Image
    """
    sample = sample / vae.config.scaling_factor
    pix = vae.decode(sample.to(vae.dtype)).sample
    return image_processor.postprocess(pix, output_type="pil")[0]


def build_base_image(device):
    """
    Build the oval stick-figure base conditioning image (512x512).

    Returns:
        (PIL.Image, [1, 3, 512, 512] float32 tensor on `device`)
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), color="black")
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 45, 362, 257), fill="white")
    draw.rectangle((234, 241, 278, 262), fill="white")
    draw.polygon([(244, 262), (268, 262), (274, 475), (238, 475)], fill="white")

    transform = T.Compose([T.ToTensor()])
    tensor = transform(img).unsqueeze(0).to(device).to(torch.float32)
    return img, tensor
