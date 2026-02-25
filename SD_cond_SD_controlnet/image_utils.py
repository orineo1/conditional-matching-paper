import torch
import torch.nn.functional as F
import torchvision.transforms as T

def sobel_proxy(img_tensor, device):
    gray = img_tensor.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3)
    ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=device).view(1,1,3,3)
    grad_x = F.conv2d(gray, kx, padding=1)
    grad_y = F.conv2d(gray, ky, padding=1)
    edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-8)
    return edge_mag.repeat(1, 3, 1, 1)

def latent_to_pil(sample, vae, image_processor):
    sample = sample / vae.config.scaling_factor
    pix = vae.decode(sample.to(vae.dtype)).sample
    return image_processor.postprocess(pix, output_type="pil")[0]
def build_portrait_image(device):
    from PIL import Image, ImageDraw
    img  = Image.new('RGB', (512, 512), color='black')
    draw = ImageDraw.Draw(img)

    offset = 140  # shift everything down so head isn't clipped at top

    # Head
    draw.ellipse((181, 20  + offset, 331, 190  + offset), fill='white')
    # Neck
    draw.rectangle((231, 180 + offset, 281, 245 + offset), fill='white')
    # Shoulders
    draw.polygon([
        (30,  370 + offset),
        (482, 370 + offset),
        (380, 245 + offset),
        (132, 245 + offset),
    ], fill='white')

    transform = T.Compose([T.ToTensor()])
    tensor = transform(img).unsqueeze(0).to(device).to(torch.float32)
    return img, tensor