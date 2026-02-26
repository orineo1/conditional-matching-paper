"""LoRA fine-tune SDXL Turbo U-Net on QuickDraw scribble data."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class ScribbleDataset(Dataset):
    """Dataset of rendered QuickDraw scribbles with fixed captions."""

    def __init__(self, data_dir, caption, resolution=512):
        self.data_dir = Path(data_dir)
        self.caption = caption
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        metadata_path = self.data_dir / "metadata.jsonl"
        self.entries = []
        with open(metadata_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self.entries.append(entry["file_name"])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path = self.data_dir / self.entries[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image, "caption": self.caption}


def encode_prompt(tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, prompt, device):
    """Encode prompt using both SDXL text encoders."""
    # Tokenizer 1 (CLIP ViT-L)
    tokens_1 = tokenizer_1(
        prompt, padding="max_length", max_length=tokenizer_1.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
    prompt_embeds_1 = encoder_output_1.hidden_states[-2]

    # Tokenizer 2 (CLIP ViT-bigG)
    tokens_2 = tokenizer_2(
        prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
    prompt_embeds_2 = encoder_output_2.hidden_states[-2]
    pooled_prompt_embeds = encoder_output_2[0]

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SDXL Turbo on scribbles")
    parser.add_argument("--config", type=str, default="scribble_tune/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=cfg["output_dir"],
    )

    set_seed(cfg["seed"])

    # Load models
    model_name = cfg["model_name"]
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=torch.float16)
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=torch.float16)

    # Freeze everything except LoRA layers
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Apply LoRA to U-Net
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Cast LoRA params to float32 for training stability
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Dataset and dataloader
    dataset = ScribbleDataset(cfg["data_dir"], cfg["caption"], cfg["resolution"])
    dataloader = DataLoader(
        dataset, batch_size=cfg["train_batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Optimizer and scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["learning_rate"], weight_decay=1e-2)

    # Compute total training steps from epochs (use len(dataloader) which accounts for drop_last)
    num_epochs = cfg["num_epochs"]
    steps_per_epoch = len(dataloader) // cfg["gradient_accumulation_steps"]
    max_train_steps = num_epochs * steps_per_epoch
    print(f"Dataset: {len(dataset)} images, {steps_per_epoch} steps/epoch, {max_train_steps} total steps ({num_epochs} epochs)")

    lr_scheduler = get_scheduler(
        cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    # Prepare with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    vae.to(accelerator.device)
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    # Pre-compute text embeddings (fixed prompt)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
            cfg["caption"], accelerator.device,
        )

    # SDXL add_time_ids: (original_size, crop_coords, target_size)
    res = cfg["resolution"]
    add_time_ids = torch.tensor([[res, res, 0, 0, res, res]], dtype=torch.float16).to(accelerator.device)

    # Training loop
    accelerator.init_trackers("scribble_lora")
    global_step = 0
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    unet.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)

                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device, dtype=torch.long,
                )

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents.to(dtype=torch.float16)

                # Expand embeddings to batch size
                batch_prompt_embeds = prompt_embeds.expand(bsz, -1, -1)
                batch_pooled = pooled_prompt_embeds.expand(bsz, -1)
                batch_time_ids = add_time_ids.expand(bsz, -1)

                added_cond_kwargs = {
                    "text_embeds": batch_pooled,
                    "time_ids": batch_time_ids,
                }

                # Predict noise
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=batch_prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # MSE loss (predict noise)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.detach().item()
            epoch_steps += 1

            if global_step % 100 == 0:
                accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}, step=global_step)
                if accelerator.is_main_process:
                    print(f"Epoch {epoch+1}/{num_epochs} | Step {global_step}/{max_train_steps} | Loss: {loss.item():.4f}")

            if global_step % cfg["checkpointing_steps"] == 0 and accelerator.is_main_process:
                save_dir = Path(cfg["output_dir"]) / f"checkpoint-{global_step}"
                accelerator.unwrap_model(unet).save_pretrained(save_dir)
                print(f"Saved checkpoint to {save_dir}")

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"Epoch {epoch+1}/{num_epochs} complete | Avg loss: {avg_loss:.4f}")

    # Save final LoRA weights
    if accelerator.is_main_process:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(output_dir)
        print(f"Saved final LoRA weights to {output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
