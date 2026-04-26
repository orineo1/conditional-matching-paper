# ── dependencies ─────────────────────────────────────────────────────────────
# If running fresh:  pip install diffusers transformers accelerate

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed on cluster

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    DDIMScheduler,
)


# ── device ───────────────────────────────────────────────────────────────────
SEED   = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Device: {device}  |  dtype: {dtype}")


# ── experiment hyper-parameters ───────────────────────────────────────────────
HEIGHT, WIDTH = 512, 512        # image size (latent: H/8 × W/8 = 64×64)

# image sampling (§3)
N_TEACHER_SAMPLE_STEPS = 20     # DDIM steps for teacher images
N_STUDENT_SAMPLE_STEPS = 4      # LCM  steps for student images

# §5 experiment
K_SKIP     = 4                  # teacher ODE segment length  (k in the paper) ## REMOVE
N_RANDOM_V = 50                 # random VJP directions per (prompt, t_n)
GUIDANCE_W_TEACHER = 7.5
GUIDANCE_W_STUDENT = 1.0

PROMPTS = [
    "a golden retriever playing in a sunny park",
    "a red sports car on a mountain road",
    "a bowl of colorful fruit on a wooden table",
    "a snowy mountain peak at sunset",
    "an astronaut floating in outer space",
    "a cat sitting on a windowsill in the rain",
    "a bustling Tokyo street at night",
    "a lone lighthouse on a rocky coast",
    "a field of lavender under a purple sky",
    "a steaming cup of coffee on a wooden desk",
    "a polar bear walking on Arctic ice",
    "a vintage train passing through autumn forest",
    "a crowded beach on a summer day",
    "a medieval castle on a hilltop at dusk",
    "a hummingbird hovering near a red flower",
    "a neon-lit cyberpunk alley in the rain",
    "a waterfall in a tropical rainforest",
    "a child flying a kite in an open meadow",
    "a wolf howling at the full moon",
    "a glass of water with ice and lemon",
    "a row of colorful houses in a Scandinavian village",
    "a dragon perched on a mountain peak",
    "a farmer working in a rice paddy at sunrise",
    "a spaceship landing on a red desert planet",
    "a cozy fireplace in a snowy cabin interior",
    "a grand piano in an empty concert hall",
    "a market stall overflowing with spices",
    "a surfer riding a massive ocean wave",
    "a hot air balloon over a desert canyon",
    "a fox in a snowy pine forest",
    "a futuristic city skyline at golden hour",
    "a close-up of a bee on a sunflower",
    "an old man reading a book in a library",
    "a fishing boat at sea during a storm",
    "a tiger drinking from a jungle river",
    "a cobblestone street in a rainy European city",
    "a chef plating an elegant dish in a restaurant",
    "a glacier calving into a turquoise sea",
    "a young girl dancing in a flower field",
    "a robot walking through a futuristic museum",
    "a deer grazing in a misty morning forest",
    "a sailing yacht on a calm blue ocean",
    "a beekeeper tending hives in golden afternoon light",
    "a crowded subway platform in New York City",
    "a snowy owl perched on a branch at night",
    "a volcano erupting at night over the ocean",
    "a samurai standing in a bamboo forest",
    "a penguin colony on an Antarctic shore",
    "a street musician playing violin in the rain",
    "a sunflower field stretching to the horizon",
    "a neon sign reflected in a wet city street",
    "a horse galloping across an open plain",
    "a deep sea diver exploring a coral reef",
    "an abandoned greenhouse overgrown with vines",
    "a crowded night market in Southeast Asia",
    "a shepherd and his flock on a misty hillside",
    "a library with floor-to-ceiling bookshelves",
    "a bridge over a river in autumn",
    "a ballerina dancing on an empty stage",
    "a lighthouse beam sweeping through thick fog",
    "a campfire under a starry sky in the desert",
    "a child looking through a telescope at night",
    "a giant redwood forest with shafts of light",
    "a falcon in mid-dive against a blue sky",
    "a gondola on a Venetian canal at dawn",
    "a laboratory with glowing test tubes",
    "a monk meditating in a mountain temple",
    "a polar aurora over a frozen lake",
    "a jazz band playing in a dimly lit club",
    "a sunlit wheat field with a red barn",
    "a lone tree on a cliff above the ocean",
    "a blacksmith hammering glowing metal",
    "an elephant herd crossing the Serengeti",
    "a child splashing in rain puddles",
    "a lantern festival over a river at night",
    "a mountain biker on a forest trail",
    "a vintage camera on a wooden table",
    "a thunderstorm over a flat prairie",
    "a hermit crab on a tropical beach",
    "a street artist painting a large mural",
    "a snowy village with smoke rising from chimneys",
    "a peacock displaying its feathers in a garden",
    "a flooded Venetian piazza at high tide",
    "a baker pulling fresh bread from a stone oven",
    "a climber reaching a rocky summit at sunrise",
    "a firefly-lit forest clearing at dusk",
    "a dog running on a winter beach",
    "a desert caravan of camels at sunset",
    "a butterfly landing on a child's hand",
    "a futuristic greenhouse on Mars",
    "an open-air fish market by the harbor",
    "a cellist performing in a candlelit church",
    "a hot spring steaming in a winter landscape",
    "a paper lantern floating up into the night sky",
    "a wolf pack moving through a snowy forest",
    "a pirate ship on a stormy sea",
    "a tea ceremony in a traditional Japanese room",
    "a golden wheat field under a dramatic storm sky",
    "a child building a sandcastle at low tide",
    "a red fox watching the city lights from a hilltop at dusk",
]


TEACHER_ID = "stabilityai/stable-diffusion-xl-base-1.0"
STUDENT_ID = "latent-consistency/lcm-sdxl"

print("Loading teacher pipeline (SDXL-Base + DDIM scheduler)...")
teacher_pipe = StableDiffusionXLPipeline.from_pretrained(
    TEACHER_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
    use_safetensors=True,
).to(device)
teacher_pipe.scheduler = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
teacher_pipe.set_progress_bar_config(disable=True)
print("  ✓ Teacher ready.")


print("Loading student UNet (LCM-SDXL)...")
student_unet = UNet2DConditionModel.from_pretrained(
    STUDENT_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
).to(device)

# Build student pipeline: identical to teacher except UNet + scheduler
student_pipe = StableDiffusionXLPipeline(
    vae               = teacher_pipe.vae,
    text_encoder      = teacher_pipe.text_encoder,
    text_encoder_2    = teacher_pipe.text_encoder_2,
    tokenizer         = teacher_pipe.tokenizer,
    tokenizer_2       = teacher_pipe.tokenizer_2,
    unet              = student_unet,
    scheduler         = LCMScheduler.from_config(teacher_pipe.scheduler.config),
).to(device)
student_pipe.set_progress_bar_config(disable=True)
print("  ✓ Student ready.")


teacher_images, student_images = [], []

print("Sampling images...")
for i, prompt in enumerate(PROMPTS):

    gen = torch.Generator(device=device).manual_seed(SEED + i)
    with torch.no_grad():
        t_img = teacher_pipe(
            prompt, num_inference_steps=N_TEACHER_SAMPLE_STEPS,
            guidance_scale=7.5, generator=gen, height=HEIGHT, width=WIDTH,
        ).images[0]

    gen = torch.Generator(device=device).manual_seed(SEED + i)   # reset same seed
    with torch.no_grad():
        s_img = student_pipe(
            prompt, num_inference_steps=N_STUDENT_SAMPLE_STEPS,
            guidance_scale=1.0, generator=gen, height=HEIGHT, width=WIDTH,
        ).images[0]

    teacher_images.append(t_img)
    student_images.append(s_img)
    print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:50]}")


fig, axes = plt.subplots(2, len(PROMPTS), figsize=(4 * len(PROMPTS), 9))

for i, prompt in enumerate(PROMPTS):
    for row, (img, label) in enumerate([
        (teacher_images[i], f"Teacher\n({N_TEACHER_SAMPLE_STEPS} DDIM steps)"),
        (student_images[i], f"Student LCM\n({N_STUDENT_SAMPLE_STEPS} steps)"),
    ]):
        axes[row, i].imshow(img)
        axes[row, i].set_title(f'"{prompt[:26]}…"', fontsize=8)
        axes[row, i].axis("off")

axes[0, 0].set_ylabel("Teacher", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Student (LCM)", fontsize=11, fontweight="bold")
plt.suptitle(
    "Teacher (SDXL-Base) vs Student (LCM-SDXL) — same seed η per prompt",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("fig1_samples.png", dpi=120, bbox_inches="tight")
plt.close()


def encode_prompt(pipe, prompt):
    """Return (prompt_embeds, pooled_embeds) — the x in the paper.  No grad attached."""
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
    return pe.to(dtype), pooled.to(dtype)


def unet_forward(unet, latent, prompt_embeds, pooled_embeds, timestep):
    """Single UNet denoising step.  Returns the predicted noise tensor."""
    added_cond = {
        "text_embeds": pooled_embeds,
        "time_ids": torch.tensor(
            [[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]], dtype=dtype, device=device
        ),
    }
    return unet(
        latent, timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond,
    ).sample


pe0, pooled0 = encode_prompt(teacher_pipe, PROMPTS[0])

torch.manual_seed(SEED)
eta0 = torch.randn(1, 4, HEIGHT // 8, WIDTH // 8, dtype=dtype, device=device)
t500 = torch.tensor([500], device=device)

print("§4 Gradient sanity check")
print("-" * 55)
for name, unet in [("Teacher", teacher_pipe.unet), ("Student", student_unet)]:
    unet.train()                            # required for gradient computation
    x = pe0.detach().requires_grad_(True)
    out = unet_forward(unet, eta0, x, pooled0, t500)
    out.sum().backward()

    g = x.grad
    print(f"  {name:8s} | shape {list(g.shape)} | "
          f"NaN: {torch.isnan(g).any().item()} | "
          f"norm: {g.norm().item():.5f}")

print("\n  ✓ Both UNets are differentiable w.r.t. prompt embeddings.")


def noise_latent(z0, t_val, noise, scheduler):
    """
    Forward diffusion: z_{t_val} = sqrt(ᾱ_t)·z0 + sqrt(1−ᾱ_t)·noise.
    Mimics the noise level at timestep t_val using the scheduler's alpha schedule.
    """
    alpha = scheduler.alphas_cumprod[t_val].to(z0.device, z0.dtype)
    return (alpha ** 0.5) * z0 + ((1 - alpha) ** 0.5) * noise


def teacher_full(scheduler, z_nk, t_nk_idx, prompt_embeds, pooled_embeds):
    """
    Run all remaining DDIM steps from z_nk to z_0.
    This matches what the student does in a single step.
    """
    added_cond = {
        "text_embeds": pooled_embeds,
        "time_ids": torch.tensor(
            [[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]], dtype=dtype, device=device
        ),
    }
    latent = z_nk
    for t in scheduler.timesteps[t_nk_idx:]:      # run to the end, not just k steps
        noise_pred = teacher_pipe.unet(
            latent, t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond,
        ).sample

        if GUIDANCE_W_TEACHER > 0:
            noise_uncond = teacher_pipe.unet(
                latent, t,
                encoder_hidden_states=torch.zeros_like(prompt_embeds),
                added_cond_kwargs=added_cond,
            ).sample
            noise_pred = (1 + GUIDANCE_W_TEACHER) * noise_pred - GUIDANCE_W_TEACHER * noise_uncond

        latent = scheduler.step(noise_pred, t, latent).prev_sample
    return latent


def student_n_steps(scheduler, z_nk, t_nk_idx, prompt_embeds, pooled_embeds, n_steps):
    """
    Run n_steps LCM steps of the student UNet starting from z_nk.
    No classifier-free guidance — LCM is trained to work without it.
    """
    added_cond = {
        "text_embeds": pooled_embeds,
        "time_ids": torch.tensor(
            [[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]], dtype=dtype, device=device
        ),
    }
    latent = z_nk
    for t in scheduler.timesteps[t_nk_idx : t_nk_idx + n_steps]:
        t_gpu = torch.tensor([t], device=device)
        t_cpu = torch.tensor([t], device="cpu")

        noise_pred = student_unet(
            latent, t_gpu,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond,
        ).sample

        latent = scheduler.step(
            noise_pred.cpu().to(torch.float32),
            t_cpu,
            latent.cpu().to(torch.float32)
        ).prev_sample.to(device=device, dtype=dtype)

    return latent

def compute_vjp(fn, x_embed, v):
    """
    Compute the vector-Jacobian product  ∂_x (v^T fn(x))  via one backward pass.
    """
    x = x_embed.detach().requires_grad_(True)
    out = fn(x)
    (out * v).sum().backward()
    return x.grad.detach().clone()


# Schedulers
sched = DDIMScheduler.from_config(teacher_pipe.scheduler.config)
sched.set_timesteps(N_TEACHER_SAMPLE_STEPS, device=device)

lcm_sched = LCMScheduler.from_config(teacher_pipe.scheduler.config)
lcm_sched.set_timesteps(N_STUDENT_SAMPLE_STEPS, device=device)

# Make sure all models are in eval mode
teacher_pipe.unet.eval()
teacher_pipe.vae.eval()
student_unet.eval()

def decode(latent):
    with torch.no_grad():
        teacher_pipe.vae.to(torch.float32)
        img = teacher_pipe.vae.decode(
            latent.float() / teacher_pipe.vae.config.scaling_factor
        ).sample
        teacher_pipe.vae.to(dtype)   # cast back after decode
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.squeeze(0).permute(1, 2, 0).float().detach().cpu().numpy()
    return (img * 255).astype("uint8")

records = []

print("ε_s / ε_g experiment")
print(f"  Teacher: {N_TEACHER_SAMPLE_STEPS} DDIM steps, z_T → z_0, guidance={GUIDANCE_W_TEACHER}")
print(f"  Student: {N_STUDENT_SAMPLE_STEPS} LCM  steps, z_T → z_0, guidance={GUIDANCE_W_STUDENT}")
print(f"  VJP directions: {N_RANDOM_V}")

for i, prompt in enumerate(PROMPTS):
    print(f"\n[{i+1}/{len(PROMPTS)}] \"{prompt}\"")

    pe, pooled = encode_prompt(teacher_pipe, prompt)

    # Pure noise — both models always denoise from t=T to z_0
    torch.manual_seed(SEED + i)
    z_T = torch.randn(1, 4, HEIGHT // 8, WIDTH // 8, dtype=dtype, device=device)

    # Closures — only x (prompt embedding) varies; everything else is fixed
    def f_teacher(x): return teacher_full(sched,     z_T, 0, x, pooled)
    def f_student(x): return student_n_steps(lcm_sched, z_T, 0, x, pooled, N_STUDENT_SAMPLE_STEPS)

    # ── Output gap  ε_s = ‖f★(x) − fφ(x)‖ ──────────────────────────────
    with torch.no_grad():
        out_t = f_teacher(pe)
        out_s = f_student(pe)

    # ── Sanity check: decode and save first prompt only ───────────────────
    if i == 0:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(decode(out_t))
        axes[0].set_title(f"Teacher ({N_TEACHER_SAMPLE_STEPS} DDIM steps)")
        axes[0].axis("off")
        axes[1].imshow(decode(out_s))
        axes[1].set_title(f"Student ({N_STUDENT_SAMPLE_STEPS} LCM steps)")
        axes[1].axis("off")
        plt.suptitle(f'Sanity check — "{PROMPTS[0]}"', fontsize=11)
        plt.tight_layout()
        plt.savefig("fig2_sanity.png", dpi=120, bbox_inches="tight")
        plt.close()  # plt.show() replaced for headless cluster execution

    eps_s      = (out_t - out_s).float().norm().item()
    f_phi_norm = out_s.float().norm().item()
    out_shape  = out_t.shape
    del out_t, out_s;  torch.cuda.empty_cache()

    # ── Jacobian gap  ε_g via random VJPs ───────────────────────────────
    vjp_norms   = []
    J_phi_norms = []
    for kk in range(N_RANDOM_V):
        torch.manual_seed(4000 + 100 * i + kk)
        v = torch.randn(*out_shape, dtype=dtype, device=device)
        v = v / v.norm()

        g_t = compute_vjp(f_teacher, pe, v)
        g_s = compute_vjp(f_student, pe, v)
        vjp_norms.append((g_t - g_s).float().norm().item())
        J_phi_norms.append(g_s.float().norm().item())
        del g_t, g_s;  torch.cuda.empty_cache()

    eps_g      = float(np.mean(vjp_norms))
    eps_g_std  = float(np.std(vjp_norms))
    J_phi_mean = float(np.mean(J_phi_norms))

    records.append(dict(
        prompt_idx=i, prompt=prompt,
        eps_s=eps_s, eps_g=eps_g, eps_g_std=eps_g_std,
        f_phi_norm=f_phi_norm, J_phi_mean=J_phi_mean,
    ))
    print(f"  ε_s={eps_s:7.4f} | ε_g={eps_g:.4f}±{eps_g_std:.4f}| "
          f"‖f_φ‖={f_phi_norm:.4f} | ‖∂_x[vᵀf_φ]‖={J_phi_mean:.4f}")

df = pd.DataFrame(records)
print("\nDone.")

y_bar = float(np.sqrt(np.mean(df["f_phi_norm"].values ** 2)))
J_bar = float(np.sqrt(np.mean(df["J_phi_mean"].values ** 2)))

df["hat_r_s"] = df["eps_s"] / y_bar
df["hat_r_g"] = df["eps_g"] / J_bar

print(f"y_bar (output scale)   = {y_bar:.4f}")
print(f"J_bar (gradient scale) = {J_bar:.4f}\n")

print("=" * 90)
print("SUMMARY  (mean ± std across prompts)")
print("=" * 90)
print(f"{'ε_s':>14} | {'ε_g':>20} | {'raw ratio':>10} | {'r̂_s (ε_s/ȳ)':>14} | {'r̂_g (ε_g/J̄)':>14}")
print("-" * 90)
print(
    f"{df['eps_s'].mean():>6.4f} ± {df['eps_s'].std():>5.4f} | "
    f"{df['eps_g'].mean():>6.4f} ± {df['eps_g_std'].mean():>5.4f} | "
    f"{df['hat_r_s'].mean():>14.4f} | "
    f"{df['hat_r_g'].mean():>14.4f}"
)
print("=" * 90)

if df['hat_r_g'].mean() < df['hat_r_s'].mean():
    print("✓ r̂_g < r̂_s: gradients align better than outputs — supports using f_φ as guidance surrogate")
else:
    print("✗ r̂_g ≥ r̂_s: no gradient advantage observed")

print()
print(df[["prompt", "eps_s", "eps_g", "hat_r_s", "hat_r_g"]].to_string(index=False))
