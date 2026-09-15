"""
common.py — shared helpers for the point-vs-distributional SD experiment.

Pure-python/numpy parts (seeds, projection, CIs, PC1 stats, cache IO) are
importable without torch-GPU/diffusers and are CPU-tested in tests/.
GPU loaders live behind functions so build_report can run on a laptop.
"""
import json
import os
import sys

import numpy as np

EXP_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO      = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))
SD_SRC    = os.path.join(REPO, "SD_cond_SD_controlnet", "src")
SD_SCRIPTS = os.path.join(REPO, "SD_cond_SD_controlnet", "scripts")
for _p in (SD_SRC, SD_SCRIPTS):
    if _p not in sys.path:
        sys.path.append(_p)

NEUTRAL_PROMPT = "a superrealistic professional photograph of"
MAN_PROMPT   = "a superrealistic portrait photograph of a man, studio lighting"
WOMAN_PROMPT = "a superrealistic portrait photograph of a woman, studio lighting"
SEED = 1          # Ori's working configuration uses --seed 1. (The earlier
                  # Scenario-B runs used 42; those directories keep their own seed,
                  # recorded per-run in metrics.json["args"]["seed"].)

# ---------------------------------------------------------------------------
# Seed conventions (all derived from SEED; ranges must stay disjoint)
# ---------------------------------------------------------------------------
def eval_seed(seed, i):
    """Photo i of the fresh-sample eval — the pipeline's --seeded_rng rule."""
    return seed * 1_000_003 + 7_000_000 + i


def guidance_seed(seed, step, i):
    """Variation i at guided step `step` in run_mlgd_f --seeded_rng."""
    return seed * 1_000_003 + step * 10_000 + i


def pgd_seed(seed, step):
    """The single inner sample of PGD step `step` (arm 1)."""
    return seed * 1_000_003 + 3_000_000 + step


def seed_ranges_disjoint(seed=SEED, n_guided_steps=125, n_var=100, n_pgd=200, n_eval=2000):
    g = {guidance_seed(seed, s, i) for s in range(1, n_guided_steps + 1) for i in range(n_var)}
    p = {pgd_seed(seed, s) for s in range(n_pgd)}
    e = {eval_seed(seed, i) for i in range(n_eval)}
    return not (g & p) and not (g & e) and not (p & e)


# ---------------------------------------------------------------------------
# Arm-1 projection (pure)
# ---------------------------------------------------------------------------
def project_linf(delta, x0, eps):
    """L_inf ball of radius eps around x0, intersected with [0,1] pixel range.
    Returns the projected delta (same type as input; works for torch and numpy)."""
    d = delta.clip(-eps, eps) if isinstance(delta, np.ndarray) else delta.clamp(-eps, eps)
    x = (x0 + d)
    x = x.clip(0.0, 1.0) if isinstance(x, np.ndarray) else x.clamp(0.0, 1.0)
    return x - x0


# ---------------------------------------------------------------------------
# Statistics (pure)
# ---------------------------------------------------------------------------
def mean_ci95(values):
    """(mean, lo, hi): normal-approx 95% CI on the mean of `values`."""
    v = np.asarray(values, dtype=np.float64)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    return m, m - 1.96 * se, m + 1.96 * se


def wilson_ci95(k, n):
    """Wilson interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def p_male(embs, text_features):
    """Per-sample male probability: softmax over (man, woman) text features at
    logit scale 100 (the repo's compute_clip_softmax convention).
    embs: [n,768] L2-normalised image embeddings; text_features: [2,768] normalised."""
    e = np.asarray(embs, dtype=np.float64)
    t = np.asarray(text_features, dtype=np.float64)
    logits = 100.0 * e @ t.T                       # [n, 2]  (row 0 = man)
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex[:, 0] / ex.sum(axis=1)


def pc1_stats(target_embs, gen_embs, anchor_embs=None):
    """PC1 fitted on `anchor_embs` (default: the target set — pass the two anchor
    groups for scenario B, mirroring build_targets_gender's PCA); both sets
    projected onto it. Returns the deliverable numbers."""
    A = np.asarray(anchor_embs if anchor_embs is not None else target_embs, np.float64)
    mu = A.mean(axis=0)
    _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
    pc1 = Vt[0]
    t = (np.asarray(target_embs, np.float64) - mu) @ pc1
    g = (np.asarray(gen_embs, np.float64) - mu) @ pc1
    var_t, var_g = float(t.var(ddof=1)), float(g.var(ddof=1))
    return {"target_mean": float(t.mean()), "target_var": var_t,
            "gen_mean": float(g.mean()), "gen_var": var_g,
            "var_ratio": var_g / var_t if var_t > 0 else float("nan"),
            "target_proj": t, "gen_proj": g}


# ---------------------------------------------------------------------------
# Target-cache IO (format = run_mlgd_f.py's --target_cache, verbatim)
# ---------------------------------------------------------------------------
def cache_npz_path(cache_dir):
    return os.path.join(cache_dir, "targets_cache.npz")     # run_mlgd_f._target_cache_path


def write_cache(cache_dir, images_per_group, source_img, scribble_img, target_groups):
    """images_per_group: {name: [HxWx3 uint8 arrays]}; target_groups: [(name,prompt,n)]."""
    os.makedirs(cache_dir, exist_ok=True)
    arrays = {f"group__{name}": np.stack(imgs) for name, imgs in images_per_group.items()}
    arrays["source_portrait"] = np.asarray(source_img)
    arrays["scribble"] = np.asarray(scribble_img)
    np.savez(cache_npz_path(cache_dir), **arrays)
    with open(os.path.join(cache_dir, "target_groups.json"), "w") as f:
        json.dump([[n, p, int(k)] for n, p, k in target_groups], f, indent=2)


def read_cache(cache_dir):
    data = np.load(cache_npz_path(cache_dir))
    with open(os.path.join(cache_dir, "target_groups.json")) as f:
        groups = json.load(f)
    imgs = {n: data[f"group__{n}"] for n, _, _ in groups}
    return imgs, data["source_portrait"], data["scribble"], groups


# ---------------------------------------------------------------------------
# GPU loaders (imported lazily)
# ---------------------------------------------------------------------------
def load_sprinter_grad(device, sprinter_id="stabilityai/sdxl-turbo",
                       controlnet_id="xinsir/controlnet-scribble-sdxl-1.0"):
    """Sprinter pipeline prepared for GRADIENT flow (arm 1): no_grad decorator
    stripped from __call__ (as models.load_models does), weights frozen, block
    gradient checkpointing on, VAE fp32."""
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    from models import freeze_module
    cn = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16,
                                         use_safetensors=True).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sprinter_id, controlnet=cn, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    original_call = StableDiffusionXLControlNetPipeline.__call__
    if hasattr(original_call, "__wrapped__"):
        StableDiffusionXLControlNetPipeline.__call__ = \
            lambda self, *a, **k: original_call.__wrapped__(self, *a, **k)
    pipe.vae.to(dtype=torch.float32)
    pipe.unet.enable_gradient_checkpointing()
    pipe.controlnet.enable_gradient_checkpointing()
    for m in (pipe.unet, pipe.controlnet, pipe.vae):
        freeze_module(m)
    return pipe


def sprinter_clip_embed(sprinter, clip_model, clip_processor, ctrl, seed, prompt,
                        cn_scale, device):
    """One differentiable sprinter sample -> CLIP embedding [1,768].  The
    generator is built from `seed` INSIDE the call so a checkpoint recompute
    (or a repeat call) replays identical noise — the pipeline's convention."""
    import torch
    from clip_utils import encode_images_clip
    gen = torch.Generator(device=device).manual_seed(int(seed))
    lat = sprinter(prompt=[prompt], image=ctrl, num_inference_steps=2,
                   guidance_scale=0.0, controlnet_conditioning_scale=cn_scale,
                   output_type="latent", return_dict=True, generator=[gen]).images
    px = sprinter.vae.decode((lat.float() / sprinter.vae.config.scaling_factor)
                             .to(sprinter.vae.dtype)).sample
    px = torch.clamp((px.float() + 1.0) / 2.0, 0.0, 1.0)
    with torch.amp.autocast("cuda", enabled=False):
        return encode_images_clip(px.float(), clip_model, clip_processor)


# ---------------------------------------------------------------------------
# Measured descent bands (||applied step||) — one per ARM, from its own probe.
# NEVER use one arm's band for another: they differ by ~4x and doing so
# mislabels the point arm's steps (bug fixed 2026-09-12).
#   mmd   : debug/B/debug_gradient.json      (job 46134134) lam 0.5-2.0 improving
#   point : debug/B_point/debug_gradient.json(job 46147886) lam 0.5 improving, 1.0 degrading
# ---------------------------------------------------------------------------
PROBE_BANDS = {"mmd": (5.9, 23.5), "point": (22.3, 44.7)}
# Per-step guidance-loss noise, needed to judge whether a P-C1 delta is resolvable:
# the mmd arm logs a 100-sample MMD (sd ~0.008), the point arm a 1-SAMPLE loss (sd ~0.09).
PER_STEP_SD = {"mmd": 0.0075, "point": 0.09}


def band_for(args):
    """Probe band for the arm described by a run's `args` dict (default: mmd)."""
    return PROBE_BANDS.get(args.get("loss_fn", "mmd"), PROBE_BANDS["mmd"])
