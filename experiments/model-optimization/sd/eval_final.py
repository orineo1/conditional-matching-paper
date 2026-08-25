"""
eval_final.py — standalone final MMD evaluation for an sd_perf arm directory.

Reconstructs the final state from what run_mlgd_f.py persists BEFORE its own
final eval (step 9b): final_scribble_mlgd_f.png / final_scribble_regular.png
(exact: evaluate_distribution_mmd itself decodes the latent to this PIL before
the sprinter sees it) or, as fallback, final_latents.pt decoded through the
architect VAE. Targets come from target_clip_embeddings.pt (saved by the run)
or are re-encoded from the --target_cache npz. Only the sprinter + CLIP are
loaded (no architect UNet), so the eval fits easily.

Eval seeds: photo j uses Generator(seed*1000003 + 7000000 + j) — the same
formula as run_mlgd_f.py --seeded_rng, disjoint from every guidance seed
(guidance: seed*1000003 + step*10000 + i, step <= 500) and identical across arms
with the same --seed.

Usage:
  python experiments/model-optimization/sd/eval_final.py --run_dir output/sd_perf/baseline_seed1
     [--eval_n 2000 --eval_batch_size 8 --clip_batch_size 32 --seed 1 --target_cache DIR]
Writes <run_dir>/metrics.json (merging metrics_partial.json) and photos_*/ + npy/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SRC = os.path.join(_REPO, "SD_cond_SD_controlnet", "src")
sys.path.insert(0, _SRC)

from clip_utils import encode_images_clip, load_clip_model  # noqa: E402
from metrics import compute_mmd, evaluate_distribution_mmd  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--eval_n", type=int, default=2000)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--clip_batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=None, help="default: the run's --seed")
    p.add_argument("--target_cache", type=str, default=None,
                   help="targets npz dir (default: run args.target_cache); only needed "
                        "if target_clip_embeddings.pt is missing")
    p.add_argument("--eval_prompt", type=str, default=None, help="default: run's sprinter_eval_prompt")
    p.add_argument("--sprinter_model_id", default="stabilityai/sdxl-turbo")
    p.add_argument("--controlnet_model_id", default="xinsir/controlnet-scribble-sdxl-1.0")
    p.add_argument("--architect_model_id", default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="only its VAE is loaded, and only if the final PNGs are missing")
    p.add_argument("--save_photos", type=int, default=100, help="how many eval photos to save per path")
    return p.parse_args()


def load_run_args(run_dir):
    for name in ("metrics_partial.json", "metrics.json"):
        f = os.path.join(run_dir, name)
        if os.path.exists(f):
            with open(f) as fh:
                return json.load(fh)
    lp = os.path.join(run_dir, "final_latents.pt")
    if os.path.exists(lp):
        return {"args": torch.load(lp, map_location="cpu")["args"], "steps": None}
    raise FileNotFoundError(f"{run_dir}: no metrics_partial.json / metrics.json / final_latents.pt "
                            "— this run predates the persistence fix and cannot be evaluated post hoc.")


def load_sprinter(device, sprinter_id, controlnet_id):
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    cn = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16,
                                         use_safetensors=True).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sprinter_id, controlnet=cn, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def final_scribbles(run_dir, device, architect_id):
    pm = os.path.join(run_dir, "final_scribble_mlgd_f.png")
    pr = os.path.join(run_dir, "final_scribble_regular.png")
    if os.path.exists(pm) and os.path.exists(pr):
        return Image.open(pm).convert("RGB"), Image.open(pr).convert("RGB"), "png"
    lp = os.path.join(run_dir, "final_latents.pt")
    if not os.path.exists(lp):
        raise FileNotFoundError("neither final_scribble_*.png nor final_latents.pt in run_dir")
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from image_utils import latent_to_pil
    d = torch.load(lp, map_location="cpu")
    vae = AutoencoderKL.from_pretrained(architect_id, subfolder="vae",
                                        torch_dtype=torch.float32).to(device)
    proc = VaeImageProcessor(vae_scale_factor=8)
    with torch.no_grad():
        a = latent_to_pil(d["latents"].to(device), vae, proc)
        b = latent_to_pil(d["latents_regular"].to(device), vae, proc)
    del vae; torch.cuda.empty_cache()
    return a, b, "latents.pt"


def target_embeddings(run_dir, args, run_args, clip_model, clip_processor, device):
    tp = os.path.join(run_dir, "target_clip_embeddings.pt")
    if os.path.exists(tp):
        return torch.load(tp, map_location="cpu")["all_clip_embeddings"].to(device), "target_clip_embeddings.pt"
    cache = args.target_cache or run_args.get("target_cache")
    npz = os.path.join(cache, "targets_cache.npz") if cache else None
    if not npz or not os.path.exists(npz):
        raise FileNotFoundError("no target_clip_embeddings.pt and no target cache npz")
    data = np.load(npz)
    with open(os.path.join(cache, "target_groups.json")) as f:
        groups = json.load(f)
    import torchvision.transforms.functional as TF
    embs = []
    with torch.no_grad():
        for name, _, _ in groups:
            arr = data[f"group__{name}"]
            for s in range(0, len(arr), args.clip_batch_size):
                t = torch.stack([TF.to_tensor(Image.fromarray(a)) for a in arr[s:s + args.clip_batch_size]]).to(device)
                embs.append(encode_images_clip(t, clip_model, clip_processor))
    return torch.cat(embs, 0), npz


def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = load_run_args(args.run_dir)
    run_args = run["args"]
    seed = args.seed if args.seed is not None else run_args.get("seed")
    if seed is None:
        raise ValueError("no seed: pass --seed")
    eval_prompt = args.eval_prompt or run_args.get("sprinter_eval_prompt",
                                                   "a superrealistic professional photograph of")
    eval_seed = seed * 1_000_003 + 7_000_000
    print(f"run_dir={args.run_dir} seed={seed} eval_seed_base={eval_seed} eval_n={args.eval_n}", flush=True)

    scr_mlgd, scr_reg, src = final_scribbles(args.run_dir, device, args.architect_model_id)
    print(f"final scribbles from {src}", flush=True)

    clip_model, clip_processor = load_clip_model(device)
    targets, tsrc = target_embeddings(args.run_dir, args, run_args, clip_model, clip_processor, device)
    print(f"targets {tuple(targets.shape)} from {tsrc}", flush=True)
    sprinter = load_sprinter(device, args.sprinter_model_id, args.controlnet_model_id)

    # evaluate_distribution_mmd accepts the stored PIL scribble directly (metrics.py).
    t0 = time.time()
    out = {}
    for key, scr in (("regular", scr_reg), ("mlgd_f", scr_mlgd)):
        print(f"evaluating {key}...", flush=True)
        mmd, photos, embs = evaluate_distribution_mmd(
            scr, None, None, sprinter, clip_model, clip_processor, targets, eval_prompt,
            n_eval=args.eval_n, device=device, batch_size=args.eval_batch_size,
            seed=eval_seed, clip_batch_size=args.clip_batch_size)
        out[key] = {"mmd": float(mmd), "embs": embs.cpu(), "photos": photos}
        print(f"  {key}: MMD={mmd:.6f}", flush=True)

    reg, mf = out["regular"]["mmd"], out["mlgd_f"]["mmd"]
    # eval-noise floor: MMD of two disjoint halves of the same sample vs targets
    def half_mmd(e):
        n = e.shape[0] // 2
        return (compute_mmd(e[:n], targets.cpu()).item(), compute_mmd(e[n:], targets.cpu()).item())
    floor = {k: half_mmd(v["embs"]) for k, v in out.items()}

    npy = os.path.join(args.run_dir, "npy"); os.makedirs(npy, exist_ok=True)
    for k, v in out.items():
        np.save(os.path.join(npy, f"eval_clip_{k}.npy"), v["embs"].numpy())
        pd = os.path.join(args.run_dir, f"photos_{k}"); os.makedirs(pd, exist_ok=True)
        for i, ph in enumerate(v["photos"][:args.save_photos]):
            ph.save(os.path.join(pd, f"photo_{i:03d}.png"))

    result = dict(run)
    result.update({
        "final_mlgd_f_mmd": mf, "final_regular_mmd": reg, "mmd_delta": reg - mf,
        "eval_n_final": args.eval_n, "eval_seed_base": eval_seed,
        "eval_batch_size": args.eval_batch_size, "eval_source": {"scribbles": src, "targets": tsrc},
        "eval_half_split_mmd": floor, "eval_time_sec": time.time() - t0,
        "evaluated_by": "experiments/model-optimization/sd/eval_final.py",
    })
    with open(os.path.join(args.run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Regular MMD : {reg:.6f}\nMLGD-F MMD  : {mf:.6f}\nDelta       : {reg - mf:.6f}\n"
          f"half-split floor: {floor}\nmetrics.json written to {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
