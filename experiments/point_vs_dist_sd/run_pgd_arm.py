"""
run_pgd_arm.py — Arm 1: AdvI2I-style adversarial optimization of the scribble
(no diffusion prior).  Direct Adam/PGD on the PIXEL-space scribble x inside an
L_inf eps-ball around x0 with per-step clipping (AdvI2I's constraint + eps grid;
our adaptation optimizes x directly instead of a VAE generator and matches the
CLIP image embedding of f_phi's output to y* — mapping table in README.md).

Per step: ONE sprinter sample (seed = pgd_seed(seed, step), fresh each step) ->
CLIP -> ||e - y*||^2 -> backward to x -> Adam -> project (eps-ball AND [0,1]).

Output dir follows the run_mlgd_f contract so eval_final.py works unchanged:
final_scribble_mlgd_f.png (optimized x*), final_scribble_regular.png (x0),
target_clip_embeddings.pt (copied from the cache), metrics_partial.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import NEUTRAL_PROMPT, SEED, pgd_seed, project_linf


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="scenario cache dir from setup_targets.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eps_255", type=int, default=32, help="L_inf radius in /255 units (AdvI2I grid: 32|64|128)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=None, help="Adam lr; default eps/10")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cn_scale", type=float, default=0.5)
    p.add_argument("--prompt", default=NEUTRAL_PROMPT)
    return p.parse_args()


def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    eps = args.eps_255 / 255.0
    lr = args.lr if args.lr is not None else eps / 10.0

    from PIL import Image
    import torchvision.transforms.functional as TF
    from clip_utils import load_clip_model

    _, _, scribble_arr, _ = common.read_cache(args.cache)
    x0_pil = Image.fromarray(scribble_arr)
    x0 = TF.to_tensor(x0_pil).unsqueeze(0).to(device).float()          # [1,3,512,512] in [0,1]
    y_star = torch.load(os.path.join(args.cache, "point_target.pt"),
                        map_location="cpu")["y_star"].to(device).float()

    sprinter = common.load_sprinter_grad(device)
    clip_model, clip_processor = load_clip_model(device)

    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    curve = []
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        x = (x0 + delta).clamp(0.0, 1.0)
        emb = torch.utils.checkpoint.checkpoint(
            lambda ctrl: common.sprinter_clip_embed(
                sprinter, clip_model, clip_processor, ctrl,
                pgd_seed(args.seed, step), args.prompt, args.cn_scale, device),
            x, use_reentrant=False)
        loss = ((emb - y_star) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data = project_linf(delta.data, x0, eps)
        curve.append({"step": step, "loss": float(loss),
                      "delta_linf": float(delta.detach().abs().max()),
                      "delta_l2": float(delta.detach().norm())})
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  pgd step {step}/{args.steps} loss={float(loss):.5f} "
                  f"|d|_inf={curve[-1]['delta_linf']:.4f}", flush=True)

    with torch.no_grad():
        x_star = (x0 + delta).clamp(0.0, 1.0)
    TF.to_pil_image(x_star.squeeze(0).cpu()).save(
        os.path.join(args.output_dir, "final_scribble_mlgd_f.png"))
    x0_pil.save(os.path.join(args.output_dir, "final_scribble_regular.png"))
    shutil.copy(os.path.join(args.cache, "targets_clip.pt"),
                os.path.join(args.output_dir, "target_clip_embeddings.pt"))
    with open(os.path.join(args.output_dir, "metrics_partial.json"), "w") as f:
        json.dump({"args": {"arm": "pgd", "seed": args.seed, "eps_255": args.eps_255,
                            "eps": eps, "lr": lr, "steps": args.steps,
                            "variation_cn_scale": args.cn_scale,
                            "sprinter_eval_prompt": args.prompt,
                            "target_cache": os.path.abspath(args.cache)},
                   "steps": curve, "final_loss": curve[-1]["loss"],
                   "optimization_time_sec": time.time() - t0}, f, indent=2)
    print(f"done: final loss {curve[-1]['loss']:.5f} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
