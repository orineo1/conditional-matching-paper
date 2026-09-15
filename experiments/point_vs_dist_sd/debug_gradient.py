"""
debug_gradient.py — three cheap GPU probes that decide WHY the guidance objective
does not descend (diagnosis: DEBUG_B.md). Runs at ONE fixed latent; no trajectory.

P1  GRADIENT SNR. Compute the guidance gradient R times at the SAME latent with
    DIFFERENT sprinter seed sets. Report pairwise cosines and
    SNR = ||mean(g)|| / mean(||g||). Noise-dominated <=> cos ~ 0, SNR ~ 1/sqrt(R).
P2  PAIRED LINE SEARCH. Move x <- x - lambda*zeta*g and re-evaluate the loss
    (a) with the SAME sprinter seeds used for the gradient (common random numbers:
    isolates the true local descent) and (b) with FRESH seeds (what the run sees).
    A correct, useful gradient descends in (a) at some lambda.
P3  fp16 UNDERFLOW. Recompute the gradient with loss_scale in {1, 1e2, 1e4}
    (rescaling afterwards). Directions must agree; if they do not, the fp16
    backward is underflowing and --loss_scale is not optional.

Usage (see submit_debug.sh):
  python debug_gradient.py --cache cache/B --out debug/B [--n_cond 100 --repeats 4]
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                        # noqa: E402  (puts SD src on path)
from common import NEUTRAL_PROMPT, SEED              # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_cond", type=int, default=100,
                   help="conditional samples per GRADIENT (the arm's own setting: 100 for mmd, 1 for point)")
    p.add_argument("--eval_n_cond", type=int, default=None,
                   help="conditional samples used only to MEASURE the loss along the P2 line "
                        "search (default = --n_cond). With n_cond=1 (point arm) a single sample "
                        "cannot resolve the landscape, so use e.g. 32: the gradient stays the "
                        "arm's own 1-sample estimate, only the readout is averaged.")
    p.add_argument("--loss_fn", default="mmd", choices=["mmd", "point"],
                   help="point: ||CLIP(y) - y*||^2 against cache/point_target.pt (arm 2)")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--base_zeta", type=float, default=4.0)
    p.add_argument("--cn_scale", type=float, default=0.5)
    p.add_argument("--n_steps", type=int, default=250)
    p.add_argument("--start_step", type=int, default=125)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    import torchvision.transforms.functional as TF
    from PIL import Image
    from clip_utils import load_clip_model
    from generation import compute_pred_x0_direct, predict_noise_cfg, variation_objective
    from metrics import compute_mmd
    from models import load_models, setup_gradient_checkpointing

    torch.manual_seed(args.seed)
    architect, sprinter = load_models(dev)
    clip_model, clip_processor = load_clip_model(dev)
    sprinter.vae.to(dtype=torch.float32)
    setup_gradient_checkpointing(architect, sprinter)

    T = torch.load(os.path.join(args.cache, "targets_clip.pt"), map_location="cpu")
    targets = T["all_clip_embeddings"].to(dev)
    scribble = Image.fromarray(np.load(os.path.join(args.cache, "targets_cache.npz"))["scribble"])

    with torch.no_grad():
        pe, npe, ppe, nppe = architect.encode_prompt(prompt="", negative_prompt="", device=dev,
                                                     do_classifier_free_guidance=True,
                                                     num_images_per_prompt=1)
    architect.scheduler.set_timesteps(args.n_steps, device=dev)
    ts = architect.scheduler.timesteps
    add_time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=pe.dtype, device=dev)
    added = {"text_embeds": torch.cat([nppe, ppe], 0), "time_ids": add_time_ids.repeat(2, 1)}
    enc = torch.cat([npe, pe], 0)

    with torch.no_grad():
        xt = (TF.to_tensor(scribble).unsqueeze(0).to(dev).float() * 2 - 1)
        lat0 = architect.vae.encode(xt).latent_dist.mean * architect.vae.config.scaling_factor
    t = ts[args.start_step]
    ab = architect.scheduler.alphas_cumprod.to(dev)[t.long()].float()
    noise = torch.randn(lat0.shape, generator=torch.Generator(device=dev).manual_seed(args.seed * 7919 + 1),
                        device=dev, dtype=lat0.dtype)
    latents = ((ab ** 0.5) * lat0 + ((1 - ab) ** 0.5) * noise).to(torch.float16)
    print(f"probe latent at step {args.start_step}/{args.n_steps} t={int(t)} "
          f"||x||={float(latents.norm()):.1f}", flush=True)

    if args.loss_fn == "point":
        from functools import partial
        from metrics import compute_point_loss
        y_star = torch.load(os.path.join(args.cache, "point_target.pt"),
                            map_location="cpu")["y_star"].to(dev)
        loss_fn = partial(compute_point_loss, point_target=y_star)
        print(f"point loss vs y* (norm {float(y_star.norm()):.4f})", flush=True)
    else:
        loss_fn = compute_mmd
    eval_n_cond = args.eval_n_cond or args.n_cond

    def grad_at(x, seeds, loss_scale=1.0, want_loss_only=False):
        xs = x.detach().requires_grad_(True)
        eps = predict_noise_cfg(architect.unet, architect.scheduler, xs, t, enc, added, 0.0)
        x0 = compute_pred_x0_direct(architect.scheduler, eps, t, xs)
        px = torch.utils.checkpoint.checkpoint(
            lambda l: architect.vae.decode(l.to(architect.vae.dtype)).sample,
            x0 / architect.vae.config.scaling_factor, use_reentrant=False)
        px = torch.clamp((px + 1.0) / 2.0, 0.0, 1.0)
        if want_loss_only:
            with torch.no_grad():
                out = variation_objective(
                    latents=None, latents_step=None, noise_pred=None, pixel_x0_norm=px,
                    sprinter=sprinter, all_clip_embeddings=targets, num_variations=len(seeds),
                    variation_batch_size=1, base_zeta_prime=args.base_zeta, clip_model=clip_model,
                    clip_processor=clip_processor, vae=sprinter.vae,
                    vae_scaling_factor=sprinter.vae.config.scaling_factor,
                    variation_prompt=NEUTRAL_PROMPT, loss_fn=loss_fn, loss_scale=1.0,
                    cn_scale=args.cn_scale, variation_seeds=seeds, verbose=False)
            return float(out[1])
        out = variation_objective(
            latents=None, latents_step=None, noise_pred=None, pixel_x0_norm=px,
            sprinter=sprinter, all_clip_embeddings=targets, num_variations=len(seeds),
            variation_batch_size=1, base_zeta_prime=args.base_zeta, clip_model=clip_model,
            clip_processor=clip_processor, vae=sprinter.vae,
            vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=NEUTRAL_PROMPT, loss_fn=loss_fn, loss_scale=loss_scale,
            cn_scale=args.cn_scale, variation_seeds=seeds, verbose=False)
        g = torch.autograd.grad(out[0], xs)[0]
        return g.detach() / loss_scale, float(out[1]), float(out[2])

    res = {"config": vars(args)}
    seedset = lambda r, n=None: [args.seed * 1_000_003 + 900_000 + r * 10_000 + j
                                 for j in range(n or args.n_cond)]

    # ---------- P1 ----------
    grads, losses = [], []
    for r in range(args.repeats):
        g, l, z = grad_at(latents, seedset(r))
        grads.append(g.float().flatten()); losses.append(l)
        print(f"P1 rep {r}: loss {l:.4f} ||g|| {float(g.norm()):.4f}", flush=True)
    G = torch.stack(grads)
    cos = [(float(torch.nn.functional.cosine_similarity(G[i], G[j], dim=0)))
           for i in range(len(G)) for j in range(i + 1, len(G))]
    snr = float(G.mean(0).norm() / G.norm(dim=1).mean())
    res["P1"] = {"losses": losses, "pairwise_cos": cos, "mean_cos": float(np.mean(cos)),
                 "snr": snr, "snr_pure_noise": 1 / np.sqrt(len(G)), "grad_norms": [float(g.norm()) for g in grads]}
    print(f"P1: mean pairwise cos {np.mean(cos):+.4f} | SNR {snr:.3f} "
          f"(pure noise would be {1/np.sqrt(len(G)):.3f})", flush=True)

    # ---------- P2 ----------
    g0, l0, z0 = grad_at(latents, seedset(0))
    step = (-z0 * g0).float()
    step_unit_norm = step.norm()
    print(f"P2: ||zeta*g|| = {float(step_unit_norm):.2f}  (probe band for the mmd arm was 5.9-23.5)", flush=True)
    p2 = {"lambdas": args.lambdas, "crn": [], "fresh": [], "zeta": z0, "base_loss": l0,
          "eval_n_cond": eval_n_cond, "step_unit_norm": float(step_unit_norm)}
    for lam in args.lambdas:
        xl = (latents.float() + lam * step).to(torch.float16)
        p2["crn"].append(grad_at(xl, seedset(0, eval_n_cond), want_loss_only=True))
        p2["fresh"].append(grad_at(xl, seedset(99, eval_n_cond), want_loss_only=True))
        print(f"P2 lambda {lam:6.1f} | CRN loss {p2['crn'][-1]:.4f} | fresh loss {p2['fresh'][-1]:.4f} "
              f"| ||step|| {lam*float(step.norm()):.2f}", flush=True)
    res["P2"] = p2

    # ---------- P3 ----------
    p3 = {}
    ref = None
    for ls in (1.0, 1e2, 1e4):
        g, l, _ = grad_at(latents, seedset(0), loss_scale=ls)
        gf = g.float().flatten()
        ref = gf if ref is None else ref
        p3[str(ls)] = {"norm": float(gf.norm()),
                       "cos_to_ls1": float(torch.nn.functional.cosine_similarity(gf, ref, dim=0)),
                       "n_zero_elems": int((g == 0).sum())}
        print(f"P3 loss_scale {ls:g}: ||g|| {p3[str(ls)]['norm']:.4f} "
              f"cos-to-ls1 {p3[str(ls)]['cos_to_ls1']:+.4f} zeros {p3[str(ls)]['n_zero_elems']}", flush=True)
    res["P3"] = p3

    with open(os.path.join(args.out, "debug_gradient.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("written", os.path.join(args.out, "debug_gradient.json"), flush=True)


if __name__ == "__main__":
    main()
