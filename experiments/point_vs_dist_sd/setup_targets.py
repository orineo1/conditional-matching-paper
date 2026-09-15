"""
setup_targets.py — build the SHARED, byte-identical inputs of one scenario
(run once per scenario on GPU, before any arm):

Scenario A (oracle recovery — RUN FIRST; target reachable by construction):
    y_f  = ONE generated female portrait,  y* = CLIP(y_f),
    x_f  = HED scribble of y_f (a zero-loss solution by construction),
    G    = CLIP of 100 samples of f_phi(x_f, .) with the neutral prompt.
    oracle.pt: L(x_f) (mean point loss over fresh samples), x_f VAE latent.

Scenario B (the gender/G_bal case; run after A):
    x0   = HED scribble of a generated male source portrait (the pipeline's
           build_targets_gender recipe, seeded),
    G    = G_bal: 50 Man + 50 Woman portraits conditioned on x0,
    y*   = CLIP centroid of G_bal (= 0.5 mu_male + 0.5 mu_female at 50/50).
Outputs in <cache_root>/<scenario>/:
    targets_cache.npz + target_groups.json   (run_mlgd_f --target_cache format)
    targets_clip.pt   {all_clip_embeddings [N,768], group_names, group_sizes}
    point_target.pt   {y_star [1,768], mode, meta}
    text_features.pt  {features [2,768] (man, woman), prompts}   (for build_report)
    x0.png            (the scribble arm 1 optimizes; == npz['scribble'])
    scenario A only:  y_f.png, oracle.pt {L_xf, L_xf_samples, x_f_latent, y_f_clip}

Usage: python setup_targets.py --scenario A [--smoke]   # A FIRST, then B
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import (EXP_DIR, MAN_PROMPT, NEUTRAL_PROMPT, WOMAN_PROMPT, SEED,
                    eval_seed, write_cache)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["A", "B"])
    p.add_argument("--cache_root", default=os.path.join(EXP_DIR, "cache"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cn_scale", type=float, default=0.5, help="f_phi ControlNet scale (Appendix E)")
    p.add_argument("--n_per_group", type=int, default=10,
                   help="Scenario B targets per group. Ori's working config uses 10+10 "
                        "(the earlier 50+50 was ours). Scenario A is unaffected.")
    p.add_argument("--n_cond", type=int, default=100,
                   help="Scenario A: samples of f_phi(x_f) forming S_G.")
    p.add_argument("--smoke", action="store_true", help="8 per group / oracle 8 (format check)")
    p.add_argument("--oracle_only", action="store_true",
                   help="Scenario A: compute ONLY oracle.pt from the EXISTING cache "
                        "(scribble + y* are read back, nothing else is rewritten). Use this "
                        "when arms have already run against that cache.")
    return p.parse_args()


def _write_oracle(out, sprinter, clip_model, clip_processor, scribble, y_star,
                  args, device, n_oracle):
    """oracle.pt: L(x_f) on fresh f_phi(x_f) samples drawn with the ARMS' eval seeds,
    plus x_f's VAE latent (for the ||x* - x_f|| columns)."""
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from diffusers import AutoencoderKL
    from clip_utils import encode_images_clip
    from metrics import compute_point_loss

    def clip_of(pil_list):
        from run_mlgd_f import pil_images_to_tensor
        with torch.no_grad():
            return encode_images_clip(pil_images_to_tensor(pil_list, device),
                                      clip_model, clip_processor)

    losses = []
    # The sprinter VAE is fp32 here (common.load_sprinter_grad) but the pipeline hands it
    # fp16 latents for a PIL decode -> "Input type (c10::Half) and bias type (float)".
    # generate_and_store_cs / evaluate_distribution_mmd both cast around the call; do the
    # same. (This crash is why cache/A/oracle.pt was missing for the first A setup.)
    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    with torch.no_grad():
        for j in range(n_oracle):
            gen = torch.Generator(device=device).manual_seed(eval_seed(args.seed, j))
            r = sprinter(prompt=[NEUTRAL_PROMPT], image=[scribble],
                         num_inference_steps=2, guidance_scale=0.0,
                         controlnet_conditioning_scale=args.cn_scale,
                         output_type="pil", generator=[gen])
            e = clip_of(r.images)
            losses.append(float(compute_point_loss(e, None, point_target=y_star)))
    sprinter.vae.to(dtype=original_vae_dtype)

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                        subfolder="vae", torch_dtype=torch.float32).to(device)
    xt = (TF.to_tensor(scribble).unsqueeze(0).to(device) * 2 - 1).float()
    with torch.no_grad():
        x_f_latent = vae.encode(xt).latent_dist.mean * vae.config.scaling_factor
    torch.save({"L_xf": float(np.mean(losses)), "L_xf_samples": losses,
                "n_oracle": n_oracle, "x_f_latent": x_f_latent.cpu(),
                "y_f_clip": clip_of([scribble]).cpu()},
               os.path.join(out, "oracle.pt"))
    print(f"oracle: L(x_f) = {np.mean(losses):.5f} +- {np.std(losses)/max(len(losses)-1,1)**0.5:.5f} "
          f"over {n_oracle} fresh samples -> {out}/oracle.pt", flush=True)


def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(args.cache_root, args.scenario)
    os.makedirs(out, exist_ok=True)
    n_grp = 8 if args.smoke else args.n_per_group
    n_cond = 16 if args.smoke else args.n_cond
    n_oracle = 8 if args.smoke else 64

    from PIL import Image
    from clip_utils import encode_images_clip, load_clip_model
    from generation import generate_and_store_cs
    from image_utils import build_base_image, sobel_proxy
    from run_mlgd_f import extract_scribble_hed, pil_images_to_tensor
    import torchvision.transforms as T

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sprinter = common.load_sprinter_grad(device)          # grad-capable; used no_grad here
    clip_model, clip_processor = load_clip_model(device)

    if args.oracle_only:
        if args.scenario != "A":
            raise ValueError("--oracle_only is Scenario A only")
        _, _, scribble_arr, _ = common.read_cache(out)
        scribble = Image.fromarray(scribble_arr)
        y_star = torch.load(os.path.join(out, "point_target.pt"),
                            map_location="cpu")["y_star"].to(device)
        print(f"--oracle_only: reusing the EXISTING cache in {out} "
              f"(scribble {scribble.size}, y* norm {float(y_star.norm()):.4f}); "
              "nothing else is rewritten.", flush=True)
        _write_oracle(out, sprinter, clip_model, clip_processor, scribble, y_star,
                      args, device, n_oracle)
        return

    base_pil, base_tensor = build_base_image(device)
    with torch.no_grad():
        sobel_pil = T.ToPILImage()(sobel_proxy(base_tensor, device).squeeze(0).cpu())

    def clip_of(pil_list):
        with torch.no_grad():
            return encode_images_clip(pil_images_to_tensor(pil_list, device),
                                      clip_model, clip_processor)

    if args.scenario == "B":
        # -- x0: the build_targets_gender recipe (3 male portraits, HED of #2) --
        with torch.no_grad():
            init_imgs, _ = generate_and_store_cs(sprinter, MAN_PROMPT, sobel_pil, 3,
                                                 batch_size=2, cn_scale=args.cn_scale)
        source_img = init_imgs[2]
        scribble = extract_scribble_hed(source_img)
        groups = [("Man", MAN_PROMPT, n_grp), ("Woman", WOMAN_PROMPT, n_grp)]
        images = {}
        with torch.no_grad():
            for name, prompt, n in groups:
                imgs, _ = generate_and_store_cs(sprinter, prompt, scribble, n,
                                                batch_size=2, cn_scale=args.cn_scale)
                images[name] = [np.array(im) for im in imgs]
        embs = {n: clip_of([Image.fromarray(a) for a in images[n]]) for n, _, _ in groups}
        all_embs = torch.cat([embs[n] for n, _, _ in groups], 0)
        y_star = all_embs.mean(dim=0, keepdim=True)       # = 0.5 mu_M + 0.5 mu_W at 50/50
        mu = {n: embs[n].mean(0) for n, _, _ in groups}
        meta = {"mode": "centroid", "inter_mode_dist": float((mu["Man"] - mu["Woman"]).norm()),
                "n_per_group": n_grp}
    else:
        # -- y_f: one female portrait; x_f = HED(y_f); G = f_phi(x_f, neutral) --
        with torch.no_grad():
            yf_imgs, _ = generate_and_store_cs(sprinter, WOMAN_PROMPT, sobel_pil, 1,
                                               batch_size=1, cn_scale=args.cn_scale)
        y_f = yf_imgs[0]
        y_f.save(os.path.join(out, "y_f.png"))
        scribble = extract_scribble_hed(y_f)
        source_img = y_f
        groups = [("Cond", NEUTRAL_PROMPT, n_cond)]
        with torch.no_grad():
            cond_imgs, _ = generate_and_store_cs(sprinter, NEUTRAL_PROMPT, scribble, n_cond,
                                                 batch_size=2, cn_scale=args.cn_scale)
        images = {"Cond": [np.array(im) for im in cond_imgs]}
        all_embs = clip_of(cond_imgs)
        y_f_clip = clip_of([y_f])
        y_star = y_f_clip.clone()
        meta = {"mode": "oracle_clip", "n_cond": n_cond}

    write_cache(out, images, source_img, scribble, groups)
    scribble.save(os.path.join(out, "x0.png"))
    torch.save({"all_clip_embeddings": all_embs.cpu(),
                "group_names": [g[0] for g in groups],
                "group_sizes": [g[2] for g in groups]},
               os.path.join(out, "targets_clip.pt"))
    torch.save({"y_star": y_star.cpu(), **meta}, os.path.join(out, "point_target.pt"))

    # gender text features for build_report's p(male) (compute_clip_softmax convention)
    ti = clip_processor.tokenizer([MAN_PROMPT, WOMAN_PROMPT], return_tensors="pt",
                                  padding=True).to(device)
    with torch.no_grad():
        tf = clip_model.get_text_features(input_ids=ti["input_ids"],
                                          attention_mask=ti["attention_mask"])
        if not torch.is_tensor(tf):
            # transformers >=5 returns a model-output object here
            emb = getattr(tf, "text_embeds", None)
            if emb is None and getattr(tf, "pooler_output", None) is not None:
                # BaseModelOutputWithPooling: project the pooled text state ourselves
                emb = clip_model.text_projection(tf.pooler_output)
            if emb is None:
                raise TypeError(f"get_text_features returned {type(tf).__name__} "
                                "with neither text_embeds nor pooler_output")
            tf = emb
        tf = tf / tf.norm(dim=-1, keepdim=True)
    torch.save({"features": tf.cpu(), "prompts": [MAN_PROMPT, WOMAN_PROMPT]},
               os.path.join(out, "text_features.pt"))

    if args.scenario == "A":
        _write_oracle(out, sprinter, clip_model, clip_processor, scribble, y_star,
                      args, device, n_oracle)

    with open(os.path.join(out, "setup_meta.json"), "w") as f:
        json.dump({"scenario": args.scenario, "seed": args.seed, "cn_scale": args.cn_scale,
                   "smoke": args.smoke, "groups": [[g[0], g[2]] for g in groups],
                   "y_star_mode": meta["mode"]}, f, indent=2)
    print(f"cache written to {out}", flush=True)


if __name__ == "__main__":
    main()
