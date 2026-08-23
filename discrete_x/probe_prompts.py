"""Probe hand-crafted prompts against the mixture target: can ANY single
prompt induce a distribution that straddles the 50/50 old-man/young-woman
mixture, or does SDXL-Turbo collapse ambiguous prompts to one mode?

Evaluates L (MMD^2 vs mixture G) and demographics (p_male, p_old over 64
fresh generations) for each probe prompt.

Run on cluster:  python discrete_x/probe_prompts.py --outdir output/probe_<jobid>
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_smc import TurboImageMMDLoss

TARGETS = ["a photo of an old man smiling",
           "a photo of a young woman smiling"]

PROBES = [
    # the two components and the SMC-recovered collapse
    "a photo of an old man smiling",
    "a photo of a young woman smiling",
    "a photo of a woman smiling in the image",
    # ambiguity: does SDXL mix, or pick a mode?
    "a photo of a person smiling",
    "a photo of an adult smiling at the camera",
    # explicit disjunction / union phrasings
    "a photo of an old man or a young woman smiling",
    "a photo of an elderly man or a young woman smiling",
    "a photo of a smiling person, either an old man or a young woman",
    # compositional straddle attempts
    "a photo of an old person smiling",
    "a photo of a young person smiling",
    "a photo of an androgynous middle aged person smiling",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output/probe_prompts")
    ap.add_argument("--n_target", type=int, default=64)
    ap.add_argument("--n_cond", type=int, default=8)
    ap.add_argument("--model", choices=["turbo", "base"], default="turbo",
                    help="turbo: distilled 1-step; base: non-distilled SDXL "
                         "(tests whether distillation causes the demographic "
                         "mode collapse)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.model == "base":
        name = "stabilityai/stable-diffusion-xl-base-1.0"
        n_steps = args.steps if args.steps is not None else 12
        cfg = args.cfg if args.cfg is not None else 3.0
    else:
        name = "stabilityai/sdxl-turbo"
        n_steps = args.steps if args.steps is not None else 1
        cfg = args.cfg if args.cfg is not None else 0.0
    print(f"model={name} steps={n_steps} cfg={cfg}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = TurboImageMMDLoss(TARGETS, device, n_cond=args.n_cond,
                                n_target=args.n_target, save_dir=args.outdir,
                                turbo_name=name, n_steps=n_steps, cfg=cfg)
    tgt_demo = loss_fn.demographics(loss_fn.S_G)
    print(f"target demographics: {tgt_demo}", flush=True)

    results = []
    for p in PROBES:
        l = loss_fn(p)
        demo = loss_fn.eval_demographics(p)
        results.append({"prompt": p, "L": l, **demo})
        print(f"L={l:.3f} p_male={demo['p_male']:.2f} "
              f"p_old={demo['p_old']:.2f}  '{p}'", flush=True)

    with open(os.path.join(args.outdir, "probe_results.json"), "w") as f:
        json.dump({"targets": TARGETS, "target_demographics": tgt_demo,
                   "probes": results}, f, indent=2)
    print(f"wrote {args.outdir}/probe_results.json")


if __name__ == "__main__":
    main()
