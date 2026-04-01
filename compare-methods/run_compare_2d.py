"""
run_compare_2d.py — Run LGD and LGD-CM on the 2D MoG benchmark.

Usage:
    python run_compare_2d.py --models_dir compare-methods/output/models_2d_44381622
    python run_compare_2d.py --models_dir compare-methods/output/models_2d_NEW_JOB_ID
    python run_compare_2d.py --models_dir compare-methods/output/models_2d_44381622 --n_attempts 10 --nsamples 100
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
compare_dir = script_dir
sys.path.insert(0, compare_dir)

from dist_utils import compute_conditionals, compute_alpha, filter_and_normalize
from Diffusion import DiffusionModel
from ConsistencyModels import ConsistencyModeliCT
from Optimization import optimize_LGD


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mog(device):
    mog_json = os.path.join(compare_dir, "mog_2d.json")
    with open(mog_json) as f:
        mog_cfg = json.load(f)
    mu_list    = [torch.tensor(m, dtype=torch.float32, device=device) for m in mog_cfg["mu_list"]]
    Sigma_list = [torch.tensor(s, dtype=torch.float32, device=device) for s in mog_cfg["Sigma_list"]]
    alpha      = torch.tensor(mog_cfg["alpha"], dtype=torch.float32, device=device)
    x_star_val = mog_cfg["splits"]["cond1_y1"]["x_star"][0]  # e.g. -5.0
    return mu_list, Sigma_list, alpha, x_star_val


def load_models(models_dir, device):
    """Support both new (split subdir) and old (flat) layout."""
    split_dir = os.path.join(models_dir, "cond1_y1")
    if os.path.exists(os.path.join(split_dir, "split_config.json")):
        model_dir = split_dir
        with open(os.path.join(model_dir, "split_config.json")) as f:
            cfg = json.load(f)
    else:
        model_dir = models_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

    nblocks, nunits, diff_steps = cfg["nblocks"], cfg["nunits"], cfg["diffusion_steps"]
    print(f"Model dir:  {model_dir}")
    print(f"nblocks={nblocks}, nunits={nunits}, diffusion_steps={diff_steps}")

    model_uncond = DiffusionModel(
        nfeatures=1, nblocks=nblocks, nunits=nunits,
        condition=False, diffusion_steps=diff_steps,
    )
    model_uncond.load_state_dict(torch.load(
        os.path.join(model_dir, "model_uncond.pt"), map_location=device, weights_only=False))
    model_uncond.to(device).eval()

    model_cond = DiffusionModel(
        nfeatures=2, nblocks=nblocks, nunits=nunits,
        condition=True, condition_on=1, diffusion_steps=diff_steps,
    )
    model_cond.load_state_dict(torch.load(
        os.path.join(model_dir, "model_cond.pt"), map_location=device, weights_only=False))
    model_cond.to(device).eval()

    model_cm = ConsistencyModeliCT(nfeatures=1, condition_on=1, nunits=nunits)
    model_cm.load_state_dict(torch.load(
        os.path.join(model_dir, "model_cm.pt"), map_location=device, weights_only=False))
    model_cm.to(device).eval()

    return model_uncond, model_cond, model_cm


def run_once(model_uncond, model_cond_or_cm, mog_means, mog_variances, weights,
             mu_list, Sigma_list, alpha, seed, nsamples, num_x_t, cm_flag, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cond_or_cm,
        mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, num_x_t=num_x_t,
        loss="MMD", CM=cm_flag,
        device=device, FLAG=False,
    )
    return best_x_t.flatten()[0].item(), final_loss.item()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True,
                   help="Path to the trained models directory, e.g. compare-methods/output/models_2d_44381622")
    p.add_argument("--n_attempts", type=int, default=25)
    p.add_argument("--nsamples",   type=int, default=250)
    p.add_argument("--num_x_t",   type=int, default=3)
    p.add_argument("--base_seed", type=int, default=0,
                   help="Seeds used are base_seed, base_seed+1, ..., base_seed+n_attempts-1")
    p.add_argument("--skip_lgd",   action="store_true")
    p.add_argument("--skip_lgdcm", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    mu_list, Sigma_list, alpha, x_star_val = load_mog(device)
    model_uncond, model_cond, model_cm     = load_models(args.models_dir, device)

    x_star = torch.tensor([x_star_val], device=device)
    mu_temp, Sigma_temp = compute_conditionals(mu_list, Sigma_list, x_star)
    alpha_temp          = compute_alpha(mu_list, Sigma_list, alpha, x_star)
    mog_means, mog_variances, weights = filter_and_normalize(
        mu_temp, Sigma_temp, alpha_temp, threshold=0.01)

    print(f"x* = {x_star_val}")
    print(f"Target: {len(mog_means)} active components")
    for i, (m, w) in enumerate(zip(mog_means, weights)):
        print(f"  component {i}: mean={m.tolist()}  weight={w.item():.4f}")
    print()

    seeds = list(range(args.base_seed, args.base_seed + args.n_attempts))

    # ── LGD (diffusion) ───────────────────────────────────────────────────────
    lgd_results = []
    if not args.skip_lgd:
        print(f"{'='*50}")
        print(f"Running LGD (diffusion)  x{args.n_attempts}")
        print(f"{'='*50}")
        for i, seed in enumerate(seeds):
            x_pred, loss = run_once(
                model_uncond, model_cond, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha,
                seed=seed, nsamples=args.nsamples, num_x_t=args.num_x_t,
                cm_flag=False, device=device,
            )
            l1 = abs(x_pred - x_star_val)
            lgd_results.append({"seed": seed, "x_pred": x_pred, "loss": loss, "l1": l1})
            print(f"  [{i+1:02d}/{args.n_attempts}] seed={seed}  x={x_pred:.4f}  "
                  f"loss={loss:.6f}  L1={l1:.4f}")

        l1s  = [r["l1"]  for r in lgd_results]
        losses = [r["loss"] for r in lgd_results]
        print(f"\nLGD summary  —  L1: {np.mean(l1s):.4f} ± {np.std(l1s):.4f} "
              f"| loss: {np.mean(losses):.6f} ± {np.std(losses):.6f}\n")

    # ── LGD-CM (consistency model) ────────────────────────────────────────────
    lgdcm_results = []
    if not args.skip_lgdcm:
        print(f"{'='*50}")
        print(f"Running LGD-CM (consistency model)  x{args.n_attempts}")
        print(f"{'='*50}")
        for i, seed in enumerate(seeds):
            x_pred, loss = run_once(
                model_uncond, model_cm, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha,
                seed=seed, nsamples=args.nsamples, num_x_t=args.num_x_t,
                cm_flag=True, device=device,
            )
            l1 = abs(x_pred - x_star_val)
            lgdcm_results.append({"seed": seed, "x_pred": x_pred, "loss": loss, "l1": l1})
            print(f"  [{i+1:02d}/{args.n_attempts}] seed={seed}  x={x_pred:.4f}  "
                  f"loss={loss:.6f}  L1={l1:.4f}")

        l1s    = [r["l1"]   for r in lgdcm_results]
        losses = [r["loss"] for r in lgdcm_results]
        print(f"\nLGD-CM summary  —  L1: {np.mean(l1s):.4f} ± {np.std(l1s):.4f} "
              f"| loss: {np.mean(losses):.6f} ± {np.std(losses):.6f}\n")

    # ── save results ──────────────────────────────────────────────────────────
    out = {
        "models_dir": args.models_dir,
        "x_star":     x_star_val,
        "n_attempts": args.n_attempts,
        "nsamples":   args.nsamples,
        "num_x_t":    args.num_x_t,
        "seeds":      seeds,
        "lgd":        lgd_results,
        "lgdcm":      lgdcm_results,
    }
    out_path = os.path.join(args.models_dir, "run_compare_2d_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
