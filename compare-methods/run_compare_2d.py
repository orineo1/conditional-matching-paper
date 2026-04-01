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
import time

import torch
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
compare_dir = script_dir
sys.path.insert(0, compare_dir)

from dist_utils import compute_conditionals, compute_alpha, filter_and_normalize, generate_mog_samples_not_differentiable
from Diffusion import DiffusionModel
from ConsistencyModels import ConsistencyModeliCT
from Optimization import optimize_LGD
from LossFunctions import MMDLoss, RBF


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mog(device):
    mog_json = os.path.join(compare_dir, "mog_2d.json")
    with open(mog_json) as f:
        mog_cfg = json.load(f)
    mu_list    = [torch.tensor(m, dtype=torch.float32, device=device) for m in mog_cfg["mu_list"]]
    Sigma_list = [torch.tensor(s, dtype=torch.float32, device=device) for s in mog_cfg["Sigma_list"]]
    alpha      = torch.tensor(mog_cfg["alpha"], dtype=torch.float32, device=device)
    x_star_val = mog_cfg["splits"]["cond1_y1"]["x_star"][0]
    return mu_list, Sigma_list, alpha, x_star_val


def load_models(models_dir, device):
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


def compute_conditional_mmd(x_pred_val, x_star_val, mu_list, Sigma_list, alpha,
                             device, n_samples=1000):
    """
    Real distance metric:
    sample 1000 points from p(y | x=x_pred) and p(y | x=x_star)
    and compute MMD between them.
    """
    mmd_fn = MMDLoss(kernel=RBF(device=device), device=device)

    x_pred_t = torch.tensor([x_pred_val], device=device)
    x_star_t = torch.tensor([x_star_val], device=device)

    # p(y | x = x_pred)
    mu_pred, Sigma_pred = compute_conditionals(mu_list, Sigma_list, x_pred_t)
    alpha_pred          = compute_alpha(mu_list, Sigma_list, alpha, x_pred_t)
    mu_pred, Sigma_pred, alpha_pred = filter_and_normalize(
        mu_pred, Sigma_pred, alpha_pred, threshold=0.01)
    samples_pred = generate_mog_samples_not_differentiable(
        n_samples, mu_pred, Sigma_pred, alpha_pred).to(device)

    # p(y | x = x_star)
    mu_star, Sigma_star = compute_conditionals(mu_list, Sigma_list, x_star_t)
    alpha_star          = compute_alpha(mu_list, Sigma_list, alpha, x_star_t)
    mu_star, Sigma_star, alpha_star = filter_and_normalize(
        mu_star, Sigma_star, alpha_star, threshold=0.01)
    samples_star = generate_mog_samples_not_differentiable(
        n_samples, mu_star, Sigma_star, alpha_star).to(device)

    return mmd_fn(samples_pred, samples_star).item()


def print_summary(label, results, x_star_val):
    l1s       = [r["l1"]        for r in results]
    l2s       = [r["l2"]        for r in results]
    cond_mmds = [r["cond_mmd"]  for r in results]
    opt_mmds  = [r["opt_mmd"]   for r in results]
    times     = [r["time"]      for r in results]

    # top-10 selected by optimization MMD (proxy used during optimization)
    top10        = sorted(results, key=lambda r: r["opt_mmd"])[:10]
    top10_l1s       = [r["l1"]       for r in top10]
    top10_l2s       = [r["l2"]       for r in top10]
    top10_cond_mmds = [r["cond_mmd"] for r in top10]
    top10_opt_mmds  = [r["opt_mmd"]  for r in top10]
    top10_times     = [r["time"]     for r in top10]

    print(f"\n{'='*60}")
    print(f"{label}  —  ALL {len(results)} attempts:")
    print(f"  L1  (|x_pred - x*|):              {np.mean(l1s):.4f} ± {np.std(l1s):.4f}")
    print(f"  L2  (|x_pred - x*|^2):            {np.mean(l2s):.4f} ± {np.std(l2s):.4f}")
    print(f"  cond MMD  (real distance):         {np.mean(cond_mmds):.6f} ± {np.std(cond_mmds):.6f}")
    print(f"  opt  MMD  (optim proxy):           {np.mean(opt_mmds):.6f} ± {np.std(opt_mmds):.6f}")
    print(f"  time:                              {np.mean(times):.1f}s ± {np.std(times):.1f}s")
    print(f"\n{label}  —  TOP-10 by opt MMD:")
    print(f"  L1  (|x_pred - x*|):              {np.mean(top10_l1s):.4f} ± {np.std(top10_l1s):.4f}")
    print(f"  L2  (|x_pred - x*|^2):            {np.mean(top10_l2s):.4f} ± {np.std(top10_l2s):.4f}")
    print(f"  cond MMD  (real distance):         {np.mean(top10_cond_mmds):.6f} ± {np.std(top10_cond_mmds):.6f}")
    print(f"  opt  MMD  (optim proxy):           {np.mean(top10_opt_mmds):.6f} ± {np.std(top10_opt_mmds):.6f}")
    print(f"  time:                              {np.mean(top10_times):.1f}s ± {np.std(top10_times):.1f}s")
    print(f"{'='*60}\n")


def run_once(model_uncond, model_cond_or_cm, mog_means, mog_variances, weights,
             mu_list, Sigma_list, alpha, seed, nsamples, num_x_t, cm_flag, device,
             attempt_idx, n_attempts, method_label, x_star_val):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  [{attempt_idx+1:02d}/{n_attempts}] {method_label} | seed={seed} | running optimization...")

    t0 = time.time()
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cond_or_cm,
        mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, num_x_t=num_x_t,
        loss="MMD", CM=cm_flag,
        device=device, FLAG=False,
    )
    elapsed = time.time() - t0

    x_pred   = best_x_t.flatten()[0].item()
    opt_mmd  = final_loss.item()
    l1       = abs(x_pred - x_star_val)
    l2       = (x_pred - x_star_val) ** 2

    # real distance: MMD between p(y|x_pred) and p(y|x_star)
    cond_mmd = compute_conditional_mmd(
        x_pred, x_star_val, mu_list, Sigma_list, alpha, device, n_samples=1000)

    print(f"  [{attempt_idx+1:02d}/{n_attempts}] {method_label} | seed={seed} | "
          f"x_pred={x_pred:.4f}  x*={x_star_val:.4f}  "
          f"L1={l1:.4f}  L2={l2:.4f}  "
          f"cond_MMD={cond_mmd:.6f}  opt_MMD={opt_mmd:.6f}  "
          f"t={elapsed:.1f}s")

    return x_pred, opt_mmd, cond_mmd, l1, l2, elapsed


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True,
                   help="Path to trained models dir, e.g. compare-methods/output/models_2d_44381622")
    p.add_argument("--n_attempts", type=int, default=25)
    p.add_argument("--nsamples",   type=int, default=250)
    p.add_argument("--num_x_t",   type=int, default=3)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--n_samples_mmd", type=int, default=1000,
                   help="Samples for conditional MMD evaluation")
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
        print(f"{'='*60}")
        print(f"Running LGD (diffusion)  x{args.n_attempts}")
        print(f"{'='*60}")
        for i, seed in enumerate(seeds):
            x_pred, opt_mmd, cond_mmd, l1, l2, elapsed = run_once(
                model_uncond, model_cond, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha,
                seed=seed, nsamples=args.nsamples, num_x_t=args.num_x_t,
                cm_flag=False, device=device,
                attempt_idx=i, n_attempts=args.n_attempts,
                method_label="LGD", x_star_val=x_star_val,
            )
            lgd_results.append({"seed": seed, "x_pred": x_pred,
                                 "opt_mmd": opt_mmd, "cond_mmd": cond_mmd,
                                 "l1": l1, "l2": l2, "time": elapsed})
        print_summary("LGD", lgd_results, x_star_val)

    # ── LGD-CM (consistency model) ────────────────────────────────────────────
    lgdcm_results = []
    if not args.skip_lgdcm:
        print(f"{'='*60}")
        print(f"Running LGD-CM (consistency model)  x{args.n_attempts}")
        print(f"{'='*60}")
        for i, seed in enumerate(seeds):
            x_pred, opt_mmd, cond_mmd, l1, l2, elapsed = run_once(
                model_uncond, model_cm, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha,
                seed=seed, nsamples=args.nsamples, num_x_t=args.num_x_t,
                cm_flag=True, device=device,
                attempt_idx=i, n_attempts=args.n_attempts,
                method_label="LGD-CM", x_star_val=x_star_val,
            )
            lgdcm_results.append({"seed": seed, "x_pred": x_pred,
                                   "opt_mmd": opt_mmd, "cond_mmd": cond_mmd,
                                   "l1": l1, "l2": l2, "time": elapsed})
        print_summary("LGD-CM", lgdcm_results, x_star_val)

    # ── save results ──────────────────────────────────────────────────────────
    out = {
        "models_dir":    args.models_dir,
        "x_star":        x_star_val,
        "n_attempts":    args.n_attempts,
        "nsamples":      args.nsamples,
        "num_x_t":       args.num_x_t,
        "n_samples_mmd": args.n_samples_mmd,
        "seeds":         seeds,
        "lgd":           lgd_results,
        "lgdcm":         lgdcm_results,
    }
    out_path = os.path.join(args.models_dir, "run_compare_2d_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()