"""
run_compare_10d.py — Run LGD and LGD-CM on the 10D MoG benchmark (cond9_y1 split).

x = dims 1-9  (9D, what we optimize over, unconditional model is 9D)
y = dim 0     (1D, the target we condition on)

The MoG is reordered so x-dims come first, matching how train_models.py
built the cond9_y1 split: condition_on=9 means the first 9 dims are x,
the last 1 dim is y.

Usage:
    python run_compare_10d.py --models_dir compare-methods/output/models_10d_<JOB_ID>
    python run_compare_10d.py --models_dir compare-methods/output/models_10d_<JOB_ID> --n_attempts 10 --nsamples 100
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

SPLIT_NAME = "cond9_y1"
MOG_JSON   = os.path.join(compare_dir, "mog_10d.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mog(device):
    with open(MOG_JSON) as f:
        mog_cfg = json.load(f)

    # Original MoG: dim0=y, dims1-9=x
    # Reorder to [dims1-9, dim0] so the first 9 dims are x (what we optimize)
    # and the last dim is y — matching how train_models.py built cond9_y1.
    mu_list_raw    = [torch.tensor(m, dtype=torch.float32, device=device) for m in mog_cfg["mu_list"]]
    Sigma_list_raw = [torch.tensor(s, dtype=torch.float32, device=device) for s in mog_cfg["Sigma_list"]]
    alpha          = torch.tensor(mog_cfg["alpha"], dtype=torch.float32, device=device)

    # Permutation: [1,2,3,4,5,6,7,8,9,0]
    perm = list(range(1, 10)) + [0]
    mu_list    = [m[perm] for m in mu_list_raw]
    Sigma_list = [S[perm][:, perm] for S in Sigma_list_raw]

    split_cfg  = mog_cfg["splits"][SPLIT_NAME]
    # x_star from JSON is dims 1-9 of the sampled joint point (already in x order)
    x_star_val = split_cfg["x_star"]   # list of length 9

    return mu_list, Sigma_list, alpha, x_star_val, split_cfg


def load_models(models_dir, device):
    split_dir = os.path.join(models_dir, SPLIT_NAME)
    cfg_path  = os.path.join(split_dir, "split_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"split_config.json not found at {cfg_path}. "
            f"Did you run train_models.py with --mog_json mog_10d.json?"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    nblocks      = cfg["nblocks"]
    nunits       = cfg["nunits"]
    diff_steps   = cfg["diffusion_steps"]
    condition_on = cfg["condition_on"]   # 9  (x dims)
    nfeatures    = cfg["nfeatures"]      # 10
    nfeatures_y  = cfg["nfeatures_y"]    # 1  (y dim)

    print(f"Split dir:    {split_dir}")
    print(f"condition_on={condition_on} (x=9D), nfeatures_y={nfeatures_y} (y=1D), "
          f"nblocks={nblocks}, nunits={nunits}, diffusion_steps={diff_steps}")

    # Unconditional diffusion over x (9 dims)
    model_uncond = DiffusionModel(
        nfeatures=condition_on, nblocks=nblocks, nunits=nunits,
        condition=False, diffusion_steps=diff_steps,
    )
    model_uncond.load_state_dict(torch.load(
        os.path.join(split_dir, "model_uncond.pt"), map_location=device, weights_only=False))
    model_uncond.to(device).eval()

    # Conditional diffusion p(joint | x) — nfeatures=10, condition_on=9
    model_cond = DiffusionModel(
        nfeatures=nfeatures, nblocks=nblocks, nunits=nunits,
        condition=True, condition_on=condition_on, diffusion_steps=diff_steps,
    )
    model_cond.load_state_dict(torch.load(
        os.path.join(split_dir, "model_cond.pt"), map_location=device, weights_only=False))
    model_cond.to(device).eval()

    # Consistency model over y (1 dim), conditioned on x (9 dims)
    model_cm = ConsistencyModeliCT(
        nfeatures=nfeatures_y, condition_on=condition_on, nunits=nunits)
    model_cm.load_state_dict(torch.load(
        os.path.join(split_dir, "model_cm.pt"), map_location=device, weights_only=False))
    model_cm.to(device).eval()

    return model_uncond, model_cond, model_cm, cfg


def compute_conditional_mmd(x_pred_tensor, x_star_tensor,
                             mu_list, Sigma_list, alpha, device, n_samples=1000):
    """
    MMD between p(y | x=x_pred) and p(y | x=x_star).
    Both tensors are shape (9,) in the reordered space (x first).
    compute_conditionals treats first 9 dims as x, last 1 as y.
    """
    mmd_fn = MMDLoss(kernel=RBF(device=device), device=device)

    mu_pred, Sigma_pred = compute_conditionals(mu_list, Sigma_list, x_pred_tensor)
    alpha_pred          = compute_alpha(mu_list, Sigma_list, alpha, x_pred_tensor)
    mu_pred, Sigma_pred, alpha_pred = filter_and_normalize(
        mu_pred, Sigma_pred, alpha_pred, threshold=0.01)
    samples_pred = generate_mog_samples_not_differentiable(
        n_samples, mu_pred, Sigma_pred, alpha_pred).to(device)

    mu_star, Sigma_star = compute_conditionals(mu_list, Sigma_list, x_star_tensor)
    alpha_star          = compute_alpha(mu_list, Sigma_list, alpha, x_star_tensor)
    mu_star, Sigma_star, alpha_star = filter_and_normalize(
        mu_star, Sigma_star, alpha_star, threshold=0.01)
    samples_star = generate_mog_samples_not_differentiable(
        n_samples, mu_star, Sigma_star, alpha_star).to(device)

    return mmd_fn(samples_pred, samples_star).item()


def print_summary(label, results, x_star_val):
    l1s       = [r["l1"]       for r in results]
    l2s       = [r["l2"]       for r in results]
    cond_mmds = [r["cond_mmd"] for r in results]
    opt_mmds  = [r["opt_mmd"]  for r in results]
    times     = [r["time"]     for r in results]

    top10           = sorted(results, key=lambda r: r["opt_mmd"])[:10]
    top10_l1s       = [r["l1"]       for r in top10]
    top10_l2s       = [r["l2"]       for r in top10]
    top10_cond_mmds = [r["cond_mmd"] for r in top10]
    top10_opt_mmds  = [r["opt_mmd"]  for r in top10]
    top10_times     = [r["time"]     for r in top10]

    print(f"\n{'='*60}")
    print(f"{label}  —  ALL {len(results)} attempts:")
    print(f"  L1  |x_pred - x*| (9D):       {np.mean(l1s):.4f} ± {np.std(l1s):.4f}")
    print(f"  L2  |x_pred - x*|² (9D):      {np.mean(l2s):.4f} ± {np.std(l2s):.4f}")
    print(f"  cond MMD  (real distance):     {np.mean(cond_mmds):.6f} ± {np.std(cond_mmds):.6f}")
    print(f"  opt  MMD  (optim proxy):       {np.mean(opt_mmds):.6f} ± {np.std(opt_mmds):.6f}")
    print(f"  time:                          {np.mean(times):.1f}s ± {np.std(times):.1f}s")
    print(f"\n{label}  —  TOP-10 by opt MMD:")
    print(f"  L1  |x_pred - x*| (9D):       {np.mean(top10_l1s):.4f} ± {np.std(top10_l1s):.4f}")
    print(f"  L2  |x_pred - x*|² (9D):      {np.mean(top10_l2s):.4f} ± {np.std(top10_l2s):.4f}")
    print(f"  cond MMD  (real distance):     {np.mean(top10_cond_mmds):.6f} ± {np.std(top10_cond_mmds):.6f}")
    print(f"  opt  MMD  (optim proxy):       {np.mean(top10_opt_mmds):.6f} ± {np.std(top10_opt_mmds):.6f}")
    print(f"  time:                          {np.mean(top10_times):.1f}s ± {np.std(top10_times):.1f}s")
    print(f"{'='*60}\n")


def run_once(model_uncond, model_cond_or_cm, mog_means, mog_variances, weights,
             mu_list, Sigma_list, alpha, x_star_tensor,
             seed, nsamples, num_x_t, cm_flag, device,
             attempt_idx, n_attempts, method_label, x_star_val):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  [{attempt_idx+1:02d}/{n_attempts}] {method_label} | seed={seed} | running optimization...")

    t0 = time.time()
    # optimize_LGD returns (best_x_t, best_x_t, final_loss)
    # best_x_t shape: (1, 9) — the predicted x in 9D
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cond_or_cm,
        mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, num_x_t=num_x_t,
        loss="MMD", CM=cm_flag,
        device=device, FLAG=False,
    )
    elapsed = time.time() - t0

    x_pred_tensor = best_x_t.flatten()[:9].to(device)   # shape (9,)
    opt_mmd       = final_loss.item() if isinstance(final_loss, torch.Tensor) else float(final_loss)

    # L1 / L2 in 9D x-space
    x_star_t = x_star_tensor.to(x_pred_tensor.device)
    l1 = (x_pred_tensor - x_star_t).abs().sum().item()
    l2 = (x_pred_tensor - x_star_t).pow(2).sum().sqrt().item()

    # Real quality: MMD between p(y | x_pred) and p(y | x_star)
    cond_mmd = compute_conditional_mmd(
        x_pred_tensor, x_star_tensor,
        mu_list, Sigma_list, alpha,
        device=device, n_samples=1000,
    )

    print(f"  [{attempt_idx+1:02d}/{n_attempts}] {method_label} | seed={seed} | "
          f"L1={l1:.4f}  L2={l2:.4f}  "
          f"cond_MMD={cond_mmd:.6f}  opt_MMD={opt_mmd:.6f}  "
          f"t={elapsed:.1f}s")

    return x_pred_tensor.cpu().tolist(), opt_mmd, cond_mmd, l1, l2, elapsed


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True,
                   help="Path to trained models dir, e.g. compare-methods/output/models_10d_<JOB_ID>")
    p.add_argument("--n_attempts",    type=int, default=25)
    p.add_argument("--nsamples",      type=int, default=250)
    p.add_argument("--num_x_t",       type=int, default=3)
    p.add_argument("--base_seed",     type=int, default=0)
    p.add_argument("--n_samples_mmd", type=int, default=1000,
                   help="Samples for conditional MMD evaluation")
    p.add_argument("--skip_lgd",   action="store_true")
    p.add_argument("--skip_lgdcm", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    mu_list, Sigma_list, alpha, x_star_val, split_cfg = load_mog(device)
    model_uncond, model_cond, model_cm, cfg = load_models(args.models_dir, device)

    # x_star: length-9 tensor in reordered space (dims 1-9 of the original joint)
    x_star_tensor = torch.tensor(x_star_val, dtype=torch.float32, device=device)  # shape (9,)

    # Build target conditional distribution q(y | x = x_star)
    mu_temp, Sigma_temp = compute_conditionals(mu_list, Sigma_list, x_star_tensor)
    alpha_temp          = compute_alpha(mu_list, Sigma_list, alpha, x_star_tensor)
    mog_means, mog_variances, weights = filter_and_normalize(
        mu_temp, Sigma_temp, alpha_temp, threshold=0.01)

    print(f"x* = {x_star_val}  (cond9_y1 — x=dims1-9, y=dim0)")
    print(f"Target conditional p(y|x*): {len(mog_means)} active components")
    for i, (m, w) in enumerate(zip(mog_means, weights)):
        print(f"  component {i}: mean={[f'{v:.3f}' for v in m.tolist()]}  weight={w.item():.4f}")
    print()

    seeds = list(range(args.base_seed, args.base_seed + args.n_attempts))

    # ── LGD (diffusion) ───────────────────────────────────────────────────────
    lgd_results = []
    if not args.skip_lgd:
        print(f"{'='*60}")
        print(f"Running LGD (diffusion)  x{args.n_attempts}  [10D cond9_y1]")
        print(f"{'='*60}")
        for i, seed in enumerate(seeds):
            x_pred, opt_mmd, cond_mmd, l1, l2, elapsed = run_once(
                model_uncond, model_cond,
                mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha, x_star_tensor,
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
        print(f"Running LGD-CM (consistency model)  x{args.n_attempts}  [10D cond9_y1]")
        print(f"{'='*60}")
        for i, seed in enumerate(seeds):
            x_pred, opt_mmd, cond_mmd, l1, l2, elapsed = run_once(
                model_uncond, model_cm,
                mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha, x_star_tensor,
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
        "split":         SPLIT_NAME,
        "x_star":        x_star_val,
        "n_attempts":    args.n_attempts,
        "nsamples":      args.nsamples,
        "num_x_t":       args.num_x_t,
        "n_samples_mmd": args.n_samples_mmd,
        "seeds":         seeds,
        "lgd":           lgd_results,
        "lgdcm":         lgdcm_results,
    }
    out_path = os.path.join(args.models_dir, "run_compare_10d_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
