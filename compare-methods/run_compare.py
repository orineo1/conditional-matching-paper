"""
run_compare.py — Load trained models and run D-Flow / LGD / LGD-CM comparison.

Requires checkpoints from train_models.py.

Usage (from repo root, no wandb):
    python compare_methods/run_compare.py \
        --models_dir compare_methods/output/models_2d \
        --output_dir compare_methods/output/compare_2d \
        --no_wandb

Usage (quick local smoke-test):
    python compare_methods/run_compare.py \
        --models_dir /tmp/test_models --output_dir /tmp/test_compare \
        --n_attempts 3 --nsamples_mmd 50 --no_wandb

Usage (cluster):
    sbatch compare_methods/submit_compare.sh compare_methods/output/models_2d_<JOB_ID>
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
# Works whether you run from repo root OR from inside compare_methods/
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
for p in [script_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dist_utils import (
    generate_mog_samples_not_differentiable,
    compute_conditionals,
    compute_alpha,
    filter_and_normalize,
    mog_covariance,
    warpper_L1_distance,
)


# ── model loaders ─────────────────────────────────────────────────────────────

def load_diffusion_models(cfg, models_dir, device):
    from Diffusion import DiffusionModel
    model_uncond = DiffusionModel(
        nfeatures=cfg["condition_on"], nblocks=cfg["nblocks"], nunits=cfg["nunits"],
        condition=False, diffusion_steps=cfg["diffusion_steps"],
    )
    model_uncond.load_state_dict(
        torch.load(os.path.join(models_dir, "model_uncond.pt"), map_location=device))
    model_uncond.to(device).eval()

    model_cond = DiffusionModel(
        nfeatures=cfg["dim"], nblocks=cfg["nblocks"], nunits=cfg["nunits"],
        condition=True, condition_on=cfg["condition_on"],
        diffusion_steps=cfg["diffusion_steps"],
    )
    model_cond.load_state_dict(
        torch.load(os.path.join(models_dir, "model_cond.pt"), map_location=device))
    model_cond.to(device).eval()
    return model_uncond, model_cond


def load_cm_model(cfg, models_dir, device):
    from ConsistencyModels import ConsistencyModeliCT
    model = ConsistencyModeliCT(
        nfeatures=cfg["nfeatures_y"], condition_on=cfg["condition_on"], nunits=cfg["nunits"],
    )
    model.load_state_dict(
        torch.load(os.path.join(models_dir, "model_cm.pt"), map_location=device))
    model.to(device).eval()
    return model


def load_fm_model(cfg, models_dir, device):
    from FlowMatching import FMModel
    model = FMModel(
        nfeatures=cfg["nfeatures_y"], condition_on=cfg["condition_on"],
        nunits=cfg["nunits"], nblocks=cfg["nblocks"], device=device,
    )
    model.load_state_dict(
        torch.load(os.path.join(models_dir, "model_fm.pt"), map_location=device))
    model.to(device).eval()
    return model


# ── SWD metric ────────────────────────────────────────────────────────────────

# BEFORE
def compute_swd_simple(x_pred, mu_list, Sigma_list, alpha,
                        mog_means, mog_variances, weights,
                        nsamples=10_000, num_projections=500):
    """Normalized Sliced Wasserstein Distance at x_pred vs known target."""
    import ot

    x_pred_t = (x_pred.float().view(-1) if isinstance(x_pred, torch.Tensor)
                else torch.tensor(x_pred, dtype=torch.float32).view(-1))
    d_x = len(mu_list[0]) - len(mog_means[0])

    mu_opt, Sigma_opt = compute_conditionals(mu_list, Sigma_list, x_pred_t[:d_x])
    alpha_opt = compute_alpha(mu_list, Sigma_list, alpha, x_pred_t[:d_x])
    samples_opt    = generate_mog_samples_not_differentiable(nsamples, mu_opt, Sigma_opt, alpha_opt)
    samples_target = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

    cov = mog_covariance(mu_list, Sigma_list, alpha)
    norm_coef = torch.sqrt(torch.trace(cov) / cov.shape[0]).item()

    d_y = samples_opt.shape[1]
    projections = torch.randn(num_projections, d_y)
    projections = projections / projections.norm(dim=1, keepdim=True)

    swd = 0.0
    for proj in projections:
        X_proj = (samples_opt    @ proj).cpu().numpy(); X_proj.sort()
        Y_proj = (samples_target @ proj).cpu().numpy(); Y_proj.sort()
        swd += ot.wasserstein_1d(X_proj, Y_proj)
    swd /= num_projections

    return float(swd), float(swd / (norm_coef + 1e-8))

# AFTER
def compute_mmd_eval(x_pred, mu_list, Sigma_list, alpha,
                     mog_means, mog_variances, weights,
                     nsamples=10_000):
    """MMD between p(y|x=x_pred) and target G(y), using MMDLoss for consistency with optimization."""
    from LossFunctions import MMDLoss, RBF

    x_pred_t = (x_pred.float().view(-1) if isinstance(x_pred, torch.Tensor)
                else torch.tensor(x_pred, dtype=torch.float32).view(-1))
    d_x = len(mu_list[0]) - len(mog_means[0])

    mu_opt, Sigma_opt = compute_conditionals(mu_list, Sigma_list, x_pred_t[:d_x])
    alpha_opt = compute_alpha(mu_list, Sigma_list, alpha, x_pred_t[:d_x])
    samples_opt    = generate_mog_samples_not_differentiable(nsamples, mu_opt, Sigma_opt, alpha_opt)
    samples_target = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

    mmd_loss = MMDLoss(kernel=RBF())
    mmd = mmd_loss(samples_opt, samples_target)

    cov = mog_covariance(mu_list, Sigma_list, alpha)
    norm_coef = torch.sqrt(torch.trace(cov) / cov.shape[0]).item()

    mmd_val = mmd.item()
    return float(mmd_val), float(mmd_val)
    # return float(mmd_val), float(mmd_val / (norm_coef + 1e-8))


# ── algorithm runners ─────────────────────────────────────────────────────────

def run_lgd(model_uncond, model_cond, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha, nsamples, num_x_t, device):
    from Optimization import optimize_LGD
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cond, mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, num_x_t=num_x_t, loss="MMD", CM=False,
        device=device, FLAG=False,
    )
    return best_x_t, final_loss


def run_lgd_cm(model_uncond, model_cm, mog_means, mog_variances, weights,
               mu_list, Sigma_list, alpha, nsamples, num_x_t, device):
    from Optimization import optimize_LGD
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cm, mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha,
        nsamples=nsamples, num_x_t=num_x_t, loss="MMD", CM=True,
        device=device, FLAG=False,
    )
    return best_x_t, final_loss


def run_dflow(model_fm, model_fm_x, mog_means, mog_variances, weights,
              mu_list, Sigma_list, alpha, nsamples, device):
    from Optimization import optimize_DFLOW
    best_x1, final_loss = optimize_DFLOW(
        vf_y_cond_x=model_fm, vf_X=model_fm_x,
        device=device,
        mog_means=mog_means, mog_variances=mog_variances, weights=weights,
        n_sample=nsamples, loss_method="MMD", FLAG=False,
    )
    return best_x1.detach(), final_loss


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_result(x_pred, mu_list, Sigma_list, alpha,
                    mog_means, mog_variances, weights,
                    nsamples_swd=10_000, num_projections_swd=500):
    l1 = warpper_L1_distance(x_pred, mu_list, Sigma_list, alpha,
                             mog_means, mog_variances, weights)
    mmd, norm_mmd = compute_mmd_eval(
        x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
        nsamples=nsamples_swd,
    )
    return l1, mmd, norm_mmd


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_results(results, output_dir):
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle("Method Comparison", fontsize=14, fontweight="bold")
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, metric, label in zip(axes[:2],
                                 ["mmd", "l1"],
                                 ["MMD", "L1 to optimal"]):
        data = [results[m][metric] for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(label); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_boxplot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Boxplot saved: {path}", flush=True)


def print_summary(results):
    print("\n" + "=" * 65)
    print(f"{'Method':<12} {'MMD mean':>10} {'±':>4} {'L1 mean':>9}")
    print("=" * 45)
    for method, data in results.items():
        print(f"{method:<12} {np.mean(data['mmd']):>10.4f} {np.std(data['mmd']):>4.3f} "
              f"{np.mean(data['l1']):>9.4f}")
    print("=" * 65)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare D-Flow / LGD / LGD-CM on MoG")
    p.add_argument("--models_dir",   type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="compare_methods/output/compare")
    p.add_argument("--n_attempts",   type=int, default=25)
    p.add_argument("--nsamples_mmd", type=int, default=250)
    p.add_argument("--num_x_t",      type=int, default=3)
    p.add_argument("--nsamples_swd", type=int, default=10_000)
    p.add_argument("--num_projections_swd", type=int, default=500)
    # x_star defaults to None — if not given, uses the value baked into config.json
    p.add_argument("--x_star", type=float, nargs="+", default=None,
                   help="Override target x* (default: use value from training config)")
    p.add_argument("--skip_lgd",   action="store_true")
    p.add_argument("--skip_lgdcm", action="store_true")
    p.add_argument("--skip_dflow", action="store_true")
    p.add_argument("--wandb_project", type=str, default="compare-methods")
    p.add_argument("--wandb_entity",  type=str, default="conditional-matching")
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load config & MoG ─────────────────────────────────────────────────────
    with open(os.path.join(args.models_dir, "config.json")) as f:
        cfg = json.load(f)

    mog_data = torch.load(os.path.join(args.models_dir, "mog_params.pt"), map_location=device)
    mu_list    = [m.to(device) for m in mog_data["mu_list"]]
    Sigma_list = [s.to(device) for s in mog_data["Sigma_list"]]
    alpha      = mog_data["alpha"].to(device)

    # ── x_star — use CLI override or baked-in value ────────────────────────────
    if args.x_star is not None:
        x_star_vals = args.x_star
    else:
        x_star_vals = cfg.get("target_info", {}).get("x_star", [-5.0])
    x_star = torch.tensor(x_star_vals, dtype=torch.float32, device=device)
    print(f"x_star = {x_star.tolist()}", flush=True)

    # ── target conditional q(Y | X = x_star) ──────────────────────────────────
    mu_temp, Sigma_temp = compute_conditionals(mu_list, Sigma_list, x_star)
    alpha_temp = compute_alpha(mu_list, Sigma_list, alpha, x_star)
    mog_means, mog_variances, weights = filter_and_normalize(
        mu_temp, Sigma_temp, alpha_temp, threshold=0.01)
    print(f"Target conditional: {len(mog_means)} active components", flush=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         config={**vars(args), **cfg, "x_star": x_star_vals})
        print(f"wandb run: {run.name}", flush=True)

    # ── load models ───────────────────────────────────────────────────────────
    print("\nLoading models...", flush=True)
    model_uncond = model_cond = model_cm = model_fm = None

    if not (args.skip_lgd and args.skip_lgdcm):
        model_uncond, model_cond = load_diffusion_models(cfg, args.models_dir, device)
        print("  Diffusion loaded.", flush=True)
    if not args.skip_lgdcm:
        model_cm = load_cm_model(cfg, args.models_dir, device)
        print("  CM loaded.", flush=True)
    if not args.skip_dflow:
        model_fm = load_fm_model(cfg, args.models_dir, device)
        print("  FM loaded.", flush=True)

    # ── run methods ───────────────────────────────────────────────────────────
    results = {}

    def _run_method(name, run_fn):
        print(f"\n── {name} ({args.n_attempts} attempts) ──", flush=True)
        data = {"mmd": [], "l1": [], "time": [], "x_pred": []}
        for i in range(args.n_attempts):
            t0 = time.time()
            x_pred, _ = run_fn()
            elapsed = time.time() - t0
            l1, mmd, _ = evaluate_result(
                x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                args.nsamples_swd, args.num_projections_swd,
            )
            x_pred_list = x_pred.detach().cpu().tolist() if isinstance(x_pred, torch.Tensor) else x_pred

            data["mmd"].append(mmd)
            data["l1"].append(l1)
            data["time"].append(elapsed)
            data["x_pred"].append(
                x_pred.detach().cpu().tolist() if isinstance(x_pred, torch.Tensor) else x_pred)
            print(f"  [{i + 1:2d}/{args.n_attempts}] mmd={mmd:.4f}  l1={l1:.4f}  t={elapsed:.1f}s", flush=True)
            if not args.no_wandb:
                import wandb
                wandb.log({f"{name.lower()}/mmd": mmd,
                           f"{name.lower()}/l1": l1, f"{name.lower()}/time": elapsed,
                            f"{name.lower()}/x_pred": x_pred_list,
                           "attempt": i + 1},)
        return data

    if not args.skip_lgd:
        results["LGD"] = _run_method("LGD", lambda: run_lgd(
            model_uncond, model_cond, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha, args.nsamples_mmd, args.num_x_t, device))

    if not args.skip_lgdcm:
        results["LGD-CM"] = _run_method("LGD-CM", lambda: run_lgd_cm(
            model_uncond, model_cm, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha, args.nsamples_mmd, args.num_x_t, device))

    if not args.skip_dflow:
        results["D-Flow"] = _run_method("D-Flow", lambda: run_dflow(
            model_fm, model_fm, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha, args.nsamples_mmd, device))

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(results)
    plot_results(results, args.output_dir)

    summary = {
        method: {
            "mmd_mean": float(np.mean(d["mmd"])),
            "mmd_std": float(np.std(d["mmd"])),
            "l1_mean": float(np.mean(d["l1"])),
            "l1_std": float(np.std(d["l1"])),
            "time_mean": float(np.mean(d["time"])),
        } for method, d in results.items()
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    results_ser = {
        m: {k: ([float(v) for v in vals] if k != "x_pred" else vals)
            for k, vals in d.items()}
        for m, d in results.items()
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "cfg": cfg, "results": results_ser}, f, indent=2)

    if not args.no_wandb:
        import wandb
        for method, stats in summary.items():
            for k, v in stats.items():
                wandb.summary[f"{method}/{k}"] = v
        wandb.log({"comparison_boxplot": wandb.Image(
            os.path.join(args.output_dir, "comparison_boxplot.png"))})
        wandb.finish()

    print(f"\n✅ Done. Outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()