"""
run_compare.py — Load trained models and run D-Flow / LGD / LGD-CM comparison.

Requires model checkpoints produced by train_models.py.

Usage (cluster):
    python compare_methods/run_compare.py \
        --models_dir compare_methods/output/models_2d \
        --output_dir compare_methods/output/compare_2d \
        --n_attempts 25 \
        --nsamples_mmd 250

Usage (quick smoke-test):
    python compare_methods/run_compare.py \
        --models_dir /tmp/test_models \
        --output_dir /tmp/test_compare \
        --n_attempts 3 --nsamples_mmd 50
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
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
for p in [script_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_methods.dist_utils import (
    generate_mog_samples_not_differentiable,
    compute_conditionals,
    compute_alpha,
    filter_and_normalize,
    mog_covariance,
    warpper_L1_distance,
)


# ── model imports ─────────────────────────────────────────────────────────────

def load_diffusion_models(cfg, models_dir, device):
    from Diffusion import DiffusionModel
    model_uncond = DiffusionModel(
        nfeatures=cfg["condition_on"],
        nblocks=cfg["nblocks"], nunits=cfg["nunits"],
        condition=False, diffusion_steps=cfg["diffusion_steps"],
    )
    model_uncond.load_state_dict(torch.load(
        os.path.join(models_dir, "model_uncond.pt"), map_location=device))
    model_uncond.to(device).eval()

    model_cond = DiffusionModel(
        nfeatures=cfg["dim"],
        nblocks=cfg["nblocks"], nunits=cfg["nunits"],
        condition=True, condition_on=cfg["condition_on"],
        diffusion_steps=cfg["diffusion_steps"],
    )
    model_cond.load_state_dict(torch.load(
        os.path.join(models_dir, "model_cond.pt"), map_location=device))
    model_cond.to(device).eval()
    return model_uncond, model_cond


def load_cm_model(cfg, models_dir, device):
    from ConsistencyModels import ConsistencyModeliCT
    model = ConsistencyModeliCT(
        nfeatures=cfg["nfeatures_y"],
        condition_on=cfg["condition_on"],
        nunits=cfg["nunits"],
    )
    model.load_state_dict(torch.load(
        os.path.join(models_dir, "model_cm.pt"), map_location=device))
    model.to(device).eval()
    return model


def load_fm_model(cfg, models_dir, device):
    from FlowMatching import FMModel
    model = FMModel(
        nfeatures=cfg["nfeatures_y"],
        condition_on=cfg["condition_on"],
        nunits=cfg["nunits"],
        nblocks=cfg["nblocks"],
        device=device,
    )
    model.load_state_dict(torch.load(
        os.path.join(models_dir, "model_fm.pt"), map_location=device))
    model.to(device).eval()
    return model


# ── SWD metric ────────────────────────────────────────────────────────────────

def compute_swd_simple(x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                        nsamples=10_000, num_projections=500):
    """
    Sliced Wasserstein Distance between conditional samples from x_pred
    vs samples from the target conditional distribution.
    """
    import ot

    x_pred_t = x_pred.float().view(-1) if isinstance(x_pred, torch.Tensor) else torch.tensor(x_pred, dtype=torch.float32).view(-1)
    d_x = len(mu_list[0]) - len(mog_means[0])

    # Samples from conditional at x_pred (optimized distribution)
    mu_opt, Sigma_opt = compute_conditionals(mu_list, Sigma_list, x_pred_t[:d_x])
    alpha_opt = compute_alpha(mu_list, Sigma_list, alpha, x_pred_t[:d_x])
    samples_opt = generate_mog_samples_not_differentiable(nsamples, mu_opt, Sigma_opt, alpha_opt)

    # Samples from true target conditional
    samples_target = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

    # Normalize SWD by sqrt(trace(cov)/d)
    cov = mog_covariance(mu_list, Sigma_list, alpha)
    d = cov.shape[0]
    norm_coef = torch.sqrt(torch.trace(cov) / d).item()

    # Sliced Wasserstein
    d_y = samples_opt.shape[1]
    projections = torch.randn(num_projections, d_y)
    projections = projections / projections.norm(dim=1, keepdim=True)

    swd = 0.0
    for proj in projections:
        X_proj = (samples_opt @ proj).cpu().numpy()
        Y_proj = (samples_target @ proj).cpu().numpy()
        X_proj.sort(); Y_proj.sort()
        swd += ot.wasserstein_1d(X_proj, Y_proj)
    swd /= num_projections

    return float(swd), float(swd / (norm_coef + 1e-8))


# ── algorithm runners ─────────────────────────────────────────────────────────

def run_lgd(model_uncond, model_cond, mog_means, mog_variances, weights,
            mu_list, Sigma_list, alpha, nsamples, num_x_t, device):
    from Optimization import optimize_LGD
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cond, mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha, nsamples=nsamples, num_x_t=num_x_t,
        loss="MMD", CM=False, device=device, FLAG=False,
    )
    return best_x_t, final_loss


def run_lgd_cm(model_uncond, model_cm, mog_means, mog_variances, weights,
               mu_list, Sigma_list, alpha, nsamples, num_x_t, device):
    """LGD-CM: use consistency model as the conditional model."""
    from Optimization import optimize_LGD
    best_x_t, _, final_loss = optimize_LGD(
        model_uncond, model_cm, mog_means, mog_variances, weights,
        mu_list, Sigma_list, alpha, nsamples=nsamples, num_x_t=num_x_t,
        loss="MMD", CM=True, device=device, FLAG=False,
    )
    return best_x_t, final_loss


def run_dflow(model_fm, model_fm_x, mog_means, mog_variances, weights,
              mu_list, Sigma_list, alpha, nsamples, device):
    """
    D-Flow: optimize x in FM latent space.
    model_fm   = p(y | x) conditional FM
    model_fm_x = p(x) unconditional FM over x-marginal (used for init / transport)
    """
    from Optimization import optimize_DFLOW
    best_x1, final_loss = optimize_DFLOW(
        vf_y_cond_x=model_fm, vf_X=model_fm_x,
        device=device,
        mog_means=mog_means, mog_variances=mog_variances, weights=weights,
        n_sample=nsamples, loss_method="MMD", FLAG=False,
    )
    return best_x1, final_loss


# ── per-run evaluation ────────────────────────────────────────────────────────

def evaluate_result(x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                    nsamples_swd=10_000, num_projections_swd=500):
    """Returns (l1, swd, norm_swd)."""
    l1 = warpper_L1_distance(x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights)
    swd, norm_swd = compute_swd_simple(
        x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
        nsamples=nsamples_swd, num_projections=num_projections_swd,
    )
    return l1, swd, norm_swd


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_results(results, output_dir):
    """Box plots and summary stats for all methods."""
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Method Comparison", fontsize=14, fontweight="bold")

    for ax, metric, label in zip(
        axes,
        ["swd", "norm_swd", "l1"],
        ["SWD", "Normalized SWD", "L1 to optimal"],
    ):
        data = [results[m][metric] for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(label); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_boxplot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Boxplot saved to: {path}", flush=True)


def print_summary(results):
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'SWD mean':>10} {'SWD std':>8} {'NormSWD mean':>13} {'L1 mean':>10}")
    print("=" * 60)
    for method, data in results.items():
        print(f"{method:<15} {np.mean(data['swd']):>10.4f} {np.std(data['swd']):>8.4f} "
              f"{np.mean(data['norm_swd']):>13.4f} {np.mean(data['l1']):>10.4f}")
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare D-Flow / LGD / LGD-CM on MoG")
    p.add_argument("--models_dir", type=str, required=True,
                   help="Directory with model checkpoints from train_models.py")
    p.add_argument("--output_dir", type=str, default="compare_methods/output/compare")

    # Optimization
    p.add_argument("--n_attempts", type=int, default=25,
                   help="Independent optimization runs per method")
    p.add_argument("--nsamples_mmd", type=int, default=250,
                   help="Samples for MMD during optimization")
    p.add_argument("--num_x_t", type=int, default=3,
                   help="MC samples for LGD/LGD-CM expected loss")
    p.add_argument("--nsamples_swd", type=int, default=10_000,
                   help="Samples for final SWD evaluation")
    p.add_argument("--num_projections_swd", type=int, default=500)

    # Target x* (conditioning value for the comparison)
    p.add_argument("--x_star", type=float, nargs="+", default=[-5.0],
                   help="Target conditioning value x* (space-separated floats)")

    # Which methods to run
    p.add_argument("--skip_lgd",   action="store_true")
    p.add_argument("--skip_lgdcm", action="store_true")
    p.add_argument("--skip_dflow", action="store_true")

    # wandb
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

    # ── load config & MoG params ───────────────────────────────────────────────
    with open(os.path.join(args.models_dir, "config.json")) as f:
        cfg = json.load(f)

    mog_data = torch.load(os.path.join(args.models_dir, "mog_params.pt"), map_location=device)
    mu_list    = [m.to(device) for m in mog_data["mu_list"]]
    Sigma_list = [s.to(device) for s in mog_data["Sigma_list"]]
    alpha      = mog_data["alpha"].to(device)

    # ── compute target conditional distribution at x_star ─────────────────────
    x_star = torch.tensor(args.x_star, dtype=torch.float32, device=device)
    print(f"x_star = {x_star.tolist()}", flush=True)

    mu_temp, Sigma_temp = compute_conditionals(mu_list, Sigma_list, x_star)
    alpha_temp = compute_alpha(mu_list, Sigma_list, alpha, x_star)
    mog_means, mog_variances, weights = filter_and_normalize(mu_temp, Sigma_temp, alpha_temp, threshold=0.01)
    print(f"Target conditional: {len(mog_means)} active components", flush=True)

    # ── wandb init ─────────────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            config={**vars(args), **cfg, "x_star": args.x_star},
        )
        print(f"wandb run: {run.name}", flush=True)

    # ── load models ────────────────────────────────────────────────────────────
    results = {}
    method_times = {}

    print("\nLoading models...", flush=True)
    needs_diff = not (args.skip_lgd and args.skip_lgdcm)
    needs_cm   = not args.skip_lgdcm
    needs_fm   = not args.skip_dflow

    model_uncond = model_cond = model_cm = model_fm = model_fm_x = None

    if needs_diff:
        model_uncond, model_cond = load_diffusion_models(cfg, args.models_dir, device)
        print("  Diffusion models loaded.", flush=True)

    if needs_cm:
        model_cm = load_cm_model(cfg, args.models_dir, device)
        print("  CM model loaded.", flush=True)

    if needs_fm:
        model_fm = load_fm_model(cfg, args.models_dir, device)
        # For D-Flow we also need an unconditional FM over x-marginal
        # Reuse the conditional FM as model_fm_x (condition_on=0) if a separate one wasn't trained;
        # alternatively fall back to using the same model with y=None.
        # Here we share model_fm with model_fm_x (D-Flow typically only needs vf_y_cond_x; vf_X is for init)
        model_fm_x = model_fm
        print("  FM model loaded.", flush=True)

    # ── LGD ───────────────────────────────────────────────────────────────────
    if not args.skip_lgd:
        print(f"\n── Running LGD ({args.n_attempts} attempts) ──", flush=True)
        lgd_results = {"swd": [], "norm_swd": [], "l1": [], "time": [], "x_pred": []}
        for i in range(args.n_attempts):
            t0 = time.time()
            x_pred, final_loss = run_lgd(
                model_uncond, model_cond, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha, args.nsamples_mmd, args.num_x_t, device,
            )
            elapsed = time.time() - t0
            l1, swd, norm_swd = evaluate_result(
                x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                args.nsamples_swd, args.num_projections_swd,
            )
            lgd_results["swd"].append(swd)
            lgd_results["norm_swd"].append(norm_swd)
            lgd_results["l1"].append(l1)
            lgd_results["time"].append(elapsed)
            lgd_results["x_pred"].append(x_pred.detach().cpu().tolist())
            print(f"  [{i+1:2d}/{args.n_attempts}] swd={swd:.4f}  norm_swd={norm_swd:.4f}  l1={l1:.4f}  t={elapsed:.1f}s", flush=True)

            if not args.no_wandb:
                import wandb
                wandb.log({"lgd/swd": swd, "lgd/norm_swd": norm_swd, "lgd/l1": l1,
                           "lgd/time": elapsed, "attempt": i + 1}, commit=False)

        results["LGD"] = lgd_results
        method_times["LGD"] = lgd_results["time"]

    # ── LGD-CM ────────────────────────────────────────────────────────────────
    if not args.skip_lgdcm:
        print(f"\n── Running LGD-CM ({args.n_attempts} attempts) ──", flush=True)
        lgdcm_results = {"swd": [], "norm_swd": [], "l1": [], "time": [], "x_pred": []}
        for i in range(args.n_attempts):
            t0 = time.time()
            x_pred, final_loss = run_lgd_cm(
                model_uncond, model_cm, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha, args.nsamples_mmd, args.num_x_t, device,
            )
            elapsed = time.time() - t0
            l1, swd, norm_swd = evaluate_result(
                x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                args.nsamples_swd, args.num_projections_swd,
            )
            lgdcm_results["swd"].append(swd)
            lgdcm_results["norm_swd"].append(norm_swd)
            lgdcm_results["l1"].append(l1)
            lgdcm_results["time"].append(elapsed)
            lgdcm_results["x_pred"].append(x_pred.detach().cpu().tolist())
            print(f"  [{i+1:2d}/{args.n_attempts}] swd={swd:.4f}  norm_swd={norm_swd:.4f}  l1={l1:.4f}  t={elapsed:.1f}s", flush=True)

            if not args.no_wandb:
                import wandb
                wandb.log({"lgdcm/swd": swd, "lgdcm/norm_swd": norm_swd, "lgdcm/l1": l1,
                           "lgdcm/time": elapsed, "attempt": i + 1}, commit=False)

        results["LGD-CM"] = lgdcm_results
        method_times["LGD-CM"] = lgdcm_results["time"]

    # ── D-Flow ────────────────────────────────────────────────────────────────
    if not args.skip_dflow:
        print(f"\n── Running D-Flow ({args.n_attempts} attempts) ──", flush=True)
        dflow_results = {"swd": [], "norm_swd": [], "l1": [], "time": [], "x_pred": []}
        for i in range(args.n_attempts):
            t0 = time.time()
            x_pred, final_loss = run_dflow(
                model_fm, model_fm_x, mog_means, mog_variances, weights,
                mu_list, Sigma_list, alpha, args.nsamples_mmd, device,
            )
            elapsed = time.time() - t0
            l1, swd, norm_swd = evaluate_result(
                x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
                args.nsamples_swd, args.num_projections_swd,
            )
            dflow_results["swd"].append(swd)
            dflow_results["norm_swd"].append(norm_swd)
            dflow_results["l1"].append(l1)
            dflow_results["time"].append(elapsed)
            dflow_results["x_pred"].append(x_pred.detach().cpu().tolist() if isinstance(x_pred, torch.Tensor) else x_pred)
            print(f"  [{i+1:2d}/{args.n_attempts}] swd={swd:.4f}  norm_swd={norm_swd:.4f}  l1={l1:.4f}  t={elapsed:.1f}s", flush=True)

            if not args.no_wandb:
                import wandb
                wandb.log({"dflow/swd": swd, "dflow/norm_swd": norm_swd, "dflow/l1": l1,
                           "dflow/time": elapsed, "attempt": i + 1}, commit=False)

        results["D-Flow"] = dflow_results
        method_times["D-Flow"] = dflow_results["time"]

    # ── Summary & plots ────────────────────────────────────────────────────────
    print_summary(results)
    plot_results(results, args.output_dir)

    # Save full results JSON
    results_serializable = {}
    for method, data in results.items():
        results_serializable[method] = {k: [float(v) for v in vals] if k != "x_pred" else vals
                                         for k, vals in data.items()}

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"args": vars(args), "cfg": cfg, "results": results_serializable}, f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)

    # Summary stats JSON
    summary = {}
    for method, data in results.items():
        summary[method] = {
            "swd_mean":      float(np.mean(data["swd"])),
            "swd_std":       float(np.std(data["swd"])),
            "norm_swd_mean": float(np.mean(data["norm_swd"])),
            "norm_swd_std":  float(np.std(data["norm_swd"])),
            "l1_mean":       float(np.mean(data["l1"])),
            "l1_std":        float(np.std(data["l1"])),
            "time_mean":     float(np.mean(data["time"])),
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_wandb:
        import wandb
        for method, stats in summary.items():
            for k, v in stats.items():
                wandb.summary[f"{method}/{k}"] = v
        wandb.log({"comparison_boxplot": wandb.Image(
            os.path.join(args.output_dir, "comparison_boxplot.png"))})
        wandb.finish()

    print(f"\n✅ Comparison complete. Outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
