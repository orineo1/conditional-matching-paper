"""
run_compare.py — Load trained models and run D-Flow / LGD / LGD-CM comparison.

Supports:
  - 2D: x* read from JSON (x_star=[-5.0])
  - 10D: two splits (cond1_y9, cond9_y1), x* fixed in JSON (from one sampled joint point).
    Models for each split live in <models_dir>/<split_name>/.

Results for all splits are saved together in one results.json keyed by scenario.

Usage (no wandb):
    python compare_methods/run_compare.py \
        --models_dir compare_methods/output/models_2d \
        --output_dir compare_methods/output/compare_2d \
        --no_wandb

    python compare_methods/run_compare.py \
        --models_dir compare_methods/output/models_10d \
        --output_dir compare_methods/output/compare_10d \
        --no_wandb

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

def load_diffusion_models(split_cfg, split_dir, device):
    from Diffusion import DiffusionModel
    condition_on = split_cfg["condition_on"]
    nfeatures    = split_cfg["nfeatures"]
    nblocks      = split_cfg["nblocks"]
    nunits       = split_cfg["nunits"]
    diff_steps   = split_cfg["diffusion_steps"]

    model_uncond = DiffusionModel(
        nfeatures=condition_on, nblocks=nblocks, nunits=nunits,
        condition=False, diffusion_steps=diff_steps,
    )
    model_uncond.load_state_dict(
        torch.load(os.path.join(split_dir, "model_uncond.pt"), map_location=device))
    model_uncond.to(device).eval()

    model_cond = DiffusionModel(
        nfeatures=nfeatures, nblocks=nblocks, nunits=nunits,
        condition=True, condition_on=condition_on, diffusion_steps=diff_steps,
    )
    model_cond.load_state_dict(
        torch.load(os.path.join(split_dir, "model_cond.pt"), map_location=device))
    model_cond.to(device).eval()
    return model_uncond, model_cond


def load_cm_model(split_cfg, split_dir, device):
    from ConsistencyModels import ConsistencyModeliCT
    model = ConsistencyModeliCT(
        nfeatures=split_cfg["nfeatures_y"],
        condition_on=split_cfg["condition_on"],
        nunits=split_cfg["nunits"],
    )
    model.load_state_dict(
        torch.load(os.path.join(split_dir, "model_cm.pt"), map_location=device))
    model.to(device).eval()
    return model


def load_fm_model(split_cfg, split_dir, device):
    from FlowMatching import FMModel
    model = FMModel(
        nfeatures=split_cfg["nfeatures_y"],
        condition_on=split_cfg["condition_on"],
        nunits=split_cfg["nunits"],
        nblocks=split_cfg["nblocks"],
        device=device,
    )
    model.load_state_dict(
        torch.load(os.path.join(split_dir, "model_fm.pt"), map_location=device))
    model.to(device).eval()
    return model


# ── x* helpers ────────────────────────────────────────────────────────────────

def get_x_star(split_cfg_json, device):
    """Return x* as a tensor from the fixed list stored in the JSON."""
    x_star_spec = split_cfg_json.get("x_star")
    if x_star_spec is None or not isinstance(x_star_spec, list):
        raise ValueError(f"split config must have 'x_star' as a list, got: {x_star_spec!r}")
    return torch.tensor(x_star_spec, dtype=torch.float32, device=device)


# ── evaluation ────────────────────────────────────────────────────────────────

def compute_mmd_eval(x_pred, mu_list, Sigma_list, alpha,
                     mog_means, mog_variances, weights,
                     condition_on, nsamples=10_000):
    from LossFunctions import MMDLoss, RBF

    x_pred_t = (x_pred.float().view(-1) if isinstance(x_pred, torch.Tensor)
                else torch.tensor(x_pred, dtype=torch.float32).view(-1))

    mu_opt, Sigma_opt = compute_conditionals(mu_list, Sigma_list, x_pred_t[:condition_on])
    alpha_opt = compute_alpha(mu_list, Sigma_list, alpha, x_pred_t[:condition_on])
    samples_opt    = generate_mog_samples_not_differentiable(nsamples, mu_opt, Sigma_opt, alpha_opt)
    samples_target = generate_mog_samples_not_differentiable(nsamples, mog_means, mog_variances, weights)

    mmd_loss = MMDLoss(kernel=RBF())
    mmd_val  = mmd_loss(samples_opt, samples_target).item()
    return float(mmd_val), float(mmd_val)


def evaluate_result(x_pred, mu_list, Sigma_list, alpha,
                    mog_means, mog_variances, weights, condition_on,
                    nsamples_eval=10_000):
    l1 = warpper_L1_distance(x_pred, mu_list, Sigma_list, alpha,
                             mog_means, mog_variances, weights)
    mmd, norm_mmd = compute_mmd_eval(
        x_pred, mu_list, Sigma_list, alpha, mog_means, mog_variances, weights,
        condition_on=condition_on, nsamples=nsamples_eval,
    )
    return l1, mmd, norm_mmd


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


def run_dflow(model_fm, mog_means, mog_variances, weights,
              mu_list, Sigma_list, alpha, nsamples, device):
    from Optimization import optimize_DFLOW
    best_x1, final_loss = optimize_DFLOW(
        vf_y_cond_x=model_fm, vf_X=model_fm,
        device=device,
        mog_means=mog_means, mog_variances=mog_variances, weights=weights,
        n_sample=nsamples, loss_method="MMD", FLAG=False,
    )
    return best_x1.detach(), final_loss


# ── run one split scenario ────────────────────────────────────────────────────

def run_scenario(scenario_name, split_cfg_json, split_dir,
                 mu_list, Sigma_list, alpha, args, device, wandb_run):
    """
    Run all methods for one split scenario and return a dict of per-method results.
    x* is determined from split_cfg_json (fixed list or sampled).
    """
    print(f"\n{'='*65}", flush=True)
    print(f"Scenario: {scenario_name}", flush=True)
    print(f"  {split_cfg_json.get('description','')}", flush=True)
    print(f"{'='*65}", flush=True)

    condition_on = split_cfg_json["condition_on"]
    dim          = len(mu_list[0])
    nfeatures_y  = dim - condition_on

    # ── x* ────────────────────────────────────────────────────────────────────
    x_star = get_x_star(split_cfg_json, device)
    print(f"x_star = {x_star.tolist()}", flush=True)

    # ── MoG view for this split ───────────────────────────────────────────────
    # For cond9_y1: conditioning dims are 1..9, target dim is 0.
    # compute_conditionals always treats the first condition_on dims as x,
    # so we reorder the MoG (as in the notebook) when condition_on == dim-1
    # and the target is dim0 (not the natural last block).
    if condition_on == dim - 1:
        # x=dims1..9, y=dim0 → reorder so y (dim0) is last
        mu_list_view    = [torch.cat([m[1:], m[:1]])    for m in mu_list]
        Sigma_list_view = [
            torch.cat([
                torch.cat([S[1:, 1:], S[1:, :1]], dim=1),
                torch.cat([S[:1, 1:], S[:1, :1]], dim=1),
            ], dim=0)
            for S in Sigma_list
        ]
    else:
        # Natural order: x=first condition_on dims, y=remaining (2D and cond1_y9)
        mu_list_view    = mu_list
        Sigma_list_view = Sigma_list

    # ── target conditional q(Y | X = x_star) ──────────────────────────────────
    mu_temp, Sigma_temp = compute_conditionals(mu_list_view, Sigma_list_view, x_star)
    alpha_temp = compute_alpha(mu_list_view, Sigma_list_view, alpha, x_star)
    mog_means, mog_variances, weights = filter_and_normalize(
        mu_temp, Sigma_temp, alpha_temp, threshold=0.01)
    print(f"Target conditional: {len(mog_means)} active components", flush=True)

    # ── load split config (has nblocks, nunits, etc.) ─────────────────────────
    with open(os.path.join(split_dir, "split_config.json")) as f:
        split_cfg = json.load(f)

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    model_uncond = model_cond = model_cm = model_fm = None

    if not (args.skip_lgd and args.skip_lgdcm):
        model_uncond, model_cond = load_diffusion_models(split_cfg, split_dir, device)
        print("  Diffusion loaded.", flush=True)
    if not args.skip_lgdcm:
        model_cm = load_cm_model(split_cfg, split_dir, device)
        print("  CM loaded.", flush=True)
    if not args.skip_dflow:
        model_fm = load_fm_model(split_cfg, split_dir, device)
        print("  FM loaded.", flush=True)

    # ── method runner helper ──────────────────────────────────────────────────
    scenario_results = {}

    def _run_method(name, run_fn):
        print(f"\n── {name} ({args.n_attempts} attempts) ──", flush=True)
        data = {"mmd": [], "l1": [], "time": [], "x_pred": [], "final_loss": [], "seed": []}
        for i in range(args.n_attempts):
            attempt_seed = args.seed + i
            torch.manual_seed(attempt_seed)
            np.random.seed(attempt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(attempt_seed)

            t0 = time.time()
            x_pred, final_loss = run_fn()
            elapsed = time.time() - t0

            l1, mmd, _ = evaluate_result(
                x_pred, mu_list_view, Sigma_list_view, alpha,
                mog_means, mog_variances, weights,
                condition_on=condition_on,
                nsamples_eval=args.nsamples_eval,
            )
            fl = final_loss.item() if isinstance(final_loss, torch.Tensor) else float(final_loss)
            x_pred_vals = x_pred.detach().cpu().flatten().tolist()

            data["mmd"].append(mmd)
            data["l1"].append(l1)
            data["time"].append(elapsed)
            data["final_loss"].append(fl)
            data["seed"].append(attempt_seed)
            data["x_pred"].append(x_pred.detach().cpu().tolist()
                                   if isinstance(x_pred, torch.Tensor) else x_pred)
            print(
                f"  [{i+1:2d}/{args.n_attempts}] seed={attempt_seed}  "
                f"pred_x0:{x_pred_vals}  mmd={mmd:.4f}  l1={l1:.4f}  "
                f"loss={fl:.4f}  t={elapsed:.1f}s",
                flush=True)

        # Top-10 by final loss
        k = min(10, len(data["final_loss"]))
        top10_idx = np.argsort(data["final_loss"])[:k]
        data["top10_mmd"] = [data["mmd"][i] for i in top10_idx]
        data["top10_l1"]  = [data["l1"][i]  for i in top10_idx]
        data["top10_seed"]= [data["seed"][i] for i in top10_idx]
        print(f"  Top-{k} MMD mean±std: "
              f"{np.mean(data['top10_mmd']):.4f} ± {np.std(data['top10_mmd']):.4f}", flush=True)

        if wandb_run is not None:
            import wandb
            wandb_run.log({
                f"{scenario_name}/{name}/mmd_mean":      float(np.mean(data["mmd"])),
                f"{scenario_name}/{name}/top10_mmd_mean": float(np.mean(data["top10_mmd"])),
                f"{scenario_name}/{name}/l1_mean":       float(np.mean(data["l1"])),
            })
        return data

    if not args.skip_lgd:
        scenario_results["LGD"] = _run_method("LGD", lambda: run_lgd(
            model_uncond, model_cond, mog_means, mog_variances, weights,
            mu_list_view, Sigma_list_view, alpha, args.nsamples_mmd, args.num_x_t, device))

    if not args.skip_lgdcm:
        scenario_results["LGD-CM"] = _run_method("LGD-CM", lambda: run_lgd_cm(
            model_uncond, model_cm, mog_means, mog_variances, weights,
            mu_list_view, Sigma_list_view, alpha, args.nsamples_mmd, args.num_x_t, device))

    if not args.skip_dflow:
        scenario_results["D-Flow"] = _run_method("D-Flow", lambda: run_dflow(
            model_fm, mog_means, mog_variances, weights,
            mu_list_view, Sigma_list_view, alpha, args.nsamples_mmd, device))

    return scenario_results, x_star.tolist()


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_results(all_results, output_dir):
    """One boxplot figure per scenario."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for scenario_name, results in all_results.items():
        methods = list(results.keys())
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Method Comparison — {scenario_name}", fontsize=13, fontweight="bold")

        for ax, metric, label in zip(axes, ["mmd", "l1"], ["MMD", "L1 to optimal"]):
            data = [results[m][metric] for m in methods]
            bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.6)
            ax.set_title(label); ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(output_dir, f"comparison_boxplot_{scenario_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  Boxplot saved: {path}", flush=True)


def print_summary(all_results):
    for scenario_name, results in all_results.items():
        print(f"\n{'='*65}")
        print(f"Scenario: {scenario_name}")
        print(f"{'Method':<12} {'MMD (all)':>10} {'MMD (top10)':>12} {'L1 (top10)':>11}")
        print("=" * 65)
        for method, data in results.items():
            print(f"{method:<12} {np.mean(data['mmd']):>10.4f} "
                  f"{np.mean(data['top10_mmd']):>12.4f} "
                  f"{np.mean(data['top10_l1']):>11.4f}")
        print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare D-Flow / LGD / LGD-CM on MoG")
    p.add_argument("--models_dir",   type=str, required=True,
                   help="Output dir from train_models.py")
    p.add_argument("--output_dir",   type=str, default="compare_methods/output/compare")
    p.add_argument("--scenarios",    type=str, nargs="*", default=None,
                   help="Which splits/scenarios to run (default: all from config.json)")
    p.add_argument("--n_attempts",   type=int, default=25)
    p.add_argument("--nsamples_mmd", type=int, default=250)
    p.add_argument("--num_x_t",      type=int, default=3)
    p.add_argument("--nsamples_eval", type=int, default=10_000)
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

    # ── load top-level config + MoG ───────────────────────────────────────────
    with open(os.path.join(args.models_dir, "config.json")) as f:
        cfg = json.load(f)

    mog_data = torch.load(os.path.join(args.models_dir, "mog_params.pt"), map_location=device)
    mu_list    = [m.to(device) for m in mog_data["mu_list"]]
    Sigma_list = [s.to(device) for s in mog_data["Sigma_list"]]
    alpha      = mog_data["alpha"].to(device)

    all_splits = cfg["splits"]  # from JSON, e.g. {"cond1_y1": {...}} or {"cond1_y9":..., "cond9_y1":...}
    scenarios_to_run = args.scenarios if args.scenarios else list(all_splits.keys())
    print(f"Scenarios to run: {scenarios_to_run}", flush=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               config={**vars(args), **cfg})
        print(f"wandb run: {wandb_run.name}", flush=True)

    # ── run all scenarios ─────────────────────────────────────────────────────
    all_results = {}
    all_x_stars = {}

    for scenario_name in scenarios_to_run:
        if scenario_name not in all_splits:
            print(f"WARNING: scenario '{scenario_name}' not in config, skipping.", flush=True)
            continue

        split_dir = os.path.join(args.models_dir, scenario_name)
        if not os.path.isdir(split_dir):
            print(f"WARNING: no model dir found at {split_dir}, skipping.", flush=True)
            continue

        scenario_results, x_star_used = run_scenario(
            scenario_name, all_splits[scenario_name], split_dir,
            mu_list, Sigma_list, alpha, args, device, wandb_run,
        )
        all_results[scenario_name]  = scenario_results
        all_x_stars[scenario_name]  = x_star_used

    # ── summary + plots ───────────────────────────────────────────────────────
    print_summary(all_results)
    plot_results(all_results, args.output_dir)

    # ── save results.json ─────────────────────────────────────────────────────
    def _serialize(d):
        if isinstance(d, dict):
            return {k: _serialize(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_serialize(v) for v in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        return d

    output_payload = {
        "args":    vars(args),
        "cfg":     cfg,
        "x_stars": all_x_stars,
        "results": _serialize(all_results),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(output_payload, f, indent=2)

    summary = {}
    for scenario_name, results in all_results.items():
        summary[scenario_name] = {
            method: {
                "mmd_mean":       float(np.mean(d["mmd"])),
                "mmd_std":        float(np.std(d["mmd"])),
                "top10_mmd_mean": float(np.mean(d["top10_mmd"])),
                "top10_mmd_std":  float(np.std(d["top10_mmd"])),
                "l1_mean":        float(np.mean(d["l1"])),
                "l1_std":         float(np.std(d["l1"])),
                "time_mean":      float(np.mean(d["time"])),
            }
            for method, d in results.items()
        }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        import wandb
        for scenario_name, results in all_results.items():
            for method, stats in summary[scenario_name].items():
                for k, v in stats.items():
                    wandb.summary[f"{scenario_name}/{method}/{k}"] = v
        for scenario_name in all_results:
            img_path = os.path.join(args.output_dir, f"comparison_boxplot_{scenario_name}.png")
            if os.path.exists(img_path):
                wandb.log({f"boxplot/{scenario_name}": wandb.Image(img_path)})
        wandb.finish()

    print(f"\n✅ Done. Outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
