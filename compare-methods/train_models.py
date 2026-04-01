"""
train_models.py — Train Diffusion (uncond + cond), ConsistencyModel, and FlowMatching
models on a MoG distribution and save checkpoints for D-Flow / LGD / LGD-CM comparison.

Reads MoG parameters from a JSON file (mog_2d.json or mog_10d.json).
For multi-split configs (e.g. 10D with cond1_y9 and cond9_y1), trains separate
model sets for each split and saves them in subdirectories.

Usage (from repo root):
    python compare_methods/train_models.py --mog_json compare_methods/mog_2d.json \
        --output_dir compare_methods/output/models_2d

    python compare_methods/train_models.py --mog_json compare_methods/mog_10d.json \
        --output_dir compare_methods/output/models_10d

Usage (quick local smoke-test):
    python compare_methods/train_models.py --mog_json compare_methods/mog_2d.json \
        --output_dir /tmp/test_models \
        --nepochs_diff 200 --nepochs_cm 100 --nepochs_fm 100

Usage (on cluster):
    sbatch compare_methods/submit_train.sh compare_methods/mog_2d.json
"""

import argparse
import os
import sys
import json

import torch
import numpy as np
from functools import partial

# ── path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
for p in [script_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dist_utils import generate_mog_samples_not_differentiable


# ── JSON loader ───────────────────────────────────────────────────────────────

def load_mog_json(json_path, device):
    """Load MoG parameters from JSON. Returns (mu_list, Sigma_list, alpha, mog_cfg)."""
    with open(json_path) as f:
        mog_cfg = json.load(f)

    mu_list    = [torch.tensor(m, dtype=torch.float32, device=device) for m in mog_cfg["mu_list"]]
    Sigma_list = [torch.tensor(s, dtype=torch.float32, device=device) for s in mog_cfg["Sigma_list"]]
    alpha      = torch.tensor(mog_cfg["alpha"], dtype=torch.float32, device=device)
    return mu_list, Sigma_list, alpha, mog_cfg


# ── model imports ─────────────────────────────────────────────────────────────

def import_models():
    from Diffusion import DiffusionModel
    from ConsistencyModels import ConsistencyModeliCT
    from FlowMatching import FMModel
    return DiffusionModel, ConsistencyModeliCT, FMModel


# ── training helpers ──────────────────────────────────────────────────────────

def make_data_gen(mu_list, Sigma_list, alpha, kernel_func=None):
    return partial(generate_mog_samples_not_differentiable,
                   means=mu_list, variances=Sigma_list, weights=alpha,
                   kernel_func=kernel_func)


def train_diffusion_uncond(mu_list, Sigma_list, alpha, condition_on,
                            nblocks, nunits, diffusion_steps, nepochs, batch_size, device,
                            wandb_run=None, wandb_prefix="diff_uncond", log_every=100):
    """Unconditional diffusion over the x-marginal (first `condition_on` dims)."""
    DiffusionModel, _, _ = import_models()
    model = DiffusionModel(
        nfeatures=condition_on, nblocks=nblocks, nunits=nunits,
        condition=False, diffusion_steps=diffusion_steps,
    )
    data_gen = make_data_gen(mu_list, Sigma_list, alpha,
                             kernel_func=lambda X: X[:, :condition_on])
    model.train_model(X=None, data_generator=data_gen,
                      nepochs=nepochs, batch_size=batch_size, device=device,
                      wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


def train_diffusion_cond(mu_list, Sigma_list, alpha, condition_on, nfeatures,
                          nblocks, nunits, diffusion_steps, nepochs, batch_size, device,
                          wandb_run=None, wandb_prefix="diff_cond", log_every=100):
    """Conditional diffusion p(y|x) over all `nfeatures` dims."""
    DiffusionModel, _, _ = import_models()
    model = DiffusionModel(
        nfeatures=nfeatures, nblocks=nblocks, nunits=nunits,
        condition=True, condition_on=condition_on, diffusion_steps=diffusion_steps,
    )
    data_gen = make_data_gen(mu_list, Sigma_list, alpha)
    model.train_model(X=None, data_generator=data_gen,
                      nepochs=nepochs, batch_size=batch_size,
                      condition_on=condition_on, device=device,
                      wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


def train_cm(mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
              nunits, nepochs, batch_size, device,
              wandb_run=None, wandb_prefix="ict", log_every=100):
    _, ConsistencyModeliCT, _ = import_models()
    model = ConsistencyModeliCT(nfeatures=nfeatures_y, condition_on=condition_on, nunits=nunits)
    data_gen = make_data_gen(mu_list, Sigma_list, alpha)
    model.train_model(X=None, nepochs=nepochs, batch_size=batch_size,
                      device=device, condition=condition_on,
                      data_generator=data_gen, use_improved_training=True,
                      wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


def train_fm(mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
              nunits, nblocks, nepochs, batch_size, device,
              wandb_run=None, wandb_prefix="fm", log_every=100):
    _, _, FMModel = import_models()
    model = FMModel(nfeatures=nfeatures_y, condition_on=condition_on,
                    nunits=nunits, nblocks=nblocks, device=device)
    data_gen = make_data_gen(mu_list, Sigma_list, alpha)
    model.train_FM(lr=1e-3, batch_size=batch_size, data_generator=data_gen, nepochs=nepochs,
                   wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


# ── train one split ────────────────────────────────────────────────────────────

def train_split(split_name, split_cfg, mu_list, Sigma_list, alpha, args,
                output_dir, device, wandb_run):
    """Train all models for a single condition_on split and save to output_dir/split_name/."""
    condition_on = split_cfg["condition_on"]
    nfeatures    = len(mu_list[0])
    nfeatures_y  = nfeatures - condition_on

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"Split: {split_name}  |  condition_on={condition_on}  |  nfeatures_y={nfeatures_y}", flush=True)
    print(f"{'='*60}", flush=True)

    # Save split config
    split_info = {
        "split_name":   split_name,
        "condition_on": condition_on,
        "nfeatures_y":  nfeatures_y,
        "nfeatures":    nfeatures,
        "nblocks":      args.nblocks,
        "nunits":       args.nunits,
        "diffusion_steps": args.diffusion_steps,
        "x_star_mode":  split_cfg.get("x_star", "sample"),
        **split_cfg,
    }
    with open(os.path.join(split_dir, "split_config.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    prefix = split_name  # used as wandb prefix

    if not args.skip_diff:
        print(f"\n── [{split_name}] Training unconditional diffusion ──", flush=True)
        model_uncond = train_diffusion_uncond(
            mu_list, Sigma_list, alpha, condition_on,
            args.nblocks, args.nunits, args.diffusion_steps,
            args.nepochs_diff, args.batch_size_diff, device,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/diff_uncond",
            log_every=args.log_every,
        )
        torch.save(model_uncond.state_dict(), os.path.join(split_dir, "model_uncond.pt"))
        print(f"  Saved {split_dir}/model_uncond.pt", flush=True)

        print(f"\n── [{split_name}] Training conditional diffusion p(y|x) ──", flush=True)
        model_cond = train_diffusion_cond(
            mu_list, Sigma_list, alpha, condition_on, nfeatures,
            args.nblocks, args.nunits, args.diffusion_steps,
            args.nepochs_diff, args.batch_size_diff, device,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/diff_cond",
            log_every=args.log_every,
        )
        torch.save(model_cond.state_dict(), os.path.join(split_dir, "model_cond.pt"))
        print(f"  Saved {split_dir}/model_cond.pt", flush=True)
    else:
        print(f"[{split_name}] Skipping Diffusion.", flush=True)

    if not args.skip_cm:
        print(f"\n── [{split_name}] Training Consistency Model (iCT) ──", flush=True)
        model_cm = train_cm(
            mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
            args.nunits, args.nepochs_cm, args.batch_size_cm, device,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/ict",
            log_every=args.log_every,
        )
        torch.save(model_cm.state_dict(), os.path.join(split_dir, "model_cm.pt"))
        print(f"  Saved {split_dir}/model_cm.pt", flush=True)
    else:
        print(f"[{split_name}] Skipping CM.", flush=True)

    if not args.skip_fm:
        print(f"\n── [{split_name}] Training Flow Matching ──", flush=True)
        model_fm = train_fm(
            mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
            args.nunits, args.nblocks, args.nepochs_fm, args.batch_size_fm, device,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/fm",
            log_every=args.log_every,
        )
        torch.save(model_fm.state_dict(), os.path.join(split_dir, "model_fm.pt"))
        print(f"  Saved {split_dir}/model_fm.pt", flush=True)
    else:
        print(f"[{split_name}] Skipping FM.", flush=True)

    print(f"\n✅ [{split_name}] Done. Models in: {split_dir}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train models for compare-methods pipeline")
    p.add_argument("--mog_json",    type=str, required=True,
                   help="Path to mog_2d.json or mog_10d.json")
    p.add_argument("--output_dir",  type=str, default="compare_methods/output/models")
    p.add_argument("--splits",      type=str, nargs="*", default=None,
                   help="Which splits to train (default: all splits in the JSON)")

    p.add_argument("--nblocks",         type=int, default=3)
    p.add_argument("--nunits",          type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=100)

    p.add_argument("--nepochs_diff",    type=int, default=20_000)
    p.add_argument("--nepochs_cm",      type=int, default=7_500)
    p.add_argument("--nepochs_fm",      type=int, default=10_000)
    p.add_argument("--batch_size_diff", type=int, default=512)
    p.add_argument("--batch_size_cm",   type=int, default=1024)
    p.add_argument("--batch_size_fm",   type=int, default=1024)

    p.add_argument("--skip_diff", action="store_true")
    p.add_argument("--skip_cm",   action="store_true")
    p.add_argument("--skip_fm",   action="store_true")
    p.add_argument("--seed",      type=int, default=42)

    p.add_argument("--wandb_project", type=str, default="compare-methods-train")
    p.add_argument("--wandb_entity",  type=str, default="conditional-matching")
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--log_every",     type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load MoG from JSON ────────────────────────────────────────────────────
    print(f"Loading MoG from: {args.mog_json}", flush=True)
    mu_list, Sigma_list, alpha, mog_cfg = load_mog_json(args.mog_json, device)
    print(f"  dim={mog_cfg['dim']}, K={len(mu_list)} components", flush=True)

    # ── Save MoG params as .pt for run_compare ────────────────────────────────
    torch.save({
        "mu_list":    [m.cpu() for m in mu_list],
        "Sigma_list": [s.cpu() for s in Sigma_list],
        "alpha":      alpha.cpu(),
    }, os.path.join(args.output_dir, "mog_params.pt"))

    # Save a top-level config for run_compare to read
    top_config = {
        "dim":             mog_cfg["dim"],
        "nblocks":         args.nblocks,
        "nunits":          args.nunits,
        "diffusion_steps": args.diffusion_steps,
        "seed":            args.seed,
        "mog_json":        args.mog_json,
        "splits":          mog_cfg["splits"],
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(top_config, f, indent=2)
    print("Config + MoG params saved.", flush=True)

    # ── Determine which splits to train ───────────────────────────────────────
    all_splits = mog_cfg["splits"]
    splits_to_train = args.splits if args.splits else list(all_splits.keys())
    print(f"Splits to train: {splits_to_train}", flush=True)

    # ── wandb init ────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={**vars(args), **top_config},
            name=f"train_{mog_cfg['dim']}d_seed{args.seed}",
        )
        print(f"wandb run: {wandb_run.name}", flush=True)

    # ── Train each split ──────────────────────────────────────────────────────
    for split_name in splits_to_train:
        if split_name not in all_splits:
            print(f"WARNING: split '{split_name}' not found in JSON, skipping.", flush=True)
            continue
        train_split(split_name, all_splits[split_name],
                    mu_list, Sigma_list, alpha,
                    args, args.output_dir, device, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n✅ All done. Outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
