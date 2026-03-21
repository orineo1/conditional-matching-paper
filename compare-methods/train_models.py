"""
train_models.py — Train Diffusion (uncond + cond), ConsistencyModel, and FlowMatching
models on a MoG distribution and save checkpoints for D-Flow / LGD / LGD-CM comparison.

Usage (cluster):
    python compare_methods/train_models.py \
        --output_dir compare_methods/output/models_2d \
        --dim 2 --condition_on 1 \
        --nepochs_diff 20000 --nepochs_cm 7500 --nepochs_fm 10000

Usage (quick smoke-test):
    python compare_methods/train_models.py --dim 2 --nepochs_diff 500 --nepochs_cm 200 --nepochs_fm 200 --output_dir /tmp/test_models
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
repo_root = os.path.dirname(script_dir)
for p in [script_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_methods.dist_utils import (
    generate_mog_samples_not_differentiable,
    compute_conditionals,
    compute_alpha,
    filter_and_normalize,
)

# ── default MoG configs ───────────────────────────────────────────────────────

def get_mog_2d(device):
    """2D MoG conditioned on 1st coordinate (condition_on=1)."""
    mu_list = [
        torch.tensor([-5,  5], dtype=torch.float32),
        torch.tensor([-5, -5], dtype=torch.float32),
        torch.tensor([ 5,  3], dtype=torch.float32),
        torch.tensor([ 5, -1], dtype=torch.float32),
        torch.tensor([ 0, -3], dtype=torch.float32),
        torch.tensor([-2,  4], dtype=torch.float32),
        torch.tensor([-2, -3], dtype=torch.float32),
        torch.tensor([ 1,  2], dtype=torch.float32),
        torch.tensor([-7,  1], dtype=torch.float32),
        torch.tensor([ 7,  5], dtype=torch.float32),
        torch.tensor([ 0, -5], dtype=torch.float32),
    ]
    Sigma_list = [
        torch.tensor([[0.5, 0.195], [0.195, 0.2]], dtype=torch.float32)
    ] * len(mu_list)
    alpha = torch.ones(len(mu_list)) / len(mu_list)
    return [m.to(device) for m in mu_list], [s.to(device) for s in Sigma_list], alpha.to(device)


def get_mog_10d(device):
    """10D MoG conditioned on first 1 coordinate (condition_on=1)."""
    torch.manual_seed(0)
    K = 8
    d = 10
    mu_list = [torch.randn(d) * 5 for _ in range(K)]
    Sigma_list = []
    for _ in range(K):
        A = torch.randn(d, d) * 0.3
        Sigma_list.append(A @ A.T + 0.5 * torch.eye(d))
    alpha = torch.ones(K) / K
    return [m.to(device) for m in mu_list], [s.to(device) for s in Sigma_list], alpha.to(device)


# ── model imports ─────────────────────────────────────────────────────────────

def import_models():
    """Lazy import so missing optional deps don't break --help."""
    from Diffusion import DiffusionModel
    from ConsistencyModels import ConsistencyModeliCT
    from FlowMatching import FMModel
    return DiffusionModel, ConsistencyModeliCT, FMModel


# ── training helpers ──────────────────────────────────────────────────────────

def train_diffusion_cond(mu_list, Sigma_list, alpha, condition_on, nfeatures,
                          nblocks, nunits, diffusion_steps, nepochs, batch_size, device):
    DiffusionModel, _, _ = import_models()
    model = DiffusionModel(
        nfeatures=nfeatures, nblocks=nblocks, nunits=nunits,
        condition=True, condition_on=condition_on, diffusion_steps=diffusion_steps,
    )
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha, kernel_func=None)
    model.train_model(
        X=None, data_generator=data_gen,
        nepochs=nepochs, batch_size=batch_size,
        condition_on=condition_on, device=device,
    )
    return model


def train_diffusion_uncond(mu_list, Sigma_list, alpha, condition_on,
                            nblocks, nunits, diffusion_steps, nepochs, batch_size, device):
    """Unconditional diffusion over the x (conditioning variable) marginal only."""
    DiffusionModel, _, _ = import_models()
    nfeatures_x = condition_on
    kernel_func = lambda X: X[:, :condition_on]
    model = DiffusionModel(
        nfeatures=nfeatures_x, nblocks=nblocks, nunits=nunits,
        condition=False, diffusion_steps=diffusion_steps,
    )
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha, kernel_func=kernel_func)
    model.train_model(
        X=None, data_generator=data_gen,
        nepochs=nepochs, batch_size=batch_size,
        device=device,
    )
    return model


def train_cm(mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
              nunits, nepochs, batch_size, device):
    _, ConsistencyModeliCT, _ = import_models()
    model = ConsistencyModeliCT(
        nfeatures=nfeatures_y, condition_on=condition_on, nunits=nunits,
    )
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha)
    model.train_model(
        X=None, nepochs=nepochs, batch_size=batch_size,
        device=device, condition=condition_on,
        data_generator=data_gen, use_improved_training=True,
    )
    return model


def train_fm(mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
              nunits, nblocks, nepochs, batch_size, device):
    _, _, FMModel = import_models()
    model = FMModel(
        nfeatures=nfeatures_y, condition_on=condition_on,
        nunits=nunits, nblocks=nblocks, device=device,
    )
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha)
    model.train_FM(
        lr=1e-3, batch_size=batch_size, data_generator=data_gen,
        nepochs=nepochs,
    )
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train models for compare-methods pipeline")
    p.add_argument("--output_dir", type=str, default="compare_methods/output/models")
    p.add_argument("--dim", type=int, default=2, choices=[2, 10],
                   help="Joint distribution dimensionality (2 or 10)")
    p.add_argument("--condition_on", type=int, default=1,
                   help="Number of x (conditioning) dimensions")

    # Architecture
    p.add_argument("--nblocks", type=int, default=3)
    p.add_argument("--nunits",  type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=100)

    # Training lengths
    p.add_argument("--nepochs_diff", type=int, default=20_000,
                   help="Epochs for unconditional + conditional diffusion")
    p.add_argument("--nepochs_cm",   type=int, default=7_500)
    p.add_argument("--nepochs_fm",   type=int, default=10_000)
    p.add_argument("--batch_size_diff", type=int, default=512)
    p.add_argument("--batch_size_cm",   type=int, default=1024)
    p.add_argument("--batch_size_fm",   type=int, default=1024)

    # Which models to train
    p.add_argument("--skip_diff", action="store_true", help="Skip Diffusion training")
    p.add_argument("--skip_cm",   action="store_true", help="Skip CM training")
    p.add_argument("--skip_fm",   action="store_true", help="Skip FM training")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Distribution setup ──────────────────────────────────────────────────
    print(f"Setting up {args.dim}D MoG distribution...", flush=True)
    if args.dim == 2:
        mu_list, Sigma_list, alpha = get_mog_2d(device)
    else:
        mu_list, Sigma_list, alpha = get_mog_10d(device)

    condition_on = args.condition_on
    nfeatures = args.dim           # joint dimensionality
    nfeatures_y = nfeatures - condition_on

    # Save distribution config for run_compare.py to reload
    config = {
        "dim": args.dim,
        "condition_on": condition_on,
        "nfeatures_y": nfeatures_y,
        "nblocks": args.nblocks,
        "nunits": args.nunits,
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved.", flush=True)

    # Also persist the MoG parameters as tensors for the comparison script
    torch.save({
        "mu_list":    [m.cpu() for m in mu_list],
        "Sigma_list": [s.cpu() for s in Sigma_list],
        "alpha":      alpha.cpu(),
    }, os.path.join(args.output_dir, "mog_params.pt"))
    print("MoG params saved.", flush=True)

    # ── 2. Train Diffusion (uncond + cond) ─────────────────────────────────────
    if not args.skip_diff:
        print("\n── Training unconditional diffusion (x-marginal) ──", flush=True)
        model_uncond = train_diffusion_uncond(
            mu_list, Sigma_list, alpha, condition_on,
            args.nblocks, args.nunits, args.diffusion_steps,
            args.nepochs_diff, args.batch_size_diff, device,
        )
        torch.save(model_uncond.state_dict(), os.path.join(args.output_dir, "model_uncond.pt"))
        print("  Saved model_uncond.pt", flush=True)

        print("\n── Training conditional diffusion p(y|x) ──", flush=True)
        model_cond = train_diffusion_cond(
            mu_list, Sigma_list, alpha, condition_on, nfeatures,
            args.nblocks, args.nunits, args.diffusion_steps,
            args.nepochs_diff, args.batch_size_diff, device,
        )
        torch.save(model_cond.state_dict(), os.path.join(args.output_dir, "model_cond.pt"))
        print("  Saved model_cond.pt", flush=True)
    else:
        print("Skipping Diffusion training.", flush=True)

    # ── 3. Train Consistency Model ─────────────────────────────────────────────
    if not args.skip_cm:
        print("\n── Training Consistency Model (iCT) ──", flush=True)
        model_cm = train_cm(
            mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
            args.nunits, args.nepochs_cm, args.batch_size_cm, device,
        )
        torch.save(model_cm.state_dict(), os.path.join(args.output_dir, "model_cm.pt"))
        print("  Saved model_cm.pt", flush=True)
    else:
        print("Skipping CM training.", flush=True)

    # ── 4. Train Flow Matching ─────────────────────────────────────────────────
    if not args.skip_fm:
        print("\n── Training Flow Matching model ──", flush=True)
        model_fm = train_fm(
            mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
            args.nunits, args.nblocks, args.nepochs_fm, args.batch_size_fm, device,
        )
        torch.save(model_fm.state_dict(), os.path.join(args.output_dir, "model_fm.pt"))
        print("  Saved model_fm.pt", flush=True)
    else:
        print("Skipping FM training.", flush=True)

    print(f"\n✅ Training complete. Outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
