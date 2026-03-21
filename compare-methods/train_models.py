"""
train_models.py — Train Diffusion (uncond + cond), ConsistencyModel, and FlowMatching
models on a MoG distribution and save checkpoints for D-Flow / LGD / LGD-CM comparison.

Usage (from repo root):
    python compare_methods/train_models.py --dim 2 --output_dir compare_methods/output/models_2d

Usage (quick local smoke-test, from repo root):
    python compare_methods/train_models.py --dim 2 --output_dir /tmp/test_models \
        --nepochs_diff 200 --nepochs_cm 100 --nepochs_fm 100

Usage (on cluster):
    sbatch compare_methods/submit_train.sh 2
"""

import argparse
import os
import sys
import json

import torch
import numpy as np
from functools import partial

# ── path setup ────────────────────────────────────────────────────────────────
# Works whether you run from repo root OR from inside compare_methods/
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
for p in [script_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# dist_utils lives in the same folder as this script
from dist_utils import (
    generate_mog_samples_not_differentiable,
    compute_conditionals,
    compute_alpha,
    filter_and_normalize,
)


# ── exact MoG configs from mog_2d_full.txt / mog_10d_full.txt ────────────────

def get_mog_2d(device):
    """2D MoG — exact params from mog_2d_full.txt. condition_on=1 (x=first coord)."""
    mu_list = [
        torch.tensor([-5.0,  5.0]),
        torch.tensor([-5.0, -5.0]),
        torch.tensor([ 5.0,  3.0]),
        torch.tensor([ 5.0, -1.0]),
        torch.tensor([ 0.0, -3.0]),
        torch.tensor([-2.0,  4.0]),
        torch.tensor([-2.0, -3.0]),
        torch.tensor([ 1.0,  2.0]),
        torch.tensor([-7.0,  1.0]),
        torch.tensor([ 7.0,  5.0]),
        torch.tensor([ 0.0, -5.0]),
    ]
    Sigma = torch.tensor([[0.5,    0.195],
                          [0.195,  0.2  ]])
    Sigma_list = [Sigma.clone() for _ in mu_list]
    alpha = torch.tensor([0.09090909361839294] * 11)

    # Known target: q(Y | X = -5.0)
    target_info = {
        "x_star":           [-5.0],
        "target_means":     [ [5.0], [-5.0] ],      # list of 1D means
        "target_variances": [ [[0.12395]], [[0.12395]] ],
        "target_weights":   [0.5, 0.5],
    }
    return ([m.float().to(device) for m in mu_list],
            [s.float().to(device) for s in Sigma_list],
            alpha.float().to(device),
            target_info)


def get_mog_10d(device):
    """10D MoG — exact params from mog_10d_full.txt. condition_on=1 (x=first coord)."""
    mu_list = [
        torch.tensor([-4.5403668291751025,  2.711368262291067,   0.551274469447325,
                      -11.295048017829952,   3.0334978489759252, -0.6915483043410363,
                        4.155081389212047,  -1.2385372712593403, -4.014704711504637,
                       -2.815263993268942]),
        torch.tensor([-4.46145254390517,  -0.2912508954184818, -0.9775478869981234,
                       -4.828179655841714,   2.1120766059107767,  1.3365849897368283,
                       -2.105975783773068,  -2.5534999964159826, -7.863326067815296,
                       -0.616238788101648]),
        torch.tensor([17.93494655158687,  -9.156450682114421,   7.993501574381028,
                       -6.38503517402635,   1.627507711787779,  -2.3957245810989263,
                        6.8950413071064105, 12.642786526878373,   2.05370831840688,
                       -4.940035594813712]),
        torch.tensor([-4.5403668291751025,  2.711368262291067,   0.551274469447325,
                      -11.295048017829952,   3.0334978489759252, -0.6915483043410363,
                        4.155081389212047,  -1.2385372712593403, -4.014704711504637,
                        1.182873383979139]),
    ]

    Sigma_list = [
        torch.tensor([
            [ 2.2145,  0.7875,  0.8889, -0.7362,  1.7716, -2.4913, -0.7966,  0.9866,  0.4105, -1.1323],
            [ 0.7875, 12.6431, -3.0298, -3.9632,  1.1073,  8.7146, -4.6469, -9.4260,  0.9222, -0.5017],
            [ 0.8889, -3.0298, 14.1583,  7.6042, -1.6193, -6.3715,  0.9514, -1.0965, -1.5863,  4.3088],
            [-0.7362, -3.9632,  7.6042,  7.7098, -0.4385,  0.7378, -1.0503, -1.2580, -2.3861,  3.1586],
            [ 1.7716,  1.1073, -1.6193, -0.4385,  7.5157, -1.6259, -0.3640, -0.1964,  2.1728, -2.1201],
            [-2.4913,  8.7146, -6.3715,  0.7378, -1.6259, 24.8138, -6.3766,-11.1967, -2.4354,  1.7954],
            [-0.7966, -4.6469,  0.9514, -1.0503, -0.3640, -6.3766, 12.5876,  5.5673, -3.7757,  0.6924],
            [ 0.9866, -9.4260, -1.0965, -1.2580, -0.1964,-11.1967,  5.5673, 14.4055,  0.1591, -1.0547],
            [ 0.4105,  0.9222, -1.5863, -2.3861,  2.1728, -2.4354, -3.7757,  0.1591,  7.2865, -0.4520],
            [-1.1323, -0.5017,  4.3088,  3.1586, -2.1201,  1.7954,  0.6924, -1.0547, -0.4520,  3.6325],
        ]),
        torch.tensor([
            [ 7.4996,  0.0871, -1.6611,  1.3706,  0.0550, -2.2879,  0.1811,  3.0109, -0.8862,  1.4527],
            [ 0.0871,  5.4189, -1.9394,  0.1206, -0.4398, -1.7742, -1.6257,  2.7045,  0.5872, -3.1182],
            [-1.6611, -1.9394, 10.1114,  0.3951, -0.5614, -2.4620,  3.8336,  4.7854, -3.0333,  7.2679],
            [ 1.3706,  0.1206,  0.3951, 10.5065, -2.0386, -1.8694,  4.7913,  1.5607, -2.1745, -1.3382],
            [ 0.0550, -0.4398, -0.5614, -2.0386,  1.6220,  3.1300, -1.9319, -1.8859,  1.8141, -0.4319],
            [-2.2879, -1.7742, -2.4620, -1.8694,  3.1300, 12.7104,  1.6860, -6.4587,  4.7036, -5.7996],
            [ 0.1811, -1.6257,  3.8336,  4.7913, -1.9319,  1.6860, 11.3179,  2.0064, -2.0780,  0.4777],
            [ 3.0109,  2.7045,  4.7854,  1.5607, -1.8859, -6.4587,  2.0064, 12.5731, -5.8153,  4.8034],
            [-0.8862,  0.5872, -3.0333, -2.1745,  1.8141,  4.7036, -2.0780, -5.8153,  6.6628, -2.5879],
            [ 1.4527, -3.1182,  7.2679, -1.3382, -0.4319, -5.7996,  0.4777,  4.8034, -2.5879, 11.2330],
        ]),
        torch.tensor([
            [ 4.9340,  1.6848,  0.5804,  3.8902,  3.5687, -0.0937, -0.5947, -3.6283,  0.9518, -0.6734],
            [ 1.6848, 13.0009, -4.8762, -0.3861, -6.5611,  3.8542, -2.0594,  1.0585, -0.9587, -0.3687],
            [ 0.5804, -4.8762, 10.5357, -0.7961,  2.3913,  0.0824,  0.6873,  3.5850,  1.2967,  0.7105],
            [ 3.8902, -0.3861, -0.7961, 11.1083, -0.0510,  3.0937, -1.4238, -4.9671, -0.2028,  0.8851],
            [ 3.5687, -6.5611,  2.3913, -0.0510, 16.8237, -8.5156,  1.5091, -3.6466,  0.1362, -0.1647],
            [-0.0937,  3.8542,  0.0824,  3.0937, -8.5156, 12.1743, -1.3181, -0.7680,  7.9209,  1.6320],
            [-0.5947, -2.0594,  0.6873, -1.4238,  1.5091, -1.3181,  8.9229,  0.2006,  1.3853, -1.4049],
            [-3.6283,  1.0585,  3.5850, -4.9671, -3.6466, -0.7680,  0.2006, 14.3054, -5.3788,  1.2659],
            [ 0.9518, -0.9587,  1.2967, -0.2028,  0.1362,  7.9209,  1.3853, -5.3788, 12.1119,  0.4970],
            [-0.6734, -0.3687,  0.7105,  0.8851, -0.1647,  1.6320, -1.4049,  1.2659,  0.4970,  2.1738],
        ]),
        torch.tensor([
            [ 9.6519,  4.7700,  4.2343, -3.3847,  3.4350,  2.2656, -5.5452, -3.6149,  0.7715,  1.7369],
            [ 4.7700,  7.9226,  0.0392, -4.0536,  6.6204,  1.1808, -5.2269, -0.9617,  1.0631, -0.6315],
            [ 4.2343,  0.0392, 22.7346,  8.0819, -0.6512,  7.5987, -1.6651,  7.0512, -0.2359,  4.0643],
            [-3.3847, -4.0536,  8.0819, 11.0285,  0.2939,  0.4066,  3.6676,  2.1717, -3.0936, -0.4415],
            [ 3.4350,  6.6204, -0.6512,  0.2939, 13.1022,  2.9650, -2.2575, -3.2725, -1.0926, -6.6981],
            [ 2.2656,  1.1808,  7.5987,  0.4066,  2.9650, 12.3785, -0.3964,  5.6983,  3.1714, -1.9920],
            [-5.5452, -5.2269, -1.6651,  3.6676, -2.2575, -0.3964,  9.3911,  1.6889, -2.4758, -1.8364],
            [-3.6149, -0.9617,  7.0512,  2.1717, -3.2725,  5.6983,  1.6889,  9.2574,  3.0359,  0.5280],
            [ 0.7715,  1.0631, -0.2359, -3.0936, -1.0926,  3.1714, -2.4758,  3.0359,  5.9602,  0.1954],
            [ 1.7369, -0.6315,  4.0643, -0.4415, -6.6981, -1.9920, -1.8364,  0.5280,  0.1954, 10.0691],
        ]),
    ]
    alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])

    # Known target: q(Y | X = x*) from mog_10d_full.txt
    # x* is the first coordinate of mu[0]: -4.5404
    # y* is the remaining 9 coordinates
    target_info = {
        "x_star":           [-4.5404],
        "target_means":     [ [-2.815263534200911], [1.1828735569838675] ],
        "target_variances": [ [[0.1718879873583199]], [[1.489962496721585]] ],
        "target_weights":   [0.5842540264129639, 0.41574597358703613],
    }
    return ([m.float().to(device) for m in mu_list],
            [s.float().to(device) for s in Sigma_list],
            alpha.float().to(device),
            target_info)


# ── model imports ─────────────────────────────────────────────────────────────

def import_models():
    """Lazy import — keeps --help fast even if deps are missing."""
    from Diffusion import DiffusionModel
    from ConsistencyModels import ConsistencyModeliCT
    from FlowMatching import FMModel
    return DiffusionModel, ConsistencyModeliCT, FMModel


# ── training helpers ──────────────────────────────────────────────────────────

def train_diffusion_cond(mu_list, Sigma_list, alpha, condition_on, nfeatures,
                          nblocks, nunits, diffusion_steps, nepochs, batch_size, device,
                          wandb_run=None, wandb_prefix="diff_cond", log_every=100):
    DiffusionModel, _, _ = import_models()
    model = DiffusionModel(
        nfeatures=nfeatures, nblocks=nblocks, nunits=nunits,
        condition=True, condition_on=condition_on, diffusion_steps=diffusion_steps,
    )
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha, kernel_func=None)
    model.train_model(X=None, data_generator=data_gen,
                      nepochs=nepochs, batch_size=batch_size,
                      condition_on=condition_on, device=device,
                      wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


def train_diffusion_uncond(mu_list, Sigma_list, alpha, condition_on,
                            nblocks, nunits, diffusion_steps, nepochs, batch_size, device,
                            wandb_run=None, wandb_prefix="diff_uncond", log_every=100):
    """Unconditional diffusion over the x-marginal (first `condition_on` dims)."""
    DiffusionModel, _, _ = import_models()
    kernel_func = lambda X: X[:, :condition_on]
    model = DiffusionModel(
        nfeatures=condition_on, nblocks=nblocks, nunits=nunits,
        condition=False, diffusion_steps=diffusion_steps,
    )
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha, kernel_func=kernel_func)
    model.train_model(X=None, data_generator=data_gen,
                      nepochs=nepochs, batch_size=batch_size, device=device,
                      wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


def train_cm(mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
              nunits, nepochs, batch_size, device,
              wandb_run=None, wandb_prefix="ict", log_every=100):
    _, ConsistencyModeliCT, _ = import_models()
    model = ConsistencyModeliCT(nfeatures=nfeatures_y, condition_on=condition_on, nunits=nunits)
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha)
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
    data_gen = partial(generate_mog_samples_not_differentiable,
                       means=mu_list, variances=Sigma_list, weights=alpha)
    model.train_FM(lr=1e-3, batch_size=batch_size, data_generator=data_gen, nepochs=nepochs,
                   wandb_run=wandb_run, wandb_prefix=wandb_prefix, log_every=log_every)
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train models for compare-methods pipeline")
    p.add_argument("--output_dir", type=str, default="compare_methods/output/models")
    p.add_argument("--dim", type=int, default=2, choices=[2, 10],
                   help="Joint distribution dimensionality")
    p.add_argument("--condition_on", type=int, default=1)

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

    # wandb
    p.add_argument("--wandb_project", type=str, default="compare-methods-train")
    p.add_argument("--wandb_entity",  type=str, default="conditional-matching")
    p.add_argument("--no_wandb",      action="store_true",
                   help="Disable wandb logging (e.g. for local runs)")
    p.add_argument("--log_every",     type=int, default=100,
                   help="Log loss to wandb every N epochs/steps")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Distribution ────────────────────────────────────────────────────────
    print(f"Setting up {args.dim}D MoG...", flush=True)
    if args.dim == 2:
        mu_list, Sigma_list, alpha, target_info = get_mog_2d(device)
    else:
        mu_list, Sigma_list, alpha, target_info = get_mog_10d(device)

    condition_on = args.condition_on
    nfeatures    = args.dim
    nfeatures_y  = nfeatures - condition_on

    # ── 2. Save config + MoG params ────────────────────────────────────────────
    config = {
        "dim":             args.dim,
        "condition_on":    condition_on,
        "nfeatures_y":     nfeatures_y,
        "nblocks":         args.nblocks,
        "nunits":          args.nunits,
        "diffusion_steps": args.diffusion_steps,
        "seed":            args.seed,
        "target_info":     target_info,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    torch.save({
        "mu_list":    [m.cpu() for m in mu_list],
        "Sigma_list": [s.cpu() for s in Sigma_list],
        "alpha":      alpha.cpu(),
    }, os.path.join(args.output_dir, "mog_params.pt"))
    print("Config + MoG params saved.", flush=True)

    # ── 3. wandb init ──────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={**vars(args), **config},
            name=f"train_{args.dim}d_seed{args.seed}",
        )
        print(f"wandb run: {wandb_run.name}", flush=True)

    # ── 4. Diffusion ───────────────────────────────────────────────────────────
    if not args.skip_diff:
        print("\n── Training unconditional diffusion (x-marginal) ──", flush=True)
        model_uncond = train_diffusion_uncond(
            mu_list, Sigma_list, alpha, condition_on,
            args.nblocks, args.nunits, args.diffusion_steps,
            args.nepochs_diff, args.batch_size_diff, device,
            wandb_run=wandb_run, wandb_prefix="diff_uncond",
            log_every=args.log_every,
        )
        torch.save(model_uncond.state_dict(), os.path.join(args.output_dir, "model_uncond.pt"))
        print("  Saved model_uncond.pt", flush=True)

        print("\n── Training conditional diffusion p(y|x) ──", flush=True)
        model_cond = train_diffusion_cond(
            mu_list, Sigma_list, alpha, condition_on, nfeatures,
            args.nblocks, args.nunits, args.diffusion_steps,
            args.nepochs_diff, args.batch_size_diff, device,
            wandb_run=wandb_run, wandb_prefix="diff_cond",
            log_every=args.log_every,
        )
        torch.save(model_cond.state_dict(), os.path.join(args.output_dir, "model_cond.pt"))
        print("  Saved model_cond.pt", flush=True)
    else:
        print("Skipping Diffusion.", flush=True)

    # ── 5. Consistency Model ───────────────────────────────────────────────────
    if not args.skip_cm:
        print("\n── Training Consistency Model (iCT) ──", flush=True)
        model_cm = train_cm(
            mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
            args.nunits, args.nepochs_cm, args.batch_size_cm, device,
            wandb_run=wandb_run, wandb_prefix="ict",
            log_every=args.log_every,
        )
        torch.save(model_cm.state_dict(), os.path.join(args.output_dir, "model_cm.pt"))
        print("  Saved model_cm.pt", flush=True)
    else:
        print("Skipping CM.", flush=True)

    # ── 6. Flow Matching ───────────────────────────────────────────────────────
    if not args.skip_fm:
        print("\n── Training Flow Matching model ──", flush=True)
        model_fm = train_fm(
            mu_list, Sigma_list, alpha, condition_on, nfeatures_y,
            args.nunits, args.nblocks, args.nepochs_fm, args.batch_size_fm, device,
            wandb_run=wandb_run, wandb_prefix="fm",
            log_every=args.log_every,
        )
        torch.save(model_fm.state_dict(), os.path.join(args.output_dir, "model_fm.pt"))
        print("  Saved model_fm.pt", flush=True)
    else:
        print("Skipping FM.", flush=True)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n✅ Training complete. Outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()