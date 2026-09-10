"""
exp_witness_unimodal.py — mean-blindness vs. MMD guidance, and witness back-selection
on concentrated unimodal targets. See HYPOTHESIS.md (pre-registered) and README.md.

Setting: the repo's 2D_cond_1D toy (1-D y | 1-D x). We reuse the branch's own
machinery end to end:
  - GMM params / pretrained models: simulations/scripts/witness_sweep_common.py
    (params from simulations/params/, checkpoints from simulations/checkpoints/
    with the same HuggingFace fallback the sweep scripts use).
  - MMD guidance loop + back-selection hooks: Optimization.optimize_LGD with
    backsel_k / backsel_rule / witness_floor / backsel_generator, unmodified.
  - Kernel/MMD: LossFunctions.{RBF, MMDLoss}. Targets: dist_utils
    generate_mog_samples_not_differentiable on custom 1-D GMM params.

Documented deviations from "just call optimize_LGD" (see README.md):
  1. The `mean` arm: optimize_LGD hardcodes MMDLoss (its `loss` argument is
     ignored), so the mean-matching arm uses optimize_mean_matching() below — a
     line-for-line twin of optimize_LGD's loop in which ONLY the per-j loss is
     replaced by ||mean(y_gen) - mean(y_target)||^2 (no backsel, no diagnostics;
     everything else — DDIM step, r_t noise, -logsumexp aggregation over num_x_t,
     x_t update — is identical).
  2. `import ot` stub: LossFunctions.py imports POT (`ot`) at module level but
     never uses it; POT is not installed in the local base env and pip installs
     are off-limits, so we register an empty placeholder module before importing.
     On the cluster (env with POT installed) the stub is skipped automatically.
  3. Arm 5 (witness + trust-style step cap) is dropped: optimize_LGD on this
     branch has no step-cap toggle, and porting one was out of scope.
  4. The mean arm clips its gradient norm (--mean_grad_clip, default 1.0): the
     unbounded quadratic mean loss diverges to NaN at zeta=1 without it (the
     bounded MMD never does). Direction unchanged — see optimize_mean_matching.
     Superseded by --step_cap_tau when that is set (clip is turned OFF so all
     four arms are symmetric under the cap).
  5. --step_cap_tau (round 2, off by default): opt-in noise-level trust region
     ||Delta_t|| <= tau * sqrt(1 - alphabar_t) on the guidance correction
     Delta_t = zeta*grad, the trust_noise1 semantics of the perf campaign's
     IMPROVEMENTS.md section 1 (branch tfg-generalization-v2; that file is not
     present on this branch). optimize_LGD applies x_t = x_{t-1} - zeta*grad
     inline with no hook to intercept, so when the cap is on ALL FOUR arms run
     through optimize_capped() below — a replicated minimal loop (same approach
     as the mean arm, NOT a monkeypatch), reusing witness_utils.apply_backsel
     for the backsel arms. Ori's files are untouched.

Targets (all share mean 0):
  bimodal_c{1,2}     : 50/50 at ±c, component std 0.1
  unimodal_s{010,025,050}: N(0, s^2), s in {0.10, 0.25, 0.50}

Arms: mean | mmd | mmd_uniform | mmd_witness  (n=32, k=8 by default).

Pairing: experiment_utils.set_run_seed(seed, i) is called immediately before
every (target, arm, restart-i) run, and the post-hoc evaluation reseeds with a
fixed function of i — so restart i is seed-matched across all arms and targets.

Outputs: one JSON in --out_dir with every run's row + meta. Build RESULTS.md
with build_report.py afterwards.

Example (smoke, ~local):
    python exp_witness_unimodal.py --smoke
Full (cluster):
    python exp_witness_unimodal.py --n_restarts 40
"""

import os
import sys
import json
import math
import time
import types
import argparse
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "simulations", "src"),
           os.path.join(_ROOT, "simulations", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Deviation 2 (see module docstring): stub POT if absent — LossFunctions only
# imports it, never uses it.
if importlib.util.find_spec("ot") is None:
    sys.modules["ot"] = types.ModuleType("ot")

import torch

import dist_utils
import experiment_utils
import Optimization
from LossFunctions import MMDLoss, RBF
from witness_sweep_common import EXPERIMENT_CONFIGS, load_or_generate_gmm_params

EXPERIMENT = "2D_cond_1D"
BIMODAL_COMPONENT_STD = 0.1


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def make_targets():
    """All target configs, keyed by name. Every target has mean 0."""
    targets = {}
    for c in (1.0, 2.0):
        targets[f"bimodal_c{int(c)}"] = dict(
            kind="bimodal", c=c, s=None,
            means=[torch.tensor([-c]), torch.tensor([c])],
            variances=[torch.tensor([BIMODAL_COMPONENT_STD ** 2])] * 2,
            weights=torch.tensor([0.5, 0.5]),
            modes=[-c, c], mode_scale=BIMODAL_COMPONENT_STD,
        )
    for s in (0.10, 0.25, 0.50):
        targets[f"unimodal_s{int(round(s * 100)):03d}"] = dict(
            kind="unimodal", c=None, s=s,
            means=[torch.tensor([0.0])],
            variances=[torch.tensor([s ** 2])],
            weights=torch.tensor([1.0]),
            modes=[0.0], mode_scale=s,
        )
    return targets


# ---------------------------------------------------------------------------
# Model loading (checkpoints only — never train locally by accident)
# ---------------------------------------------------------------------------

def ensure_checkpoints(checkpoint_dir, seed, allow_train):
    """Pre-fetch the three 2D_cond_1D checkpoints (local or HF) so that
    load_or_train_models never silently falls into a 20k-epoch local training
    run. Raises if unavailable, unless --allow_train was passed."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    missing = []
    for model_name in ("Diffusion_uncond", "Diffusion_cond", "CM"):
        fname = f"{EXPERIMENT}_{model_name}_seed{seed}.pt"
        local = os.path.join(checkpoint_dir, fname)
        if os.path.exists(local):
            continue
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=experiment_utils.HF_REPO_ID,
                filename=f"simulations/checkpoints/{EXPERIMENT}/{fname}",
                local_dir=os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_dir))),
                local_dir_use_symlinks=False,
            )
            print(f"[Checkpoint] fetched {fname} from HF")
        except Exception as e:
            print(f"[Checkpoint] could not fetch {fname}: {e}")
            missing.append(fname)
    if missing and not allow_train:
        raise RuntimeError(
            f"Missing checkpoints {missing} and --allow_train not set; refusing "
            f"to launch a local 20k-epoch training run."
        )


def load_models(checkpoint_dir, seed, device, allow_train):
    from witness_sweep_common import load_or_train_models
    cfg = EXPERIMENT_CONFIGS[EXPERIMENT]
    params_dir = os.path.join(_ROOT, "simulations", "params")
    results_dir = os.path.join(_ROOT, "simulations", "results", EXPERIMENT)
    mu_list, Sigma_list, alpha, _, _, _, x_star = load_or_generate_gmm_params(
        cfg, params_dir, results_dir, EXPERIMENT, seed
    )
    ensure_checkpoints(checkpoint_dir, seed, allow_train)
    model_uncond, model_cond, _model_cm = load_or_train_models(
        cfg, mu_list, Sigma_list, alpha, checkpoint_dir, EXPERIMENT,
        seed, device, force_retrain=False,
    )
    model_uncond.eval()
    model_cond.eval()
    return model_uncond, model_cond, mu_list, Sigma_list, alpha


# ---------------------------------------------------------------------------
# Mean-matching arm — deviation 1 (thin twin of Optimization.optimize_LGD)
# ---------------------------------------------------------------------------

def optimize_mean_matching(model_uncond, model_cond, target_mean, nsamples=32,
                           num_x_t=1, device="cpu", zeta=1.0, grad_clip=1.0):
    """Line-for-line copy of optimize_LGD's loop with ONLY the loss changed to
    ||mean(y_gen) - target_mean||^2. No backsel (mean of all samples is the
    statistic being matched), no history/diagnostics. Returns x_t_final [1, dx].

    grad_clip (deviation 3, see README): unlike the multi-bandwidth-RBF MMD,
    which is bounded (so optimize_LGD's raw zeta*grad update is always tame),
    the quadratic mean loss is unbounded and creates positive feedback through
    the sampled conditional chain — at zeta=1 the trajectory reaches NaN within
    ~10 diffusion steps. Clipping the gradient NORM (default 1.0) caps the step
    size without changing its direction; the mean arm still optimizes pure
    mean-matching, which is the property under test.
    """
    target_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device).view(-1)

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)

    for t in range(model_uncond.diffusion_steps - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)

        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0)
        current_var = model_uncond.betas[t].to(device)
        r_t = current_var / torch.sqrt(1 + current_var ** 2)

        losses = []
        for _j in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            condition = x0_sample.view(1, -1).repeat(nsamples, 1)
            target_samples, _, _ = model_cond.sample(
                nsamples=nsamples, condition_x=condition, device=device)
            target_samples = target_samples[:, model_cond.condition_on:]

            loss_val = (target_samples.mean(dim=0) - target_mean).pow(2).sum()
            losses.append(-loss_val)

        log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        grad = torch.autograd.grad(log_mean_exp_loss, x_t)[0]

        with torch.no_grad():
            if torch.isnan(grad).any():          # same guard as run_mlgd_f.py
                grad = torch.zeros_like(grad)
            gn = grad.norm()
            if grad_clip is not None and gn > grad_clip:
                grad = grad * (grad_clip / gn)
            x_t = x_t_minus_1.detach().clone() - zeta * grad

    return x_t.detach().clone()


# ---------------------------------------------------------------------------
# Round 2: capped loop for ALL arms — deviation 5 (see module docstring)
# ---------------------------------------------------------------------------

def optimize_capped(model_uncond, model_cond, mog_means, mog_variances, weights,
                    nsamples=32, num_x_t=1, device="cpu", zeta=1.0,
                    step_cap_tau=1.0, inv_sqrt_alpha=False,
                    loss_mode="mmd", target_mean=None,
                    backsel_k=None, backsel_rule="uniform", witness_floor=0.3,
                    backsel_generator=None):
    """Replicates optimize_LGD's loop exactly (incl. witness_utils.apply_backsel
    for the backsel arms; loss_mode='mean' gives the mean arm, no norm clip) and
    adds ONLY the noise-level trust region on the applied correction:
        ||Delta_t|| <= step_cap_tau * sqrt(1 - alphabar_t),  Delta_t = zeta*grad.
    Rescales (direction preserved), never zeroes. NaN grads are dropped, matching
    run_mlgd_f.py's guard."""
    from witness_utils import apply_backsel

    mmd_loss = MMDLoss(kernel=RBF())
    if target_mean is not None:
        target_mean = torch.as_tensor(target_mean, dtype=torch.float32,
                                      device=device).view(-1)

    x_t = torch.zeros(model_uncond.nfeatures, device=device, requires_grad=True)
    x_t = x_t.unsqueeze(0)

    for t in range(model_uncond.diffusion_steps - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)

        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0)
        current_var = model_uncond.betas[t].to(device)
        r_t = current_var / torch.sqrt(1 + current_var ** 2)

        losses = []
        for _j in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            condition = x0_sample.view(1, -1).repeat(nsamples, 1)
            target_samples, _, _ = model_cond.sample(
                nsamples=nsamples, condition_x=condition, device=device)
            target_samples = target_samples[:, model_cond.condition_on:]

            if loss_mode == "mean":
                loss_val = (target_samples.mean(dim=0) - target_mean).pow(2).sum()
            else:
                mog_samples = dist_utils.generate_mog_samples_not_differentiable(
                    nsamples, mog_means, mog_variances, weights)
                if backsel_k is not None:
                    target_samples, _info = apply_backsel(
                        target_samples, mog_samples, backsel_k, rule=backsel_rule,
                        witness_floor=witness_floor, generator=backsel_generator)
                loss_val = mmd_loss(target_samples, mog_samples)
            losses.append(-loss_val)

        log_mean_exp_loss = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        grad = torch.autograd.grad(log_mean_exp_loss, x_t)[0]

        with torch.no_grad():
            if torch.isnan(grad).any():
                grad = torch.zeros_like(grad)
            step_scale = (1.0 / torch.sqrt(model_uncond.alphas[t].to(device))) \
                if inv_sqrt_alpha else zeta
            delta = step_scale * grad
            cap = step_cap_tau * torch.sqrt(1.0 - model_uncond.baralphas[t].to(device))
            dn = delta.norm()
            if dn > cap:
                delta = delta * (cap / dn)
            x_t = x_t_minus_1.detach().clone() - delta

    return x_t.detach().clone()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_final(x_final, model_cond, tgt, n_eval, eval_seed, device):
    """Fresh-sample metrics at the returned x. Uses its own seed so evaluation
    randomness is identical across arms for a given restart index."""
    torch.manual_seed(eval_seed)
    with torch.no_grad():
        condition = x_final.view(1, -1).repeat(n_eval, 1)
        y, _, _ = model_cond.sample(nsamples=n_eval, condition_x=condition, device=device)
        y = y[:, model_cond.condition_on:].detach().cpu()
        tgt_samples = dist_utils.generate_mog_samples_not_differentiable(
            n_eval, tgt["means"], tgt["variances"], tgt["weights"])
        mmd = MMDLoss(kernel=RBF())(y, tgt_samples).item()

        yv = y.view(-1)
        modes = torch.tensor(tgt["modes"], dtype=torch.float32)
        dist_to_mode = (yv.view(-1, 1) - modes.view(1, -1)).abs().min(dim=1).values
        return {
            "final_mmd": mmd,
            "abs_mean_err": abs(yv.mean().item()),   # every target has mean 0
            "gen_std": yv.std().item(),
            "frac_within_2s": (dist_to_mode <= 2.0 * tgt["mode_scale"]).float().mean().item(),
            "x_final": x_final.view(-1).cpu().tolist(),
        }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(arm, tgt, restart_idx, args, model_uncond, model_cond,
            mu_list, Sigma_list, alpha, device):
    run_seed = experiment_utils.set_run_seed(args.seed, restart_idx)
    t0 = time.time()

    if args.step_cap_tau is not None:
        # Round 2: symmetric capped loop for ALL arms (deviation 5).
        backsel_k, backsel_rule = None, "uniform"
        if arm == "mmd_uniform":
            backsel_k = args.backsel_k
        elif arm == "mmd_witness":
            backsel_k, backsel_rule = args.backsel_k, "witness"
        elif arm not in ("mean", "mmd"):
            raise ValueError(f"unknown arm {arm!r}")
        x_final = optimize_capped(
            model_uncond, model_cond, tgt["means"], tgt["variances"], tgt["weights"],
            nsamples=args.nsamples, num_x_t=args.num_x_t, device=device,
            zeta=args.zeta, step_cap_tau=args.step_cap_tau,
            inv_sqrt_alpha=args.inv_sqrt_alpha,
            loss_mode=("mean" if arm == "mean" else "mmd"),
            target_mean=([0.0] if arm == "mean" else None),
            backsel_k=backsel_k, backsel_rule=backsel_rule,
            witness_floor=args.witness_floor,
            backsel_generator=torch.Generator().manual_seed(run_seed))
        internal_final_loss = float("nan")
    elif arm == "mean":
        x_final = optimize_mean_matching(
            model_uncond, model_cond, target_mean=[0.0],
            nsamples=args.nsamples, num_x_t=args.num_x_t, device=device,
            zeta=args.zeta, grad_clip=args.mean_grad_clip)
        internal_final_loss = float("nan")
    else:
        backsel_k = None
        backsel_rule = "uniform"
        if arm == "mmd_uniform":
            backsel_k = args.backsel_k
        elif arm == "mmd_witness":
            backsel_k, backsel_rule = args.backsel_k, "witness"
        elif arm != "mmd":
            raise ValueError(f"unknown arm {arm!r}")
        backsel_generator = torch.Generator().manual_seed(run_seed)
        x_final, _, internal_loss = Optimization.optimize_LGD(
            model_uncond, model_cond,
            tgt["means"], tgt["variances"], tgt["weights"],
            mu_list, Sigma_list, alpha,
            nsamples=args.nsamples, num_x_t=args.num_x_t, loss="MMD", CM=False,
            device=device, zeta=args.zeta,
            backsel_k=backsel_k, backsel_rule=backsel_rule,
            witness_floor=args.witness_floor, backsel_generator=backsel_generator,
            use_inv_sqrt_alpha_scale=args.inv_sqrt_alpha,
        )
        internal_final_loss = internal_loss.item()

    elapsed = time.time() - t0
    metrics = evaluate_final(x_final, model_cond, tgt, args.n_eval,
                             eval_seed=args.seed + 100_000 + restart_idx, device=device)
    row = {
        "target": tgt["name"], "kind": tgt["kind"], "c": tgt["c"], "s": tgt["s"],
        "arm": arm, "restart": restart_idx, "run_seed": run_seed,
        "internal_final_loss": internal_final_loss, "time_s": elapsed, **metrics,
    }
    print(f"[{tgt['name']} | {arm} | {restart_idx + 1}/{args.n_restarts}] "
          f"MMD={row['final_mmd']:.5f} |mean|={row['abs_mean_err']:.4f} "
          f"std={row['gen_std']:.4f} frac2s={row['frac_within_2s']:.3f} "
          f"({elapsed:.1f}s)", flush=True)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_ARMS = ["mean", "mmd", "mmd_uniform", "mmd_witness"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of target names (default: all five). "
                        "Known: bimodal_c1 bimodal_c2 unimodal_s010 unimodal_s025 unimodal_s050")
    p.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=ALL_ARMS)
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--backsel_k", type=int, default=8)
    p.add_argument("--num_x_t", type=int, default=1)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--witness_floor", type=float, default=0.3)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--inv_sqrt_alpha", action="store_true",
                   help="Use optimize_LGD's use_inv_sqrt_alpha_scale (TFG line-9 "
                        "1/sqrt(alpha_t) step convention, Ori's protocol) instead "
                        "of constant zeta. Round-1 array ran WITHOUT it.")
    p.add_argument("--step_cap_tau", type=float, default=None,
                   help="Opt-in noise-level trust region on the guidance step, "
                        "||Delta_t|| <= tau*sqrt(1-alphabar_t), applied to ALL "
                        "four arms via optimize_capped (mean arm's norm clip is "
                        "then off). Default None = round-1 behaviour.")
    p.add_argument("--mean_grad_clip", type=float, default=1.0,
                   help="Gradient-norm clip for the mean arm only (deviation 3; "
                        "the unbounded quadratic mean loss diverges without it)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--tag", default="", help="suffix for the output JSON filename")
    p.add_argument("--smoke", action="store_true",
                   help="8 restarts, targets {bimodal_c1, unimodal_s025}, all arms")
    p.add_argument("--allow_train", action="store_true",
                   help="Permit local training if checkpoints are unavailable (heavy!)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_restarts = min(args.n_restarts, 8)
        if args.targets is None:
            args.targets = ["bimodal_c1", "unimodal_s025"]
        if not args.tag:
            args.tag = "smoke"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    experiment_utils.set_global_seed(args.seed)

    checkpoint_dir = os.path.join(_ROOT, "simulations", "checkpoints", EXPERIMENT)
    model_uncond, model_cond, mu_list, Sigma_list, alpha = load_models(
        checkpoint_dir, args.seed, device, args.allow_train)

    all_targets = make_targets()
    names = args.targets or list(all_targets.keys())
    unknown = [n for n in names if n not in all_targets]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; known: {list(all_targets)}")

    rows = []
    for name in names:
        tgt = dict(all_targets[name], name=name)
        for arm in args.arms:
            for i in range(args.n_restarts):
                rows.append(run_one(arm, tgt, i, args, model_uncond, model_cond,
                                    mu_list, Sigma_list, alpha, device))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.out_dir, f"witness_unimodal_seed{args.seed}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "experiment": EXPERIMENT,
                "targets": names, "arms": args.arms,
                "n_restarts": args.n_restarts, "nsamples": args.nsamples,
                "backsel_k": args.backsel_k, "num_x_t": args.num_x_t,
                "n_eval": args.n_eval, "witness_floor": args.witness_floor,
                "zeta": args.zeta, "step_cap_tau": args.step_cap_tau,
                "inv_sqrt_alpha": args.inv_sqrt_alpha,
                "seed": args.seed, "device": device,
                "bimodal_component_std": BIMODAL_COMPONENT_STD,
                "environment": experiment_utils.get_environment_info(),
            },
            "rows": rows,
        }, f, indent=2)
    print(f"[Results] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
