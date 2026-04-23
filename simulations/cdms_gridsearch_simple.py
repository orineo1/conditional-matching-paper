"""
cdms_gridsearch_simple.py
=========================
Minimal adaptation of the notebook for cluster grid search.
Sweeps: grad_clamp × diffusion_steps
Logs per-beta plots + JS divergence (empirical vs analytical Q) to wandb.

Launch:
    sbatch cdms_gridsearch_simple.sh
Smoke test:
    python cdms_gridsearch_simple.py --config_id 0 --smoke_test
List configs:
    python cdms_gridsearch_simple.py --list_configs
"""

import os, sys, math, json, argparse, itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    "/sci/labs/orzuk/ori_m/conditional-matching-paper"
)
BASE_DIR = os.path.join(REPO_ROOT, "simulations")
SRC_PATH = os.path.join(BASE_DIR, "src")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import importlib
import Diffusion, ConsistencyModels, dist_utils, experiment_utils
for mod in [Diffusion, ConsistencyModels, dist_utils, experiment_utils]:
    importlib.reload(mod)

# ─────────────────────────────────────────────────────────────────────────────
# Fixed constants — identical to notebook
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME  = "2D_cond_1D"
GLOBAL_SEED      = 42
CONDITION_ON     = 1

ZETA_VALUES_FULL  = [0.0, 1.0, 4.0, 6.0]
ZETA_VALUES_SMOKE = [0.0, 1.0]

NBLOCKS    = 3
NUNITS     = 128
NEPOCHS    = 40_000
BATCH_SIZE = 1024

NUM_X_T        = 1
N_CDMS_SAMPLES = 1_000

# ─────────────────────────────────────────────────────────────────────────────
# Grid — only clamp + diffusion_steps swept
# ─────────────────────────────────────────────────────────────────────────────
GRID = {
    "grad_clamp":      [0.000005, 0.00003, 0.00007, 0.00015,0.03,0.1,1],
    "diffusion_steps": [1000],
}

def all_configs():
    keys, values = zip(*GRID.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ─────────────────────────────────────────────────────────────────────────────
# GMM — identical to notebook
# ─────────────────────────────────────────────────────────────────────────────
def build_gmm():
    mu_list = [
        torch.tensor([-5.,  5.]),
        torch.tensor([-5., -5.]),
        torch.tensor([ 5.,  3.]),
        torch.tensor([ 5., -1.]),
        torch.tensor([ 0., -3.]),
        torch.tensor([-2.,  4.]),
        torch.tensor([-2., -3.]),
        torch.tensor([ 1.,  2.]),
        torch.tensor([-8.,  1.]),
        torch.tensor([ 7.,  5.]),
        torch.tensor([ 0., -5.]),
    ]
    Sigma_list = [
        torch.tensor([[0.5000, 0.1950],
                      [0.1950, 0.2000]])
    ] * len(mu_list)
    alpha  = torch.tensor([1. / len(mu_list)] * len(mu_list))
    x_star = torch.tensor([-5.])

    mu_temp, Sigma_temp = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star)
    temp_alpha          = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star)
    mog_means, mog_variances, weights = dist_utils.filter_and_normalize(
        mu_temp, Sigma_temp, temp_alpha, threshold=0.01
    )
    return mu_list, Sigma_list, alpha, x_star, mog_means, mog_variances, weights


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers — identical to notebook
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_3d_cov(S):
    if S.dim() == 1:
        return S.view(-1, 1, 1)
    if S.dim() == 2:
        return S.unsqueeze(-1) if S.shape[-1] == 1 else torch.diag_embed(S)
    return S


def gmm_l2_diff(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q):
    if mu_p.dim() == 1: mu_p = mu_p.unsqueeze(0)
    if mu_q.dim() == 1: mu_q = mu_q.unsqueeze(0)
    Sigma_p = _ensure_3d_cov(Sigma_p)
    Sigma_q = _ensure_3d_cov(Sigma_q)
    D = mu_p.shape[-1]

    def inner(m1, S1, w1, m2, S2, w2):
        diff  = m1.unsqueeze(1) - m2.unsqueeze(0)
        S_sum = S1.unsqueeze(1) + S2.unsqueeze(0)
        _, logdet = torch.linalg.slogdet(S_sum)
        quad = torch.einsum('ijk,ijkl,ijl->ij', diff,
                            torch.linalg.inv(S_sum), diff)
        log_val = -0.5 * (D * math.log(2 * math.pi) + logdet + quad)
        log_w   = torch.log(w1).unsqueeze(1) + torch.log(w2).unsqueeze(0)
        return torch.exp(log_w + log_val).sum()

    pp = inner(mu_p, Sigma_p, w_p, mu_p, Sigma_p, w_p)
    qq = inner(mu_q, Sigma_q, w_q, mu_q, Sigma_q, w_q)
    pq = inner(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q)
    return pp - 2 * pq + qq


def compute_l2_gmm_loss(x0_sample, mu_list, Sigma_list, alpha,
                        mog_means, mog_variances, weights, device):
    x0_cpu = torch.clamp(x0_sample.detach(), -15., 15.).view(-1).cpu()

    mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x0_cpu)
    w_pred              = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x0_cpu)

    Sig_p = _ensure_3d_cov(Sigma_pred.to(device).detach())
    w_p   = w_pred.to(device).detach()
    mu_q  = mog_means.squeeze(1).to(device).detach()
    Sig_q = _ensure_3d_cov(mog_variances.squeeze(1).to(device).detach())
    w_q   = weights.to(device).detach()

    mu_p_list = []
    for k in range(len(mu_list)):
        mu_k    = mu_list[k].to(device)
        Sigma_k = Sigma_list[k].to(device)
        mu_xk   = mu_k[:CONDITION_ON]
        mu_yk   = mu_k[CONDITION_ON:]
        Syx     = Sigma_k[CONDITION_ON:, :CONDITION_ON]
        Sxx     = Sigma_k[:CONDITION_ON, :CONDITION_ON]
        x0_dev  = x0_sample.view(-1)[:CONDITION_ON]
        shift   = Syx @ torch.linalg.solve(Sxx, (x0_dev - mu_xk).unsqueeze(-1))
        mu_p_list.append(mu_yk + shift.squeeze(-1))

    mu_p = torch.stack(mu_p_list, dim=0)
    return gmm_l2_diff(mu_p, Sig_p, w_p, mu_q, Sig_q, w_q)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Q — identical to notebook
# ─────────────────────────────────────────────────────────────────────────────
def compute_analytical_Q(mu_list, Sigma_list, alpha,
                         mog_means, mog_variances, weights,
                         x_grid, zeta_values):

    def marginal_density(x_val):
        xv  = torch.tensor([float(x_val)])
        lps = [
            torch.log(w) + torch.distributions.Normal(
                mu[0], Sigma[0, 0].sqrt()
            ).log_prob(xv[0])
            for mu, Sigma, w in zip(mu_list, Sigma_list, alpha)
        ]
        return torch.logsumexp(torch.stack(lps), 0).exp().item()

    def l2gmm_loss_at_x(x_val):
        xt             = torch.tensor([float(x_val)])
        mu_pred, Sig_p = dist_utils.compute_conditionals(mu_list, Sigma_list, xt)
        w_pred         = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, xt)
        return dist_utils.gmm_l2_distance(mu_pred, Sig_p, w_pred,
                                          mog_means, mog_variances, weights)

    print("Pre-computing P(x) and L(x) on grid ...")
    px = np.array([marginal_density(xv) for xv in x_grid])
    lx = np.array([l2gmm_loss_at_x(xv) for xv in x_grid])

    Q = {}
    for z in zeta_values:
        q    = px * np.exp(-z * lx)
        Q[z] = q / np.trapezoid(q, x_grid)
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# JS divergence — the filterable metric in wandb
# ─────────────────────────────────────────────────────────────────────────────
def js_divergence(samples, analytical_q, x_grid):
    if len(samples) < 2 or np.std(samples) < 1e-6:
        return 1.0
    kde = gaussian_kde(samples, bw_method="scott")
    p   = np.maximum(kde(x_grid), 1e-10); p /= np.trapezoid(p, x_grid)
    q   = np.maximum(analytical_q, 1e-10); q /= np.trapezoid(q, x_grid)
    return float(jensenshannon(p, q))


# ─────────────────────────────────────────────────────────────────────────────
# CDMS sampler — identical to notebook (eta=1.0, L2_GMM only)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD_cdms(model_uncond, mu_list, Sigma_list, alpha,
                      mog_means, mog_variances, weights,
                      grad_clamp, num_x_t, zeta, device):

    effective_clamp = (0.25 * math.sqrt(zeta + 1.0)
                       if grad_clamp == "adaptive" else float(grad_clamp))

    x_t = torch.randn((1, 1, model_uncond.nfeatures), device=device,
                      requires_grad=True)

    for t in range(model_uncond.diffusion_steps - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)

        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=1.0
        )
        current_var = model_uncond.betas[t].to(device)
        r_t         = current_var / torch.sqrt(1 + current_var ** 2)

        if zeta == 0.0:
            with torch.no_grad():
                x_t = x_t_minus_1.detach().clone()
            continue

        losses = []
        for _ in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            loss_val  = compute_l2_gmm_loss(
                x0_sample, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights, device
            )
            losses.append(-loss_val)

        log_me = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        log_me.backward()

        grad = torch.clamp(x_t.grad.clone(), -effective_clamp, effective_clamp)

        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - (zeta * grad)

    return x_t.detach().float().view(-1).cpu()[0].item()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def log_beta_plot(zeta, samples, analytical_q, x_grid, x_star_val, js, cfg):
    """Upload one plot per beta to wandb immediately after sampling."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_grid, analytical_q, color="#E53935", lw=2, ls="--",
            label=r"Analytical $\mathcal{Q}_\beta$ (L2-GMM)")
    ax.fill_between(x_grid, analytical_q, alpha=0.12, color="#E53935")
    if len(samples) > 1 and np.std(samples) > 1e-6:
        ax.hist(samples, bins=40, density=True, color="#43A047",
                alpha=0.6, edgecolor="white", label=f"CDMS (N={len(samples)})")
    else:
        ax.axvline(float(samples[0]), color="#43A047", lw=2, label="Single sample")
    ax.axvline(x_star_val, color="k", ls=":", lw=1.5, label=f"x*={x_star_val:.1f}")
    ax.set_title(
        rf"$\beta={zeta}$  JS={js:.4f}  "
        f"clamp={cfg['grad_clamp']}  steps={cfg['diffusion_steps']}",
        fontsize=10
    )
    ax.set_xlabel("x"); ax.set_ylabel("Density")
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({f"plot/beta_{zeta}": wandb.Image(fig)})
    plt.close(fig)


def log_summary_plot(cdms_samples_dict, analytical_Q, x_grid,
                     x_star_val, zeta_values, js_per_beta, cfg):
    """Upload one summary grid plot (all betas) after all sampling is done."""
    n   = len(zeta_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, zeta in zip(axes, zeta_values):
        samples = cdms_samples_dict[zeta]
        q       = analytical_Q[zeta]
        js      = js_per_beta[zeta]
        ax.plot(x_grid, q, color="#E53935", lw=2, ls="--",
                label=r"Analytical $\mathcal{Q}_\beta$")
        ax.fill_between(x_grid, q, alpha=0.12, color="#E53935")
        if len(samples) > 1 and np.std(samples) > 1e-6:
            ax.hist(samples, bins=40, density=True, color="#43A047",
                    alpha=0.6, edgecolor="white", label=f"CDMS (N={len(samples)})")
        else:
            ax.axvline(float(samples[0]), color="#43A047", lw=2)
        ax.axvline(x_star_val, color="k", ls=":", lw=1.5, label=f"x*={x_star_val:.1f}")
        ax.set_title(rf"$\beta={zeta}$  JS={js:.3f}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("Density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"clamp={cfg['grad_clamp']}  steps={cfg['diffusion_steps']}  "
        f"mean_JS={np.mean(list(js_per_beta.values())):.4f}",
        fontsize=11
    )
    plt.tight_layout()
    wandb.log({"plot/summary_all_betas": wandb.Image(fig)})
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg, args, zeta_values, n_cdms_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"Config: {cfg}")
    print(f"Device: {device}  |  Zetas: {zeta_values}  |  N={n_cdms_samples}")
    print(f"{'='*65}\n")

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_NAME)
    RESULTS_DIR    = os.path.join(BASE_DIR, "results", EXPERIMENT_NAME, "gridsearch")
    for d in [CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity or None,
        config  = {
            **cfg,
            "zeta_values":    zeta_values,
            "seed":           GLOBAL_SEED,
            "loss":           "L2_GMM",
            "num_x_t":        NUM_X_T,
            "n_cdms_samples": n_cdms_samples,
            "eta":            1.0,
        },
        name   = f"clamp={cfg['grad_clamp']}_steps={cfg['diffusion_steps']}",
        tags   = ["cdms", "gridsearch", "L2_GMM"],
        reinit = True,
    )

    # GMM
    experiment_utils.set_global_seed(GLOBAL_SEED)
    (mu_list, Sigma_list, alpha,
     x_star, mog_means, mog_variances, weights) = build_gmm()

    # Unconditional diffusion — separate checkpoint per diffusion_steps value
    data_gen_uncond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
        kernel_func=lambda X: X[:, :CONDITION_ON]
    )
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=CONDITION_ON, nblocks=NBLOCKS, nunits=NUNITS,
        condition=False, diffusion_steps=cfg["diffusion_steps"]
    )
    ckpt_tag = f"Diffusion_uncond_steps{cfg['diffusion_steps']}"
    if not experiment_utils.load_model_checkpoint(
        model_uncond, ckpt_tag, CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED, device
    ):
        print(f"Training unconditional diffusion (steps={cfg['diffusion_steps']}) ...")
        experiment_utils.set_global_seed(GLOBAL_SEED)
        model_uncond.train_model(
            None, data_generator=data_gen_uncond,
            nepochs=NEPOCHS, batch_size=BATCH_SIZE,
            condition_on=CONDITION_ON
        )
        experiment_utils.save_model_checkpoint(
            model_uncond, ckpt_tag, CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED
        )
    else:
        print(f"Loaded: {ckpt_tag}")

    # Analytical Q
    x_grid       = np.linspace(-12, 12, 300)
    analytical_Q = compute_analytical_Q(
        mu_list, Sigma_list, alpha,
        mog_means, mog_variances, weights,
        x_grid, zeta_values
    )

    # Sample + log
    cdms_samples_dict = {}
    js_per_beta       = {}

    for zeta in zeta_values:
        print(f"\n[β={zeta}] sampling {n_cdms_samples} points ...")
        samples = []
        for i in trange(n_cdms_samples):
            experiment_utils.set_run_seed(GLOBAL_SEED, i)
            s = optimize_LGD_cdms(
                model_uncond, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights,
                grad_clamp = cfg["grad_clamp"],
                num_x_t    = NUM_X_T,
                zeta       = zeta,
                device     = device,
            )
            samples.append(s)

        samples_arr             = np.array(samples)
        cdms_samples_dict[zeta] = samples_arr
        js                      = js_divergence(samples_arr, analytical_Q[zeta], x_grid)
        js_per_beta[zeta]       = js

        print(f"   mean={np.mean(samples_arr):.3f}  "
              f"std={np.std(samples_arr):.3f}  JS={js:.4f}")

        # Per-beta filterable columns in wandb
        wandb.log({
            f"js/beta_{zeta}":          js,
            f"sample_mean/beta_{zeta}": float(np.mean(samples_arr)),
            f"sample_std/beta_{zeta}":  float(np.std(samples_arr)),
        })

        # Plot immediately after this beta finishes
        log_beta_plot(
            zeta, samples_arr, analytical_Q[zeta],
            x_grid, x_star[0].item(), js, cfg
        )

    # Summary metrics — the main columns to filter/sort by in wandb
    mean_js         = float(np.mean(list(js_per_beta.values())))
    mean_js_nonzero = float(np.mean([js_per_beta[z] for z in zeta_values if z > 0]))

    print(f"\n>>> mean JS (all β):  {mean_js:.4f}")
    print(f">>> mean JS (β > 0):  {mean_js_nonzero:.4f}")

    wandb.log({
        "mean_js":         mean_js,
        "mean_js_nonzero": mean_js_nonzero,  # ← sort/filter by this in wandb UI
        "grad_clamp_str":  str(cfg["grad_clamp"]),
        "diffusion_steps": cfg["diffusion_steps"],
    })

    # Summary plot after all betas
    log_summary_plot(
        cdms_samples_dict, analytical_Q, x_grid,
        x_star[0].item(), zeta_values, js_per_beta, cfg
    )

    # Save JSON
    slug      = f"clamp{cfg['grad_clamp']}_steps{cfg['diffusion_steps']}"
    json_path = os.path.join(RESULTS_DIR, f"cdms_{slug}.json")
    with open(json_path, "w") as f:
        json.dump({
            "config":          cfg,
            "mean_js":         mean_js,
            "mean_js_nonzero": mean_js_nonzero,
            "js_per_beta":     {str(k): v for k, v in js_per_beta.items()},
            "samples":         {str(k): v.tolist() for k, v in cdms_samples_dict.items()},
        }, f, indent=2)
    wandb.save(json_path)
    print(f"JSON saved → {json_path}")

    wandb.finish()
    return mean_js


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_id",     type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="cdms-gridsearch")
    p.add_argument("--wandb_entity",  type=str, default="")
    p.add_argument("--list_configs",  action="store_true")
    p.add_argument("--smoke_test",    action="store_true",
                   help="5 samples, 2 betas, quick end-to-end check")
    args = p.parse_args()

    configs   = all_configs()
    n_configs = len(configs)

    if args.list_configs:
        print(f"Total configs: {n_configs}  "
              f"({len(GRID['grad_clamp'])} clamps × "
              f"{len(GRID['diffusion_steps'])} step values)")
        for i, c in enumerate(configs):
            print(f"  [{i:3d}] {c}")
        sys.exit(0)

    print(f"Config {args.config_id} / {n_configs - 1}")
    cfg = configs[args.config_id]

    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ***")
        zeta_values    = ZETA_VALUES_SMOKE
        n_cdms_samples = 5
        print(f"cfg: {cfg} | zetas: {zeta_values} | n_samples: {n_cdms_samples}\n")
    else:
        zeta_values    = ZETA_VALUES_FULL
        n_cdms_samples = N_CDMS_SAMPLES

    run_config(cfg, args, zeta_values, n_cdms_samples)
