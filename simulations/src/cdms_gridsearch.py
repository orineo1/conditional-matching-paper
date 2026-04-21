"""
cdms_gridsearch.py
==================
Grid search over CDMS hyperparameters using L2_GMM loss only.

Objective: minimise mean JS divergence between sampled histogram
           and analytical Q(x;β) across β in [0, 1, 2, 4, 8, 16].

IMPORTANT — gradient fix:
  The original compute_l2_gmm_loss moves x0_sample to CPU which breaks
  autograd. Here we keep everything on device so gradients flow properly:
      x0_sample (on device) → gmm_l2_diff (on device) → .backward()

What is NOT changed vs original notebook:
  - GMM parameters (mu_list, Sigma_list, alpha, x_star)
  - Diffusion model architecture and training
  - DDIM sampling step  (model_uncond.sample_ddim_step)
  - log-mean-exp gradient estimator formula
  - r_t definition

What IS swept:
  grad_clamp    — float or "adaptive" (0.25 * sqrt(zeta+1))
  noise_scale   — 0.0 → deterministic  x_t = x_{t-1} - zeta*grad
                  0.5 → stochastic     x_t = x_{t-1} - zeta*grad + 0.5*sqrt(2*lr)*eps
  num_x_t       — x0 samples per diffusion step
  nunits_cm     — CM hidden units  (architecture)
  depth_cm      — CM depth         (architecture)
  nepochs_cm    — CM training epochs

Launch via SLURM array:
    sbatch cdms_gridsearch.sh
or test one config locally:
    python cdms_gridsearch.py --config_id 0 --wandb_project cdms-test
"""

import os, sys, math, json, argparse, itertools
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Paths — override REPO_ROOT in environment on cluster
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    "/sci/labs/orzuk/ori_m/conditional-matching-paper"
)
BASE_DIR = os.path.join(REPO_ROOT, "simulations")
SRC_PATH = BASE_DIR          # .py modules live directly in simulations/
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import importlib
import Diffusion, LossFunctions, ConsistencyModels
import dist_utils, Optimization, experiment_utils
from ConsistencyModels import ConsistencyModeliCT

for mod in [Diffusion, LossFunctions, ConsistencyModels,
            dist_utils, Optimization, experiment_utils]:
    importlib.reload(mod)

# ─────────────────────────────────────────────────────────────────────────────
# Fixed constants  (never change these — must match original notebook)
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME  = "2D_cond_1D"
GLOBAL_SEED      = 42
CONDITION_ON     = 1
ZETA_VALUES_FULL = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]   # used in full run
ZETA_VALUES_SMOKE = [0.0, 1.0, 4.0]                    # used in smoke test

LR              = 0.05   # gradient step size — fixed (not swept)
NBLOCKS         = 3
NUNITS          = 128
NEPOCHS_DIFF    = 20_000
BATCH_SIZE_DIFF = 512
DIFFUSION_STEPS = 100

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# ─────────────────────────────────────────────────────────────────────────────
GRID = {
    "grad_clamp":     [0.25, 1.0, 3.0, "adaptive"],
    "noise_scale":    [0.0, 0.5],
    "num_x_t":        [3, 7],
    "nunits_cm":      [128, 256],
    "depth_cm":       [3, 5],
    "nepochs_cm":     [80_000],
    "batch_cm":       [8_192],
    "n_cdms_samples": [100],
}


def all_configs():
    keys, values = zip(*GRID.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ─────────────────────────────────────────────────────────────────────────────
# GMM parameters  (identical to original notebook — do not change)
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
# L2 GMM distance  — differentiable, stays on device
# ─────────────────────────────────────────────────────────────────────────────
def gmm_l2_diff_device(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q):
    if mu_p.dim() == 1:    mu_p    = mu_p.unsqueeze(0)
    if mu_q.dim() == 1:    mu_q    = mu_q.unsqueeze(0)
    if Sigma_p.dim() == 2: Sigma_p = Sigma_p.unsqueeze(0)
    if Sigma_q.dim() == 2: Sigma_q = Sigma_q.unsqueeze(0)

    D = mu_p.shape[-1]

    def inner(m1, S1, w1, m2, S2, w2):
        diff      = m1.unsqueeze(1) - m2.unsqueeze(0)
        S_sum     = S1.unsqueeze(1) + S2.unsqueeze(0)
        _, logdet = torch.linalg.slogdet(S_sum)
        inv_S     = torch.linalg.inv(S_sum)
        quad      = torch.einsum('ijk,ijkl,ijl->ij', diff, inv_S, diff)
        log_val   = -0.5 * (D * math.log(2 * math.pi) + logdet + quad)
        log_w     = torch.log(w1).unsqueeze(1) + torch.log(w2).unsqueeze(0)
        return torch.exp(log_w + log_val).sum()

    pp = inner(mu_p, Sigma_p, w_p, mu_p, Sigma_p, w_p)
    qq = inner(mu_q, Sigma_q, w_q, mu_q, Sigma_q, w_q)
    pq = inner(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q)
    return pp - 2 * pq + qq


def compute_l2_gmm_loss_on_device(x0_sample, mu_list, Sigma_list, alpha,
                                  mog_means, mog_variances, weights, device):
    x0_val = x0_sample.detach().view(-1).cpu().float()

    mu_pred, Sigma_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, x0_val)
    w_pred              = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x0_val)

    Sigma_p = Sigma_pred.to(device).detach()
    w_p     = w_pred.to(device).detach()
    mu_q    = mog_means.squeeze(1).to(device).detach()
    Sigma_q = mog_variances.squeeze(1).to(device).detach()
    w_q     = weights.to(device).detach()

    x0_dev = x0_sample.view(-1).to(device)
    D_x    = CONDITION_ON

    mu_p_diff_list = []
    for k in range(len(mu_list)):
        mu_k    = mu_list[k].to(device)
        Sigma_k = Sigma_list[k].to(device)
        mu_xk   = mu_k[:D_x]
        mu_yk   = mu_k[D_x:]
        Syx     = Sigma_k[D_x:, :D_x]
        Sxx     = Sigma_k[:D_x, :D_x]
        shift   = Syx @ torch.linalg.solve(Sxx, (x0_dev - mu_xk).unsqueeze(-1))
        mu_p_diff_list.append(mu_yk + shift.squeeze(-1))

    mu_p_diff = torch.stack(mu_p_diff_list, dim=0)

    return gmm_l2_diff_device(mu_p_diff, Sigma_p, w_p, mu_q, Sigma_q, w_q)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Q(x; β)
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

    def loss_at_x(x_val):
        xt             = torch.tensor([float(x_val)])
        mu_pred, Sig_p = dist_utils.compute_conditionals(mu_list, Sigma_list, xt)
        w_pred         = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, xt)
        return dist_utils.gmm_l2_distance(mu_pred, Sig_p, w_pred,
                                          mog_means, mog_variances, weights)

    print("Pre-computing P(x) and L(x) on grid ...")
    px = np.array([marginal_density(xv) for xv in x_grid])
    lx = np.array([loss_at_x(xv)       for xv in x_grid])

    Q = {}
    for z in zeta_values:
        q    = px * np.exp(-z * lx)
        Q[z] = q / np.trapz(q, x_grid)
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# JS divergence
# ─────────────────────────────────────────────────────────────────────────────
def js_divergence(samples, analytical_q, x_grid):
    if len(samples) < 2 or np.std(samples) < 1e-6:
        return 1.0
    kde = gaussian_kde(samples, bw_method="scott")
    p   = np.maximum(kde(x_grid), 1e-10);  p /= np.trapz(p, x_grid)
    q   = np.maximum(analytical_q, 1e-10); q /= np.trapz(q, x_grid)
    return float(jensenshannon(p, q))


# ─────────────────────────────────────────────────────────────────────────────
# optimize_LGD_cdms
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD_cdms(model_uncond, mu_list, Sigma_list, alpha,
                      mog_means, mog_variances, weights,
                      grad_clamp, noise_scale, num_x_t, zeta, device):

    effective_clamp = (0.25 * math.sqrt(zeta + 1.0)
                       if grad_clamp == "adaptive" else float(grad_clamp))

    x_t = torch.randn((1, 1, model_uncond.nfeatures), device=device,
                      requires_grad=True)

    for t in range(model_uncond.diffusion_steps - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)

        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0
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
            loss_val  = compute_l2_gmm_loss_on_device(
                x0_sample, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights, device
            )
            losses.append(-loss_val)

        log_me = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        log_me.backward()

        grad = torch.clamp(x_t.grad.clone(), -effective_clamp, effective_clamp)

        with torch.no_grad():
            noise = torch.randn_like(x_t) * (noise_scale * math.sqrt(2 * LR))
            x_t   = x_t_minus_1.detach().clone() - (zeta * grad) + noise

    x_t_final = x_t.detach().clone()
    if str(device) != "cpu":
        torch.cuda.empty_cache()
    return x_t_final


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def make_beta_grid_plot(cdms_samples_dict, analytical_Q, x_grid,
                        x_star_val, zeta_values, js_per_beta, cfg):
    n_cols = 3
    n_rows = math.ceil(len(zeta_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), sharey=False)
    axes_flat = np.array(axes).flatten()

    for ax, zeta in zip(axes_flat, zeta_values):
        samples = cdms_samples_dict.get(zeta, np.array([]))
        q       = analytical_Q[zeta]
        js      = js_per_beta.get(zeta, float("nan"))

        ax.plot(x_grid, q, color="#E53935", lw=2, ls="--",
                label=r"Analytical $\mathcal{Q}_\beta$")
        ax.fill_between(x_grid, q, alpha=0.12, color="#E53935")

        if len(samples) > 1 and np.std(samples) > 1e-6:
            ax.hist(samples, bins=40, density=True, color="#1E88E5",
                    alpha=0.6, edgecolor="white", label="Sampled $x$")
        elif len(samples) >= 1:
            ax.axvline(float(samples[0]), color="#1E88E5", lw=2,
                       label="Sampled $x$")

        ax.axvline(x_star_val, color="k", ls=":", lw=1.5,
                   label=f"$x^*={x_star_val:.0f}$")
        ax.set_title(rf"$\beta={zeta}$   JS={js:.3f}", fontsize=12)
        ax.set_xlabel("$x$"); ax.set_ylabel("Density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(zeta_values):]:
        ax.set_visible(False)

    fig.suptitle(
        r"CDMS  $\mathcal{Q}_\beta\!\propto\!\mathcal{P}(x)e^{-\beta\mathcal{L}(x)}$"
        f"   [L2_GMM loss]\n"
        f"clamp={cfg['grad_clamp']}  noise={cfg['noise_scale']}  "
        f"num_x_t={cfg['num_x_t']}  CM=({cfg['nunits_cm']}u×{cfg['depth_cm']}d)",
        fontsize=10
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Run one config
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg, args, zeta_values):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nConfig: {cfg}\nDevice: {device}\nZetas:  {zeta_values}\n{'='*65}\n")

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_NAME)
    RESULTS_DIR    = os.path.join(BASE_DIR, "results", EXPERIMENT_NAME, "gridsearch")
    PARAMS_DIR     = os.path.join(BASE_DIR, "params")
    for d in [CHECKPOINT_DIR, RESULTS_DIR, PARAMS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity or None,
        config  = {**cfg, "zeta_values": zeta_values,
                   "seed": GLOBAL_SEED, "loss": "L2_GMM"},
        name    = (f"clamp={cfg['grad_clamp']}_noise={cfg['noise_scale']}"
                   f"_nxt={cfg['num_x_t']}_CM={cfg['nunits_cm']}u{cfg['depth_cm']}d"),
        tags    = ["cdms", "gridsearch", "L2_GMM"],
        reinit  = True,
    )

    # ── GMM ───────────────────────────────────────────────────────────────
    experiment_utils.set_global_seed(GLOBAL_SEED)
    (mu_list, Sigma_list, alpha,
     x_star, mog_means, mog_variances, weights) = build_gmm()
    print(f"x_star = {x_star.tolist()}")

    # ── Unconditional diffusion ────────────────────────────────────────────
    data_gen_uncond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
        kernel_func=lambda X: X[:, :CONDITION_ON]
    )
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=CONDITION_ON, nblocks=NBLOCKS, nunits=NUNITS,
        condition=False, diffusion_steps=DIFFUSION_STEPS
    )
    if not experiment_utils.load_model_checkpoint(
        model_uncond, "Diffusion_uncond",
        CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED, device
    ):
        print("[Diffusion_uncond] training from scratch ...")
        experiment_utils.set_global_seed(GLOBAL_SEED)
        model_uncond.train_model(
            None, data_generator=data_gen_uncond,
            nepochs=NEPOCHS_DIFF, batch_size=BATCH_SIZE_DIFF,
            condition_on=CONDITION_ON
        )
        experiment_utils.save_model_checkpoint(
            model_uncond, "Diffusion_uncond",
            CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED
        )

    # ── Consistency model ──────────────────────────────────────────────────
    cm_tag = f"CM_u{cfg['nunits_cm']}_d{cfg['depth_cm']}_ep{cfg['nepochs_cm']}_bs{cfg['batch_cm']}"
    cm_model = ConsistencyModeliCT(
        nfeatures=2 - CONDITION_ON, condition_on=CONDITION_ON,
        nunits=cfg["nunits_cm"], depth=cfg["depth_cm"]
    )
    if not experiment_utils.load_model_checkpoint(
        cm_model, cm_tag, CHECKPOINT_DIR,
        EXPERIMENT_NAME, GLOBAL_SEED, device
    ):
        print(f"[CM {cm_tag}] training from scratch ...")
        experiment_utils.set_global_seed(GLOBAL_SEED)
        cm_model.train_model(
            X=None, nepochs=cfg["nepochs_cm"], batch_size=cfg["batch_cm"],
            device=device, condition=CONDITION_ON,
            data_generator=partial(
                dist_utils.generate_mog_samples_not_differentiable,
                means=mu_list, variances=Sigma_list, weights=alpha
            ),
            use_improved_training=True
        )
        experiment_utils.save_model_checkpoint(
            cm_model, cm_tag, CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED
        )

    # ── Analytical Q ──────────────────────────────────────────────────────
    x_grid       = np.linspace(-12, 12, 500)
    analytical_Q = compute_analytical_Q(
        mu_list, Sigma_list, alpha,
        mog_means, mog_variances, weights,
        x_grid, zeta_values
    )

    # ── Sample x ~ Q(x;β) per β ───────────────────────────────────────────
    cdms_samples_dict = {}
    js_per_beta       = {}

    for zeta in zeta_values:
        print(f"\n[β={zeta}] sampling {cfg['n_cdms_samples']} points ...")
        samples = []
        for i in trange(cfg["n_cdms_samples"]):
            experiment_utils.set_run_seed(GLOBAL_SEED, i)
            x_pred = optimize_LGD_cdms(
                model_uncond, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights,
                grad_clamp  = cfg["grad_clamp"],
                noise_scale = cfg["noise_scale"],
                num_x_t     = cfg["num_x_t"],
                zeta        = zeta,
                device      = device,
            )
            samples.append(x_pred.float().view(-1).cpu()[0].item())

        samples               = np.array(samples)
        cdms_samples_dict[zeta] = samples
        js                    = js_divergence(samples, analytical_Q[zeta], x_grid)
        js_per_beta[zeta]     = js
        print(f"   mean={np.mean(samples):.3f}  std={np.std(samples):.3f}  JS={js:.4f}")
        wandb.log({
            f"js/beta_{zeta}":          js,
            f"sample_mean/beta_{zeta}": float(np.mean(samples)),
            f"sample_std/beta_{zeta}":  float(np.std(samples)),
        })

    # ── Headline metric ────────────────────────────────────────────────────
    mean_js = float(np.mean(list(js_per_beta.values())))
    print(f"\n>>> mean JS across all β: {mean_js:.4f}")
    wandb.log({"mean_js": mean_js, "lr": LR})

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = make_beta_grid_plot(
        cdms_samples_dict, analytical_Q,
        x_grid, x_star[0].item(), zeta_values, js_per_beta, cfg
    )
    wandb.log({"beta_grid_plot": wandb.Image(fig)})

    slug      = f"clamp{cfg['grad_clamp']}_noise{cfg['noise_scale']}_nxt{cfg['num_x_t']}_CM{cfg['nunits_cm']}u{cfg['depth_cm']}d"
    fig_path  = os.path.join(RESULTS_DIR, f"cdms_{slug}.png")
    json_path = os.path.join(RESULTS_DIR, f"cdms_{slug}.json")

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {fig_path}")

    with open(json_path, "w") as f:
        json.dump({
            "config": cfg, "lr": LR, "mean_js": mean_js,
            "js_per_beta": {str(k): v for k, v in js_per_beta.items()},
            "samples":     {str(k): v.tolist() for k, v in cdms_samples_dict.items()},
        }, f, indent=2)
    wandb.save(json_path)
    print(f"JSON  saved → {json_path}")

    wandb.finish()
    return mean_js


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_id",     type=int, required=True)
    p.add_argument("--wandb_project", type=str, default="cdms-sweep")
    p.add_argument("--wandb_entity",  type=str, default="")
    p.add_argument("--list_configs",  action="store_true")
    p.add_argument("--smoke_test",    action="store_true",
                   help="Tiny settings to verify pipeline end-to-end")
    args = p.parse_args()

    configs = all_configs()

    if args.list_configs:
        print(f"Total configs: {len(configs)}")
        for i, c in enumerate(configs):
            print(f"  [{i:3d}] {c}")
        sys.exit(0)

    print(f"Config {args.config_id} / {len(configs) - 1}")
    cfg = configs[args.config_id]

    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ***")
        cfg = {**cfg, "nepochs_cm": 500, "batch_cm": 256,
               "n_cdms_samples": 5, "num_x_t": 1}
        zeta_values = ZETA_VALUES_SMOKE
        print(f"cfg: {cfg}\nzetas: {zeta_values}\n")
    else:
        zeta_values = ZETA_VALUES_FULL

    run_config(cfg, args, zeta_values)
