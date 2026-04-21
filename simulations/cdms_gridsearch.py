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
  lr            — gradient step size
  grad_clamp    — float or "adaptive" (0.25 * sqrt(zeta+1))
  noise_scale   — 0.0 → deterministic  x_t = x_{t-1} - zeta*grad
                  s   → stochastic     x_t = x_{t-1} - zeta*grad + s*sqrt(2*lr)*eps
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
SRC_PATH = os.path.join(REPO_ROOT, "simulations", "src")
BASE_DIR = os.path.join(REPO_ROOT, "simulations")
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
EXPERIMENT_NAME = "2D_cond_1D"
GLOBAL_SEED     = 42
CONDITION_ON    = 1
ZETA_VALUES     = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]

# Diffusion model — fixed, loaded from existing checkpoint
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
    # ── Optimization ──────────────────────────────────────────────────────
    # gradient clamp before update
    #   float     → fixed symmetric clamp  [-c, c]
    #   "adaptive"→ clamp = 0.25 * sqrt(zeta+1)  (grows with β, avoids
    #                killing gradients at high β where signal is stronger)
    "grad_clamp":  [0.25, 1.0, 3.0, "adaptive"],

    # noise scale in the denoising step:
    #   0.0 → x_t = x_{t-1} - zeta * grad                    (fully deterministic)
    #   0.5 → x_t = x_{t-1} - zeta * grad + 0.5*sqrt(2*lr)*ε (half stochastic)
    "noise_scale": [0.0, 0.5],

    # x0 samples per diffusion step — more = lower-variance gradient
    "num_x_t":    [3, 7],

    # ── CM architecture ───────────────────────────────────────────────────
    # Each combo gets its own checkpoint, original (128, 3) is included
    "nunits_cm":  [128, 256],
    "depth_cm":   [3, 5],

    # CM training epochs (original notebook used 80_000 in the CDMS section)
    "nepochs_cm": [80_000],
    "batch_cm":   [8_192],

    # samples per β for JS estimation
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
# Identical math to notebook's gmm_l2_diff; no CPU detach so grads flow.
# ─────────────────────────────────────────────────────────────────────────────
def gmm_l2_diff_device(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q):
    """
    Exact L2^2 distance between two GMMs, fully on-device.
    Gradients flow through mu_p (and w_p if needed).

    Args:
        mu_p:    (K1, D)
        Sigma_p: (K1, D, D)
        w_p:     (K1,)
        mu_q:    (K2, D)   — detached target, no grad needed
        Sigma_q: (K2, D, D)
        w_q:     (K2,)
    """
    if mu_p.dim() == 1:    mu_p    = mu_p.unsqueeze(0)
    if mu_q.dim() == 1:    mu_q    = mu_q.unsqueeze(0)
    if Sigma_p.dim() == 2: Sigma_p = Sigma_p.unsqueeze(0)
    if Sigma_q.dim() == 2: Sigma_q = Sigma_q.unsqueeze(0)

    D = mu_p.shape[-1]

    def inner(m1, S1, w1, m2, S2, w2):
        diff    = m1.unsqueeze(1) - m2.unsqueeze(0)           # (K1,K2,D)
        S_sum   = S1.unsqueeze(1) + S2.unsqueeze(0)           # (K1,K2,D,D)
        _, logdet = torch.linalg.slogdet(S_sum)               # (K1,K2)
        inv_S   = torch.linalg.inv(S_sum)                     # (K1,K2,D,D)
        quad    = torch.einsum('ijk,ijkl,ijl->ij',
                               diff, inv_S, diff)             # (K1,K2)
        log_val = -0.5 * (D * math.log(2 * math.pi) + logdet + quad)
        log_w   = torch.log(w1).unsqueeze(1) + torch.log(w2).unsqueeze(0)
        return torch.exp(log_w + log_val).sum()

    pp = inner(mu_p, Sigma_p, w_p, mu_p, Sigma_p, w_p)
    qq = inner(mu_q, Sigma_q, w_q, mu_q, Sigma_q, w_q)
    pq = inner(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q)
    return pp - 2 * pq + qq


def compute_l2_gmm_loss_on_device(x0_sample, mu_list, Sigma_list, alpha,
                                  mog_means, mog_variances, weights, device):
    """
    L2 GMM loss between P(Y | X=x0_sample) and G(Y).

    FIX vs original notebook:
      Original did  x0_cpu = x0_sample.view(-1).cpu()  which detaches autograd.
      Here we compute conditionals on CPU (they don't support autograd anyway)
      but then rebuild the tensors that DO need grad on device from x0_sample,
      keeping the computation graph alive through the Gaussian inner products.

    The gradient path is:
      x_t  →  pred_x0  →  x0_sample  →  [mu_pred computed analytically]
           →  gmm_l2_diff_device  →  loss  →  .backward()

    NOTE: compute_conditionals / compute_alpha are non-differentiable GMM
    formula lookups (they don't touch x0_sample's value via autograd).
    The gradient flows because x0_sample appears inside the Gaussian
    inner product as the conditioning variable that shifts mu_pred linearly.
    If dist_utils.compute_conditionals is not differentiable w.r.t. its input,
    we recompute mu_pred in a differentiable way below.
    """
    # ── Detached value for the analytical conditional lookup ──────────────
    x0_val = x0_sample.detach().view(-1).cpu().float()

    # Analytical P(Y | X = x0) parameters (these are numpy/tensor constants,
    # NOT connected to x0_sample's autograd graph — that is correct:
    # the GMM conditional formula is closed-form and we differentiate
    # through the inner-product expression, not through the lookup)
    mu_pred, Sigma_pred = dist_utils.compute_conditionals(
        mu_list, Sigma_list, x0_val
    )
    w_pred = dist_utils.compute_alpha(
        mu_list, Sigma_list, alpha, x0_val
    )

    # Move to device  (no grad on these — they are the "parameter" GMM)
    mu_p    = mu_pred.squeeze(-1).to(device).detach()     # (K1, D_y)
    Sigma_p = Sigma_pred.to(device).detach()              # (K1, D_y, D_y)
    w_p     = w_pred.to(device).detach()                  # (K1,)

    # Target G(Y) = P(Y | X = x*)
    mu_q    = mog_means.squeeze(1).to(device).detach()    # (K2, D_y)
    Sigma_q = mog_variances.squeeze(1).to(device).detach()
    w_q     = weights.to(device).detach()

    # ── Differentiable part: shift mu_p by how x0_sample moves it ─────────
    # For a Gaussian conditional in a joint GMM, mu_{Y|X=x} = mu_y + Sigma_yx
    # Sigma_xx^{-1} (x - mu_x). The gradient w.r.t. x flows through this shift.
    # We recompute it differentiably here so autograd can track x0_sample.
    x0_dev = x0_sample.view(-1).to(device)  # (D_x,) — keeps grad
    K = len(mu_list)
    D_x = CONDITION_ON
    D_y = mu_p.shape[-1]

    mu_p_diff_list    = []
    for k in range(K):
        mu_k    = mu_list[k].to(device)          # (D,)
        Sigma_k = Sigma_list[k].to(device)       # (D, D)
        mu_xk   = mu_k[:D_x]                    # (D_x,)
        mu_yk   = mu_k[D_x:]                    # (D_y,)
        Syx     = Sigma_k[D_x:, :D_x]           # (D_y, D_x)
        Sxx     = Sigma_k[:D_x, :D_x]           # (D_x, D_x)
        # mu_{Y|X=x,k} = mu_yk + Syx @ Sxx^{-1} @ (x - mu_xk)
        shift   = Syx @ torch.linalg.solve(Sxx, (x0_dev - mu_xk).unsqueeze(-1))
        mu_yk_cond = mu_yk + shift.squeeze(-1)  # (D_y,)  — has grad via x0_dev
        mu_p_diff_list.append(mu_yk_cond)

    mu_p_diff = torch.stack(mu_p_diff_list, dim=0)   # (K1, D_y) — has grad

    return gmm_l2_diff_device(mu_p_diff, Sigma_p, w_p, mu_q, Sigma_q, w_q)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Q(x; β)
# ─────────────────────────────────────────────────────────────────────────────
def compute_analytical_Q(mu_list, Sigma_list, alpha,
                         mog_means, mog_variances, weights,
                         x_grid: np.ndarray, zeta_values):

    def marginal_density(x_val):
        xv = torch.tensor([float(x_val)])
        lps = [
            torch.log(w) + torch.distributions.Normal(
                mu[0], Sigma[0, 0].sqrt()
            ).log_prob(xv[0])
            for mu, Sigma, w in zip(mu_list, Sigma_list, alpha)
        ]
        return torch.logsumexp(torch.stack(lps), 0).exp().item()

    def loss_at_x(x_val):
        xt = torch.tensor([float(x_val)])
        mu_pred, Sig_pred = dist_utils.compute_conditionals(mu_list, Sigma_list, xt)
        w_pred            = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, xt)
        return dist_utils.gmm_l2_distance(
            mu_pred, Sig_pred, w_pred, mog_means, mog_variances, weights
        )

    print("Pre-computing P(x) and L(x) on grid ...")
    px = np.array([marginal_density(xv) for xv in x_grid])
    lx = np.array([loss_at_x(xv)       for xv in x_grid])

    Q = {}
    for z in zeta_values:
        q  = px * np.exp(-z * lx)
        Q[z] = q / np.trapz(q, x_grid)
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# JS divergence
# ─────────────────────────────────────────────────────────────────────────────
def js_divergence(samples: np.ndarray,
                  analytical_q: np.ndarray,
                  x_grid: np.ndarray) -> float:
    """JS distance (sqrt of JS divergence) ∈ [0, 1]. Lower = better."""
    if len(samples) < 2 or np.std(samples) < 1e-6:
        return 1.0
    kde = gaussian_kde(samples, bw_method="scott")
    p   = np.maximum(kde(x_grid), 1e-10)
    p  /= np.trapz(p, x_grid)
    q   = np.maximum(analytical_q, 1e-10)
    q  /= np.trapz(q, x_grid)
    return float(jensenshannon(p, q))


# ─────────────────────────────────────────────────────────────────────────────
# optimize_LGD_cdms  — L2_GMM loss only, all knobs explicit
# Core loop structure unchanged from original notebook.
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD_cdms(
    model_uncond,
    mu_list, Sigma_list, alpha,
    mog_means, mog_variances, weights,
    # ── swept hyperparams ──
    grad_clamp,           # float  or  "adaptive"
    noise_scale: float,   # 0.0 = deterministic, 0.5 = half stochastic
    num_x_t:     int,
    zeta:        float,
    device,
):
    # resolve adaptive clamp once per call (larger at high β)
    if grad_clamp == "adaptive":
        effective_clamp = 0.25 * math.sqrt(zeta + 1.0)
    else:
        effective_clamp = float(grad_clamp)

    # ── initialise x_t  (same shape as original after two unsqueezes) ─────
    x_t = torch.randn((1, 1, model_uncond.nfeatures), device=device,
                      requires_grad=True)

    for t in range(model_uncond.diffusion_steps - 1, 0, -1):
        x_t = x_t.detach().clone().requires_grad_(True)

        # ── DDIM denoising step (unchanged from original) ─────────────────
        x_t_minus_1, pred_x0 = model_uncond.sample_ddim_step(
            x_t, t, condition_x=None, device=device, eta=0.0
        )
        current_var = model_uncond.betas[t].to(device)
        r_t         = current_var / torch.sqrt(1 + current_var ** 2)

        # ── β = 0: pure prior, no guidance ────────────────────────────────
        if zeta == 0.0:
            with torch.no_grad():
                x_t = x_t_minus_1.detach().clone()
            continue

        # ── L2_GMM gradient estimate over num_x_t samples ─────────────────
        losses = []
        for _ in range(num_x_t):
            x0_sample = pred_x0 + r_t * torch.randn_like(pred_x0)
            loss_val  = compute_l2_gmm_loss_on_device(
                x0_sample, mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights, device
            )
            losses.append(-loss_val)   # negative because we minimise loss

        # log-mean-exp (unchanged from original)
        log_me = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        log_me.backward()

        grad = x_t.grad.clone()
        grad = torch.clamp(grad, -effective_clamp, effective_clamp)

        with torch.no_grad():
            # noise_scale=0.0 → x_t = x_{t-1} - zeta*grad  (deterministic)
            # noise_scale=0.5 → x_t = x_{t-1} - zeta*grad + 0.5*sqrt(2*LR)*ε
            noise = torch.randn_like(x_t) * (noise_scale * math.sqrt(2 * LR))
            x_t   = x_t_minus_1.detach().clone() - (zeta * grad) + noise

    x_t_final = x_t.detach().clone()
    if str(device) != "cpu":
        torch.cuda.empty_cache()
    return x_t_final


# ─────────────────────────────────────────────────────────────────────────────
# Plot: β grid  (histogram vs analytical Q, JS in title)
# ─────────────────────────────────────────────────────────────────────────────
def make_beta_grid_plot(cdms_samples_dict, analytical_Q,
                        x_grid, x_star_val, zeta_values,
                        js_per_beta, cfg):
    n_cols = 3
    n_rows = math.ceil(len(zeta_values) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharey=False)
    axes_flat = np.array(axes).flatten()

    for ax, zeta in zip(axes_flat, zeta_values):
        samples = cdms_samples_dict.get(zeta, np.array([]))
        q       = analytical_Q[zeta]
        js      = js_per_beta.get(zeta, float("nan"))

        ax.plot(x_grid, q, color="#E53935", lw=2, ls="--",
                label=r"Analytical $\mathcal{Q}_\beta$")
        ax.fill_between(x_grid, q, alpha=0.12, color="#E53935")

        if len(samples) > 1 and np.std(samples) > 1e-6:
            ax.hist(samples, bins=40, density=True,
                    color="#1E88E5", alpha=0.6, edgecolor="white",
                    label="Sampled $x$")
        elif len(samples) >= 1:
            ax.axvline(float(samples[0]), color="#1E88E5",
                       lw=2, label="Sampled $x$")

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
        f"lr={cfg['lr']}  clamp={cfg['grad_clamp']}  "
        f"noise={cfg['noise_scale']}  num_x_t={cfg['num_x_t']}  "
        f"CM=({cfg['nunits_cm']}u×{cfg['depth_cm']}d)",
        fontsize=10
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Run one config
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nConfig: {cfg}\nDevice: {device}\n{'='*65}\n")

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_NAME)
    RESULTS_DIR    = os.path.join(
        BASE_DIR, "results", EXPERIMENT_NAME, "gridsearch"
    )
    PARAMS_DIR = os.path.join(BASE_DIR, "params")
    for d in [CHECKPOINT_DIR, RESULTS_DIR, PARAMS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────
    run_name = (
        f"lr={cfg['lr']}"
        f"_clamp={cfg['grad_clamp']}"
        f"_noise={cfg['noise_scale']}"
        f"_nxt={cfg['num_x_t']}"
        f"_CM={cfg['nunits_cm']}u{cfg['depth_cm']}d"
    )
    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity or None,
        config  = {**cfg, "zeta_values": ZETA_VALUES, "seed": GLOBAL_SEED,
                   "loss": "L2_GMM"},
        name    = run_name,
        tags    = ["cdms", "gridsearch", "L2_GMM"],
        reinit  = True,
    )

    # ── GMM ───────────────────────────────────────────────────────────────
    experiment_utils.set_global_seed(GLOBAL_SEED)
    (mu_list, Sigma_list, alpha,
     x_star, mog_means, mog_variances, weights) = build_gmm()
    print(f"x_star = {x_star.tolist()}")

    # ── Unconditional diffusion  (load from original checkpoint, no retrain)
    data_gen_uncond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha,
        kernel_func=lambda X: X[:, :CONDITION_ON]
    )
    model_uncond = Diffusion.DiffusionModel(
        nfeatures=CONDITION_ON, nblocks=NBLOCKS, nunits=NUNITS,
        condition=False, diffusion_steps=DIFFUSION_STEPS
    )
    loaded = experiment_utils.load_model_checkpoint(
        model_uncond, "Diffusion_uncond",
        CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED, device
    )
    if not loaded:
        print("[Diffusion_uncond] checkpoint not found — training from scratch")
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

    # ── Consistency model  (checkpoint keyed by arch + training budget) ───
    # Tag encodes all dimensions that affect the weights so each unique combo
    # gets its own file and the original (128, 3, 80k, 8192) is preserved.
    cm_tag = (
        f"CM"
        f"_u{cfg['nunits_cm']}"
        f"_d{cfg['depth_cm']}"
        f"_ep{cfg['nepochs_cm']}"
        f"_bs{cfg['batch_cm']}"
    )
    data_gen_cond = partial(
        dist_utils.generate_mog_samples_not_differentiable,
        means=mu_list, variances=Sigma_list, weights=alpha
    )
    cm_model = ConsistencyModeliCT(
        nfeatures=2 - CONDITION_ON, condition_on=CONDITION_ON,
        nunits=cfg["nunits_cm"], depth=cfg["depth_cm"]
    )
    loaded_cm = experiment_utils.load_model_checkpoint(
        cm_model, cm_tag, CHECKPOINT_DIR,
        EXPERIMENT_NAME, GLOBAL_SEED, device
    )
    if not loaded_cm:
        print(f"[CM {cm_tag}] checkpoint not found — training from scratch")
        experiment_utils.set_global_seed(GLOBAL_SEED)
        cm_model.train_model(
            X=None, nepochs=cfg["nepochs_cm"], batch_size=cfg["batch_cm"],
            device=device, condition=CONDITION_ON,
            data_generator=data_gen_cond, use_improved_training=True
        )
        experiment_utils.save_model_checkpoint(
            cm_model, cm_tag, CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED
        )

    # ── Analytical Q ──────────────────────────────────────────────────────
    x_grid = np.linspace(-12, 12, 500)
    analytical_Q = compute_analytical_Q(
        mu_list, Sigma_list, alpha,
        mog_means, mog_variances, weights,
        x_grid, ZETA_VALUES
    )

    # ── Sample x ~ Q(x;β) per β ───────────────────────────────────────────
    cdms_samples_dict = {}
    js_per_beta       = {}

    for zeta in ZETA_VALUES:
        print(f"\n[β={zeta}] sampling {cfg['n_cdms_samples']} points ...")
        samples = []

        for i in trange(cfg["n_cdms_samples"]):
            experiment_utils.set_run_seed(GLOBAL_SEED, i)
            x_pred = optimize_LGD_cdms(
                model_uncond,
                mu_list, Sigma_list, alpha,
                mog_means, mog_variances, weights,
                grad_clamp  = cfg["grad_clamp"],
                noise_scale = cfg["noise_scale"],
                num_x_t     = cfg["num_x_t"],
                zeta        = zeta,
                device      = device,
            )
            samples.append(x_pred.float().view(-1).cpu()[0].item())

        samples = np.array(samples)
        cdms_samples_dict[zeta] = samples

        js = js_divergence(samples, analytical_Q[zeta], x_grid)
        js_per_beta[zeta] = js
        print(f"   mean={np.mean(samples):.3f}  std={np.std(samples):.3f}  JS={js:.4f}")

        # Log per-β metrics immediately (visible in WandB while job runs)
        wandb.log({
            f"js/beta_{zeta}":          js,
            f"sample_mean/beta_{zeta}": float(np.mean(samples)),
            f"sample_std/beta_{zeta}":  float(np.std(samples)),
        })

    # ── Headline metric: mean JS across all β  (lower = better) ──────────
    mean_js = float(np.mean(list(js_per_beta.values())))
    print(f"\n>>> mean JS across all β: {mean_js:.4f}")
    wandb.log({"mean_js": mean_js, "lr": LR})

    # ── Plot & upload to WandB ─────────────────────────────────────────────
    fig = make_beta_grid_plot(
        cdms_samples_dict, analytical_Q,
        x_grid, x_star[0].item(), ZETA_VALUES, js_per_beta, cfg
    )
    wandb.log({"beta_grid_plot": wandb.Image(fig)})

    slug = (
        f"clamp{cfg['grad_clamp']}"
        f"_noise{cfg['noise_scale']}"
        f"_nxt{cfg['num_x_t']}"
        f"_CM{cfg['nunits_cm']}u{cfg['depth_cm']}d"
    )
    fig_path  = os.path.join(RESULTS_DIR, f"cdms_{slug}.png")
    json_path = os.path.join(RESULTS_DIR, f"cdms_{slug}.json")

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {fig_path}")

    with open(json_path, "w") as f:
        json.dump({
            "config":      cfg,
            "lr":          LR,
            "mean_js":     mean_js,
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
    p.add_argument("--config_id",     type=int, required=True,
                   help="Grid index — pass $SLURM_ARRAY_TASK_ID here")
    p.add_argument("--wandb_project", type=str, default="cdms-sweep")
    p.add_argument("--wandb_entity",  type=str, default="")
    p.add_argument("--list_configs",  action="store_true",
                   help="Print total config count and exit")
    args = p.parse_args()

    configs = all_configs()
    if args.list_configs:
        print(f"Total configs: {len(configs)}")
        for i, c in enumerate(configs):
            print(f"  [{i:3d}] {c}")
        import sys; sys.exit(0)

    print(f"Config {args.config_id} / {len(configs) - 1}")
    run_config(configs[args.config_id], args)
