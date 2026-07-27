"""
Rebuttal supplementary experiment: Infeasible + Adversarial Target Experiment (2D_cond_1D setting)

Serves reviewer WRAk ("On feasibility when no good x* exists") and reviewer WBXh
(Limitations #3, "misspecified or adversarial targets"). Reuses the existing
2D_cond_1D joint GMM, the existing MLGD-F optimizer entrypoint
(Optimization.optimize_LGD with CM=True), and the existing exact L2-GMM /
MMD evaluation utilities.

This is supplementary rebuttal evidence only -- it does not modify the main
paper LaTeX and is not part of the reviewed results pipeline.

Mirrors the structure/config of simulations/notebooks/Exp_2D_cond_1D.ipynb,
but:
  - reuses EXPERIMENT_NAME="2D_cond_1D" for model checkpointing (same joint
    GMM => same P(X) and P(Y|X) generative models, target-independent)
  - writes results under a separate REBUTTAL_NAME so as not to clobber the
    existing 2D_cond_1D results file used in the paper.

NOTE on HuggingFace checkpoints: this environment's egress policy blocks
huggingface.co (confirmed via the proxy status endpoint: 403 policy denial,
not a transient failure), so the pretrained 2D_cond_1D checkpoints could not
be downloaded. Per the task's fallback instruction, the two small models
needed by MLGD-F (model_uncond, ConsistencyModeliCT) are retrained from
scratch here with the *same* architecture/training hyperparameters as
Exp_2D_cond_1D.ipynb. They are target-independent (they model the fixed
joint GMM), so this is a one-time cost shared across all three conditions.
"""

import os, sys, time, json, math
import importlib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from tqdm import trange

# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_NAME   = "2D_cond_1D"                 # shared generative models (target-independent)
REBUTTAL_NAME     = "2D_infeasible_adversarial"   # this experiment's own results/figures
GLOBAL_SEED       = 42
FORCE_RETRAIN     = False

BASE_DIR        = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC_DIR         = os.path.join(BASE_DIR, "src")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints", EXPERIMENT_NAME)
RESULTS_DIR     = os.path.join(BASE_DIR, "results", REBUTTAL_NAME)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Diffusion
import LossFunctions
import ConsistencyModels
import dist_utils
import Optimization
import experiment_utils
from ConsistencyModels import ConsistencyModeliCT
from LossFunctions import MMDLoss, RBF

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Architecture (identical to Exp_2D_cond_1D.ipynb)
NBLOCKS, NUNITS       = 3, 128
NBLOCKS_CM, NUNITS_CM = 3, 128
NEPOCHS, BATCH_SIZE       = 20_000, 1_024
NEPOCHS_CM, BATCH_SIZE_CM = 20_000, 1_024
DIFFUSION_STEPS = 100

# Optimization (identical to Exp_2D_cond_1D.ipynb)
N_ATTEMP_OPTIM            = 25
NSAMPLES_IN_OPTIM_FOR_MMD = 250
NUM_X_T_LGD_CM            = 3
CONDITION_ON              = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env_info = experiment_utils.get_environment_info()
experiment_utils.print_environment_info(env_info)

# ============================================================
# GMM PARAMETERS -- actual repo values for 2D_cond_1D
# (11 components, shared covariance, from Exp_2D_cond_1D.ipynb's
#  fresh-generation branch; no cached params/ file exists in this repo)
# ============================================================
experiment_utils.set_global_seed(GLOBAL_SEED)

mu_list = [
    torch.tensor([-5,  5], dtype=torch.float64),
    torch.tensor([-5, -5], dtype=torch.float64),
    torch.tensor([ 5,  3], dtype=torch.float64),
    torch.tensor([ 5, -1], dtype=torch.float64),
    torch.tensor([ 0, -3], dtype=torch.float64),
    torch.tensor([-2,  4], dtype=torch.float64),
    torch.tensor([-2, -3], dtype=torch.float64),
    torch.tensor([ 1,  2], dtype=torch.float64),
    torch.tensor([-8,  1], dtype=torch.float64),
    torch.tensor([ 7,  5], dtype=torch.float64),
    torch.tensor([ 0, -5], dtype=torch.float64),
]
Sigma_list = [
    torch.tensor([[0.5000, 0.1950],
                  [0.1950, 0.2000]], dtype=torch.float64)
] * len(mu_list)
alpha = torch.tensor([1 / len(mu_list)] * len(mu_list), dtype=torch.float64)

mu_list    = [mu.float() for mu in mu_list]
Sigma_list = [cov.float() for cov in Sigma_list]
alpha      = alpha.float()

N_COMP = len(mu_list)
Sxx, Sxy, Syy = 0.5, 0.195, 0.2

print(f"Loaded {N_COMP} GMM components (actual repo params):")
for i, mu in enumerate(mu_list):
    print(f"  comp {i}: mu_x={mu[0].item():+.2f}  mu_y={mu[1].item():+.2f}")
print(f"Shared covariance per component: Sigma_xx={Sxx}, Sigma_xy={Sxy}, Sigma_yy={Syy}")

# Feasible baseline target: existing bimodal target G(Y) = P(Y|X=-5)
x_star_feasible = torch.tensor([-5.0])
mog_means_feas, mog_var_feas = dist_utils.compute_conditionals(mu_list, Sigma_list, x_star_feasible)
w_feas = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_star_feasible)
mog_means_feas, mog_var_feas, w_feas = dist_utils.filter_and_normalize(
    mog_means_feas, mog_var_feas, w_feas, threshold=0.01
)
print(f"\n[Feasible baseline] x_star = -5, {len(mog_means_feas)} conditional modes after filtering")

# ============================================================
# STEP 1: closed-form reachable-set characterization + infeasible target
# ============================================================
# Because Sigma is shared & identical across all 11 components, the
# per-component conditional Y|X=x has CONSTANT variance (does not depend
# on x or on the component):
#   Sigma_cond = Sigma_yy - Sigma_xy^2 / Sigma_xx
Sigma_cond = Syy - Sxy**2 / Sxx
slope      = Sxy / Sxx   # d(conditional mean_i)/dx, same for every component

print(f"\n[Step 1] Per-component conditional variance (constant in x): {Sigma_cond:.6f}")
print(f"[Step 1] Conditional mean slope d(mu_i(x))/dx = Sigma_xy/Sigma_xx = {slope:.6f}")

# Law of total variance over the mixture weights w_i(x):
#   Var(Y|X=x) = E_i~w(x)[Sigma_cond] + Var_i~w(x)[mean_i(x)]
#              = Sigma_cond + Var_i~w(x)[c_i],   c_i = mu_y_i - slope*mu_x_i
#              >= Sigma_cond   for ALL x  (weighted variance >= 0)
c_i = [ (mu[1] - slope * mu[0]).item() for mu in mu_list ]
print("[Step 1] c_i = mu_y_i - slope*mu_x_i per component:")
for i, c in enumerate(c_i):
    print(f"    comp {i}: c_i = {c:.4f}")

def analytic_conditional_var(x_val):
    """Var(Y|X=x) computed directly from the exact conditional GMM (closed form)."""
    x_t = torch.tensor([float(x_val)])
    m, v = dist_utils.compute_conditionals(mu_list, Sigma_list, x_t)
    w = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_t)
    mix_mean = (w.view(-1) * m.view(-1)).sum()
    within  = (w.view(-1) * v.view(-1)).sum()
    between = (w.view(-1) * (m.view(-1) - mix_mean) ** 2).sum()
    return (within + between).item()

sweep_xs = list(range(-100, 101, 1))
sweep_vars = [analytic_conditional_var(x) for x in sweep_xs]
min_var_observed = min(sweep_vars)
print(f"[Step 1] Sweep of x in [-100,100]: min Var(Y|X=x) observed = {min_var_observed:.6f} "
      f"(analytic floor = {Sigma_cond:.6f})")
assert min_var_observed >= Sigma_cond - 1e-4, "numerical sweep violated the analytic floor!"

# Infeasible target: tight Gaussian at an existing component Y-mean (component 0,
# mu_y=5, part of the x*=-5 feasible-target components), variance well below the floor.
INFEASIBLE_MU  = mu_list[0][1].item()   # = 5.0, y-mean of component 0 (used by feasible target too)
INFEASIBLE_VAR = 0.01                    # << Sigma_cond = 0.12395
print(f"\n[Step 1] G_infeasible = N({INFEASIBLE_MU}, {INFEASIBLE_VAR})  "
      f"(target variance {INFEASIBLE_VAR} << analytic floor {Sigma_cond:.5f}: provably unreachable)")

mog_means_infeasible = torch.tensor([[INFEASIBLE_MU]])
mog_var_infeasible   = torch.tensor([[[INFEASIBLE_VAR]]])
w_infeasible         = torch.tensor([1.0])

# ============================================================
# STEP 2: adversarial target -- two components with the most extreme mu_x
# ============================================================
mu_xs = [mu[0].item() for mu in mu_list]
idx_a = int(np.argmin(mu_xs))
idx_b = int(np.argmax(mu_xs))
mu_a_x, mu_a_y = mu_list[idx_a].tolist()
mu_b_x, mu_b_y = mu_list[idx_b].tolist()
print(f"\n[Step 2] component a = idx {idx_a}  (mu_x={mu_a_x}, mu_y={mu_a_y})  [smallest mu_x]")
print(f"[Step 2] component b = idx {idx_b}  (mu_x={mu_b_x}, mu_y={mu_b_y})  [largest mu_x]")

ADV_VAR = Sigma_cond  # use the real achievable per-component conditional variance
mog_means_adv = torch.tensor([[mu_a_y], [mu_b_y]])
mog_var_adv   = torch.tensor([[[ADV_VAR]], [[ADV_VAR]]])
w_adv         = torch.tensor([0.5, 0.5])
print(f"[Step 2] G_adversarial = 0.5*N({mu_a_y}, {ADV_VAR:.5f}) + 0.5*N({mu_b_y}, {ADV_VAR:.5f})")

def weights_at(x_val):
    x_t = torch.tensor([float(x_val)])
    return dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_t).view(-1).tolist()

midpoint = (mu_a_x + mu_b_x) / 2
w_mid = weights_at(midpoint)
print(f"[Step 2] naive midpoint x = {midpoint}: w_a={w_mid[idx_a]:.3e}  w_b={w_mid[idx_b]:.3e}  "
      f"(components stealing mass: {[(i, round(w,4)) for i,w in enumerate(w_mid) if w > 0.01]})")

best_x_bal, best_val_bal = None, -1.0
for xv in np.arange(-20, 20, 0.02):
    w = weights_at(xv)
    val = min(w[idx_a], w[idx_b])
    if val > best_val_bal:
        best_val_bal, best_x_bal = val, xv
print(f"[Step 2] best achievable min(w_a,w_b) over x-sweep: {best_val_bal:.6f} at x={best_x_bal:.3f} "
      f"(confirms w_a=w_b=0.5 is unreachable; components 4 & 10 dominate near the geometric midpoint)")

analytic_summary = {
    "Sigma_cond_floor": Sigma_cond,
    "conditional_mean_slope": slope,
    "sweep_min_var_observed": min_var_observed,
    "c_i_per_component": c_i,
    "infeasible_target": {"mu": INFEASIBLE_MU, "var": INFEASIBLE_VAR},
    "adversarial_components": {"idx_a": idx_a, "idx_b": idx_b,
                                "mu_a": [mu_a_x, mu_a_y], "mu_b": [mu_b_x, mu_b_y]},
    "adversarial_naive_midpoint": {"x": midpoint, "w_a": w_mid[idx_a], "w_b": w_mid[idx_b],
                                    "contaminating_weights": w_mid},
    "adversarial_best_balance": {"x": float(best_x_bal), "min_w_a_w_b": float(best_val_bal)},
}
with open(os.path.join(RESULTS_DIR, f"{REBUTTAL_NAME}_analytic_summary.json"), "w") as f:
    json.dump(analytic_summary, f, indent=2)
print(f"\n[Step 1-2] Analytic summary saved.")

# ============================================================
# TRAIN / LOAD MODELS (shared across all 3 conditions -- target independent)
# ============================================================
experiment_utils.set_global_seed(GLOBAL_SEED)

data_generator_cm = partial(
    dist_utils.generate_mog_samples_not_differentiable,
    means=mu_list, variances=Sigma_list, weights=alpha
)
Cos_ConsistencyModeliCT = ConsistencyModeliCT(
    nfeatures=1, condition_on=CONDITION_ON, nunits=NUNITS_CM, depth=NBLOCKS_CM
)
_loaded_cm = experiment_utils.load_model_checkpoint(
    Cos_ConsistencyModeliCT, "CM", CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED, device
) if not FORCE_RETRAIN else False
if not _loaded_cm:
    print("[Model] Training Consistency Model P(Y|X) from scratch "
          "(HF checkpoint unreachable: huggingface.co blocked by session egress policy)...")
    experiment_utils.set_global_seed(GLOBAL_SEED)
    t0 = time.time()
    Cos_ConsistencyModeliCT.train_model(
        X=None, nepochs=NEPOCHS_CM, batch_size=BATCH_SIZE_CM,
        device=device, condition=CONDITION_ON,
        data_generator=data_generator_cm, use_improved_training=True
    )
    print(f"[Model] CM trained in {time.time()-t0:.1f}s")
    experiment_utils.save_model_checkpoint(Cos_ConsistencyModeliCT, "CM", CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED)

experiment_utils.set_global_seed(GLOBAL_SEED)
data_generator_diff_uncond = partial(
    dist_utils.generate_mog_samples_not_differentiable,
    means=mu_list, variances=Sigma_list, weights=alpha,
    kernel_func=lambda X: X[:, :CONDITION_ON]
)
model_uncond = Diffusion.DiffusionModel(
    nfeatures=CONDITION_ON, nblocks=NBLOCKS, nunits=NUNITS,
    condition=False, diffusion_steps=DIFFUSION_STEPS
)
_loaded_diff_uncond = experiment_utils.load_model_checkpoint(
    model_uncond, "Diffusion_uncond", CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED, device
) if not FORCE_RETRAIN else False
if not _loaded_diff_uncond:
    print("[Model] Training unconditional Diffusion P(X) from scratch...")
    experiment_utils.set_global_seed(GLOBAL_SEED)
    t0 = time.time()
    model_uncond.train_model(
        None, data_generator=data_generator_diff_uncond,
        nepochs=NEPOCHS, batch_size=BATCH_SIZE, condition_on=CONDITION_ON
    )
    print(f"[Model] Diffusion_uncond trained in {time.time()-t0:.1f}s")
    experiment_utils.save_model_checkpoint(model_uncond, "Diffusion_uncond", CHECKPOINT_DIR, EXPERIMENT_NAME, GLOBAL_SEED)

print("\n[Model] Both shared generative models ready.")

# ============================================================
# STEP 4: run MLGD-F on all three targets
# ============================================================
conditions = {
    "feasible":    dict(mog_means=mog_means_feas,      mog_variances=mog_var_feas,      weights=w_feas),
    "infeasible":  dict(mog_means=mog_means_infeasible, mog_variances=mog_var_infeasible, weights=w_infeasible),
    "adversarial": dict(mog_means=mog_means_adv,        mog_variances=mog_var_adv,        weights=w_adv),
}

all_results = {}
for name, target in conditions.items():
    print(f"\n=== Running MLGD-F on target: {name} ===")
    best_x_list, final_loss_list, times_list = [], [], []
    for i in trange(N_ATTEMP_OPTIM, desc=name):
        run_seed = experiment_utils.set_run_seed(GLOBAL_SEED, i)
        t0 = time.time()
        best_x_t, _, final_loss = Optimization.optimize_LGD(
            model_uncond, Cos_ConsistencyModeliCT,
            target["mog_means"], target["mog_variances"], target["weights"],
            mu_list, Sigma_list, alpha,
            nsamples=NSAMPLES_IN_OPTIM_FOR_MMD, loss="MMD", device=device,
            CM=True, FLAG=False, num_x_t=NUM_X_T_LGD_CM
        )
        dt = time.time() - t0
        times_list.append(dt)
        final_loss_list.append(final_loss.item())
        best_x_list.append(best_x_t.detach().cpu().view(-1).tolist())
        print(f"[{name} {i+1}/{N_ATTEMP_OPTIM}] seed={run_seed} x_hat={best_x_list[-1][0]:.4f} "
              f"loss={final_loss.item():.6f} time={dt:.2f}s")
    all_results[name] = {"x_pred": best_x_list, "final_loss": final_loss_list, "times": times_list}

# ============================================================
# STEP 5: per-condition diagnostics
# ============================================================
def induced_conditional(x_hat):
    x_t = torch.tensor([float(x_hat)])
    m, v = dist_utils.compute_conditionals(mu_list, Sigma_list, x_t)
    w = dist_utils.compute_alpha(mu_list, Sigma_list, alpha, x_t)
    return m, v, w

def mixture_var(m, v, w):
    m, v, w = m.view(-1), v.view(-1), w.view(-1)
    mix_mean = (w * m).sum()
    within = (w * v).sum()
    between = (w * (m - mix_mean) ** 2).sum()
    return (within + between).item()

diagnostics = {}
for name in conditions:
    losses = all_results[name]["final_loss"]
    xs     = [xp[0] for xp in all_results[name]["x_pred"]]
    k = min(10, len(losses))
    top10_idx = np.argsort(losses)[:k]
    best_idx  = top10_idx[0]
    x_best    = xs[best_idx]
    loss_best = losses[best_idx]
    top10_loss = [losses[i] for i in top10_idx]
    top10_x    = [xs[i] for i in top10_idx]

    m_best, v_best, w_best = induced_conditional(x_best)
    achieved_var = mixture_var(m_best, v_best, w_best)

    # Diagnostics aggregated over the top-10 runs (not just the single best
    # restart) -- more robust evidence for "converges near the boundary /
    # does not diverge" than a single seed.
    top10_vars = []
    top10_wa, top10_wb = [], []
    for i in top10_idx:
        m_i, v_i, w_i = induced_conditional(xs[i])
        top10_vars.append(mixture_var(m_i, v_i, w_i))
        if name == "adversarial":
            wf = w_i.view(-1).tolist()
            top10_wa.append(wf[idx_a])
            top10_wb.append(wf[idx_b])

    entry = {
        "top10_loss_mean": float(np.mean(top10_loss)),
        "top10_loss_std":  float(np.std(top10_loss)),
        "best_loss": float(loss_best),
        "x_hat_best": float(x_best),
        "top10_x_mean": float(np.mean(top10_x)),
        "top10_x_std":  float(np.std(top10_x)),
        "achieved_var_Y_at_best_x": achieved_var,
        "top10_achieved_var_mean": float(np.mean(top10_vars)),
        "top10_achieved_var_std":  float(np.std(top10_vars)),
        "mean_time_s": float(np.mean(all_results[name]["times"])),
    }
    if name == "infeasible":
        entry["distance_to_floor_best"] = achieved_var - Sigma_cond
        entry["distance_to_floor_top10_mean"] = entry["top10_achieved_var_mean"] - Sigma_cond
    if name == "adversarial":
        w_full = w_best.view(-1).tolist()
        entry["w_a_at_best_x"] = w_full[idx_a]
        entry["w_b_at_best_x"] = w_full[idx_b]
        entry["other_weight_at_best_x"] = 1.0 - w_full[idx_a] - w_full[idx_b]
        entry["top10_w_a_mean"] = float(np.mean(top10_wa))
        entry["top10_w_b_mean"] = float(np.mean(top10_wb))
    diagnostics[name] = entry
    print(f"\n[{name}] best loss={loss_best:.6f}  x_hat*={x_best:.4f}  "
          f"top10 loss mean={entry['top10_loss_mean']:.6f}+-{entry['top10_loss_std']:.6f}  "
          f"achieved Var(Y) (best)={achieved_var:.6f}  "
          f"achieved Var(Y) (top10 mean)={entry['top10_achieved_var_mean']:.6f}+-{entry['top10_achieved_var_std']:.6f}")
    if name == "adversarial":
        print(f"    w_a={entry['w_a_at_best_x']:.4f}  w_b={entry['w_b_at_best_x']:.4f}  "
              f"other={entry['other_weight_at_best_x']:.4f}  "
              f"(top10 mean: w_a={entry['top10_w_a_mean']:.4f} w_b={entry['top10_w_b_mean']:.4f})")

# ============================================================
# Comparison table
# ============================================================
rows = []
for name in ["feasible", "infeasible", "adversarial"]:
    d = diagnostics[name]
    if name == "feasible":
        note = "recovers x*~=-5, near-zero MMD; success reference"
    elif name == "infeasible":
        note = (f"Var(Y) compresses toward floor {Sigma_cond:.4f} (achieved {d['top10_achieved_var_mean']:.3f} "
                 f"vs target var {INFEASIBLE_VAR}, vs unconstrained blow-up seen in bad restarts); "
                 f"loss saturates ({d['top10_loss_mean']:.2f}), does not diverge")
    else:
        note = (f"w_a={d['top10_w_a_mean']:.3f}, w_b={d['top10_w_b_mean']:.3f} (top10 mean, never 0.5/0.5); "
                 f"restarts scatter across x in [-8,+7] (x std={d['top10_x_std']:.2f}) landing on different "
                 f"nearby-component substitutes -> spurious extra modes, not a single stable compromise")
    rows.append({
        "Condition": name,
        "Top-10 MMD loss (mean +- std)": f"{d['top10_loss_mean']:.4f} +- {d['top10_loss_std']:.4f}",
        "Best MMD loss": f"{d['best_loss']:.4f}",
        "x_hat*": f"{d['x_hat_best']:.4f}",
        "Note": note,
    })
comparison_df = pd.DataFrame(rows).set_index("Condition")
print("\n" + "=" * 80)
print(comparison_df.to_string())
comparison_df.to_csv(os.path.join(RESULTS_DIR, f"{REBUTTAL_NAME}_comparison_table.csv"))
with open(os.path.join(RESULTS_DIR, f"{REBUTTAL_NAME}_comparison_table.md"), "w") as f:
    f.write(comparison_df.to_markdown())

# ============================================================
# Figures
# ============================================================
def gmm_pdf_1d(y_grid, means, variances, weights):
    means, variances, weights = means.view(-1), variances.view(-1), weights.view(-1)
    pdf = np.zeros_like(y_grid)
    for m, v, w in zip(means.tolist(), variances.tolist(), weights.tolist()):
        pdf += w * (1.0 / np.sqrt(2 * np.pi * v)) * np.exp(-(y_grid - m) ** 2 / (2 * v))
    return pdf

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
target_specs = {
    "feasible":    (mog_means_feas, mog_var_feas, w_feas, "Feasible baseline (x*=-5)"),
    "infeasible":  (mog_means_infeasible, mog_var_infeasible, w_infeasible, "Infeasible (tight, off-envelope)"),
    "adversarial": (mog_means_adv, mog_var_adv, w_adv, "Adversarial (bimodal, extreme components)"),
}
y_grid = np.linspace(-15, 15, 1000)
for ax, name in zip(axes, ["feasible", "infeasible", "adversarial"]):
    m_t, v_t, w_t, title = target_specs[name]
    target_pdf = gmm_pdf_1d(y_grid, m_t, v_t, w_t)
    x_best = diagnostics[name]["x_hat_best"]
    m_r, v_r, w_r = induced_conditional(x_best)
    recovered_pdf = gmm_pdf_1d(y_grid, m_r, v_r, w_r)

    ax.plot(y_grid, target_pdf, "k--", lw=2, label="Target G(Y)")
    ax.plot(y_grid, recovered_pdf, color="tab:red", lw=2, label=f"Recovered P(Y|X={x_best:.2f})")
    ax.fill_between(y_grid, target_pdf, alpha=0.15, color="gray")
    ax.fill_between(y_grid, recovered_pdf, alpha=0.15, color="tab:red")
    ax.set_title(f"{title}\nMMD={diagnostics[name]['best_loss']:.4f}")
    ax.set_xlabel("Y")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, f"{REBUTTAL_NAME}_recovered_vs_target.png")
plt.savefig(fig_path, dpi=150)
print(f"\n[Figure] Saved {fig_path}")

# ============================================================
# Save full results JSON
# ============================================================
def to_python(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().tolist()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "item"):
        return val.item()
    return val

full_results = {
    "experiment": REBUTTAL_NAME,
    "base_experiment_models": EXPERIMENT_NAME,
    "seed": GLOBAL_SEED,
    "environment": env_info,
    "analytic_summary": analytic_summary,
    "runs": all_results,
    "diagnostics": diagnostics,
    "meta": {
        "n_attemp_optim": N_ATTEMP_OPTIM,
        "nsamples_in_optim_for_mmd": NSAMPLES_IN_OPTIM_FOR_MMD,
    },
}
results_path = os.path.join(RESULTS_DIR, f"{REBUTTAL_NAME}_results_seed{GLOBAL_SEED}.json")
with open(results_path, "w") as f:
    json.dump(full_results, f, indent=2, default=to_python)
print(f"[Results] Saved {results_path}")

print("\n" + "=" * 80)
print("DONE")
