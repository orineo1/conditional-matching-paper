"""Figures 0, 0b, 3 (targets, joint GMM, MMD-by-arm) from the analytic module + result JSONs.

Usage: python experiments/witness_unimodal/make_extra_figures.py
(figures 1-2 come from make_figures.py)
"""
import importlib.util, json, glob, math, collections, pathlib
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("expid", HERE / "exp_identifiability.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
X_BI, X_UNI = getattr(m, 'X_BI', -2.0), getattr(m, 'X_UNI', 2.0)
(HERE / "figures").mkdir(exist_ok=True)

# ---- fig 0: the two target conditionals -----------------------------------
g = torch.Generator().manual_seed(0)
yb = m.sample_cond_hard(torch.tensor(float(X_BI)), 40000, generator=g).flatten()
yu = m.sample_cond_hard(torch.tensor(float(X_UNI)), 40000, generator=g).flatten()
ys = torch.linspace(-4, 4, 801)
def dens(x):
    w = torch.softmax(m.cond_logits(torch.tensor(float(x))).flatten(), 0)
    d = torch.zeros_like(ys)
    for k in range(len(w)):
        mu, s = float(m.MU_Y[k]), float(m.SIG_Y[k])
        d += w[k] * torch.exp(-0.5 * ((ys - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return d
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
ax.hist(yb.numpy(), bins=120, density=True, alpha=0.35, color='#0072B2',
        label=f'target: bimodal  p(y | x_bi={X_BI:+.0f})')
ax.hist(yu.numpy(), bins=120, density=True, alpha=0.35, color='#E69F00',
        label=f'target: unimodal  p(y | x_uni={X_UNI:+.0f})')
ax.plot(ys, dens(X_BI), color='#0072B2', lw=1.5); ax.plot(ys, dens(X_UNI), color='#E69F00', lw=1.5)
ax.axvline(0, color='gray', ls=':', lw=1)
ax.annotate('same mean = 0', xy=(0, ax.get_ylim()[1] * 0.95), ha='center', fontsize=9, color='gray')
ax.set_xlabel('y'); ax.set_ylabel('density')
ax.set_title('The two target distributions share the mean but differ in shape')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(HERE / 'figures/fig0_targets.png')

# ---- fig 0b: the joint GMM -------------------------------------------------
xs = torch.linspace(-4.5, 4.5, 401); ys2 = torch.linspace(-3.5, 3.5, 351)
D = torch.zeros(len(ys2), len(xs))
for j, x in enumerate(xs):
    joint_x = torch.exp(m.cond_logits(x.clone()).flatten())   # a_k N(x; m_k, sigx^2)
    for k in range(len(joint_x)):
        mu, s = float(m.MU_Y[k]), float(m.SIG_Y[k])
        D[:, j] += joint_x[k] * torch.exp(-0.5 * ((ys2 - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
im = ax.contourf(xs.numpy(), ys2.numpy(), D.numpy(), levels=30, cmap='Greys')
for xv, lab, col in ((X_BI, 'x_bi = -2\np(y|x) bimodal', '#0072B2'),
                     (X_UNI, 'x_uni = +2\np(y|x) unimodal', '#E69F00')):
    ax.axvline(xv, color=col, ls='--', lw=2)
    ax.annotate(lab, xy=(xv, 3.0), ha='center', fontsize=10, color=col,
                bbox=dict(fc='white', ec=col, alpha=0.85))
ax.set_xlabel('x  (design variable — the landing-map axis)'); ax.set_ylabel('y')
ax.set_title('Joint GMM p(x, y): slicing at a given x yields the conditional target p(y|x)')
fig.colorbar(im, ax=ax, label='density'); fig.tight_layout()
fig.savefig(HERE / 'figures/fig0b_joint.png')

# ---- fig 3: per-restart eval MMD by selection rule -------------------------
data = collections.defaultdict(list)
for f in glob.glob(str(HERE / 'results/identifiability_*cap1_inv_cell_*_mmd_*.json')):
    for r in json.load(open(f))['rows']:
        data[(r['target'], r['arm'])].append(r['mmd_to_bi'] if r['target'] == 'bi' else r['mmd_to_uni'])
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=150, sharey=True)
for ax, tgt, title in ((axes[0], 'uni', 'concentrated unimodal target'),
                       (axes[1], 'bi', 'bimodal target')):
    for i, (arm, col) in enumerate((('mmd_uniform', '#0072B2'), ('mmd_witness', '#E69F00'))):
        v = data[(tgt, arm)]
        x = torch.rand(len(v)) * 0.5 - 0.25 + i
        ax.scatter(x, v, s=14, alpha=0.6, color=col)
        n_bad = sum(1 for u in v if u > 1.0)
        ax.annotate(f"wrong basin:\n{n_bad}/{len(v)}", xy=(i, 2.0), ha='center', fontsize=9, color=col)
    ax.set_xticks([0, 1], ['uniform', 'witness']); ax.set_title(title)
    ax.axhline(1.0, color='gray', ls=':', lw=1); ax.set_ylim(-0.15, 3.9)
axes[0].set_ylabel('MMD of generated samples to the given target\n(low band = correct basin; high band = wrong basin)')
fig.suptitle('Per-restart eval MMD by selection rule (both seeds, 80 restarts/arm)', y=1.0)
fig.tight_layout(); fig.savefig(HERE / 'figures/fig3_mmd_by_arm.png')
print('figures 0, 0b, 3 written')
