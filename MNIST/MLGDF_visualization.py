"""
MLGDF_visualization.py
====================
Visualization & evaluation script for the LGD MNIST experiments.

Loads precomputed results from checkpoints_and_results/, trains or loads the
robust classifier, downloads the conditional model from HuggingFace, then
produces all plots and the final results table.

Usage (local):
    python MLGDF_visualization.py

Usage (Colab):
    Set HF_TOKEN and GITHUB_TOKEN in Colab secrets, then run.

Environment variables:
    HF_TOKEN        – HuggingFace token (required to download checkpoints)
    REPO_ROOT       – root of this repository (auto-detected if not set)
"""

import os, sys, math, pickle, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, login
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

IS_COLAB = _is_colab()

if IS_COLAB:
    import getpass
    import subprocess

    # Install dependencies
    subprocess.run(['pip', 'install', '-q',
                    'diffusers', 'transformers', 'accelerate',
                    'xformers', 'scikit-learn', 'Pillow', 'tqdm', 'wandb'],
                   check=False)

    # GitHub token — set in Colab secrets as 'GITHUB_TOKEN'
    try:
        from google.colab import userdata as _ud
        github_token = _ud.get('GITHUB_TOKEN')
    except Exception:
        github_token = None
    if not github_token:
        github_token = getpass.getpass('GitHub PAT (for private repo): ')

    # ── CONFIGURE THESE BEFORE PUBLISHING ────────────────────────────────
    GITHUB_USERNAME = 'YOUR_GITHUB_USERNAME'   # <-- fill in
    REPO_NAME       = 'YOUR_REPO_NAME'         # <-- fill in
    BRANCH          = 'main'                   # <-- change if needed
    # ─────────────────────────────────────────────────────────────────────

    repo_url = f'https://{github_token}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git'
    if not os.path.exists(REPO_NAME):
        subprocess.run(['git', 'clone', repo_url], check=True)
    else:
        subprocess.run(['git', '-C', REPO_NAME, 'pull'], check=True)
    subprocess.run(['git', '-C', REPO_NAME, 'checkout', BRANCH], check=True)

    MNIST_DIR = f'/content/{REPO_NAME}/MNIST'
    SRC_DIR   = f'{MNIST_DIR}/src'
    CKPT_DIR  = f'{MNIST_DIR}/checkpoints_and_results'
    PLOTS_DIR = f'{MNIST_DIR}/notebooks/plots'

else:
    # Local: derive paths relative to this script
    HERE      = os.path.dirname(os.path.abspath(__file__))
    MNIST_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(MNIST_DIR, 'src')
    CKPT_DIR = os.path.join(MNIST_DIR, 'checkpoints_and_results')
    PLOTS_DIR = os.path.join(MNIST_DIR, 'plots')

for p in [SRC_DIR, MNIST_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace login
# ─────────────────────────────────────────────────────────────────────────────
# Set HF_TOKEN in Colab secrets (key: 'HF_TOKEN') or as an env variable.
# Never hardcode tokens here.
if IS_COLAB:
    try:
        from google.colab import userdata as _ud
        hf_token = _ud.get('HF_TOKEN')
    except Exception:
        hf_token = None
else:
    hf_token = os.environ.get('HF_TOKEN', None)

if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    login()  # interactive prompt

# ─────────────────────────────────────────────────────────────────────────────
# Imports from repo src  (after sys.path is set)
# ─────────────────────────────────────────────────────────────────────────────
from classifier import load_or_train_classifier                          # noqa: E402
from cond_model import (                                                 # noqa: E402
    CircularAngleConsistencyModel, angles_to_circular, circular_to_angles
)

# Reuse shared utilities already defined in MNIST_MLGDF.py
# (mog_pdf, classify_generated_images, sliced_wasserstein_distance)
sys.path.insert(0, MNIST_DIR)
from MNIST_MLGDF import (                                                # noqa: E402
    mog_pdf,
    classify_generated_images,
    sliced_wasserstein_distance,
    set_seed,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global seed & device
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f'[Seed] All random seeds set to {seed}')

set_global_seed(GLOBAL_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {device}')
print(f'CKPT   : {CKPT_DIR}')
print(f'PLOTS  : {PLOTS_DIR}')
print(f'SRC    : {SRC_DIR}')

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (shared with MNIST_MLGDF.py)
# ─────────────────────────────────────────────────────────────────────────────
NORM_MEAN  = 0.1307
NORM_STD   = 0.3081
HF_REPO_ID = 'anon-submission-cdm/cdm-inverse-design'

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Load / train classifier
# ─────────────────────────────────────────────────────────────────────────────
CLF_PATH = os.path.join(CKPT_DIR, 'robust_classifier.pth')

digit_classifier = load_or_train_classifier(
    save_path  = CLF_PATH,
    device     = device,
    epochs     = 10,
    batch_size = 128,
    lr         = 1e-3,
    seed       = GLOBAL_SEED,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Load conditional model from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
COND_PT = os.path.join(CKPT_DIR, 'MnistConditional500Epoch.pt')

if not os.path.exists(COND_PT):
    print('Downloading conditional model from HuggingFace...')
    COND_PT = hf_hub_download(
        repo_id  = HF_REPO_ID,
        filename = 'MnistConditional500Epoch.pt',
        token    = hf_token or None,
    )
    print(f'Downloaded → {COND_PT}')
else:
    print(f'Found → {COND_PT}')

cond_model = CircularAngleConsistencyModel(
    nfeatures=2, img_features=784, eps=0.002,
    nunits=128, depth=5, device=device,
)
ckpt = torch.load(COND_PT, map_location=device)
cond_model.load_state_dict(ckpt['model_state_dict'])
cond_model.eval()
print(f'Conditional model loaded (epoch {ckpt["epoch"]}) ✓')

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Load pkl payloads
# ─────────────────────────────────────────────────────────────────────────────
def apply_clamp_normalization(payload: dict) -> dict:
    """
    Post-hoc contrast fix: clamps to [-1, 1] then rescales to [0, 1] for display.
    Equivalent to what in-loop clamping prevents during generation.
    """
    fixed = []
    for img in payload['results']:
        t = torch.tensor(img, dtype=torch.float32)
        t = t.clamp(-1.0, 1.0)
        t = (t + 1.0) / 2.0
        fixed.append(t.numpy())
    payload = dict(payload)
    payload['results'] = fixed
    return payload


def load_pkl(name: str) -> dict:
    path = os.path.join(CKPT_DIR, f'{name}.pkl')
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    print(f'Loaded {name}.pkl  ({len(payload["results"])} seeds)')
    return payload


payload_uniform  = apply_clamp_normalization(load_pkl('unif_ns600_st290_ssoriginal_xt3'))
payload_bimodal  = apply_clamp_normalization(load_pkl('Bimodal'))
payload_unimodal = apply_clamp_normalization(load_pkl('uni_var515_st130_ssdouble_xt3'))

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_images(payload, ncols=5, dpi=100,
                    save_path=None, save_no_title=False,
                    classifier=None, device=None):
    """Grid of all seed images with optional classifier labels."""
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    n     = len(results)
    nrows = math.ceil(n / ncols)
    preds = (classify_generated_images(results, classifier, device, threshold=0.75)
             if classifier is not None else [None] * n)

    for row in range(nrows):
        start, end  = row * ncols, min(row * ncols + ncols, n)
        row_results = results[start:end]
        row_losses  = loss_log[start:end]
        row_seeds   = seed_log[start:end]
        row_preds   = preds[start:end]
        n_in_row    = len(row_results)

        def _make(show_titles, _rp=row_preds):
            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3),
                                     gridspec_kw=dict(wspace=0.02))
            axes = np.array(axes).reshape(ncols)
            for c, (img, loss, seed, pred) in enumerate(
                    zip(row_results, row_losses, row_seeds, _rp)):
                axes[c].imshow(img, cmap='gray')
                if show_titles:
                    title = f'Seed {seed} | Loss {loss:.4f}'
                    if classifier is not None:
                        title += f' | {str(pred) if pred is not None else "None"}'
                    axes[c].set_title(title, fontsize=11, pad=2)
                axes[c].axis('off')
            for c in range(n_in_row, ncols):
                axes[c].axis('off')
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            return fig

        fig_titled = _make(show_titles=True)
        if save_path:
            base, ext = os.path.splitext(save_path)
            fig_titled.savefig(f'{base}_row{row+1}{ext}', dpi=dpi, bbox_inches='tight')
        plt.show()

        if save_no_title:
            fig_clean = _make(show_titles=False)
            base, ext = os.path.splitext(save_path) if save_path else ('all_images', '.png')
            fig_clean.savefig(f'{base}_row{row+1}_notitle{ext}', dpi=dpi, bbox_inches='tight')
            plt.close(fig_clean)


def plot_top_k_images(payload, top_k=5, dpi=100,
                      save_path=None, save_no_title=False):
    """Ranked grid of the top-k images by SWD loss. Returns top_ix."""
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    k      = min(top_k, len(results))
    top_ix = np.argsort(loss_log)[:k]
    ncols  = min(5, k)
    nrows  = math.ceil(k / ncols)

    def _make(show_titles):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3, nrows * 3),
                                 gridspec_kw=dict(wspace=0.02, hspace=0.1))
        axes = np.array(axes).reshape(nrows, ncols)
        for rank, idx in enumerate(top_ix):
            r, c = divmod(rank, ncols)
            axes[r, c].imshow(results[idx], cmap='gray')
            if show_titles:
                axes[r, c].set_title(
                    f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                    fontsize=7, pad=2
                )
            axes[r, c].axis('off')
        for rank in range(k, nrows * ncols):
            r, c = divmod(rank, ncols)
            axes[r, c].axis('off')
        if show_titles:
            plt.suptitle(f'[{experiment_name}] Top {k} Images',
                         fontsize=12, fontweight='bold', y=1.002)
        plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
        return fig

    fig_titled = _make(show_titles=True)
    if save_path:
        fig_titled.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig_titled)

    if save_no_title:
        fig_clean = _make(show_titles=False)
        base, ext = os.path.splitext(save_path) if save_path else (f'top{k}_clean', '.png')
        fig_clean.savefig(f'{base}_notitle{ext}', dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close(fig_clean)

    return top_ix


def plot_top_k_distributions(payload, model_cond, top_ix, top_k_dist=5,
                              save_path=None, dpi=150, save_no_title=False):
    """Cartesian + polar angle distribution plots for the top-k results."""
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    x_range_np      = payload['x_range']
    target_pdf_np   = payload['target_pdf']

    dist_k  = min(top_k_dist, len(top_ix))
    dist_ix = top_ix[:dist_k]

    # Pre-sample angles for all top-k images
    all_max_y = target_pdf_np.max()
    temp_angs = []
    for idx in dist_ix:
        cond_t = (torch.tensor(results[idx], dtype=torch.float32)
                  .flatten().unsqueeze(0))
        ang = circular_to_angles(
            model_cond.sample(nsamples=500, condition_x=cond_t,
                              ts=[150., 50., 20., 10., 5., 1.])[0]
        )
        temp_angs.append(ang)
        h, _ = np.histogram(ang.detach().cpu().numpy(),
                             bins=30, range=(0, 360), density=True)
        all_max_y = max(all_max_y, h.max() if len(h) else 0)
    y_lim = all_max_y * 1.1

    target_rad_x        = np.deg2rad(x_range_np)
    target_rad_y        = target_pdf_np * (360.0 / (2 * np.pi))
    target_rad_x_closed = np.append(target_rad_x, target_rad_x[0])
    target_rad_y_closed = np.append(target_rad_y, target_rad_y[0])

    local_polar_max = target_rad_y.max()
    for ang in temp_angs:
        counts, _ = np.histogram(
            np.deg2rad(ang.detach().cpu().numpy()),
            bins=np.linspace(0, 2 * np.pi, 37), density=True
        )
        local_polar_max = max(local_polar_max, counts.max() if len(counts) else 0)
    polar_ylim = local_polar_max * 1.1

    def _make_cart(show_titles, show_legend):
        fig, cart_axes = plt.subplots(1, dist_k, figsize=(dist_k * 4, 4), sharey=True)
        cart_axes = np.array(cart_axes).reshape(dist_k)
        for rank, (idx, ang) in enumerate(zip(dist_ix, temp_angs)):
            ang_np = ang.detach().cpu().numpy()
            ax     = cart_axes[rank]
            ax.hist(ang_np, bins=30, alpha=0.6, color='skyblue', edgecolor='black',
                    range=(0, 360), density=True, label='Sampled')
            ax.plot(x_range_np, target_pdf_np, color='orange', linewidth=2, label='Target')
            ax.fill_between(x_range_np, target_pdf_np, alpha=0.25, color='orange')
            ax.set_xlim(0, 360)
            ax.set_ylim(0, y_lim)
            ax.set_xlabel('Angle (°)', fontsize=13)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, alpha=0.3)
            if show_titles:
                ax.set_title(
                    f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                    fontsize=9
                )
            if rank == 0:
                ax.set_ylabel('Density', fontsize=12)
                if show_legend:
                    ax.legend(fontsize=8)
        if show_titles:
            fig.suptitle(
                f'[{experiment_name}] Top {dist_k} Angle Distributions — Cartesian',
                fontsize=13, fontweight='bold'
            )
        fig.tight_layout()
        return fig

    def _make_polar(show_titles, show_legend):
        fig, polar_axes = plt.subplots(
            1, dist_k, figsize=(dist_k * 4, 4),
            subplot_kw={'projection': 'polar'}
        )
        polar_axes = np.array(polar_axes).reshape(dist_k)
        for rank, (idx, ang) in enumerate(zip(dist_ix, temp_angs)):
            ang_rad     = np.deg2rad(ang.detach().cpu().numpy())
            bin_edges   = np.linspace(0, 2 * np.pi, 37)
            counts, _   = np.histogram(ang_rad, bins=bin_edges, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width       = bin_edges[1] - bin_edges[0]
            ax = polar_axes[rank]
            ax.bar(bin_centers, counts, width=width, alpha=0.55,
                   color='skyblue', edgecolor='black', label='Sampled', zorder=2)
            ax.fill(target_rad_x_closed, target_rad_y_closed,
                    color='orange', alpha=0.35, zorder=3)
            ax.plot(target_rad_x_closed, target_rad_y_closed,
                    color='orange', linewidth=2.5, zorder=4, label='Target')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=12)
            ax.set_ylim(0, polar_ylim)
            if show_titles:
                ax.set_title(
                    f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                    fontsize=9, pad=14
                )
            if rank == 0 and show_legend:
                ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.15))
        if show_titles:
            fig.suptitle(
                f'[{experiment_name}] Top {dist_k} Angle Distributions — Polar',
                fontsize=13, fontweight='bold'
            )
        fig.tight_layout()
        return fig

    fig_cart  = _make_cart(show_titles=True, show_legend=True)
    fig_polar = _make_polar(show_titles=True, show_legend=True)
    if save_path:
        base, ext = os.path.splitext(save_path)
        fig_cart .savefig(f'{base}_cart{ext}',  dpi=dpi, bbox_inches='tight')
        fig_polar.savefig(f'{base}_polar{ext}', dpi=dpi, bbox_inches='tight')
    plt.show()

    if save_no_title:
        base, ext   = os.path.splitext(save_path) if save_path else ('dist', '.png')
        fig_cart_c  = _make_cart(show_titles=False, show_legend=False)
        fig_polar_c = _make_polar(show_titles=False, show_legend=False)
        fig_cart_c .savefig(f'{base}_cart_notitle{ext}',  dpi=dpi, bbox_inches='tight')
        fig_polar_c.savefig(f'{base}_polar_notitle{ext}', dpi=dpi, bbox_inches='tight')
        plt.close(fig_cart_c)
        plt.close(fig_polar_c)

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Quantitative results table
# ─────────────────────────────────────────────────────────────────────────────
def compute_experiment_stats(payload, classifier, device, top_k=5, threshold=0.75):
    losses = np.array(payload['loss_log'])
    times  = np.array(payload.get('time_log', [np.nan] * len(losses)))
    n      = len(losses)
    top_ix = np.argsort(losses)[:top_k]

    # reuse classify_generated_images from MNIST_MLGDF
    preds = classify_generated_images(payload['results'], classifier, device,
                                      threshold=threshold)
    digit_counts   = {d: 0 for d in range(10)}
    n_unclassified = 0
    for p in preds:
        if p is None:
            n_unclassified += 1
        else:
            digit_counts[p] += 1

    return {
        'swd_all_mean':     losses.mean(),
        'swd_all_std':      losses.std(),
        'swd_top_mean':     losses[top_ix].mean(),
        'swd_top_std':      losses[top_ix].std(),
        'time_mean':        times.mean(),
        'time_std':         times.std(),
        'digit_counts':     digit_counts,
        'digit_pct':        {d: 100.0 * digit_counts[d] / n for d in range(10)},
        'n_seeds':          n,
        'n_classified':     n - n_unclassified,
        'n_unclassified':   n_unclassified,
        'pct_unclassified': 100.0 * n_unclassified / n,
        'predicted_labels': preds,
    }


def render_results_table(stats: dict, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for exp_name, s in stats.items():
        row = {
            'Experiment':      exp_name,
            'SWD-all':         f"{s['swd_all_mean']:.4f} ± {s['swd_all_std']:.4f}",
            f'SWD-top{top_k}': f"{s['swd_top_mean']:.4f} ± {s['swd_top_std']:.4f}",
            'Time (s)':        f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
            'N seeds':         s['n_seeds'],
        }
        for d in range(10):
            row[str(d)] = f"{s['digit_pct'][d]:.1f}%"
        row['None'] = f"{s['pct_unclassified']:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows).set_index('Experiment')

# ─────────────────────────────────────────────────────────────────────────────
# 6 · Run all plots & table
# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    experiments = [
        ('uniform',  payload_uniform),
        ('bimodal',  payload_bimodal),
        ('unimodal', payload_unimodal),
    ]

    top_ix_map = {}
    for name, payload in experiments:
        label = name.capitalize()
        print(f'\n{"="*50}\n  {label}\n{"="*50}')

        # All images
        plot_all_images(
            payload, ncols=5, dpi=100,
            save_path     = os.path.join(PLOTS_DIR, f'{label}_all.png'),
            save_no_title = True,
            classifier    = digit_classifier,
            device        = device,
        )

        # Top-5 images
        top_ix = plot_top_k_images(
            payload, top_k=5, dpi=100,
            save_path     = os.path.join(PLOTS_DIR, f'{label}_top5.png'),
            save_no_title = True,
        )
        top_ix_map[name] = top_ix

        # Top-5 distributions
        plot_top_k_distributions(
            payload, cond_model, top_ix, top_k_dist=5,
            save_path     = os.path.join(PLOTS_DIR, f'{label}_dist.png'),
            save_no_title = True,
            dpi           = 150,
        )

    # Results table
    print('\n' + '='*50 + '\n  Quantitative Results\n' + '='*50)
    TOP_K_TABLE = 5
    stats = {}
    for name, payload in experiments:
        label = name.capitalize()
        print(f'Classifying {label}...')
        stats[label] = compute_experiment_stats(
            payload, digit_classifier, device, top_k=TOP_K_TABLE
        )
        s = stats[label]
        print(f'  SWD all:     {s["swd_all_mean"]:.4f} ± {s["swd_all_std"]:.4f}')
        print(f'  SWD top-{TOP_K_TABLE}: {s["swd_top_mean"]:.4f} ± {s["swd_top_std"]:.4f}')
        print(f'  Time:        {s["time_mean"]:.1f} ± {s["time_std"]:.1f} s')
        print(f'  Digits:      {s["predicted_labels"]}')

    df_results = render_results_table(stats, top_k=TOP_K_TABLE)
    table_path = os.path.join(PLOTS_DIR, 'results_table.csv')
    df_results.to_csv(table_path)
    print(f'\nTable saved → {table_path}')
    print(df_results.to_string())


if __name__ == '__main__':
    run_all()
