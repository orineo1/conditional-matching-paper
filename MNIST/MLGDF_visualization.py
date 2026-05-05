"""
MLGDF.py
========
Visualization & evaluation script for the LGD MNIST experiments.

Accepts a results directory containing .pkl files, skips any missing
experiments, and produces all plots and the final results table.

Usage:
    python MLGDF.py --results_dir /path/to/results/unimodal_run/my_run/
    python MLGDF.py --results_dir checkpoints_and_results/
    python MLGDF.py --results_dir results/ --top_k 5 --dpi 150

Arguments:
    --results_dir   Directory containing .pkl result files (required)
    --ckpt_dir      Directory with model checkpoints (default: checkpoints_and_results/)
    --plots_dir     Where to save plots (default: plots/ inside results_dir)
    --top_k         How many top images to show (default: 5)
    --dpi           Plot DPI (default: 150)
    --no_titles     Save additional copies of plots without titles
"""

import os, sys, math, pickle, random, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from huggingface_hub import hf_hub_download, login

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', type=str, required=True,
                   help='Directory containing .pkl result files')
    p.add_argument('--ckpt_dir',    type=str, default=None,
                   help='Directory with model checkpoints (default: checkpoints_and_results/ next to script)')
    p.add_argument('--plots_dir',   type=str, default=None,
                   help='Where to save plots (default: plots/ inside results_dir)')
    p.add_argument('--top_k',       type=int, default=5)
    p.add_argument('--dpi',         type=int, default=150)
    p.add_argument('--no_titles',   action='store_true',
                   help='Also save plots without titles')
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = HERE
SRC_DIR   = os.path.join(MNIST_DIR, 'src')
for p in [SRC_DIR, MNIST_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace login
# ─────────────────────────────────────────────────────────────────────────────
hf_token = os.environ.get('HF_TOKEN', None)
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    login()

# ─────────────────────────────────────────────────────────────────────────────
# Imports from repo src
# ─────────────────────────────────────────────────────────────────────────────
from classifier import load_or_train_classifier
from cond_model import (
    CircularAngleConsistencyModel, angles_to_circular, circular_to_angles
)
from MNIST_MLGDF import (
    mog_pdf,
    classify_generated_images,
    sliced_wasserstein_distance,
    set_seed,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
NORM_MEAN   = 0.1307
NORM_STD    = 0.3081
HF_REPO_ID  = 'anon-submission-cdm/cdm-inverse-design'

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# Load pkl — skips if file not found
# ─────────────────────────────────────────────────────────────────────────────
def apply_clamp_normalization(payload):
    fixed = []
    for img in payload['results']:
        t = torch.tensor(img, dtype=torch.float32)
        t = t.clamp(-1.0, 1.0)
        t = (t + 1.0) / 2.0
        fixed.append(t.numpy())
    payload = dict(payload)
    payload['results'] = fixed
    return payload


def load_pkl(path):
    """Load a single pkl by full path. Returns None if not found."""
    if not os.path.exists(path):
        print(f'[SKIP] Not found: {path}')
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    print(f'[OK]   Loaded {os.path.basename(path)}  ({len(payload["results"])} seeds)')
    return apply_clamp_normalization(payload)


def find_pkls(results_dir):
    """
    Find all .pkl files in results_dir (recursively) and return them as
    a list of (label, payload) pairs, skipping any that can't be loaded.
    """
    found = []
    for root, _, files in os.walk(results_dir):
        for fname in sorted(files):
            if fname.endswith('.pkl'):
                full = os.path.join(root, fname)
                payload = load_pkl(full)
                if payload is not None:
                    label = os.path.splitext(fname)[0]
                    found.append((label, payload))
    return found

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_images(payload, plots_dir, ncols=5, dpi=100,
                    save_no_title=False, classifier=None, device=None):
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    n     = len(results)
    nrows = math.ceil(n / ncols)
    preds, _ = (classify_generated_images(results, classifier, device, threshold=0.75)
                if classifier is not None else ([None]*n, [None]*n))

    for row in range(nrows):
        start, end  = row * ncols, min(row * ncols + ncols, n)
        row_results = results[start:end]
        row_losses  = loss_log[start:end]
        row_seeds   = seed_log[start:end]
        row_preds   = preds[start:end]
        n_in_row    = len(row_results)

        def _make(show_titles):
            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3),
                                     gridspec_kw=dict(wspace=0.02))
            axes = np.array(axes).reshape(ncols)
            for c, (img, loss, seed, pred) in enumerate(
                    zip(row_results, row_losses, row_seeds, row_preds)):
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

        base = os.path.join(plots_dir, f'{experiment_name}_all_row{row+1}.png')
        fig  = _make(show_titles=True)
        fig.savefig(base, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {base}')

        if save_no_title:
            base_nt = os.path.join(plots_dir, f'{experiment_name}_all_row{row+1}_notitle.png')
            fig_c   = _make(show_titles=False)
            fig_c.savefig(base_nt, dpi=dpi, bbox_inches='tight')
            plt.close(fig_c)


def plot_top_k_images(payload, plots_dir, top_k=5, dpi=100, save_no_title=False):
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
                    fontsize=7, pad=2)
            axes[r, c].axis('off')
        for rank in range(k, nrows * ncols):
            r, c = divmod(rank, ncols)
            axes[r, c].axis('off')
        if show_titles:
            plt.suptitle(f'[{experiment_name}] Top {k} Images',
                         fontsize=12, fontweight='bold', y=1.002)
        plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
        return fig

    base = os.path.join(plots_dir, f'{experiment_name}_top{k}.png')
    fig  = _make(show_titles=True)
    fig.savefig(base, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {base}')

    if save_no_title:
        base_nt = os.path.join(plots_dir, f'{experiment_name}_top{k}_notitle.png')
        fig_c   = _make(show_titles=False)
        fig_c.savefig(base_nt, dpi=dpi, bbox_inches='tight')
        plt.close(fig_c)

    return top_ix


def plot_top_k_distributions(payload, model_cond, top_ix, plots_dir,
                              top_k_dist=5, dpi=150, save_no_title=False, device='cpu'):
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    x_range_np      = payload['x_range']
    target_pdf_np   = payload['target_pdf']

    dist_k  = min(top_k_dist, len(top_ix))
    dist_ix = top_ix[:dist_k]

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
            bins=np.linspace(0, 2 * np.pi, 37), density=True)
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
            ax.set_xlim(0, 360); ax.set_ylim(0, y_lim)
            ax.set_xlabel('Angle (°)', fontsize=13)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.tick_params(axis='x', labelsize=12); ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, alpha=0.3)
            if show_titles:
                ax.set_title(f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                             fontsize=9)
            if rank == 0:
                ax.set_ylabel('Density', fontsize=12)
                if show_legend:
                    ax.legend(fontsize=8)
        if show_titles:
            fig.suptitle(f'[{experiment_name}] Top {dist_k} Distributions — Cartesian',
                         fontsize=13, fontweight='bold')
        fig.tight_layout()
        return fig

    def _make_polar(show_titles, show_legend):
        fig, polar_axes = plt.subplots(1, dist_k, figsize=(dist_k * 4, 4),
                                       subplot_kw={'projection': 'polar'})
        polar_axes = np.array(polar_axes).reshape(dist_k)
        for rank, (idx, ang) in enumerate(zip(dist_ix, temp_angs)):
            ang_rad   = np.deg2rad(ang.detach().cpu().numpy())
            bin_edges = np.linspace(0, 2 * np.pi, 37)
            counts, _ = np.histogram(ang_rad, bins=bin_edges, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width       = bin_edges[1] - bin_edges[0]
            ax = polar_axes[rank]
            ax.bar(bin_centers, counts, width=width, alpha=0.55,
                   color='skyblue', edgecolor='black', label='Sampled', zorder=2)
            ax.fill(target_rad_x_closed, target_rad_y_closed,
                    color='orange', alpha=0.35, zorder=3)
            ax.plot(target_rad_x_closed, target_rad_y_closed,
                    color='orange', linewidth=2.5, zorder=4, label='Target')
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
            ax.set_yticklabels([]); ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=12)
            ax.set_ylim(0, polar_ylim)
            if show_titles:
                ax.set_title(f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                             fontsize=9, pad=14)
            if rank == 0 and show_legend:
                ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.15))
        if show_titles:
            fig.suptitle(f'[{experiment_name}] Top {dist_k} Distributions — Polar',
                         fontsize=13, fontweight='bold')
        fig.tight_layout()
        return fig

    base_cart  = os.path.join(plots_dir, f'{experiment_name}_dist_cart.png')
    base_polar = os.path.join(plots_dir, f'{experiment_name}_dist_polar.png')
    _make_cart(True, True).savefig(base_cart,  dpi=dpi, bbox_inches='tight'); plt.close()
    _make_polar(True, True).savefig(base_polar, dpi=dpi, bbox_inches='tight'); plt.close()
    print(f'  Saved: {base_cart}')
    print(f'  Saved: {base_polar}')

    if save_no_title:
        _make_cart(False, False).savefig(
            base_cart.replace('.png', '_notitle.png'),  dpi=dpi, bbox_inches='tight'); plt.close()
        _make_polar(False, False).savefig(
            base_polar.replace('.png', '_notitle.png'), dpi=dpi, bbox_inches='tight'); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────
def compute_experiment_stats(payload, classifier, device, top_k=5, threshold=0.75):
    losses = np.array(payload['loss_log'])
    times  = np.array(payload.get('time_log', [np.nan] * len(losses)))
    n      = len(losses)
    top_ix = np.argsort(losses)[:top_k]

    preds, _ = classify_generated_images(payload['results'], classifier, device,
                                         threshold=threshold)
    digit_counts   = {d: 0 for d in range(10)}
    n_unclassified = 0
    for pr in preds:
        if pr is None:
            n_unclassified += 1
        else:
            digit_counts[pr] += 1

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


def render_results_table(stats, top_k=5):
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    results_dir = os.path.abspath(args.results_dir)
    ckpt_dir    = os.path.abspath(args.ckpt_dir) if args.ckpt_dir \
                  else os.path.join(MNIST_DIR, 'checkpoints_and_results')
    plots_dir   = os.path.abspath(args.plots_dir) if args.plots_dir \
                  else os.path.join(results_dir, 'plots')

    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    set_global_seed(GLOBAL_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\nDevice     : {device}')
    print(f'Results dir: {results_dir}')
    print(f'Plots dir  : {plots_dir}')
    print(f'Ckpt dir   : {ckpt_dir}\n')

    # ── Load all pkls found in results_dir ──────────────────────────────────
    experiments = find_pkls(results_dir)
    if not experiments:
        print(f'ERROR: No .pkl files found in {results_dir}')
        sys.exit(1)
    print(f'\nFound {len(experiments)} experiment(s): {[e[0] for e in experiments]}\n')

    # ── Classifier ──────────────────────────────────────────────────────────
    clf_path = os.path.join(ckpt_dir, 'robust_classifier.pth')
    # also check default checkpoints/ dir
    if not os.path.exists(clf_path):
        alt = os.path.join(MNIST_DIR, 'checkpoints', 'robust_classifier.pth')
        if os.path.exists(alt):
            clf_path = alt
    digit_classifier = load_or_train_classifier(
        save_path=clf_path, device=device, epochs=10, batch_size=128, lr=1e-3, seed=GLOBAL_SEED)

    # ── Conditional model ────────────────────────────────────────────────────
    cond_pt = os.path.join(ckpt_dir, 'MnistConditional500Epoch.pt')
    if not os.path.exists(cond_pt):
        alt = os.path.join(MNIST_DIR, 'checkpoints_and_results', 'MnistConditional500Epoch.pt')
        if os.path.exists(alt):
            cond_pt = alt
        else:
            print('Downloading conditional model from HuggingFace...')
            cond_pt = hf_hub_download(
                repo_id=HF_REPO_ID, filename='MnistConditional500Epoch.pt',
                token=hf_token or None)
    cond_model = CircularAngleConsistencyModel(
        nfeatures=2, img_features=784, eps=0.002, nunits=128, depth=5, device=device)
    ckpt = torch.load(cond_pt, map_location=device)
    cond_model.load_state_dict(ckpt['model_state_dict'])
    cond_model.eval()
    print(f'Conditional model loaded (epoch {ckpt["epoch"]}) ✓\n')

    # ── Per-experiment plots ─────────────────────────────────────────────────
    top_ix_map = {}
    for name, payload in experiments:
        print(f'\n{"="*55}\n  {name}\n{"="*55}')

        plot_all_images(payload, plots_dir, ncols=5, dpi=args.dpi,
                        save_no_title=args.no_titles,
                        classifier=digit_classifier, device=device)

        top_ix = plot_top_k_images(payload, plots_dir, top_k=args.top_k,
                                   dpi=args.dpi, save_no_title=args.no_titles)
        top_ix_map[name] = top_ix

        plot_top_k_distributions(payload, cond_model, top_ix, plots_dir,
                                 top_k_dist=args.top_k, dpi=args.dpi,
                                 save_no_title=args.no_titles, device=device)

    # ── Results table ────────────────────────────────────────────────────────
    print(f'\n{"="*55}\n  Quantitative Results\n{"="*55}')
    stats = {}
    for name, payload in experiments:
        print(f'Classifying {name}...')
        stats[name] = compute_experiment_stats(
            payload, digit_classifier, device, top_k=args.top_k)
        s = stats[name]
        print(f'  SWD all:     {s["swd_all_mean"]:.4f} ± {s["swd_all_std"]:.4f}')
        print(f'  SWD top-{args.top_k}: {s["swd_top_mean"]:.4f} ± {s["swd_top_std"]:.4f}')
        print(f'  Time:        {s["time_mean"]:.1f} ± {s["time_std"]:.1f} s')

    df = render_results_table(stats, top_k=args.top_k)
    table_path = os.path.join(plots_dir, 'results_table.csv')
    df.to_csv(table_path)
    print(f'\nTable saved → {table_path}')
    print(df.to_string())


if __name__ == '__main__':
    main()
