"""Fig 10 — the headline 3-panel summary: random inits x three targets.

Pools round-7 runs at init offsets c in {-1.5, 0, +1.5} (x_T ~ c + N(0,1)) as a
broad random-initialization ensemble; 240 runs per panel.
Usage: python experiments/witness_unimodal/make_summary_figure.py
"""
import json, glob, pathlib, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CS = ('-1.5', '0.0', '1.5')

def load(arm):
    xs = []
    for c in CS:
        for f in glob.glob(str(HERE / f'results/initsweep_cap1_inv_cell_c{c}_{arm}.json')):
            xs += [r['x_hat'] for r in json.load(open(f))['rows']]
    return xs

panels = [('point_sq_a0', '1 — target = the mean (scalar 0)', '#0072B2',
           'lands in either basin — the target cannot say which'),
          ('mmd_bi', '2 — target = the bimodal distribution', '#D55E00',
           'lands at x_bi — the distribution says which'),
          ('mmd_uni', '3 — target = the unimodal distribution', '#E69F00',
           'lands at x_uni')]
fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), dpi=150, sharex=True)
for ax, (arm, title, col, note) in zip(axes, panels):
    xs = load(arm); n = len(xs)
    torch.manual_seed(0)
    y = torch.rand(n) * 0.62 + 0.05
    ax.scatter(xs, y.numpy(), s=12, alpha=0.55, color=col)
    for xv in (-2, 2):
        ax.axvline(xv, color='gray', ls='--', lw=1.2)
    bi = sum(1 for x in xs if x < 0)
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title(title, loc='left', fontsize=11)
    ax.text(0.01, 0.88, f"{bi/n:.0%} bimodal side / {1-bi/n:.0%} unimodal side  (n={n})",
            ha='left', transform=ax.transAxes, fontsize=9)
    ax.text(0.99, 0.88, note, ha='right', transform=ax.transAxes,
            fontsize=9, style='italic', color='dimgray')
axes[0].text(-1.93, 0.70, 'x_bi = -2', ha='left', color='gray',
             transform=axes[0].get_xaxis_transform(), fontsize=9)
axes[0].text(2.07, 0.70, 'x_uni = +2', ha='left', color='gray',
             transform=axes[0].get_xaxis_transform(), fontsize=9)
axes[-1].set_xlabel('landed design point x̂   (each dot = one run from a random initialization)')
fig.suptitle('Random initializations, three targets: only a distributional target determines the answer',
             fontsize=11)
fig.tight_layout()
fig.savefig(HERE / 'figures/fig10_summary.png')
print('written')
