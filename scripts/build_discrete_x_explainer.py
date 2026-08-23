"""Two-page PDF explainer of the discrete-x CDM extension (for Ori).

Page 1: problem statement (from the continuous case he knows), what breaks
        with discrete x, the twisted-SMC method, pipeline schematic.
Page 2: validation ladder with the toy figure, image-scale results figure,
        findings and next steps.

Run:  conda run -n grassy_dit python scripts/build_discrete_x_explainer.py
Out:  output/discrete_x_explainer.pdf
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

A4 = (8.27, 11.69)
ASSETS = "output/discrete_x_assets"
TOY = "output/discrete_x_sanity_anneal"
OUT = "output/discrete_x_explainer.pdf"

BODY_FS = 9.6
HEAD_FS = 11.5


def heading(fig, y, text):
    fig.text(0.07, y, text, fontsize=HEAD_FS, fontweight="bold",
             va="top", ha="left")
    return y - 0.022


def body(fig, y, text, fs=BODY_FS, dy_per_line=0.0163):
    fig.text(0.07, y, text, fontsize=fs, va="top", ha="left",
             linespacing=1.45, wrap=False)
    return y - dy_per_line * (text.count("\n") + 1) - 0.012


def box(ax, x, y, w, h, text, fc="#eef3fa", fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        fc=fc, ec="#4a6fa5", lw=1.0, mutation_aspect=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", color="#4a6fa5"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
        color=color, lw=1.2))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.045, text,
                ha="center", va="bottom", fontsize=7.2, color="#333333")


def page1(pdf):
    fig = plt.figure(figsize=A4)

    fig.text(0.07, 0.955,
             "CDM with Discrete Inputs: Optimizing Prompts by Guided "
             "Discrete Diffusion",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.07, 0.932,
             "Status note for Ori — what I've been building on top of "
             "MLGD-F, July 2026 (Shaul)",
             fontsize=9.5, style="italic", va="top", color="#444444")

    y = 0.905
    y = heading(fig, y, "1.  Same problem, new input space")
    y = body(fig, y,
        "In the paper we solve CDM for continuous $x$: find a scribble $x^*$ whose induced conditional\n"
        "$\\mathcal{P}(Y\\,|\\,X{=}x^*)$ matches a target $\\mathcal{G}(Y)$, by running reverse diffusion over the scribble and correcting\n"
        "each step with $\\zeta_t\\,\\nabla_x\\,\\mathcal{L}(\\hat{x}_0)$, where "
        "$\\mathcal{L} = \\mathrm{MMD}^2$ between images sampled from the fast conditional\n"
        "sampler $f_\\phi(x,\\cdot)$ and $\\mathcal{G}$.  The extension: $x$ is now a PROMPT — a sequence of discrete tokens.\n"
        "The inner loop is unchanged (SDXL-Turbo is now literally a text-to-image $f_\\phi$; CLIP-space MMD as before).\n"
        "We want the prompt whose induced image distribution matches $\\mathcal{G}$ — e.g. a target with several modes\n"
        "that no obvious prompt generates. Everything still runs at inference time, nothing is trained.")

    y = heading(fig, y, "2.  What breaks: no gradient through tokens")
    y = body(fig, y,
        "Both MLGD components need a replacement on the outer side:\n"
        "$\\bullet$  Outer prior. The score network over scribbles becomes MASKED (absorbing) DISCRETE DIFFUSION over\n"
        "    token sequences: the forward process replaces tokens with [MASK]; the reverse process unmasks a few\n"
        "    positions per step, filling each from the denoiser posterior $p_\\theta(x_0^i\\,|\\,x_t)$. Key convenience: a plain\n"
        "    masked LM (we use BERT) IS this denoiser — MLM training is exactly the denoising objective (MDLM).\n"
        "$\\bullet$  Guidance. $\\nabla_x$ is meaningless on tokens, and we deliberately use NO relaxations (no Gumbel-softmax,\n"
        "    no straight-through). Instead the loss enters ONLY as a scalar, via Feynman--Kac reweighting — the\n"
        "    'twisted SMC corrector' from our Future Work paragraph, promoted to the main mechanism.")

    y = heading(fig, y, "3.  The method in four lines")
    y = body(fig, y,
        "Run $N$ particles (partially masked prompts) through the reverse process. After each step, reweight:\n"
        "        $w_n \\;\\propto\\; h_{t-1}(x_{t-1}^n)\\,/\\,h_t(x_t^n)$,     "
        "$h_t(x_t) \\,=\\, \\mathbb{E}[e^{-\\beta_t\\,\\mathcal{L}(x_0)}\\,|\\,x_t]$   (plug-in estimate),\n"
        "and resample (systematic) whenever ESS $< N/2$. The final particle population targets "
        "$\\mathcal{P}(x_0)\\,e^{-\\beta\\,\\mathcal{L}(x_0)}$ —\n"
        "the same tempered posterior $\\mathcal{Q}_\\beta$ as Problem 1, with $\\beta\\to$ large giving the CDMO optimum.\n"
        "Since the discrete $\\hat{x}_0$ is a distribution over sequences (not a point like Tweedie), we tested two twist\n"
        "estimators:  MODE = per-position argmax decode, one $\\mathcal{L}$ eval;   SAMPLED = $n_{dec}$ decodes,\n"
        "$h_t = \\frac{1}{n_{dec}}\\sum_j e^{-\\beta \\mathcal{L}(x_0^{(j)})}$ (unbiased for the exact twist). "
        "Two tricks matter in practice:\n"
        "$\\bullet$  $\\beta$-ANNEALING $\\beta_s = \\beta\\,(T{-}s)/T$: weak guidance early (twist estimates are noise when nearly\n"
        "    everything is masked), full strength at the end. Final target unchanged.\n"
        "$\\bullet$  SDEDIT ANALOGUE: instead of starting all-masked, start every particle from a SOURCE prompt with\n"
        "    each token re-masked w.p. $t_0/T$ (exact forward corruption), and run the reverse chain from $t_0$.\n"
        "    This is precisely the paper's scribble-editing setup, transplanted to text.")

    # pipeline schematic
    ax = fig.add_axes([0.05, 0.055, 0.90, 0.26])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("One outer step of the guided reverse chain "
                 "(everything gradient-free)",
                 fontsize=9, pad=4)

    box(ax, 0.1, 1.7, 1.6, 0.95,
        "source prompt\n're-masked' state\n$x_t$ ($N$ particles)")
    box(ax, 2.15, 1.7, 1.6, 0.95,
        "BERT denoiser\nunmask $\\sim 1/t$ of\npositions")
    box(ax, 4.2, 1.7, 1.6, 0.95,
        "decode $\\hat{x}_0$\n(mode / sampled)")
    box(ax, 6.25, 1.7, 1.6, 0.95,
        "SDXL-Turbo\n$n_{cond}$ images\n(fixed latents)")
    box(ax, 8.3, 1.7, 1.6, 0.95,
        "CLIP + MMD$^2$\nvs. target set\n$S_\\mathcal{G}$")
    arrow(ax, 1.7, 2.17, 2.15, 2.17)
    arrow(ax, 3.75, 2.17, 4.2, 2.17)
    arrow(ax, 5.8, 2.17, 6.25, 2.17)
    arrow(ax, 7.85, 2.17, 8.3, 2.17)

    box(ax, 4.2, 0.25, 3.7, 0.8,
        "reweight  $w \\propto h_{t-1}/h_t$,   $h = e^{-\\beta_t \\mathcal{L}}$"
        "  ;   resample if ESS $< N/2$", fc="#faf0e6")
    arrow(ax, 9.1, 1.7, 7.9, 1.05)
    arrow(ax, 4.2, 0.65, 0.9, 1.7, text="$x_{t-1}$, go to next step")

    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=A4)

    y = 0.955
    y = heading(fig, y, "4.  Validation ladder: toy world  →  real LM  →  "
                        "real images")
    y = body(fig, y,
        "I validated bottom-up, isolating one failure mode per rung.  "
        "(All numbers: biased MMD$^2$; lower is better.)\n"
        "$\\bullet$  TOY (5-slot grammar, 3,750 prompts, EXACT denoiser from a mixture prior, brute-forceable optimum):\n"
        "    with $\\beta$-annealing the SAMPLED estimator recovers the generating prompt 5/5 seeds; MODE only 1/5 —\n"
        "    its deterministic argmax twist ignores denoiser uncertainty and all seeds collapse to the same wrong\n"
        "    prompt. Left figure: best loss per step vs. the brute-force optimum (dotted).\n"
        "$\\bullet$  REAL LM, from scratch (BERT over $\\sim$20k tokens, CLIP-text loss, then SDXL image loss): guidance pulls\n"
        "    in the right semantic direction but never finds rare target tokens ('rain'): with a bootstrap proposal,\n"
        "    SMC can only SELECT among what the prior proposes — proposal coverage is the binding constraint,\n"
        "    not the twist. From-scratch image run: best $\\mathcal{L}=0.50$ vs. optimum $0.07$ (middle row, right figure).\n"
        "$\\bullet$  REAL LM, SDEDIT EDITING (the paper-consistent setup): source 'a photo of a man walking down the\n"
        "    street', re-mask 75%, target $\\mathcal{G}$ = images of 'a photo of a person standing in the rain'.  Result:\n"
        "    $\\mathcal{L}$: 0.80 (source) $\\to$ 0.39, recovered prompt 'a photo of a man in the rain and' — the guidance\n"
        "    inserted the rain semantics using nothing but scalar MMD reweighting (bottom row, right figure).")

    # left: toy loss trajectories; right: results table-ish
    ax1 = fig.add_axes([0.07, 0.50, 0.42, 0.185])
    img = Image.open(os.path.join(TOY, "loss_trajectories.png"))
    ax1.imshow(np.asarray(img))
    ax1.axis("off")
    ax1.set_title("Toy world, $\\beta$-annealed: sampled (orange) reaches "
                  "the exact optimum", fontsize=8)

    ax2 = fig.add_axes([0.53, 0.50, 0.44, 0.185])
    ax2.axis("off")
    rows = [
        ["stage", "estimator", "result"],
        ["toy, from scratch", "sampled", "100% exact (5/5 seeds)"],
        ["toy, from scratch", "mode", "20% (systematic bias)"],
        ["image, from scratch", "both", "fail: 0.5 vs opt 0.07"],
        ["image, SDEdit 75%", "mode", "0.80 → 0.39, 'in the rain'"],
        ["image, SDEdit 75%", "sampled", "0.80 → 0.48, off-target"],
    ]
    tbl = ax2.table(cellText=rows, loc="center", cellLoc="left",
                    colWidths=[0.33, 0.20, 0.47])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.4)
    tbl.scale(1.0, 1.5)
    for j in range(3):
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_facecolor("#eef3fa")
    ax2.set_title("Summary of runs (seed 0 at image scale)", fontsize=8)

    # image rows: target / from-scratch / sdedit
    row_files = [
        ("target_rain_row.png",
         "Target $\\mathcal{G}$:  64 SDXL-Turbo images of 'a photo of a person "
         "standing in the rain' (first 8 shown), $\\mathcal{L}(x_{true})=0.07$"),
        ("best_mode_seed0.png",
         "From scratch (fail):  'a photo of a guy in big trouble or', "
         "$\\mathcal{L}=0.54$ — style matched, content lost"),
        ("sdedit_best_mode_seed0.png",
         "SDEdit editing (works):  'a photo of a man in the rain and', "
         "$\\mathcal{L}=0.39$ from source 0.80"),
    ]
    # crop first row of the 8x8 target grid once
    tgt = Image.open(os.path.join(ASSETS, "target_rain.png"))
    w, h = tgt.size
    tgt.crop((0, 0, w, h // 8)).save(
        os.path.join(ASSETS, "target_rain_row.png"))

    y0 = 0.39
    for fname, caption in row_files:
        ax = fig.add_axes([0.07, y0, 0.88, 0.078])
        ax.imshow(np.asarray(Image.open(os.path.join(ASSETS, fname))))
        ax.axis("off")
        fig.text(0.07, y0 - 0.006, caption, fontsize=8, va="top")
        y0 -= 0.108

    y = 0.145
    y = heading(fig, y, "5.  What we learned, and what's next")
    y = body(fig, y,
        "Selection-based guidance WORKS and needs no gradients — but it is only as good as its proposals: editing\n"
        "(re-masking a source prompt) beats generation from scratch for exactly the reason SDEdit does in the paper.\n"
        "The estimator ranking FLIPPED at image scale (mode $>$ sampled; toy said the opposite) — greedy decodes are\n"
        "coherent sentences, factorized samples are not; needs a multi-seed check before we believe it.  Next: (i) seeds;\n"
        "(ii) MIXTURE targets $\\mathcal{G}$ with no single generating prompt — the real CDM story; (iii) locally-twisted\n"
        "proposals (score top-$k$ candidate tokens by $\\mathcal{L}$ at unmask time) to attack the coverage bottleneck.",
        fs=8.8, dy_per_line=0.0135)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        page1(pdf)
        page2(pdf)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
