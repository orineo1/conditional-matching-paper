"""
witness_utils.py — MMD witness function and backprop-subsample selection for the
toy LGD guidance loop (Optimization.optimize_LGD).

Mirrors SD_cond_SD_controlnet/src/metrics.py's compute_witness_scores, reusing
this repo's existing multi-bandwidth RBF kernel (LossFunctions.RBF) instead of
duplicating a single-bandwidth Gaussian kernel.
"""

import torch

from LossFunctions import RBF


def compute_witness_scores(X, Y, kernel=None):
    """
    Per-sample MMD witness function w(x_l) = mean_i k(x_l, x_i) - mean_j k(x_l, y_j).

    Positive and large where X has "too much mass" relative to Y -- the samples
    whose removal/change would most reduce the MMD. |w| is the natural importance
    score for choosing which of X to backprop through: cheap (no_grad, row-means
    of a kernel matrix already computed the same way MMDLoss computes it), and
    doesn't require gradients through whatever produced X.

    Args:
        X:      [n, d] candidate samples (detached internally regardless).
        Y:      [m, d] target samples (detached internally regardless).
        kernel: callable kernel(Z) -> [n+m, n+m] Gram matrix (e.g. LossFunctions.RBF()).
                Defaults to a fresh RBF() to match MMDLoss's default kernel.

    Returns:
        [n] tensor of w(x_l), one score per row of X.
    """
    kernel = kernel or RBF()
    with torch.no_grad():
        X = X.detach()
        Y = Y.detach()
        n = X.shape[0]
        K = kernel(torch.vstack([X, Y]))
        K_xx = K[:n, :n]
        K_xy = K[:n, n:]
        scores = K_xx.mean(dim=1) - K_xy.mean(dim=1)
    return scores


def select_backsel_mask(scores, k, rule="uniform", witness_floor=0.3, generator=None,
                        replacement=False):
    """
    Choose which of len(scores) rows to keep differentiable (True) vs. detach (False).

    Args:
        scores:        [n] witness scores (signed; only used when rule='witness').
        k:             how many rows to keep differentiable (clamped to [0, n]).
        rule:          'uniform' -- k uniformly-random rows (or k random draws with
                       replacement, each row's count reflected in `counts`).
                       'witness' -- sample proportional to |scores|, blended with a
                       witness_floor fraction of uniform (the defensive mixture
                       p_i = floor/n + (1-floor)*|w_i|/sum(|w|); see chat/SD
                       generation.py for the same construction).
        witness_floor: uniform-mixing floor for rule='witness' (0 = pure importance
                       sampling, 1 = uniform). Ignored for rule='uniform'.
        generator:     optional torch.Generator (CPU) for reproducible sampling.
        replacement:   False (default) -- k distinct rows, removes the "same
                       sample picked repeatedly" failure mode entirely (recommended
                       unless k is tiny relative to n). True -- draw with
                       replacement; a row drawn c times gets counts[row] = c.

    Returns:
        (mask, counts, probs)
        mask:   [n] bool tensor, True = kept differentiable at least once.
        counts: [n] long tensor, how many times each row was drawn (0 or 1 when
                replacement=False; can exceed 1 when replacement=True).
        probs:  [n] float tensor, the selection probabilities actually used
                (uniform 1/n for rule='uniform').
    """
    n = scores.shape[0]
    k = max(0, min(int(k), n))

    if rule == "uniform":
        probs = torch.full((n,), 1.0 / n, dtype=torch.float64)
    elif rule == "witness":
        p = scores.abs().double()
        p = (1.0 - witness_floor) * p / p.sum().clamp_min(1e-12) + witness_floor / n
        probs = p / p.sum()
    else:
        raise ValueError(f"unknown backsel_rule {rule!r} (known: 'uniform', 'witness')")

    if k == 0:
        counts = torch.zeros(n, dtype=torch.long)
    elif k >= n and not replacement:
        counts = torch.ones(n, dtype=torch.long)
    else:
        idx = torch.multinomial(probs, k, replacement=replacement, generator=generator)
        counts = torch.bincount(idx, minlength=n)

    mask = counts > 0
    return mask, counts, probs.float()


def apply_backsel(samples, target_samples_for_scoring, k, rule="uniform",
                  witness_floor=0.3, generator=None, replacement=False):
    """
    Build the differentiable-subsample batch used for the guidance loss: rows not
    selected are individually .detach()'d (their gradient is exactly zero, but the
    full n-row batch is still used for the loss VALUE / MMD statistic), rows
    selected keep their gradient (duplicated `counts[row]` times if replacement=True
    and that row was drawn more than once -- autograd correctly sums a tensor's
    gradient across however many times it's reused in the graph).

    Args:
        samples:                     [n, d] differentiable candidate samples.
        target_samples_for_scoring:  [m, d] samples to score `samples` against
                                     (e.g. the MoG samples used in the MMD loss).
                                     Only used when rule='witness'.
        k, rule, witness_floor, generator, replacement: see select_backsel_mask.

    Returns:
        (batch, info) where batch is [n or more, d] (grows beyond n only when
        replacement=True produces duplicates) and info is a dict with
        {"mask", "counts", "probs", "scores"} for logging/diagnostics.
    """
    n = samples.shape[0]
    scores = compute_witness_scores(samples, target_samples_for_scoring) \
        if rule == "witness" else torch.zeros(n)
    mask, counts, probs = select_backsel_mask(
        scores, k, rule=rule, witness_floor=witness_floor,
        generator=generator, replacement=replacement,
    )

    rows = []
    for i in range(n):
        c = int(counts[i])
        if c > 0:
            rows.extend([samples[i:i + 1]] * c)
        else:
            rows.append(samples[i:i + 1].detach())
    batch = torch.cat(rows, dim=0)

    info = {"mask": mask, "counts": counts, "probs": probs, "scores": scores}
    return batch, info
