"""
witness_utils.py — MMD witness function and backprop-subsample selection for the
toy LGD guidance loop (Optimization.optimize_LGD).

Mirrors SD_cond_SD_controlnet/src/metrics.py's compute_witness_scores, reusing
this repo's existing multi-bandwidth RBF kernel (LossFunctions.RBF) instead of
duplicating a single-bandwidth Gaussian kernel.
"""

import torch

from LossFunctions import RBF


def witness_scenario_stats(scores):
    """Per-step heterogeneity diagnostics on witness scores (independent of any
    subsampling rule/floor): witness_std, witness_skew_proxy (max/mean of |w|,
    peakedness), ess_raw (effective sample size of the pure |w|-proportional
    distribution, 1=peaked to n=flat -- upper bound on the runtime ESS once
    mixed with witness_floor), and n=len(scores)."""
    with torch.no_grad():
        n = scores.shape[0]
        abs_scores = scores.abs().double()
        mean_abs = abs_scores.mean().clamp_min(1e-12)
        p = abs_scores / abs_scores.sum().clamp_min(1e-12)
        ess = 1.0 / (p ** 2).sum().clamp_min(1e-12)
        return {
            "witness_std": scores.std().item(),
            "witness_skew_proxy": (abs_scores.max() / mean_abs).item(),
            "ess_raw": ess.item(),
            "n": n,
        }


def compute_witness_scores(X, Y, kernel=None):
    """Per-sample MMD witness function w(x_l) = mean_i k(x_l,x_i) - mean_j k(x_l,y_j)
    -- large where X has too much mass relative to Y; |w| is the importance score
    for backsel selection. X, Y detached and moved to CPU internally regardless
    of input device."""
    kernel = kernel or RBF()
    with torch.no_grad():
        # CPU here matches MMDLoss/RBF's own default device and avoids a device
        # mismatch with select_backsel_mask's CPU-only torch.Generator downstream.
        X = X.detach().cpu()
        Y = Y.detach().cpu()
        n = X.shape[0]
        K = kernel(torch.vstack([X, Y]))
        K_xx = K[:n, :n]
        K_xy = K[:n, n:]
        scores = K_xx.mean(dim=1) - K_xy.mean(dim=1)
    return scores


def select_backsel_mask(scores, k, rule="uniform", witness_floor=0.3, witness_temperature=1.0,
                        generator=None, replacement=False):
    """Choose which of len(scores) rows to keep differentiable.

    rule='uniform': k uniformly-random rows. rule='witness': sample proportional
    to |scores|^(1/T) (T=witness_temperature; T=1 is plain |w|, T>1 flattens
    toward uniform, T<1 sharpens toward the top-|score| rows), blended with
    witness_floor toward uniform (defensive mixture p_i = floor/n +
    (1-floor)*|w_i|^(1/T)/sum|w|^(1/T)). replacement=False (default) draws k
    distinct rows; True allows repeats, tracked in `counts`.

    Returns (mask, counts, probs): mask[n] bool = kept at least once; counts[n]
    long = times drawn; probs[n] float = selection probabilities used.
    """
    n = scores.shape[0]
    k = max(0, min(int(k), n))

    if rule == "uniform":
        probs = torch.full((n,), 1.0 / n, dtype=torch.float64)
    elif rule == "witness":
        p = scores.abs().double()
        if witness_temperature != 1.0:
            p = p.clamp_min(1e-12) ** (1.0 / witness_temperature)
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
                  witness_floor=0.3, witness_temperature=1.0, generator=None, replacement=False):
    """Build the differentiable-subsample batch for the guidance loss: unselected
    rows are individually detached (zero gradient, still counted in the loss
    VALUE), selected rows keep gradient (duplicated if drawn >1x with
    replacement -- autograd sums across reuses). See select_backsel_mask for
    k/rule/witness_floor/witness_temperature/generator/replacement.

    Returns (batch, info): batch is samples' rows reassembled per selection
    (grows beyond n only with replacement duplicates); info has
    {"mask", "counts", "probs", "scores"} for logging.
    """
    n = samples.shape[0]
    scores = compute_witness_scores(samples, target_samples_for_scoring) \
        if rule == "witness" else torch.zeros(n)
    mask, counts, probs = select_backsel_mask(
        scores, k, rule=rule, witness_floor=witness_floor, witness_temperature=witness_temperature,
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
