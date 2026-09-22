"""
witness_utils.py — MMD witness function and backprop-subsample selection for the
toy LGD guidance loop (Optimization.optimize_LGD).

Mirrors SD_cond_SD_controlnet/src/metrics.py's compute_witness_scores exactly,
via the same rbf_kernel/estimate_bandwidth functions LossFunctions.compute_mmd
uses (single Gaussian kernel, detached median-heuristic bandwidth).
"""

import torch

from LossFunctions import rbf_kernel, estimate_bandwidth


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


def compute_witness_scores(X, Y, bandwidth=None):
    """Per-sample MMD witness function w(x_l) = mean_i k(x_l,x_i) - mean_j k(x_l,y_j)
    -- large where X has too much mass relative to Y; |w| is the importance score
    for backsel selection. X, Y detached and moved to CPU internally regardless
    of input device."""
    with torch.no_grad():
        # CPU here avoids a device mismatch with select_backsel_mask's
        # CPU-only torch.Generator downstream.
        X = X.detach().cpu()
        Y = Y.detach().cpu()
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, Y)
        K_xx = rbf_kernel(X, X, bandwidth)
        K_xy = rbf_kernel(X, Y, bandwidth)
        scores = K_xx.mean(dim=1) - K_xy.mean(dim=1)
    return scores


def select_backsel_mask(scores, k, rule="uniform", witness_floor=0.3, generator=None,
                        replacement=False):
    """Choose which of len(scores) rows to keep differentiable.

    rule='uniform': k uniformly-random rows. rule='witness': sample proportional
    to |scores|, blended with witness_floor toward uniform (defensive mixture
    p_i = floor/n + (1-floor)*|w_i|/sum|w|). replacement=False (default) draws k
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


class _ScaleGrad(torch.autograd.Function):
    """Identity in the forward pass; multiplies the backward-pass gradient
    through this tensor by a per-row `scale`. Used by apply_backsel_ht to make
    each selected row's gradient contribution exactly Horvitz-Thompson/
    importance-sampling-corrected without perturbing the forward loss value."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (scale,) = ctx.saved_tensors
        return grad_out * scale.view(-1, *([1] * (grad_out.dim() - 1))), None


def scale_grad(x, scale):
    return _ScaleGrad.apply(x, scale.to(x.dtype).to(x.device))


def apply_backsel_ht(samples, target_samples_for_scoring, k, rule="uniform",
                     witness_floor=0.3, generator=None):
    """Like apply_backsel, but exactly Horvitz-Thompson/importance-sampling
    rescaled, for BOTH rules -- apply_backsel's plain detach-or-keep leaves the
    selected rows' gradient unrescaled, which is only made comparable across
    k/n choices by a flat post-hoc `normalize_by_k_frac` factor (see
    Optimization._compute_step_gradients); that flat factor is exactly correct
    for 'uniform' (SRSWOR, same weight n/k on every selected row -- a global
    constant commutes with autograd summation, so post-hoc == per-row here) but
    WRONG for 'witness' (each selected row needs its OWN weight, which a single
    flat multiplier can't reproduce whenever p_i isn't uniform). This rescales
    each selected row's gradient in place (forward value untouched, via
    scale_grad) so d(loss)/dx is unbiased for the true full-n gradient (summing
    every row's chain-rule contribution with all n rows present at their real
    values -- the "what if every row were differentiable" gradient) under
    EITHER rule -- the correction the downstream Witness-vs-Uniform comparison
    needs actually wired into `zeta * grad`, not just diagnosed after the fact.

    uniform: k of n rows, SRSWOR (no replacement). Marginal inclusion probability
             is k/n for every row, so each selected row is scaled by n/k
             (E[row scaled] = (k/n)*(n/k) = 1, matching an always-included row).
    witness: k of n rows, drawn WITH replacement proportional to the
             witness_floor-mixed |scores| distribution (select_backsel_mask).
             Row i's EXPECTED draw count is k*p_i, so scaling each of its
             counts[i] actual draws by 1/(k*p_i) gives E[counts[i]/(k*p_i)] =
             (k*p_i)/(k*p_i) = 1 -- NOT 1/(n*p_i) (that denominator belongs to a
             different estimator, the self-normalized-by-n sample MEAN over k
             draws, and applied here would shrink the gradient by a spurious
             k/n factor). Row i's total scale is counts[i]/(k*p_i); autograd
             sums its (possibly >1) draws' contributions automatically.

    Do not also pass normalize_by_k_frac=True alongside this -- see
    Optimization.optimize_LGD's backsel_ht_rescale docstring.

    Returns (batch, info) like apply_backsel; info adds "row_weight" (the
    per-row rescale actually applied).
    """
    n = samples.shape[0]
    k = max(0, min(int(k), n))
    replacement = (rule == "witness")
    scores = compute_witness_scores(samples, target_samples_for_scoring) \
        if rule == "witness" else torch.zeros(n)
    mask, counts, probs = select_backsel_mask(
        scores, k, rule=rule, witness_floor=witness_floor,
        generator=generator, replacement=replacement,
    )
    if rule == "uniform":
        row_weight = torch.full((n,), n / k, dtype=torch.float64) if k > 0 else torch.zeros(n, dtype=torch.float64)
    elif rule == "witness":
        probs = probs.double().clamp_min(1e-12)
        row_weight = counts.double() / (k * probs)
    else:
        raise ValueError(f"unknown backsel_rule {rule!r} (known: 'uniform', 'witness')")

    scale = torch.where(mask, row_weight, torch.zeros_like(row_weight))
    batch = scale_grad(samples, scale)
    info = {"mask": mask, "counts": counts, "probs": probs, "scores": scores, "row_weight": row_weight}
    return batch, info


def apply_backsel(samples, target_samples_for_scoring, k, rule="uniform",
                  witness_floor=0.3, generator=None, replacement=False):
    """Build the differentiable-subsample batch for the guidance loss: unselected
    rows are individually detached (zero gradient, still counted in the loss
    VALUE), selected rows keep gradient (duplicated if drawn >1x with
    replacement -- autograd sums across reuses). See select_backsel_mask for
    k/rule/witness_floor/generator/replacement.

    Returns (batch, info): batch is samples' rows reassembled per selection
    (grows beyond n only with replacement duplicates); info has
    {"mask", "counts", "probs", "scores"} for logging.
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
