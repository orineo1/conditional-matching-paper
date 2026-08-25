"""
backsel.py — SD adapter for importance-selected backpropagation.

ALL selection / weighting logic lives in the shared ``tfg.backsel``
(simulations/src/tfg/backsel.py; theory in
experiments/model-optimization/backsel/THEORY.md).  This module only

  * maps the SD CLI names to the tfg rules
        uniform -> select_uniform            (Horvitz-Thompson n/k, unbiased)
        is      -> select_importance         (||g||-proportional + floor, unbiased)
        kcenter -> select_kcenter            (cluster-aggregated, biased; FAILED on SD,
                                              sd/BACKSEL_DIAG.md)
        strat   -> select_stratified_balanced(balanced strata, |C| g_r, unbiased)
    and ``weighting="soft"`` -> soft_tau (local | bandwidth) + soft_aggregate,
  * feeds the tfg functions' tape protocol from a ``torch.Generator``
    (``GeneratorTape``: the SD pipeline seeds selection per step from its
    seed; the synthetic engine uses a keyed NoiseTape),
  * returns the ``(idx, G_sel, info)`` triple that ``generation.run_dps_step_clip``
    consumes (indices are regenerated bit-exactly from their saved seeds and
    ``G_sel`` rows are paired with the regenerated embeddings in the surrogate
    ``sum_i <G_i, e_i>``; see PIPELINE.md sec 3).

The SD-specific pieces -- per-variation seeds, checkpointed regeneration, the
CLIP/VAE plumbing -- stay in ``generation.py``.
"""

import torch

import _tfg_path  # noqa: F401  (makes `tfg` importable)
from tfg.backsel import (balanced_assignment, kcenter_centers, select_importance,
                         select_kcenter, select_stratified_balanced, soft_aggregate,
                         soft_tau)
from tfg.backsel import select_uniform as tfg_select_uniform

IS_FLOOR = 0.25  # eps in the importance-sampling mixture (THEORY.md sec 2b)
RULES = {"uniform": "uniform", "is": "importance", "kcenter": "kcenter",
         "strat": "stratified_balanced"}


class GeneratorTape:
    """Minimal NoiseTape look-alike over a CPU ``torch.Generator``: the tfg
    selection rules ask ``tape.randn(key, shape, dtype=...)``; keys are ignored
    (the SD pipeline already seeds one generator per step)."""

    def __init__(self, generator=None):
        self.generator = generator

    def randn(self, key, shape, device=None, dtype=torch.float64):
        return torch.randn(tuple(shape), generator=self.generator, dtype=dtype)


def _identity(g):
    n = g.shape[0]
    return torch.arange(n), g.clone(), {"rule": "identity", "k": n, "n": n}


def _as_idx(idx):
    return torch.as_tensor(list(idx), dtype=torch.long)


def kcenter_greedy(E, k, start):
    """Greedy k-center on rows of E (= tfg.backsel.kcenter_centers)."""
    centers, assign = kcenter_centers(E.float(), k, start)
    return torch.tensor(centers), assign


def balanced_kcenter(E, k, start, generator=None):
    """k-center centers + capacity-constrained assignment (= tfg.backsel)."""
    centers, _ = kcenter_centers(E.float(), k, start)
    n = E.shape[0]
    return torch.tensor(centers), balanced_assignment(E.float(), centers, -(-n // k))


def select_uniform(g, k, generator=None):
    n = g.shape[0]
    if k >= n:
        return _identity(g)
    idx, G = tfg_select_uniform(g, k, GeneratorTape(generator), ())
    return _as_idx(idx), G, {"rule": "uniform", "k": k, "n": n, "weights": [n / k] * len(idx)}


def select_is(g, k, generator=None, eps=IS_FLOOR):
    n = g.shape[0]
    if k >= n:
        return _identity(g)
    idx, G = select_importance(g, k, GeneratorTape(generator), (), floor=eps)
    norms = g.detach().double().norm(dim=-1)
    s = float(norms.sum())
    p = ((1.0 - eps) * norms / s + eps / n) if s > 0 else torch.full((n,), 1.0 / n, dtype=torch.float64)
    p = p / p.sum()
    w = [float(G[j].double().norm() / g[i].double().norm()) if float(g[i].norm()) > 0 else float("nan")
         for j, i in enumerate(idx)]
    return _as_idx(idx), G, {"rule": "is", "k": k, "n": n, "n_unique": len(idx),
                             "weights": w, "p": p.tolist()}


def select_kcenter_sd(g, E, k, generator=None):
    n = g.shape[0]
    if k >= n:
        return _identity(g)
    idx, G = select_kcenter(E, g, k, GeneratorTape(generator), ())
    # cluster sizes for logging: nearest-center assignment to the chosen centers
    assign = torch.cdist(E.detach().double(), E.detach().double()[list(idx)]).argmin(dim=1)
    sizes = torch.bincount(assign, minlength=len(idx)).tolist()
    return _as_idx(idx), G, {"rule": "kcenter", "k": k, "n": n, "cluster_sizes": sizes}


def select_strat(g, E, k, generator=None):
    n = g.shape[0]
    if k >= n:
        return _identity(g)
    idx, G, sizes = select_stratified_balanced(E, g, k, GeneratorTape(generator), ())
    return _as_idx(idx), G, {"rule": "strat", "k": k, "n": n,
                             "cluster_sizes": sizes, "weights": sizes}


def soft_reweight(g, E, idx, tau_scale=1.0, tau_mode="local"):
    """Soft-assignment weighting (config: --backsel_weighting soft) =
    ``tfg.backsel.soft_tau`` + ``soft_aggregate``: every NON-selected sample j
    hands g_j to the selected representatives with a_ji = softmax_i(-||e_j-e_i||^2/tau);
    representative i carries G_i = g_i + sum_j a_ji g_j (mass conserved,
    Jacobian-substitution biased).  tau: 'local' (median squared distance of the
    skipped samples to their nearest representative) or 'bandwidth' (median
    heuristic on E), times tau_scale."""
    n = g.shape[0]
    idx = _as_idx(idx)
    if idx.numel() >= n:
        return g[idx].clone(), {"weighting": "soft", "tau": None}
    tau = soft_tau(E, idx.tolist(), mode=tau_mode, scale=tau_scale)
    G, mass = soft_aggregate(E, g, idx.tolist(), tau, return_mass=True)
    return G.to(torch.float32), {"weighting": "soft", "tau": tau, "tau_scale": float(tau_scale),
                                 "tau_mode": tau_mode, "mass": mass.tolist(),
                                 "mass_max": float(mass.max())}


def select_backprop_set(g, E, k, rule="uniform", generator=None,
                        weighting="ht", soft_tau_scale=1.0, soft_tau_mode="local"):
    """
    Args:
        g:    [N, d] output-space gradient dL/de_i (detached).
        E:    [N, d] embeddings (detached; used by kcenter / strat / soft).
        k:    number of variations to differentiate.
        rule: 'uniform' | 'is' | 'kcenter' | 'strat'  (see RULES for the tfg names).
        weighting: 'ht' (rule's own unbiased weights) | 'soft' (softmax proximity
                   reweighting on top of the selection, see soft_reweight).
        generator: CPU torch.Generator for the selection randomness.
    Returns:
        idx:   LongTensor [k'] (sorted, k' <= k) indices to regenerate with graphs.
        G_sel: [k', d] detached vectors to pair with e_idx in the surrogate.
        info:  dict for logging.
    """
    g = g.detach()
    if rule == "uniform":
        out = select_uniform(g, k, generator)
    elif rule == "is":
        out = select_is(g, k, generator)
    elif rule == "kcenter":
        out = select_kcenter_sd(g, E, k, generator)
    elif rule == "strat":
        out = select_strat(g, E, k, generator)
    else:
        raise ValueError(f"unknown backsel rule {rule!r} (known: {sorted(RULES)})")
    if weighting == "ht" or out[2].get("rule") == "identity":
        return out
    if weighting != "soft":
        raise ValueError(f"unknown backsel weighting {weighting!r}")
    idx, _, info = out
    G, winfo = soft_reweight(g, E, idx, soft_tau_scale, soft_tau_mode)
    info = dict(info); info.update(winfo); info["weights"] = winfo.get("mass")
    return idx, G, info
