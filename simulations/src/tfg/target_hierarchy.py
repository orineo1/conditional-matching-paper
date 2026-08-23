"""Deterministic nested target hierarchy G^(K) from a fixed sample set S_G.

    G^(K) = sum_{j=1..K} w_j^(K) delta_{c_j^(K)}

Construction, fully deterministic given S_G (no RNG at all):

  * K = 1: a single cluster containing every point; its centre is the target
    mean, its weight 1.
  * K -> K+1: split the cluster with the largest WEIGHTED within-cluster
    variance (w_j * var_j). In 1-D the split is at the cluster mean: points
    <= mean go left, points > mean go right. Ties in the selection are broken
    by lowest cluster index, so the hierarchy is unique.
  * A cluster with a single point (or zero variance) cannot be split and is
    skipped in favour of the next-largest.
  * At K = |S_G| every cluster holds one point, so G^(K) is exactly the
    empirical target with weights 1/|S_G|.

The hierarchy is NESTED: the assignment at K+1 refines the assignment at K.
That is what makes a curriculum meaningful -- increasing K adds resolution
rather than re-partitioning from scratch.
"""

import hashlib
import json

import torch


def _weighted_var(points, weight):
    if points.numel() <= 1:
        return 0.0
    return float(weight) * float(points.var(unbiased=False))


class TargetHierarchy:
    """Nested weighted prototypes of a fixed empirical target set."""

    def __init__(self, S_G):
        self.S_G = S_G.detach().reshape(-1, S_G.shape[-1]).double()
        self.N = self.S_G.shape[0]
        self.d = self.S_G.shape[1]
        if self.d != 1:
            raise NotImplementedError(
                "the deterministic split rule is specified for 1-D targets; "
                f"got d={self.d}"
            )
        # assignments[K] is a list of index-tensors, one per cluster
        self._levels = {1: [torch.arange(self.N)]}
        self._built_to = 1

    # -- construction ------------------------------------------------------

    def _split_once(self, clusters):
        """Split the highest weighted-variance splittable cluster."""
        best, best_score = None, -1.0
        for i, idx in enumerate(clusters):
            if idx.numel() < 2:
                continue
            pts = self.S_G[idx, 0]
            if float(pts.max() - pts.min()) == 0.0:
                continue                      # identical points: unsplittable
            score = _weighted_var(pts, idx.numel() / self.N)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return None
        idx = clusters[best]
        pts = self.S_G[idx, 0]
        mean = pts.mean()
        left = idx[pts <= mean]
        right = idx[pts > mean]
        if left.numel() == 0 or right.numel() == 0:
            # Degenerate (e.g. all points equal to the mean): fall back to a
            # deterministic median split by sorted order.
            order = idx[pts.argsort()]
            half = order.numel() // 2
            left, right = order[:half], order[half:]
        out = list(clusters)
        out[best] = left
        out.insert(best + 1, right)
        return out

    def _build_to(self, K):
        K = int(K)
        if K < 1 or K > self.N:
            raise ValueError(f"K must lie in [1, {self.N}], got {K}")
        while self._built_to < K:
            nxt = self._split_once(self._levels[self._built_to])
            if nxt is None:
                raise RuntimeError(
                    f"cannot refine beyond K={self._built_to}: every cluster is "
                    "a single point or has zero spread"
                )
            self._built_to += 1
            self._levels[self._built_to] = nxt

    # -- access ------------------------------------------------------------

    def level(self, K):
        """Return ``(centres (K,d), weights (K,), assignments list)``."""
        self._build_to(K)
        clusters = self._levels[int(K)]
        centres = torch.stack([self.S_G[idx].mean(0) for idx in clusters])
        weights = torch.tensor([idx.numel() / self.N for idx in clusters],
                               dtype=torch.float64)
        return centres, weights, clusters

    def mean(self):
        """E_{Y~G}[Y] -- the K=1 centre, used by the pointwise tier."""
        return self.S_G.mean(0)

    # -- provenance --------------------------------------------------------

    def descriptor(self, levels=(1, 2, 4, 8, 16, 32, 64, 128, 250)):
        """Hashable summary of the hierarchy, for the run record."""
        out = {"N": self.N, "d": self.d,
               "S_G_sha256": hashlib.sha256(
                   self.S_G.numpy().tobytes()).hexdigest(),
               "split_rule": "max weighted within-cluster variance; "
                             "1-D split at cluster mean; ties -> lowest index",
               "deterministic": True, "seed": None, "levels": {}}
        for K in levels:
            if K > self.N:
                continue
            c, w, _ = self.level(K)
            out["levels"][str(K)] = {
                "centres_sha256": hashlib.sha256(c.numpy().tobytes()).hexdigest(),
                "weights_sha256": hashlib.sha256(w.numpy().tobytes()).hexdigest(),
                "centre_min": float(c.min()), "centre_max": float(c.max()),
                "weight_min": float(w.min()), "weight_max": float(w.max()),
            }
        return out

    def save(self, path, **kw):
        with open(path, "w") as f:
            json.dump(self.descriptor(**kw), f, indent=2)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def pointwise_loss(y, target_mean):
    """K = n = 1 tier:  L1(x) = | h_phi(x, eta_1) - E_G[Y] |^2.

    The ordinary point-target TFG predictor. ``y`` is the single conditional
    sample, shape (1, d).
    """
    return ((y.reshape(-1) - target_mean.reshape(-1).to(y.dtype)) ** 2).sum()


def weighted_mmd2(y, centres, weights, bandwidth, n_kernels=5, mul_factor=2.0):
    """MMD^2 between n equally weighted samples and K weighted prototypes.

    Uses the repository's summed-RBF kernel family at a FIXED bandwidth:
        k(a,b) = sum_k exp( -||a-b||^2 / (bw * mul^(k - n_kernels//2)) )

    The empirical (V-statistic) convention of ``LossFunctions.MMDLoss`` is kept:
    diagonal terms are included in the self-blocks. That contributes a constant
    in x for the yy block and an O(1/n) constant for the xx block, so it does
    not bias the gradient direction, but it keeps the scale comparable to the
    repository's loss.
    """
    y = y.reshape(y.shape[0], -1)
    c = centres.reshape(centres.shape[0], -1).to(y.dtype)
    w = weights.reshape(-1).to(y.dtype)
    n = y.shape[0]
    a = torch.full((n,), 1.0 / n, dtype=y.dtype, device=y.device)

    ks = torch.arange(n_kernels, dtype=y.dtype, device=y.device) - (n_kernels // 2)
    mult = (mul_factor ** ks) * float(bandwidth)

    def K(u, v):
        d2 = torch.cdist(u, v, p=2) ** 2
        return torch.exp(-d2.unsqueeze(0) / mult[:, None, None]).sum(0)

    xx = a @ K(y, y) @ a
    xy = a @ K(y, c) @ w
    yy = w @ K(c, c) @ w
    return xx - 2.0 * xy + yy
