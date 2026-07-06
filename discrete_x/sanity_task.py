"""Sanity task for discrete-x CDM: find a prompt x* whose induced conditional
P(Y | X=x*) matches a target G, where the prompt space is small enough to
brute-force the true optimum.

Components:
  - Prompt space: 5 slots (size, color, shape, texture, pattern), each with its
    own small vocabulary. A prompt is a length-5 sequence of slot-token indices.
  - Prior P(X): mixture of K=3 product-of-categoricals (slots conditionally
    independent given the mixture component). This gives context-dependent
    denoising posteriors that are exactly computable — no training, so the
    mode-vs-sampled comparison is not confounded by learned-model error.
  - Forward model f_phi: y = mean(token embeddings) + sigma_y * eps, so
    P(Y | X=x) = N(g(x), sigma_y^2 I). The target G is the empirical conditional
    of a known ground-truth prompt x_true.
  - Loss L(x) = biased MMD^2 (RBF kernel, median-heuristic bandwidth) between
    n_cond samples from f_phi(x, .) and the fixed target set S_G. Common random
    numbers: the same n_cond noise draws are reused for every L evaluation, so
    L is a deterministic (memoized) function of x.
"""

import itertools

import numpy as np

MASK = -1

SLOTS = [
    ("size", ["tiny", "small", "big", "huge", "giant"]),
    ("color", ["red", "crimson", "blue", "azure", "green", "golden"]),
    ("shape", ["circle", "square", "triangle", "star", "hexagon"]),
    ("texture", ["smooth", "rough", "striped", "dotted", "glossy"]),
    ("pattern", ["swirl", "grid", "waves", "sparks", "rings"]),
]


def mmd2_biased(X, Y, gamma, kyy_mean=None):
    """Biased (V-statistic) MMD^2 between sample sets X (n,d) and Y (m,d)."""
    def sqdist(A, B):
        return (
            np.sum(A**2, axis=1)[:, None]
            + np.sum(B**2, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )

    kxx = np.exp(-gamma * sqdist(X, X)).mean()
    kxy = np.exp(-gamma * sqdist(X, Y)).mean()
    if kyy_mean is None:
        kyy_mean = np.exp(-gamma * sqdist(Y, Y)).mean()
    return float(kxx + kyy_mean - 2.0 * kxy)


class SanityTask:
    def __init__(
        self,
        task_seed=0,
        n_components=3,
        emb_dim=8,
        sigma_y=0.5,
        n_cond=64,
        n_target=256,
        n_eval=1024,
        dirichlet_conc=0.3,
    ):
        rng = np.random.default_rng(task_seed)
        self.slot_names = [name for name, _ in SLOTS]
        self.slot_vocab = [vocab for _, vocab in SLOTS]
        self.vocab_sizes = [len(v) for v in self.slot_vocab]
        self.L = len(SLOTS)
        self.K = n_components
        self.emb_dim = emb_dim
        self.sigma_y = sigma_y

        # Mixture-of-products prior: pi (K,), theta[i] (K, V_i)
        self.pi = np.array([0.5, 0.3, 0.2])[: self.K]
        self.pi = self.pi / self.pi.sum()
        self.theta = [
            rng.dirichlet(dirichlet_conc * np.ones(V), size=self.K)
            for V in self.vocab_sizes
        ]

        # Token embeddings per slot: emb[i] (V_i, emb_dim)
        self.emb = [
            rng.normal(size=(V, emb_dim)) for V in self.vocab_sizes
        ]

        # Ground-truth prompt: per slot, the token of median prior-marginal
        # probability — neither the prior mode (too easy) nor the anti-mode.
        self.x_true = np.array(
            [
                int(np.argsort(self.pi @ self.theta[i])[V // 2])
                for i, V in enumerate(self.vocab_sizes)
            ]
        )

        # Target sample set S_G ~ P(Y | X=x_true), fixed once.
        rng_t = np.random.default_rng(task_seed + 777)
        self.S_G = self.g(self.x_true) + sigma_y * rng_t.normal(
            size=(n_target, emb_dim)
        )

        # Median-heuristic bandwidth from S_G pairwise squared distances.
        d2 = (
            np.sum(self.S_G**2, axis=1)[:, None]
            + np.sum(self.S_G**2, axis=1)[None, :]
            - 2.0 * (self.S_G @ self.S_G.T)
        )
        med = np.median(d2[np.triu_indices(n_target, k=1)])
        self.gamma = 1.0 / med
        kyy = np.exp(-self.gamma * d2)
        self._kyy_mean = float(kyy.mean())

        # Common random numbers for L (train) and a fresh set for eval.
        self._eta_cond = np.random.default_rng(task_seed + 888).normal(
            size=(n_cond, emb_dim)
        )
        self._eta_eval = np.random.default_rng(task_seed + 999).normal(
            size=(n_eval, emb_dim)
        )
        self._loss_cache = {}
        self.n_loss_evals = 0  # non-memoized evaluations

    def g(self, x):
        return np.mean(
            [self.emb[i][x[i]] for i in range(self.L)], axis=0
        )

    def loss(self, x):
        """Deterministic (CRN + memoized) MMD^2 for a fully decoded prompt."""
        key = tuple(int(v) for v in x)
        if key not in self._loss_cache:
            Y = self.g(x) + self.sigma_y * self._eta_cond
            self._loss_cache[key] = mmd2_biased(
                Y, self.S_G, self.gamma, self._kyy_mean
            )
            self.n_loss_evals += 1
        return self._loss_cache[key]

    def eval_mmd(self, x):
        """Held-out MMD^2 with fresh noise (not used during guidance)."""
        Y = self.g(x) + self.sigma_y * self._eta_eval
        return mmd2_biased(Y, self.S_G, self.gamma, self._kyy_mean)

    def emb_dist_to_true(self, x):
        """Exact conditional-mean distance ||g(x) - g(x_true)|| (Gaussians
        with equal covariance, so this is the true distributional gap)."""
        return float(np.linalg.norm(self.g(x) - self.g(self.x_true)))

    def decode(self, x):
        return " ".join(
            "[MASK]" if x[i] == MASK else self.slot_vocab[i][x[i]]
            for i in range(self.L)
        )

    def brute_force_optimum(self):
        """Exact argmin_x L(x) over the full prompt space."""
        best_x, best_l = None, np.inf
        for x in itertools.product(*[range(V) for V in self.vocab_sizes]):
            l = self.loss(np.array(x))
            if l < best_l:
                best_x, best_l = np.array(x), l
        return best_x, best_l
