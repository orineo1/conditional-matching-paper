"""[Agent P] Preconditioning of the rho-branch guidance gradient.

Opt-in via ``TemporalConfig.precond`` (:class:`tfg.config.PrecondConfig`);
``mode="none"`` (the default) makes :func:`make_preconditioner` return ``None``
and the engine's default path stays byte-identical to the frozen reference.

Design axiom (see ``experiments/model-optimization/precond/THEORY.md``): a
preconditioner may rotate or shrink the gradient but never systematically
inflate its norm -- fixed-magnitude outputs (Adam / norm_only / unit) are the
verified 10D failure. ``whiten``/``diag``/``sign`` are therefore exactly
norm-preserving; ``median`` outputs the coordinate-wise median of a short
window, whose norm is naturally tail-robust.

All modes are deterministic, causal (state built from PAST gradients only for
``whiten``/``diag``; the ``median`` window includes the current gradient, which
is an aggregation, not a look-ahead) and float32-safe: internal statistics are
kept in float64 and the output is cast back to the input dtype. State is
carried across steps like :class:`tfg.adam_guidance.AdamGuidance`; a fresh
engine gets a fresh preconditioner.
"""

import torch


class GuidancePreconditioner:
    """Applies ``PrecondConfig.mode`` to a stream of gradients.

    Parameters mirror :class:`tfg.config.PrecondConfig`. ``apply(grad)``
    returns the preconditioned gradient in ``grad``'s dtype and shape.
    """

    MODES = ("whiten", "diag", "sign", "median")

    def __init__(self, mode, ema=0.9, eps=1e-6, window=5, warmup=5):
        if mode not in self.MODES:
            raise ValueError(f"unknown precond mode {mode!r}")
        if not (0.0 <= float(ema) < 1.0):
            raise ValueError("precond.ema must lie in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("precond.eps must be positive")
        if int(window) < 1:
            raise ValueError("precond.window must be >= 1")
        if int(warmup) < 0:
            raise ValueError("precond.warmup must be >= 0")
        self.mode = mode
        self.ema = float(ema)
        self.eps = float(eps)
        self.window = int(window)
        self.warmup = int(warmup)
        self.reset()

    def reset(self):
        self._C = None        # whiten: EMA of g g^T (float64, d x d)
        self._v = None        # diag: EMA of g^2 (float64, flat d)
        self._seen = 0        # number of PAST gradients absorbed into state
        self._buf = []        # median: window of raw gradients (float64, flat d)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _restore_norm(direction, target_norm):
        nrm = direction.norm()
        if float(nrm) == 0.0:
            return direction
        return direction * (target_norm / nrm)

    # -- modes (float64 flat vectors in, float64 flat vectors out) ----------

    def _apply_whiten(self, g):
        out = g
        if self._seen >= self.warmup and self._C is not None:
            d = g.numel()
            ridge = self.eps * (torch.diagonal(self._C).sum() / d)
            C_reg = self._C + torch.clamp(ridge, min=self.eps * 1e-30) * torch.eye(
                d, dtype=torch.float64)
            evals, evecs = torch.linalg.eigh(C_reg)
            evals = torch.clamp(evals, min=1e-300)
            w = evecs @ ((evecs.T @ g) / torch.sqrt(evals))
            out = self._restore_norm(w, g.norm())
        # causal state update with the RAW gradient, after deciding this step
        gg = torch.outer(g, g)
        self._C = gg if self._C is None else self.ema * self._C + (1 - self.ema) * gg
        self._seen += 1
        return out

    def _apply_diag(self, g):
        out = g
        if self._seen >= self.warmup and self._v is not None:
            v_reg = self._v + self.eps * torch.clamp(self._v.mean(), min=1e-300)
            u = g / torch.sqrt(v_reg)
            out = self._restore_norm(u, g.norm())
        g2 = g * g
        self._v = g2 if self._v is None else self.ema * self._v + (1 - self.ema) * g2
        self._seen += 1
        return out

    def _apply_sign(self, g):
        s = torch.sign(g)
        nnz = float((s != 0).sum())
        if nnz == 0.0:
            return g
        return s * (g.norm() / nnz ** 0.5)

    def _apply_median(self, g):
        self._buf.append(g)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        stacked = torch.stack(self._buf)          # (w, d)
        return stacked.median(dim=0).values

    # -- public entry --------------------------------------------------------

    def apply(self, grad):
        """Precondition one gradient. Detached; never mutates ``grad``."""
        g64 = grad.detach().reshape(-1).to(torch.float64)
        if self.mode == "whiten":
            out = self._apply_whiten(g64)
        elif self.mode == "diag":
            out = self._apply_diag(g64)
        elif self.mode == "sign":
            out = self._apply_sign(g64)
        else:
            out = self._apply_median(g64)
        return out.to(grad.dtype).reshape(grad.shape)

    def state(self):
        return {"mode": self.mode, "seen": self._seen,
                "window_fill": len(self._buf),
                "C_trace": None if self._C is None else float(torch.diagonal(self._C).sum()),
                "v_mean": None if self._v is None else float(self._v.mean())}


def make_preconditioner(cfg):
    """Build a :class:`GuidancePreconditioner` from a ``PrecondConfig``.

    Returns ``None`` when ``cfg`` is ``None`` or ``cfg.mode == "none"`` so the
    engine's default path is exactly the reference path.
    """
    if cfg is None or getattr(cfg, "mode", "none") == "none":
        return None
    return GuidancePreconditioner(cfg.mode, ema=cfg.ema, eps=cfg.eps,
                                  window=cfg.window, warmup=cfg.warmup)
