"""Alternative runners for ``simulations/experiments/_guided.py::run`` (Agent 5).

Every runner here is meant to be *numerically equivalent* to ``_guided.run``
(same inputs -> same trajectory up to float32 round-off).  They never modify
``simulations/``; they re-implement the loop at the call site so that

  * the per-perturbation RNG draws of ``ConsistencyModeliCT.sample`` are
    reproduced EXACTLY (``torch.Generator`` seeded with the same
    ``key_seed('cond', restart, t, j)``; same draw order:
    ``randn_like(x0)`` [LGD only], ``randn(n, d_y)``, then 5 x ``randn(n, d_y)``),
  * the M=3 LGD perturbations can be pushed through the conditional model as
    one batch of 3n rows (``batched_lgd``),
  * the M MMDs can be computed with one batched kernel call (``batched_mmd``),
  * B independent restarts can be pushed through the denoiser and the
    conditional model as one batch (``run_batched_restarts``),
  * the models can be put in ``requires_grad_(False)`` / compiled / float64 /
    moved to MPS without touching the repo.

Reference trajectory capture: ``capture_trajectory`` wraps
``model_uncond.sample_ddim_step`` to record its input, giving x_t for
t = T-1..1 without touching ``_guided``.
"""
import contextlib
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _setup import PAPER_TS, key_seed  # noqa: E402

from LossFunctions import MMDLoss, RBF          # noqa: E402
from tfg.adam_guidance import AdamGuidance      # noqa: E402
from tfg.n_schedule import n_at                 # noqa: E402
from tfg.schedule import DiffusionSchedule      # noqa: E402

M_LGD = 3


# ----------------------------------------------------------------------------
# helpers that mirror _guided.py exactly
# ----------------------------------------------------------------------------
def n_for_step(t, sched, n_max, schedule):
    if schedule == "constant":
        return int(n_max)
    return int(n_at(t, sched, n_max, 1.0, schedule))


def n_perturb(spatial):
    if spatial == "no_lgd":
        return 1
    if spatial == "lgd":
        return M_LGD
    raise ValueError(spatial)


def draw_cond_noise(gen, restart, t, j, n, d_x, d_y, lgd, ts=PAPER_TS, dtype=torch.float32):
    """Reproduce the RNG consumption of one perturbation in _guided.run:
    torch.manual_seed(key_seed('cond', restart, t, j)); [randn_like(x0) if M>1];
    ConsistencyModeliCT.sample -> randn(n, d_y) then randn_like(x) per ladder step."""
    gen.manual_seed(key_seed("cond", restart, t, j))
    eps_x0 = torch.randn(1, d_x, generator=gen, dtype=dtype) if lgd else None
    z0 = torch.randn(n, d_y, generator=gen, dtype=dtype)
    zs = [torch.randn(n, d_y, generator=gen, dtype=dtype) for _ in ts[1:]]
    return eps_x0, z0, zs


def cm_sample_with_noise(model, z0, zs, cond, ts=PAPER_TS):
    """ConsistencyModeliCT.sample (ConsistencyModels.py:249-262) with the noise
    supplied instead of drawn.  Row-wise identical to the original."""
    x = z0 * ts[0]
    for z, t in zip(zs, ts[1:]):
        x = x + math.sqrt(t ** 2 - model.eps ** 2) * z
        x = model(x, t, cond=cond)
    return x


class BatchedMMD:
    """MMD^2 of S sets X_s (S, n, d) against one fixed Y (m, d) with the
    repo's RBF (5 bandwidths, mul_factor 2, biased V-statistic), one cdist call
    per block.  YY is a constant of the run (no gradient) and is cached.

    ``compute_mode='use_mm_for_euclid_dist'`` forces the same matmul-based
    distance path that torch.cdist picks for the repo's 258-row stacked
    matrix, so individual kernel entries match the reference bit-for-bit
    (only the reduction order of .mean() differs)."""

    def __init__(self, Y, bandwidth, n_kernels=5, mul_factor=2.0, device="cpu"):
        self.Y = Y.to(device)
        self.mult = (mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2))
        self.bw = float(bandwidth)
        self.device = device
        self._yy = None

    def kernel(self, D2):
        # (5, *D2.shape): one slab per bandwidth, summed over the bandwidth axis
        scaled = D2.unsqueeze(0) / (self.bw * self.mult.view(-1, *([1] * D2.dim())))
        return torch.exp(-scaled).sum(dim=0)

    @property
    def yy(self):
        if self._yy is None:
            with torch.no_grad():
                d2 = torch.cdist(self.Y, self.Y, p=2,
                                 compute_mode="use_mm_for_euclid_dist") ** 2
                self._yy = self.kernel(d2).mean()
        return self._yy

    def __call__(self, X):
        """X: (S, n, d) -> (S,) MMD^2 values."""
        S, n, d = X.shape
        Yb = self.Y.unsqueeze(0).expand(S, -1, -1)
        d2_xx = torch.cdist(X, X, p=2, compute_mode="use_mm_for_euclid_dist") ** 2
        d2_xy = torch.cdist(X, Yb, p=2, compute_mode="use_mm_for_euclid_dist") ** 2
        XX = self.kernel(d2_xx).mean(dim=(-2, -1))
        XY = self.kernel(d2_xy).mean(dim=(-2, -1))
        return XX - 2 * XY + self.yy


class ElementwiseAdam:
    """AdamGuidance (tfg/adam_guidance.py) on a (B, d) tensor with a row mask:
    identical arithmetic per row, k shared (all rows step in lock-step)."""

    def __init__(self, beta1, beta2, delta, rho):
        self.b1, self.b2, self.delta, self.rho = float(beta1), float(beta2), float(delta), float(rho)
        self.m = self.v = None
        self.k = 0

    def step(self, g, active):
        g = g.detach()
        if self.m is None:
            self.m = torch.zeros_like(g)
            self.v = torch.zeros_like(g)
        self.k += 1
        a = active.view(-1, 1).to(g.dtype)
        self.m = torch.where(active.view(-1, 1), self.b1 * self.m + (1.0 - self.b1) * g, self.m)
        self.v = torch.where(active.view(-1, 1), self.b2 * self.v + (1.0 - self.b2) * g * g, self.v)
        m_hat = self.m / (1.0 - self.b1 ** self.k)
        v_hat = self.v / (1.0 - self.b2 ** self.k)
        return self.rho * m_hat / (torch.sqrt(v_hat) + self.delta)


@contextlib.contextmanager
def frozen_params(*models):
    """requires_grad_(False) on all parameters, restored afterwards."""
    saved = [(p, p.requires_grad) for m in models for p in m.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(False)
        yield
    finally:
        for p, rg in saved:
            p.requires_grad_(rg)


@contextlib.contextmanager
def capture_trajectory(model_uncond, store):
    """Record the x_t fed to sample_ddim_step (t = T-1 .. 1)."""
    orig = model_uncond.sample_ddim_step

    def wrapped(x_start, *a, **k):
        store.append(x_start.detach().clone())
        return orig(x_start, *a, **k)
    model_uncond.sample_ddim_step = wrapped
    try:
        yield
    finally:
        del model_uncond.sample_ddim_step


# ----------------------------------------------------------------------------
# single-restart runner (mirrors _guided.run line by line)
# ----------------------------------------------------------------------------
def run_single(model_cond, model_uncond, S_G, bandwidth, n_max, spatial, temporal,
               restart, schedule="constant", adam_rho=0.4, device="cpu",
               guidance_target="x_t", mu_strength=1.0, beta1=0.9, beta2=0.995,
               delta=1e-8, *, batched_lgd=False, batched_mmd=False,
               trajectory=None, force_traj=None, grad_log=None):
    """Same signature/semantics as _guided.run.  With both flags False this
    is a transcription that differs only in using a torch.Generator for the
    per-perturbation seeds (bit-identical stream)."""
    T = model_uncond.diffusion_steps
    sched = DiffusionSchedule(T=T)
    mmd = MMDLoss(kernel=RBF(bandwidth=bandwidth, device="cpu"), device="cpu")
    bmmd = BatchedMMD(S_G, bandwidth, device=device) if batched_mmd else None
    adam = (AdamGuidance(beta1=beta1, beta2=beta2, delta=delta, rho=adam_rho,
                         inv_sqrt_alpha=False) if temporal == "adam" else None)
    gen = torch.Generator(device="cpu")
    d_x, d_y = model_uncond.nfeatures, model_cond.nfeatures
    M = n_perturb(spatial)
    lgd = M > 1

    x = torch.zeros(1, d_x, device=device)
    calls, n_hist, diverged = 0, [], False
    for t in range(T - 1, 0, -1):
        if force_traj is not None:                 # teacher forcing (per-step equivalence)
            x = force_traj[T - 1 - t].to(device).reshape(1, -1)
        x = x.detach().clone().requires_grad_(True)
        if trajectory is not None:
            trajectory.append(x.detach().clone())
        x_prev, pred_x0 = model_uncond.sample_ddim_step(x, t, condition_x=None,
                                                        device=device, eta=0.0)
        cur_var = model_uncond.betas[t].to(device)
        r_t = cur_var / torch.sqrt(1 + cur_var ** 2)
        n = n_for_step(t, sched, n_max, schedule)
        n_hist.append(n)
        x0_leaf = pred_x0.detach().clone().requires_grad_(True)
        base_x0 = x0_leaf if guidance_target == "x0" else pred_x0

        noises = [draw_cond_noise(gen, restart, t, j, n, d_x, d_y, lgd) for j in range(M)]
        if not batched_lgd:
            ys = []
            for j, (eps_x0, z0, zs) in enumerate(noises):
                x0 = base_x0 + (r_t * eps_x0.to(device) if lgd else 0.0)
                cond = x0.reshape(1, -1).repeat(n, 1)
                ys.append(cm_sample_with_noise(model_cond, z0.to(device), zs_to(zs, device), cond))
                calls += n
        else:
            x0s = [base_x0 + (r_t * e.to(device) if lgd else 0.0) for e, _, _ in noises]
            cond = torch.cat([x0.reshape(1, -1).repeat(n, 1) for x0 in x0s], 0)
            z0 = torch.cat([z for _, z, _ in noises], 0).to(device)
            zs = [torch.cat([zz[k] for _, _, zz in noises], 0).to(device)
                  for k in range(len(PAPER_TS) - 1)]
            yall = cm_sample_with_noise(model_cond, z0, zs, cond)
            ys = list(yall.split(n, 0))
            calls += M * n
        if batched_mmd:
            vals = bmmd(torch.stack(ys, 0))              # (M,)
            terms = [-vals[j] for j in range(M)]
        else:
            terms = [-mmd(y, S_G) for y in ys]
        loss = (-terms[0] if M == 1
                else -torch.logsumexp(torch.stack(terms), 0) + math.log(M))

        if guidance_target == "x0":
            g, = torch.autograd.grad(loss, x0_leaf, allow_unused=True)
            g = torch.zeros_like(x0_leaf) if g is None else g
            upd = g.detach() if adam is None else adam.step(g)
            with torch.no_grad():
                ab_t = model_uncond.baralphas[t].to(device)
                ab_prev = model_uncond.baralphas[t - 1].to(device)
                x0_moved = x0_leaf.detach() - mu_strength * upd
                eps_eff = (x.detach() - ab_t.sqrt() * x0_moved) / (1 - ab_t).sqrt()
                x = ab_prev.sqrt() * x0_moved + (1 - ab_prev).sqrt() * eps_eff
        else:
            g, = torch.autograd.grad(loss, x, allow_unused=True)
            g = torch.zeros_like(x) if g is None else g
            upd = g.detach() if adam is None else adam.step(g)
            if grad_log is not None:
                grad_log.append((g.detach().cpu().clone(), upd.detach().cpu().clone(),
                                 float(loss.detach())))
            with torch.no_grad():
                x = x_prev.detach().clone() - upd
        if not torch.isfinite(x).all() or float(x.abs().max()) > 50.0:
            diverged = True
            if force_traj is None:
                break
    return x.detach().reshape(-1).clone(), {
        "conditional_calls": calls, "diverged": diverged,
        "n_t_mean": sum(n_hist) / len(n_hist), "steps": len(n_hist)}


def zs_to(zs, device):
    return [z.to(device) for z in zs] if device != "cpu" else zs


# ----------------------------------------------------------------------------
# batched restarts
# ----------------------------------------------------------------------------
def run_batched_restarts(model_cond, model_uncond, S_G, bandwidth, n_max, spatial,
                         temporal, restarts, schedule="constant", adam_rho=0.4,
                         device="cpu", beta1=0.9, beta2=0.995, delta=1e-8, *,
                         batched_mmd=True, trajectory=None, lean_ddim=False,
                         force_traj=None, grad_log=None):
    """B = len(restarts) independent runs of _guided.run (guidance_target='x_t')
    as one batch.  Rows never interact: the denoiser, the CM sampler, the
    per-set MMDs and the element-wise Adam are all row-separable, and the
    per-(restart, t, j) conditional noise is drawn from the same seeds.
    Diverged rows are frozen at their diverging value (as the reference
    returns them) and dropped from further compute."""
    T = model_uncond.diffusion_steps
    sched = DiffusionSchedule(T=T)
    B = len(restarts)
    d_x, d_y = model_uncond.nfeatures, model_cond.nfeatures
    M = n_perturb(spatial)
    lgd = M > 1
    gen = torch.Generator(device="cpu")
    bmmd = BatchedMMD(S_G, bandwidth, device=device)
    mmd = None if batched_mmd else MMDLoss(kernel=RBF(bandwidth=bandwidth, device=device), device=device)
    adam = (ElementwiseAdam(beta1, beta2, delta, adam_rho) if temporal == "adam" else None)
    S_Gd = S_G.to(device)

    x = torch.zeros(B, d_x, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    calls, n_hist = 0, []
    diverged = torch.zeros(B, dtype=torch.bool, device=device)
    for t in range(T - 1, 0, -1):
        if force_traj is not None:                 # teacher forcing: (B, d) per step
            x = force_traj[T - 1 - t].to(device).reshape(B, d_x)
            active = torch.ones(B, dtype=torch.bool, device=device)
        if trajectory is not None:
            trajectory.append(x.detach().clone())
        if not bool(active.any()):
            n_hist.append(n_for_step(t, sched, n_max, schedule))
            continue
        idx = active.nonzero(as_tuple=True)[0]
        Ba = int(idx.numel())
        xa = x[idx].detach().clone().requires_grad_(True)
        if lean_ddim:
            x_prev, pred_x0 = ddim_step_lean(model_uncond, xa, t)
        else:
            x_prev, pred_x0 = model_uncond.sample_ddim_step(xa, t, condition_x=None,
                                                            device=device, eta=0.0)
        cur_var = model_uncond.betas[t].to(device)
        r_t = cur_var / torch.sqrt(1 + cur_var ** 2)
        n = n_for_step(t, sched, n_max, schedule)
        n_hist.append(n)

        # noise for every (restart, j): drawn on CPU from the reference seeds
        eps_l, z0_l, zs_l = [], [], []
        for bi in idx.tolist():
            for j in range(M):
                e, z0, zs = draw_cond_noise(gen, restarts[bi], t, j, n, d_x, d_y, lgd)
                eps_l.append(e); z0_l.append(z0); zs_l.append(zs)
        z0 = torch.cat(z0_l, 0).to(device)                                 # (Ba*M*n, d_y)
        zs = [torch.cat([z[k] for z in zs_l], 0).to(device) for k in range(len(PAPER_TS) - 1)]
        if lgd:
            eps_x0 = torch.stack(eps_l, 0).view(Ba, M, d_x).to(device)       # (Ba, M, d_x)
            x0 = pred_x0.unsqueeze(1) + r_t * eps_x0                           # (Ba, M, d_x)
        else:
            x0 = pred_x0.unsqueeze(1)                                          # (Ba, 1, d_x)
        cond = x0.unsqueeze(2).expand(Ba, M, n, d_x).reshape(Ba * M * n, d_x)
        y = cm_sample_with_noise(model_cond, z0, zs, cond)                     # (Ba*M*n, d_y)
        calls += Ba * M * n
        if batched_mmd:
            vals = bmmd(y.view(Ba * M, n, d_y)).view(Ba, M)                     # MMD^2 per (b, j)
        else:
            vals = torch.stack([mmd(yy, S_Gd) for yy in y.view(Ba * M, n, d_y)]).view(Ba, M)
        terms = -vals
        if M == 1:
            loss_rows = -terms[:, 0]
        else:
            loss_rows = -torch.logsumexp(terms, 1) + math.log(M)
        g, = torch.autograd.grad(loss_rows.sum(), xa, allow_unused=True)
        g = torch.zeros_like(xa) if g is None else g
        if adam is None:
            upd = g.detach()
        else:
            g_full = torch.zeros(B, d_x, device=device, dtype=g.dtype).index_copy(0, idx, g.detach())
            upd = adam.step(g_full, active)[idx]
        if grad_log is not None:
            grad_log.append((g.detach().cpu().clone(), upd.detach().cpu().clone(),
                             loss_rows.detach().cpu().clone()))
        with torch.no_grad():
            x_new = x_prev.detach() - upd
            bad = ~torch.isfinite(x_new).all(1) | (x_new.abs().amax(1) > 50.0)
            x = x.clone()
            x[idx] = x_new
            diverged[idx[bad]] = True
            active = active & ~diverged
    return x.detach().clone(), {"conditional_calls": calls, "diverged": diverged.cpu(),
                                "steps": len(n_hist), "n_t_mean": sum(n_hist) / len(n_hist)}


def ddim_step_lean(model, x, t):
    """Diffusion.sample_ddim_step without the per-call self.to(device),
    dtype lookup and .to() of x; same arithmetic (eta=0)."""
    t_batch = torch.full([x.shape[0], 1], t, device=x.device)
    eps = model(x, t_batch, None)
    ab_t = model.baralphas[t]
    ab_prev = model.baralphas[t - 1]
    sigma_t = 0.0 * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
    pred_x0 = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
    dir_xt = torch.sqrt(1 - ab_prev - sigma_t ** 2) * eps
    return torch.sqrt(ab_prev) * pred_x0 + dir_xt, pred_x0
