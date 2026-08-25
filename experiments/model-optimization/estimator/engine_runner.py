"""Agent 4 -- the synthetic benchmark of Experiments 2-7, run THROUGH the engine.

``run_engine`` reproduces ``simulations/experiments/_guided.py::run`` with
``tfg.engine.GeneralizedTFG`` (proof: ``simulations/tests/test_engine_matches_guided.py``)
and exposes every campaign candidate as an engine/sampler option.

    python experiments/model-optimization/estimator/engine_runner.py --help
    python experiments/model-optimization/estimator/engine_runner.py --setting 2D \
        --n 8 --spatial no_lgd --temporal none --restarts 40 --candidate baseline

Conventions: python = /Users/stolk/miniconda3/bin/python, CPU; the pipeline is
float32 end to end (the checkpoints and ``_guided.py`` are float32; see
EQUIVALENCE.md) unless ``--dtype float64`` is given, in which case the models
are converted and the schedule is REBUILT in float64.
"""
import argparse
import copy
import csv
import json
import math
import platform
import resource
import statistics as st
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _common import (PENALTY, fixed_bandwidth, load, paired_stats,     # noqa: E402
                     penalised_score, target_set)
from _guided import evaluate                                          # noqa: E402
from _models import PAPER_TS, SEEDS, conditional_model, unconditional_model  # noqa: E402
from tfg.config import TFGConfig                                      # noqa: E402
from tfg.distributional import (CMSampler, DistributionalLoss, LegacyTape,   # noqa: E402
                                RandnInitTape, repository_schedule)
from tfg.engine import GeneralizedTFG                                 # noqa: E402
from tfg.noise_tape import NoiseTape                                  # noqa: E402

COMMIT = "6af2081"
TAG = {"2D": "", "5D": "_canonical", "10D": "_canonical"}
M_LGD = 3
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


# ---------------------------------------------------------------------------
# candidates: name -> (engine-config mutator, sampler kwargs, loss kwargs)
# ---------------------------------------------------------------------------

def candidate_spec(name, n, T):
    """Return ``(cfg_mutator, sampler_kw, loss_kw, notes)`` for a candidate name.

    Names (all relative to the baseline arm given by spatial/temporal):
      baseline            nothing
      norm_only           temporal adam with beta1=0 (normalisation-only rule)
      clip<c>             gradient-norm clipping to c (e.g. clip0.5)
      unit                unit-norm gradient (direction only), rho sets the step
      adapt_agree<thr>    adaptive n, agreement policy, budget = n*T, n_min=2, n_max=4n
      adapt_improve       adaptive n, improvement policy, same budget
      crn                 frozen conditional noise across steps (approximate)
      antithetic          antithetic conditional-noise pairs
      stale<k>            stale gradient reuse, refresh every k steps (approximate)
      recur<R>_<metric>   adaptive recurrence v1, up to R recurrences, early stop
      bw_target / bw_pooled / bw_pooled_floor / sqrt_abs_eps / sqrt_floor / mmd2
    Round 2 (scale-free clipping):
      relclip<c>          clip to c x running MEDIAN of past raw grad norms
      relclip_ema<c>      clip to c x EMA(0.9) of past raw grad norms
      qclip<q>            clip to the q-quantile of past raw grad norms
      trust_noise<tau>    ||Delta_t|| <= tau x sqrt(1 - alphabar_t)
      trust_ddim<tau>     ||Delta_t|| <= tau x ||x_ddim - x_t||
      sqrtfloor_clip<c>   sqrt_floor transform + absolute clip c
      sqrtfloor_relclip<c> sqrt_floor transform + relative (median) clip c
    """
    sampler_kw, loss_kw, notes = {}, {}, {}

    def mut(cfg):
        pass

    if name == "baseline":
        pass
    elif name == "norm_only":
        def mut(cfg):
            cfg.temporal.mode = "adam"
            cfg.temporal.beta1 = 0.0
    elif name.startswith("clip"):
        c = float(name[4:])

        def mut(cfg):
            cfg.temporal.grad_norm = "clip"
            cfg.temporal.grad_clip = c
    elif name.startswith("unit"):
        scale = float(name[4:]) if len(name) > 4 else 0.4

        def mut(cfg):
            cfg.temporal.grad_norm = "unit"
            cfg.rho_scalar = scale
    elif name.startswith("adapt_agree"):
        thr = float(name[len("adapt_agree"):] or 0.5)

        def mut(cfg):
            cfg.n_schedule.type = "adaptive"
            cfg.n_schedule.policy = "agreement"
            cfg.n_schedule.agreement_threshold = thr
            cfg.n_schedule.n_min = max(2, n // 4)
            cfg.n_schedule.n_max = 4 * n
            cfg.n_schedule.n_start = n
            cfg.n_schedule.budget_total = n * T
        sampler_kw["cache"] = True
    elif name.startswith("adapt_improve"):
        def mut(cfg):
            cfg.n_schedule.type = "adaptive"
            cfg.n_schedule.policy = "improvement"
            cfg.n_schedule.n_min = max(1, n // 4)
            cfg.n_schedule.n_max = 4 * n
            cfg.n_schedule.n_start = n
            cfg.n_schedule.budget_total = n * T
    elif name == "crn":
        def mut(cfg):
            cfg.n_schedule.eta_keying = "frozen"
        notes["approximate"] = True
    elif name == "antithetic":
        sampler_kw["antithetic"] = True
    elif name.startswith("stale"):
        k = int(name[5:])

        def mut(cfg):
            cfg.temporal_cache.enabled = True
            cfg.temporal_cache.implementation = "stale"
            cfg.temporal_cache.refresh_every = k
        notes["approximate"] = True
    elif name.startswith("recur"):
        body = name[5:]
        R, metric = body.split("_", 1)

        def mut(cfg):
            cfg.adaptive_recurrence.enabled = True
            cfg.adaptive_recurrence.implementation = "v1"
            cfg.adaptive_recurrence.max_recurrences = int(R)
            cfg.adaptive_recurrence.metric = metric
            cfg.adaptive_recurrence.threshold = 1e-2
    elif name.startswith("relclip_ema"):
        c = float(name[len("relclip_ema"):])

        def mut(cfg):
            cfg.temporal.grad_norm = "clip_rel"
            cfg.temporal.clip_ref = "ema"
            cfg.temporal.grad_clip = c
    elif name.startswith("relclip"):
        c = float(name[len("relclip"):])

        def mut(cfg):
            cfg.temporal.grad_norm = "clip_rel"
            cfg.temporal.clip_ref = "median"
            cfg.temporal.grad_clip = c
    elif name.startswith("qclip"):
        q = float(name[5:])

        def mut(cfg):
            cfg.temporal.grad_norm = "clip_quantile"
            cfg.temporal.grad_clip = q
    elif name.startswith("trust_noise") or name.startswith("trust_ddim"):
        kind = "noise" if name.startswith("trust_noise") else "ddim"
        tau = float(name[len("trust_") + len(kind):])

        def mut(cfg):
            cfg.temporal.step_clip = kind
            cfg.temporal.step_tau = tau
    elif name.startswith("sqrtfloor_relclip"):
        c = float(name[len("sqrtfloor_relclip"):])
        loss_kw["transform"] = "sqrt_floor"

        def mut(cfg):
            cfg.temporal.grad_norm = "clip_rel"
            cfg.temporal.clip_ref = "median"
            cfg.temporal.grad_clip = c
    elif name.startswith("sqrtfloor_clip"):
        c = float(name[len("sqrtfloor_clip"):])
        loss_kw["transform"] = "sqrt_floor"

        def mut(cfg):
            cfg.temporal.grad_norm = "clip"
            cfg.temporal.grad_clip = c
    elif name == "bw_target":
        loss_kw["bandwidth"] = "target"
    elif name == "bw_pooled":
        loss_kw["bandwidth"] = "pooled"
    elif name == "bw_pooled_floor":
        loss_kw["bandwidth"] = "pooled_floor"
    elif name in ("sqrt_abs_eps", "sqrt_floor", "mmd2"):
        loss_kw["transform"] = name
    elif name.startswith("precond_"):
        # [Agent P] round 3: precond_{cov,diag,sign,median}[_trust]
        body = name[len("precond_"):]
        with_trust = body.endswith("_trust")
        if with_trust:
            body = body[: -len("_trust")]
        p_mode = {"cov": "whiten", "diag": "diag", "sign": "sign",
                  "median": "median"}.get(body)
        if p_mode is None:
            raise ValueError(f"unknown candidate {name!r}")

        def mut(cfg, _mode=p_mode, _trust=with_trust):
            cfg.temporal.precond.mode = _mode
            if _trust:
                cfg.temporal.step_clip = "noise"
                cfg.temporal.step_tau = 1.0
    elif name.startswith("replay"):
        # [Agent M] round 3: sample-replay MMD (tfg/replay.py).  Names:
        #   replay<P>            Ori's reuse: depth 1, P% of the batch replayed,
        #                        total MMD batch = n, fresh calls = n - round(P% n)
        #   replay_w<P>          same split, weighted-V-statistic mode
        #   replay_geo<lam>d<D>  geometric decay lam, depth D, batch = n
        #   replay_wgeo<lam>d<D> weighted geometric
        #   ..._aug              augment: fresh = n (calls = baseline), batch grows
        #   replay_fill<B>_geo<lam>d<D>  [M-9] fresh = n, topped up to MMD batch B
        #                        with recycled rows ~ lam^k (ReplayConfig.fill)
        #   replay_fifo<B> / replay_cohort<B>  [M-10] fresh = n, topped up to
        #                        batch B: uniform FIFO / capped-cohort thinning
        #   ..._trust            + trust_noise1 on top (champion combination)
        from tfg.replay import replay_counts
        body = name[len("replay"):]
        r_trust = body.endswith("_trust")
        if r_trust:
            body = body[: -len("_trust")]
        r_aug = body.endswith("_aug")
        if r_aug:
            body = body[: -len("_aug")]
        r_mode = "subsample"
        r_fill = 0
        if body.startswith("_w"):
            r_mode, body = "weighted", body[2:]
        elif body.startswith("_"):
            body = body[1:]
        if body.startswith("fill"):
            fill_s, body = body[4:].split("_", 1)
            r_fill = int(fill_s)
        if body.startswith(("fifo", "cohort")):
            from tfg.replay import cohort_counts, fifo_counts
            r_policy = "fifo" if body.startswith("fifo") else "cohort"
            r_fill = int(body[len(r_policy):])
            plan = (fifo_counts if r_policy == "fifo" else cohort_counts)(r_fill, n)
            r_decay, r_depth = 0.7, max(1, len(plan) - 1)   # decay unused by these plans

            def mut(cfg, _pol=r_policy, _d=r_depth, _B=r_fill, _tr=r_trust):
                cfg.replay.enabled = True
                cfg.replay.mode = "subsample"
                cfg.replay.policy = _pol
                cfg.replay.depth = _d
                cfg.replay.batch_total = _B
                cfg.replay.fill = True
                if _tr:
                    cfg.temporal.step_clip = "noise"
                    cfg.temporal.step_tau = 1.0
            notes["approximate"] = True
            return mut, sampler_kw, loss_kw, notes
        if body.startswith("geo"):
            lam_s, d_s = body[3:].split("d")
            r_decay, r_depth = float(lam_s), int(d_s)
        else:
            p = float(body) / 100.0
            if not (0.0 < p < 0.5):
                raise ValueError(f"replay percent must be in (0,50): {name!r}")
            r_decay, r_depth = p / (1.0 - p), 1
        r_batch = r_fill if r_fill else (0 if r_aug else n)

        def mut(cfg, _m=r_mode, _lam=r_decay, _d=r_depth, _B=r_batch,
                _tr=r_trust, _fill=bool(r_fill)):
            cfg.replay.enabled = True
            cfg.replay.mode = _m
            cfg.replay.decay = _lam
            cfg.replay.depth = _d
            cfg.replay.batch_total = _B
            cfg.replay.fill = _fill
            if _B > 0 and not _fill:   # calls-saving arm: engine draws only the fresh rows
                cfg.n_schedule.n_max = replay_counts(_B, _lam, _d)[0]
            if _tr:
                cfg.temporal.step_clip = "noise"
                cfg.temporal.step_tau = 1.0
        notes["approximate"] = True
    elif name.startswith("backsel_"):
        # [Agent B] round 4: importance-selected backpropagation (tfg/backsel.py).
        #   backsel_{uni,is,clust}_k<K>[_cohort<B>|_fifo<B>][_trust]
        #     uni    uniform k of n without replacement, weights n/k (unbiased control)
        #     is     gradient-magnitude importance sampling, inverse-probability
        #            weights, floor 0.25 (unbiased)
        #     clust  greedy k-center on y_i, cluster-aggregated output gradient
        #            through the center's Jacobian (approximate)
        #     _cohort<B>/_fifo<B>  composed with Agent C's [M-10] replay top-up
        #            (fresh = n, MMD batch B); selection over the fresh rows only
        #     _trust + trust_noise1 on top (champion combination)
        #   Forward samples per evaluation = n + k (no-grad batch + regenerated
        #   subset); DIFFERENTIATED samples = k (reported as diff_samples).
        body = name[len("backsel_"):]
        b_trust = body.endswith("_trust")
        if b_trust:
            body = body[: -len("_trust")]
        b_policy, b_B = None, 0
        for pol in ("cohort", "fifo"):
            tag = f"_{pol}"
            if tag in body:
                body, B_s = body.split(tag, 1)
                b_policy, b_B = pol, int(B_s)
        rule_s, k_s = body.rsplit("_k", 1)
        b_rule = {"uni": "uniform", "is": "importance", "clust": "kcenter"}.get(rule_s)
        if b_rule is None:
            raise ValueError(f"unknown candidate {name!r}")
        if b_policy is not None:
            from tfg.replay import cohort_counts, fifo_counts
            plan = (fifo_counts if b_policy == "fifo" else cohort_counts)(b_B, n)
            b_depth = max(1, len(plan) - 1)
        else:
            b_depth = 1

        def mut(cfg, _r=b_rule, _k=int(k_s), _tr=b_trust, _pol=b_policy, _B=b_B, _d=b_depth):
            cfg.backsel.enabled = True
            cfg.backsel.rule = _r
            cfg.backsel.k = _k
            if _pol is not None:
                cfg.replay.enabled = True
                cfg.replay.mode = "subsample"
                cfg.replay.policy = _pol
                cfg.replay.depth = _d
                cfg.replay.batch_total = _B
                cfg.replay.fill = True
            if _tr:
                cfg.temporal.step_clip = "noise"
                cfg.temporal.step_tau = 1.0
        notes["approximate"] = True   # stochastic (uni/is, unbiased) or biased (clust) gradient
    else:
        raise ValueError(f"unknown candidate {name!r}")
    return mut, sampler_kw, loss_kw, notes


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def build_models(setting, dtype=torch.float32):
    params = load(setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    mu = unconditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    if dtype == torch.float64:
        mc = copy.deepcopy(mc).double()
        mu = copy.deepcopy(mu).double()
        S_G = S_G.double()
    return params, S_G, bw, mc, mu


def run_engine(mc, mu, S_G, bw, n, spatial, temporal, restart, *, rng="tape",
               dtype=torch.float32, adam_rho=0.4, beta1=0.9, beta2=0.995,
               delta=1e-8, candidate="baseline", cfg_mutator=None, sampler_kw=None,
               loss_kw=None, trace_steps=False, tape_seed=None, loss_backend="reference",
               x_init="zeros", zeta=1.0, step_clip="none", step_tau=1.0):
    """Engine-based counterpart of ``_guided.run``; same return signature plus
    ``cm_samples`` (actual generator draws), ``grad_norms``, ``agreement``.

    Protocol fields (defaults = the legacy protocol of Exp 2-7 and campaign
    rounds 1-4): ``x_init`` ``zeros`` | ``randn`` (``x_T ~ N(0,I)`` from the
    generator ``_guided.py`` uses, seed ``0x5EED0000 ^ restart``); ``zeta`` = the
    guidance step scale (``rho_scalar``; multiplies the Adam output too, as in
    ``_guided``); ``step_clip``/``step_tau`` = the noise-level trust region
    (candidates such as ``trust_noise1`` set the same fields)."""
    sch = repository_schedule(mu, dtype=dtype)
    T = sch.T
    M = M_LGD if spatial == "lgd" else 1
    mut, s_kw, l_kw, notes = candidate_spec(candidate, n, T)
    s_kw = {**s_kw, **(sampler_kw or {})}
    l_kw = {**l_kw, **(loss_kw or {})}

    if x_init not in ("zeros", "randn"):
        raise ValueError(x_init)
    cfg = TFGConfig(T=T, N_recur=1, N_iter=0, gamma_bar=0.0, rho_scalar=float(zeta),
                    mu_scalar=0.0, n_mc=M, init=x_init, guidance_scaling="raw",
                    smoothing=("lgd_beta" if M > 1 else "tfg"))
    cfg.temporal.step_clip = step_clip
    cfg.temporal.step_tau = float(step_tau)
    cfg.n_schedule.enabled = True
    cfg.n_schedule.type = "constant"
    cfg.n_schedule.n_max = int(n)
    cfg.n_schedule.eta_per_perturbation = True      # independent noise per perturbation, as _guided
    if temporal == "adam":
        cfg.temporal.mode = "adam"
        cfg.temporal.adam_rho = adam_rho
        cfg.temporal.beta1, cfg.temporal.beta2, cfg.temporal.delta = beta1, beta2, delta
    elif temporal != "none":
        raise ValueError(temporal)
    mut(cfg)
    if cfg_mutator is not None:
        cfg_mutator(cfg)
    if cfg.temporal.mode == "adam" and temporal == "none":
        cfg.temporal.adam_rho = adam_rho          # e.g. norm_only on a none arm
    cfg.validate()
    protocol = {"x_init": x_init, "zeta": float(cfg.rho_scalar),
                "step_clip": cfg.temporal.step_clip, "step_tau": float(cfg.temporal.step_tau),
                "rng": rng, "dtype": str(dtype).split(".")[-1], "loss_backend": loss_backend}

    seed = restart if tape_seed is None else tape_seed
    base_tape = NoiseTape(seed=seed, dtype=dtype)
    if rng == "legacy":
        tape = LegacyTape(base_tape, restart, with_delta=(M > 1),
                          delta_shape=(1, mu.nfeatures), dtype=dtype)
        source = "legacy"
    elif rng == "tape":
        tape, source = base_tape, "tape"
    else:
        raise ValueError(rng)
    if x_init == "randn":
        tape = RandnInitTape(tape, restart)

    sampler = CMSampler(mc, PAPER_TS, tape, source=source, dtype=dtype, **s_kw)
    loss_kw_full = {"bandwidth": "fixed", "bandwidth_value": bw, "transform": "mmd2",
                    "backend": loss_backend, **l_kw}
    if loss_kw_full["bandwidth"] != "fixed":
        loss_kw_full.pop("bandwidth_value", None)
    loss = DistributionalLoss(S_G, **loss_kw_full)

    def eps_theta(x, t):
        t_batch = torch.full([x.shape[0], 1], int(t))
        return mu(x, t_batch, None)

    if getattr(cfg, "replay", None) is not None and cfg.replay.enabled:
        from tfg.replay import wrap_log_f          # [Agent M] opt-in sample replay
        log_f = wrap_log_f(sampler, loss, base_tape, cfg.replay)
    else:
        def log_f(x, n_t=None, eta_keys=None):
            return -loss(sampler(x, eta_keys))

    if getattr(cfg, "backsel", None) is not None and cfg.backsel.enabled:
        # [Agent B] opt-in importance-selected backprop; composes with replay
        # (selection over the fresh rows of the replay-stacked batch).
        from tfg.backsel import wrap_log_f as backsel_wrap
        inner = None
        if cfg.replay.enabled:
            from tfg.replay import wrap_log_f as replay_wrap
            inner = (lambda s_, l_: replay_wrap(s_, l_, base_tape, cfg.replay))
        log_f = backsel_wrap(sampler, loss, base_tape, cfg.backsel, inner=inner)

    engine = GeneralizedTFG(eps_theta, log_f, sch, tape, cfg)
    steps = {}

    def tracer(name, t, r, k, tensor):
        if name in ("x_prev", "grad_rho_raw") and torch.is_tensor(tensor):
            steps.setdefault(name, {})[(int(t), int(r))] = tensor.detach().clone()

    t0 = time.perf_counter()
    x = engine.run((1, mu.nfeatures), trace=tracer)
    secs = time.perf_counter() - t0
    # _guided's divergence rule, applied per step post hoc
    diverged, final = False, x
    for t in range(T, 0, -1):
        xp = steps["x_prev"].get((t, engine.counter.recurrence_history[T - t]))
        if xp is None:
            continue
        if not torch.isfinite(xp).all() or float(xp.abs().max()) > 50.0:
            diverged, final = True, xp
            break
    gn = [float(g.norm()) for (_, g) in sorted(steps["grad_rho_raw"].items(), reverse=True)]
    c = engine.counter
    info = {"conditional_calls": int(c.conditional_calls), "cm_samples": int(sampler.cm_samples),
            "seconds": secs, "diverged": diverged,
            "n_t_mean": sum(c.n_t_history) / len(c.n_t_history),
            "n_t_min": min(c.n_t_history), "n_t_max": max(c.n_t_history),
            "steps": len(c.n_t_history), "grad_norms": gn,
            "agreement": list(c.agreement_history),
            "recurrences": sum(c.recurrence_history), "stale_steps": c.stale_steps,
            "approximate": bool(notes.get("approximate", False)),
            "protocol": protocol}
    # [Agent B] differentiated samples (graphs + backward): the full batch
    # unless the backsel wrapper reports its own count.
    b_stats = getattr(log_f, "stats", None)
    info["diff_samples"] = int(b_stats["diff_samples"]) if b_stats and b_stats["diff_samples"] else int(sampler.cm_samples)
    if trace_steps:
        info["x_prev_trace"] = steps["x_prev"]
        info["grad_trace"] = steps["grad_rho_raw"]
    return final.detach().reshape(-1).clone(), info


# ---------------------------------------------------------------------------
# screening driver
# ---------------------------------------------------------------------------

def cell(setting, n, spatial, temporal, candidate, restarts, offset=0, dtype="float32",
         rng="tape", verbose=True, loss_backend="reference", x_init="zeros", zeta=1.0,
         step_clip="none", step_tau=1.0):
    dt = DTYPES[dtype]
    params, S_G, bw, mc, mu = build_models(setting, dt)
    runs, xs = [], []
    t_wall = time.perf_counter()
    for r in range(offset, offset + restarts):
        x, info = run_engine(mc, mu, S_G, bw, n, spatial, temporal, r, rng=rng, dtype=dt,
                             candidate=candidate, loss_backend=loss_backend,
                             x_init=x_init, zeta=zeta, step_clip=step_clip, step_tau=step_tau)
        ev = evaluate(x, params, info)
        ev.update({k: info[k] for k in ("cm_samples", "n_t_mean", "recurrences", "stale_steps",
                                        "diff_samples")})   # [Agent B] diff_samples appended
        ev["grad_norm_median"] = st.median(info["grad_norms"]) if info["grad_norms"] else None
        ev["agreement_mean"] = (st.mean(info["agreement"]) if info["agreement"] else None)
        runs.append(ev)
    wall = time.perf_counter() - t_wall
    scores = [min(q["L2"], PENALTY) if not q["diverged"] else PENALTY for q in runs]
    fin = [q for q in runs if not q["diverged"]]
    summ = {"setting": setting, "n": n, "spatial": spatial, "temporal": temporal,
            "candidate": candidate, "restarts": restarts, "offset": offset, "dtype": dtype,
            "loss_backend": loss_backend,
            "protocol": info["protocol"],
            "score": penalised_score(runs),
            "L2_mean": st.mean([q["L2"] for q in fin]) if fin else float("nan"),
            "L2_median": st.median([q["L2"] for q in fin]) if fin else float("nan"),
            "success_rate": sum(1 for q in fin if q["abs_err"] < 0.5) / restarts,
            "diverged": len(runs) - len(fin),
            "conditional_calls_mean": st.mean([q["conditional_calls"] for q in runs]),
            "cm_samples_mean": st.mean([q["cm_samples"] for q in runs]),
            "diff_samples_mean": st.mean([q["diff_samples"] for q in runs]),   # [Agent B]
            "seconds_per_run": wall / restarts, "wall_s": wall,
            "peak_mem_mb": peak_rss_mb(),
            "grad_norm_median": st.median([q["grad_norm_median"] for q in runs
                                           if q["grad_norm_median"] is not None]),
            "scores": scores, "runs": runs}
    if verbose:
        print(f"{setting} n={n:<3} {spatial:<7} {temporal:<5} {candidate:<16} "
              f"score={summ['score']:.4f} succ={summ['success_rate']:.0%} "
              f"div={summ['diverged']} calls={summ['cm_samples_mean']:.0f} "
              f"{summ['seconds_per_run']:.2f}s/run", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="2D", choices=["2D", "5D", "10D"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--spatial", default="no_lgd", choices=["no_lgd", "lgd"])
    ap.add_argument("--temporal", default="none", choices=["none", "adam"])
    ap.add_argument("--candidate", default="baseline")
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--rng", default="tape", choices=["tape", "legacy"])
    ap.add_argument("--loss", default="reference", choices=["reference", "fast"],
                    help="MMD backend: repository MMDLoss, or the exact cached-target tfg.fast_mmd")
    ap.add_argument("--x-init", default="zeros", choices=["zeros", "randn"],
                    help="x_T = 0 (legacy) or x_T ~ N(0,I) with _guided's per-restart generator")
    ap.add_argument("--zeta", type=float, default=1.0, help="guidance step scale (rho_scalar)")
    ap.add_argument("--step-clip", default="none", choices=["none", "noise", "ddim"])
    ap.add_argument("--step-tau", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="json path for the cell summary")
    a = ap.parse_args()
    s = cell(a.setting, a.n, a.spatial, a.temporal, a.candidate, a.restarts, a.offset,
             a.dtype, a.rng, loss_backend=a.loss, x_init=a.x_init, zeta=a.zeta,
             step_clip=a.step_clip, step_tau=a.step_tau)
    if a.out:
        Path(a.out).write_text(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
