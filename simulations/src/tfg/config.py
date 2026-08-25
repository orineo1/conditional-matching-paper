"""Configuration for the generalised TFG engine.

Every extension is off by default.  With the defaults, ``tfg.engine`` executes
plain TFG Algorithm 1 and must match ``tfg.reference`` exactly; that is what
``tests/test_equivalence.py`` asserts.

Components 2 (temporal gradient cache) and 3 (improvement-adaptive recurrence)
keep their ORIGINAL gated surface: ``enabled=True`` with the original
``implementation="gated"`` still raises ``NotImplementedError`` (the design that
surface described was never built).  The performance campaign (Agent 4,
2026-08-23) added concrete, tested implementations behind
``implementation="stale"`` (component 2) and ``implementation="v1"``
(component 3); see the dataclass docstrings.

Agent-4 additions are marked ``[A4]`` below.  They are all opt-in and every one
of them leaves the default path byte-identical to the frozen reference.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class NScheduleConfig:
    """Component 1: adaptive conditional sample count.

    ``type``
        ``constant | time | noise`` (fixed schedules, see ``n_schedule.py``) or
        ``adaptive`` [A4]: ``n_t`` is chosen per step from a state dict built by
        the engine (gradient agreement between two independent half batches,
        or loss improvement) -- see ``tfg/adaptive.py``.
    ``eta_per_perturbation`` [A4]
        Key the conditional draws ``("eta", t, j, i)`` instead of
        ``("eta", t, i)`` so every spatial perturbation ``j`` uses independent
        conditional noise.  This is what ``experiments/_guided.py`` does
        (``torch.manual_seed(key_seed("cond", restart, t, j))``).  Off: the C5
        convention (draws shared across all loss evaluations within a step).
    ``eta_keying`` [A4]
        ``per_step``: fresh conditional noise at every outer step (default).
        ``frozen``: common random numbers -- the SAME conditional noise keys at
        every step (``("eta", "frozen", i)``).  APPROXIMATE: the guidance
        gradient is then a deterministic function of a single noise draw, so it
        is a biased estimate of the population gradient across the trajectory.
        Candidate 4 in the campaign; bias is measured, never assumed.
    ``n_min, n_start, policy, agreement_threshold, improvement_threshold,
      grow, budget_total`` [A4]
        Parameters of the adaptive policy; see ``tfg/adaptive.py``.
    """
    enabled: bool = False
    type: str = "constant"       # constant | time | noise | adaptive
    n_max: int = 1
    kappa: float = 1.0
    # --- [A4] ---
    eta_per_perturbation: bool = False
    eta_keying: str = "per_step"          # per_step | frozen
    n_min: int = 1
    n_start: int = 0                      # 0 -> start at n_max
    policy: str = "agreement"             # agreement | improvement
    agreement_threshold: float = 0.5
    improvement_threshold: float = 0.0
    grow: int = 2
    budget_total: int = 0                 # 0 -> unlimited


@dataclass
class TemporalCacheConfig:
    """Component 2: temporal reuse of guidance information.

    ``implementation="gated"`` (original surface): raises NotImplementedError.
    ``implementation="stale"`` [A4]: the rho-branch gradient is recomputed only
    every ``refresh_every`` outer steps and the last value is re-applied in
    between (``lambda_value`` mixes fresh and stale when both exist).  This is
    APPROXIMATE: a stale gradient is evaluated at a previous ``x_t``.  It saves
    conditional calls on the skipped steps (no predictor evaluation at all).
    Bias is measured in the campaign (candidate 6), never assumed.  Exact reuse
    (identical inputs within a step) does not live here: it is a property of
    the predictor/sampler wrapper (``tfg/distributional.py``).
    """
    enabled: bool = False
    type: str = "gradient"       # gradient (kernel-statistics caching is design-only)
    lambda_mode: str = "fixed"   # fixed | adaptive
    lambda_value: float = 0.0
    # --- [A4] ---
    implementation: str = "gated"   # gated | stale
    refresh_every: int = 1


@dataclass
class AdaptiveRecurrenceConfig:
    """Component 3: improvement-adaptive N_recur.

    ``implementation="gated"`` (original surface): raises NotImplementedError.
    ``implementation="v1"`` [A4]: the recurrence loop runs ``r = 1..max_recurrences``
    and stops early when the chosen metric says the extra recurrence is not
    paying for itself:

      ``clean_proxy``        |log f(r) - log f(r-1)| / max(|log f(r-1)|, 1)  < threshold
      ``next_state_tweedie`` ||x_prev(r) - x_prev(r-1)|| / max(||x_prev(r-1)||,1) < threshold
      ``grad_stability``     cos(grad_rho(r), grad_rho(r-1)) > 1 - threshold

    ``TFGConfig.N_recur`` must be 1 when this is enabled (the loop bound is
    ``max_recurrences``); validate() refuses the ambiguous combination.
    """
    enabled: bool = False
    max_recurrences: int = 1
    threshold: float = 1e-2
    metric: str = "next_state_tweedie"   # clean_proxy | next_state_tweedie | grad_stability
    # --- [A4] ---
    implementation: str = "gated"        # gated | v1


@dataclass
class PrecondConfig:
    """[Agent P] Preconditioning of the rho-branch gradient (opt-in).

    Applied AFTER the ``grad_norm`` pre-processing and BEFORE the temporal
    operator; the later ``rho_t`` scaling, ``step_clip`` trust region and the
    line-9 ``/sqrt(alpha_t)`` are unchanged.  See ``tfg/precond.py`` and
    ``experiments/model-optimization/precond/THEORY.md``.

    ``mode``
        ``none``    off (default; the engine path is byte-identical to the
                    frozen reference).
        ``whiten``  full-covariance whitening: EMA (factor ``ema``) of the
                    outer products of PAST raw gradients, ridge
                    ``eps * tr(C)/d``, direction ``C^{-1/2} g`` rescaled to the
                    RAW norm ``||g||`` (norm-preserving); identity for the
                    first ``warmup`` steps.
        ``diag``    diagonal second-moment direction fix: EMA of past ``g^2``,
                    direction ``g / sqrt(v_reg)`` rescaled to ``||g||``
                    (norm-preserving RMSProp); identity for ``warmup`` steps.
        ``sign``    ``sign(g) * ||g|| / sqrt(#nonzero)`` (norm-preserving
                    sign-SGD; stateless; the identity when g has one nonzero
                    coordinate).
        ``median``  per-coordinate median over a sliding window of the last
                    ``window`` raw gradients (current included); the output
                    norm is the median's own (tail-robust) norm.

    Norm-preservation is deliberate: fixed-magnitude outputs (Adam/norm_only/
    unit) are the verified 10D failure mode; these rules change the direction
    (or, for ``median``, the estimator) and leave step-size control to the raw
    norm or to ``TemporalConfig.step_clip``.  Validation happens in
    ``tfg.precond.make_preconditioner`` at engine construction.
    """
    mode: str = "none"        # none | whiten | diag | sign | median
    ema: float = 0.9          # EMA factor for whiten/diag state
    eps: float = 1e-6         # relative ridge for whiten/diag
    window: int = 5           # median window length
    warmup: int = 5           # steps before whiten/diag activate (identity)


@dataclass
class ReplayConfig:
    """[Agent M] Sample-replay MMD: mix DETACHED conditional samples from past
    outer steps into the guidance MMD batch (opt-in, off by default).

    Generalisation of the upstream ``reuse_frac`` mechanism
    (``upstream/claude/hybrid-sampling-optimization-55fv3b``,
    ``Optimization.optimize_LGD``); theory and derivations in
    ``experiments/model-optimization/replay/THEORY.md``, implementation in
    ``tfg/replay.py``.  The engine itself is untouched: the replay wrapper
    lives outside, around the ``log_f`` callable (``tfg.replay.wrap_log_f``),
    exactly like the rest of the distributional path.

    ``mode``
        ``subsample``  draw ``r_k`` rows from the step-``t+k`` cache
                       (NoiseTape-keyed ``("replay", t, j, k)``, uniform
                       without replacement) and evaluate the ordinary
                       V-statistic on the stacked fresh+replay batch;
        ``weighted``   use every cached row with per-group weights
                       ``W_k ~ decay^k`` in a weighted V-statistic (fixed
                       bandwidth only).
    ``decay``
        Geometric decay ``lambda``: the step-``t+k`` group carries weight
        ``~ lambda^k`` (``k = 0`` is the fresh group).  ``0.0`` disables
        replay exactly (fresh-only batch).
    ``depth``
        Number of past steps buffered (``k = 1..depth``).
    ``batch_total``
        ``> 0``: fixed total MMD rows ``B``; the engine's ``n_t`` must be set
        to the fresh count ``replay_counts(B, decay, depth)[0]`` (calls-saving
        arm).  ``0``: augment mode -- fresh count stays the engine's ``n_t``
        and replay rows are added on top (``r_k = round(n_t * decay^k)``).
    ``policy`` [M-10]
        Buffer plan used when ``fill=True``: ``geometric`` (top-up split
        ``~ decay^k`` -- the M-9 arm), ``fifo`` (uniform inclusion of the
        most recent ``batch_total - n_t`` rows, oldest evicted), ``cohort``
        (capped-cohort thinning: cohort ``k`` keeps ``min(n_t,
        ceil(batch_total / 2^k))`` rows -- implicit smooth decay, longer
        tail).  ``depth`` must cover the plan (``tfg.replay.fifo_counts`` /
        ``cohort_counts``); ignored when ``fill=False``.
    ``fill`` [M-9]
        With ``batch_total > 0``: instead of deriving the fresh count from the
        geometric split, keep the engine's ``n_t`` as the fresh count and TOP
        UP with ``batch_total - n_t`` recycled rows split ``~ decay^k``
        (``tfg.replay.fill_counts``), clamped by cache availability -- the
        tiny-fresh-budget arm of hypothesis M-9.  Ignored when
        ``batch_total = 0``.
    """
    enabled: bool = False
    mode: str = "subsample"      # subsample | weighted
    decay: float = 0.5
    depth: int = 1
    batch_total: int = 0
    fill: bool = False
    policy: str = "geometric"    # geometric | fifo | cohort   (fill=True only)

    def validate(self):
        if self.mode not in ("subsample", "weighted"):
            raise ValueError(f"unknown replay mode {self.mode!r}")
        if not (0.0 <= self.decay < 1.0):
            raise ValueError("replay decay must lie in [0, 1)")
        if self.depth < 1:
            raise ValueError("replay depth must be >= 1")
        if self.batch_total < 0:
            raise ValueError("replay batch_total must be >= 0")
        if self.fill and self.batch_total <= 0:
            raise ValueError("replay fill=True requires batch_total > 0")
        if self.fill and self.mode != "subsample":
            raise ValueError("replay fill=True is only defined for mode='subsample'")
        if self.policy not in ("geometric", "fifo", "cohort"):
            raise ValueError(f"unknown replay policy {self.policy!r}")
        return self


@dataclass
class BackselConfig:
    """[Agent B] Importance-selected backpropagation (opt-in, off by default).

    Per predictor evaluation: all ``n`` conditional samples are generated
    under ``torch.no_grad()`` (full-batch MMD value and output-space
    gradients ``g_i = dL/dy_i``, kernel-only), then ``k`` samples are
    selected (tape key ``("backsel", t, j)``), regenerated WITH graphs by
    replaying their per-sample eta keys (identical noise), and only those are
    backpropagated: ``dL/dx ~ sum_{i in S} w_i g_i^T dy_i/dx``.  Theory,
    unbiasedness proofs and cost accounting:
    ``experiments/model-optimization/backsel/THEORY.md``; implementation in
    ``tfg/backsel.py`` (the engine itself is untouched -- the wrapper lives
    around the ``log_f`` callable, like ``tfg/replay.py``).

    ``rule``
        ``uniform``     k of n without replacement, weights ``n/k`` (unbiased
                        control);
        ``importance``  k iid draws with ``p_i ~ (1-floor)||g_i||/sum + floor/n``,
                        inverse-probability weights ``c_i/(k p_i)`` (unbiased);
        ``kcenter``     greedy k-center on the ``y_i``, cluster-aggregated
                        output gradient through each center's Jacobian
                        (APPROXIMATE; exact at ``k = n``).
    ``k``
        Number of differentiated samples per evaluation; ``k >= n`` is the
        exact identity (all rules).
    ``floor``
        Mixture floor of the importance distribution (bounds the weights by
        ``n/(k*floor)``); ``importance`` only.
    """
    enabled: bool = False
    rule: str = "uniform"        # uniform | importance | kcenter | stratified | stratified_balanced
    k: int = 4
    floor: float = 0.25
    # --- [B-R7] soft assignment of the non-selected output gradients ---
    weighting: str = "hard"      # hard | soft
    tau_mult: float = 1.0        # tau = tau_mult x the scale chosen by tau_mode
    # [Agent S] tau_mode: "bandwidth" = tau_mult x the loss's bandwidth (synthetic
    # default, unchanged); "local" = tau_mult x median over the non-selected rows
    # of the squared distance to their nearest representative (the SD default:
    # the global bandwidth measures the TARGET spread and is far too large,
    # backsel/REPORT.md sec 7).  ``stratified_balanced``: k-center strata with
    # capacity ceil(n/k), one uniform representative per stratum, weight |C_c|
    # (unbiased, the SD "strat" rule).
    tau_mode: str = "bandwidth"  # bandwidth | local

    def validate(self):
        if self.rule not in ("uniform", "importance", "kcenter", "stratified",
                             "stratified_balanced"):
            raise ValueError(f"unknown backsel rule {self.rule!r}")
        if self.weighting not in ("hard", "soft"):
            raise ValueError(f"unknown backsel weighting {self.weighting!r}")
        if self.tau_mode not in ("bandwidth", "local"):
            raise ValueError(f"unknown backsel tau_mode {self.tau_mode!r}")
        if self.tau_mult <= 0:
            raise ValueError("backsel tau_mult must be positive")
        if self.k < 1:
            raise ValueError("backsel k must be >= 1")
        if not (0.0 < self.floor <= 1.0):
            raise ValueError("backsel floor must lie in (0, 1]")
        return self


@dataclass
class TemporalConfig:
    """Temporal treatment of the rho-branch gradient.

    ``none``   use the raw gradient (ordinary TFG).
    ``adam``   AdamDPS adaptive moments (arXiv:2603.16797v2), applied to the
               rho-branch gradient BEFORE multiplication by rho_t. The engine's
               existing ``/ sqrt(alpha_t)`` on line 9 then reproduces upstream's
               ``x_prev += guidance / alpha_t ** 0.5`` exactly, so
               ``inv_sqrt_alpha`` must stay False on the Adam object itself.
               ``beta1 = 0`` is the normalisation-only rule of Experiment 5A.
    ``lambda`` deployable adaptive temporal mixing (retained, not evaluated).

    [A4] ``grad_norm`` is an INDEPENDENT pre-processing switch applied to the
    raw rho-branch gradient before the temporal operator:
      ``none``           leave it;
      ``clip``           rescale to norm ``grad_clip`` when larger (absolute clipping);
      ``unit``           g / (||g|| + grad_eps)   (direction only; rho sets the step);
      ``clip_rel``       [round 2] scale-free: threshold = ``grad_clip`` x a running
                         statistic of the PAST raw per-step gradient norms
                         (``clip_ref``: ``median`` of all past norms, or ``ema``
                         with factor ``clip_ema``); no clipping on the first step;
      ``clip_quantile``  [round 2] threshold = the ``grad_clip``-quantile (in
                         (0,1)) of all past raw norms; no clipping on the first step.
    [round 2] ``step_clip`` is a trust region on the APPLIED step ``Delta_t``.
    Order of operations in the engine:  grad_rho -> grad_norm pre-processing ->
    temporal operator (none/adam/lambda) -> ``Delta_t = rho_t * grad_used`` ->
    **step_clip rescales Delta_t** -> line 9 adds ``Delta_t / sqrt(alpha_t)``
    (the ``/sqrt(alpha_t)`` of C4a is unchanged and is applied AFTER the clip):
      ``none``  off;
      ``noise`` ||Delta_t|| <= step_tau * sqrt(1 - alphabar_t)   (noise-level relative;
                ``step_tau=1`` is the promoted ``trust_noise1`` rule);
      ``ddim``  ||Delta_t|| <= step_tau * ||x_ddim - x_t||       (relative to the DDIM move);
      ``noise_prev_rms`` [Agent S] ||Delta_t|| <= step_tau * max(sqrt(1-alphabar_{t-1}),
                step_min_noise) * sqrt(numel(Delta_t))  (per-element RMS convention of the
                SD pipeline, shared implementation in ``tfg/trust.py``).
    The clip is a pure rescaling of the direction when the norm exceeds the
    bound, never a change of direction; it does not touch the mu branch.
    """
    mode: str = "none"                # none | adam | lambda
    adam_rho: float = 1.0             # upstream guidance_strength; rho_t also applies
    beta1: float = 0.9
    beta2: float = 0.995              # official default (utils/configs.py)
    delta: float = 1e-8
    lam_max: float = 0.95
    lam_smooth: float = 0.0
    # --- [A4] ---
    grad_norm: str = "none"           # none | clip | unit | clip_rel | clip_quantile
    grad_clip: float = 1.0
    grad_eps: float = 1e-12
    clip_ref: str = "median"          # median | ema   (clip_rel only)
    clip_ema: float = 0.9
    step_clip: str = "none"           # none | noise | ddim | noise_prev_rms
    step_tau: float = 1.0
    # [Agent S] ``noise_prev_rms``: the SD latent convention (tfg/trust.py):
    # ||Delta_t|| <= step_tau * max(sqrt(1-alphabar_{t-1}), step_min_noise) * sqrt(numel).
    step_min_noise: float = 0.0
    # --- [Agent P] gradient preconditioning (off by default) ---
    precond: PrecondConfig = field(default_factory=PrecondConfig)


@dataclass
class TFGConfig:
    """Core TFG hyper-parameters plus the extension switches.

    [A4] legacy-compatibility switches (all default to the Algorithm 1 value):

    ``init``
        ``randn``: ``x_T ~ N(0, I)`` from the tape (line 2).
        ``zeros``: ``x_T = 0``, the repository's ``Optimization.optimize_LGD``
        and ``experiments/_guided.py`` convention.
    ``guidance_scaling``
        ``tfg``: line 9 adds ``Delta_t / sqrt(alpha_t)`` (C4a).
        ``raw``: line 9 adds ``Delta_t`` with no ``1/sqrt(alpha_t)`` -- the
        ``x_{t-1} = DDIM(x_t) - g`` convention of ``optimize_LGD``/``_guided``.
    ``smoothing``
        ``tfg``: perturbation scale ``gamma_bar * sqrt(1 - alphabar_t)`` (line 4).
        ``lgd_beta``: scale ``beta_t / sqrt(1 + beta_t^2)`` -- the repository's
        LGD ``r_t`` (``_guided.py``), independent of ``gamma_bar``.
    """

    # --- core TFG (H_TFG of Definition 3.1) ---
    T: int = 100
    N_recur: int = 1
    N_iter: int = 0
    gamma_bar: float = 0.0
    rho_scalar: float = 0.0
    mu_scalar: float = 0.0
    rho_structure: str = "constant"   # constant | increase | decrease
    mu_structure: str = "constant"
    n_mc: int = 1                     # Monte-Carlo draws for the gamma_bar smoothing

    # --- [A4] legacy-compatibility switches ---
    init: str = "randn"               # randn | zeros
    guidance_scaling: str = "tfg"     # tfg | raw
    smoothing: str = "tfg"            # tfg | lgd_beta

    # --- temporal guidance treatment ---
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    # --- extensions, all off by default ---
    distributional_tfg_enabled: bool = False
    n_schedule: NScheduleConfig = field(default_factory=NScheduleConfig)
    temporal_cache: TemporalCacheConfig = field(default_factory=TemporalCacheConfig)
    adaptive_recurrence: AdaptiveRecurrenceConfig = field(default_factory=AdaptiveRecurrenceConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)  # [Agent M] sample-replay MMD (handled outside the engine, in tfg/replay.py)
    backsel: BackselConfig = field(default_factory=BackselConfig)  # [Agent B] importance-selected backprop (handled outside the engine, in tfg/backsel.py)

    def validate(self):
        if self.N_recur < 1:
            raise ValueError("N_recur must be >= 1")
        if self.N_iter < 0:
            raise ValueError("N_iter must be >= 0")
        if self.n_mc < 1:
            raise ValueError("n_mc must be >= 1")
        if self.init not in ("randn", "zeros"):
            raise ValueError(f"unknown init {self.init!r}")
        if self.guidance_scaling not in ("tfg", "raw"):
            raise ValueError(f"unknown guidance_scaling {self.guidance_scaling!r}")
        if self.smoothing not in ("tfg", "lgd_beta"):
            raise ValueError(f"unknown smoothing {self.smoothing!r}")
        if self.temporal.mode not in ("none", "adam", "lambda"):
            raise ValueError(f"unknown temporal mode {self.temporal.mode!r}")
        tm = self.temporal
        if tm.grad_norm not in ("none", "clip", "unit", "clip_rel", "clip_quantile"):
            raise ValueError(f"unknown grad_norm {tm.grad_norm!r}")
        if tm.grad_norm in ("clip", "clip_rel") and tm.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if tm.grad_norm == "clip_quantile" and not (0.0 < tm.grad_clip < 1.0):
            raise ValueError("clip_quantile needs grad_clip (the quantile) in (0,1)")
        if tm.clip_ref not in ("median", "ema"):
            raise ValueError(f"unknown clip_ref {tm.clip_ref!r}")
        if not (0.0 <= tm.clip_ema < 1.0):
            raise ValueError("clip_ema must lie in [0,1)")
        if tm.step_clip not in ("none", "noise", "ddim", "noise_prev_rms"):
            raise ValueError(f"unknown step_clip {tm.step_clip!r}")
        if tm.step_clip != "none" and tm.step_tau <= 0:
            raise ValueError("step_tau must be positive")
        ns = self.n_schedule
        if ns.eta_keying not in ("per_step", "frozen"):
            raise ValueError(f"unknown eta_keying {ns.eta_keying!r}")
        if ns.type == "adaptive":
            if not ns.enabled:
                raise ValueError("n_schedule.type='adaptive' requires n_schedule.enabled")
            if ns.policy not in ("agreement", "improvement"):
                raise ValueError(f"unknown adaptive policy {ns.policy!r}")
            if ns.n_min < 1 or ns.n_min > ns.n_max:
                raise ValueError("need 1 <= n_min <= n_max")
            if ns.grow < 2:
                raise ValueError("grow must be >= 2")
            if ns.policy == "agreement" and ns.n_max < 2:
                raise ValueError("agreement policy needs n_max >= 2 (two half batches)")
        tc = self.temporal_cache
        if tc.enabled:
            if tc.implementation == "gated":
                raise NotImplementedError(
                    "temporal_cache is not implemented yet; it is gated on Checkpoint 2 "
                    "(set implementation='stale' for the campaign's stale-gradient rule)"
                )
            if tc.implementation != "stale":
                raise ValueError(f"unknown temporal_cache implementation {tc.implementation!r}")
            if tc.refresh_every < 1:
                raise ValueError("refresh_every must be >= 1")
            if not (0.0 <= tc.lambda_value <= 1.0):
                raise ValueError("lambda_value must lie in [0, 1]")
        ar = self.adaptive_recurrence
        if ar.enabled:
            if ar.implementation == "gated":
                raise NotImplementedError(
                    "adaptive_recurrence is not implemented yet; it is gated on Checkpoint 3 "
                    "(set implementation='v1' for the campaign's early-stopping rule)"
                )
            if ar.implementation != "v1":
                raise ValueError(f"unknown adaptive_recurrence implementation {ar.implementation!r}")
            if ar.metric not in ("clean_proxy", "next_state_tweedie", "grad_stability"):
                raise ValueError(f"unknown adaptive_recurrence metric {ar.metric!r}")
            if ar.max_recurrences < 1:
                raise ValueError("max_recurrences must be >= 1")
            if self.N_recur != 1:
                raise ValueError("adaptive_recurrence sets the recurrence bound; leave N_recur=1")
            if ar.threshold < 0:
                raise ValueError("threshold must be >= 0")
        return self

    def all_extensions_disabled(self):
        legacy_default = (self.init == "randn" and self.guidance_scaling == "tfg"
                          and self.smoothing == "tfg"
                          and self.temporal.grad_norm == "none"
                          and self.temporal.step_clip == "none")
        return legacy_default and not (self.distributional_tfg_enabled
                                       or self.n_schedule.enabled
                                       or self.temporal_cache.enabled
                                       or self.adaptive_recurrence.enabled)

    def resolved(self):
        """Plain dict of the fully resolved configuration, for run records."""
        return asdict(self)
