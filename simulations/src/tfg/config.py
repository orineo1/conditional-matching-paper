"""Configuration for the generalised TFG engine.

Every extension is off by default.  With the defaults, ``tfg.engine`` executes
plain TFG Algorithm 1 and must match ``tfg.reference`` exactly; that is what
``tests/test_equivalence.py`` asserts.

Components 2 (temporal gradient cache) and 3 (improvement-adaptive recurrence)
have their configuration surface defined here so that the shape of the design
space is fixed, but the engine raises ``NotImplementedError`` if they are
enabled.  They are deliberately not implemented before their respective
checkpoints.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class NScheduleConfig:
    """Component 1: adaptive conditional sample count."""
    enabled: bool = False
    type: str = "constant"       # constant | time | noise
    n_max: int = 1
    kappa: float = 1.0


@dataclass
class TemporalCacheConfig:
    """Component 2: temporal reuse of guidance information. NOT YET IMPLEMENTED."""
    enabled: bool = False
    type: str = "gradient"       # gradient (kernel-statistics caching is design-only)
    lambda_mode: str = "fixed"   # fixed | adaptive
    lambda_value: float = 0.0


@dataclass
class AdaptiveRecurrenceConfig:
    """Component 3: improvement-adaptive N_recur. NOT YET IMPLEMENTED."""
    enabled: bool = False
    max_recurrences: int = 1
    threshold: float = 1e-2
    metric: str = "next_state_tweedie"   # clean_proxy | next_state_tweedie


@dataclass
class TemporalConfig:
    """Temporal treatment of the rho-branch gradient.

    ``none``   use the raw gradient (ordinary TFG).
    ``adam``   AdamDPS adaptive moments (arXiv:2603.16797v2), applied to the
               rho-branch gradient BEFORE multiplication by rho_t. The engine's
               existing ``/ sqrt(alpha_t)`` on line 9 then reproduces upstream's
               ``x_prev += guidance / alpha_t ** 0.5`` exactly, so
               ``inv_sqrt_alpha`` must stay False on the Adam object itself.
    ``lambda`` deployable adaptive temporal mixing (retained, not evaluated).
    """
    mode: str = "none"                # none | adam | lambda
    adam_rho: float = 1.0             # upstream guidance_strength; rho_t also applies
    beta1: float = 0.9
    beta2: float = 0.995              # official default (utils/configs.py)
    delta: float = 1e-8
    lam_max: float = 0.95
    lam_smooth: float = 0.0


@dataclass
class TFGConfig:
    """Core TFG hyper-parameters plus the extension switches."""

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

    # --- temporal guidance treatment ---
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    # --- extensions, all off by default ---
    distributional_tfg_enabled: bool = False
    n_schedule: NScheduleConfig = field(default_factory=NScheduleConfig)
    temporal_cache: TemporalCacheConfig = field(default_factory=TemporalCacheConfig)
    adaptive_recurrence: AdaptiveRecurrenceConfig = field(default_factory=AdaptiveRecurrenceConfig)

    def validate(self):
        if self.N_recur < 1:
            raise ValueError("N_recur must be >= 1")
        if self.N_iter < 0:
            raise ValueError("N_iter must be >= 0")
        if self.n_mc < 1:
            raise ValueError("n_mc must be >= 1")
        if self.temporal.mode not in ("none", "adam", "lambda"):
            raise ValueError(f"unknown temporal mode {self.temporal.mode!r}")
        if self.temporal_cache.enabled:
            raise NotImplementedError(
                "temporal_cache is not implemented yet; it is gated on Checkpoint 2"
            )
        if self.adaptive_recurrence.enabled:
            raise NotImplementedError(
                "adaptive_recurrence is not implemented yet; it is gated on Checkpoint 3"
            )
        return self

    def all_extensions_disabled(self):
        return not (self.distributional_tfg_enabled
                    or self.n_schedule.enabled
                    or self.temporal_cache.enabled
                    or self.adaptive_recurrence.enabled)

    def resolved(self):
        """Plain dict of the fully resolved configuration, for run records."""
        return asdict(self)
