# Agent P -- preconditioning the guidance gradient: theory survey

Round 3 of the CDM/TFG performance campaign, 2026-08-24, branch
`tfg-generalization-v2`. Object of study: the rho-branch gradient

    g_t = grad_{x_t} MMD^2( S(x_t), S_G ),      t = 99 .. 1,

where `S(x_t)` is the set of n conditional-model draws at `x_{0|t}` (through
the denoiser), `x` has spatial dimension d = 1..9 (2D/5D/10D settings), and the
applied step is `Delta_t = rho_t * g_t` (optionally trust-capped, then
`/sqrt(alpha_t)` unless `guidance_scaling="raw"`).

Ground truth about the regime (estimator/REPORT.md sec 4, measured):

* **SNR < 1 at every n** (0.07-0.27 at n<=8, 0.37-0.61 at n=32): the per-step
  gradient is noise-dominated; a single draw has the wrong sign 20-40% of the
  time.
* **Heavy tail**: within-run p90 of ||g|| is ~6-8x the median, the max
  100-500x. The occasional huge step is the divergence/penalty tail.
* **Dimension scaling**: raw-norm medians 2D 0.04-0.09, 5D 0.09-0.25, 10D
  0.29-0.38 -- the gradient does not shrink with n in 10D and is 3-8x the 2D
  one at an identical step multiplier.
* **Verified champion**: `trust_noise1` -- cap `||Delta_t|| <= sqrt(1-alphabar_t)`.
  A norm cap, never a direction change. Verified graveyard: Adam / norm_only
  (fixed-magnitude steps) fail 10D; absolute clips do not transfer; variance
  reduction that leaves the step rule alone (CRN/antithetic) does nothing.

"Preconditioning" here = replacing `g_t` by `P_t g_t` for some (possibly
history-dependent) linear or nonlinear map `P_t`, before `rho_t` and before the
trust cap. The graveyard dictates one design axiom used throughout:

> **Axiom N (no inflation).** A preconditioner may rotate or shrink but must
> never systematically inflate the step norm. Fixed-magnitude output
> (`||P g|| ~ const`) is what killed Adam/norm_only/unit in 10D: at n=32 near
> the optimum the raw step is already correctly small and a floored step
> over-shoots. Direction fixes are therefore made **norm-preserving**
> (`||P g|| = ||g||`) or naturally norm-robust (median), and step-SIZE control
> is left to the raw norm or to the trust cap (sec f).

---

## (a) Diagonal second-moment (RMSProp / Adam) -- why it failed, what must change

Adam's update is `rho * m_hat / (sqrt(v_hat) + delta)`: each coordinate of the
output has magnitude ~1 whenever the gradient is persistent, so the step norm
is ~`rho * sqrt(d)` **regardless of ||g||**. Round-1/2 data: norm_only
(beta1=0) and Adam win in 2D at n<=8 (the floor rescues a too-small noisy
step AND the tail is capped) but lose in 10D at n=32 (-0.12..-0.18): there the
raw gradient is already 3-8x larger and near the data end the correct step is
tiny -- the floor becomes systematic over-stepping. Clipping before Adam
cannot fix it (sec 3.3: never helps) because the problem is the *output*
magnitude, not the input tail.

**What a diagonal preconditioner must do differently**: keep the per-coordinate
*relative* equalisation, discard the absolute magnitude. That is

    P_diag g = ||g|| * u / ||u||,   u = g / sqrt(v_reg),
    v_t = beta v_{t-1} + (1-beta) g_t^2   (past steps only, causal),
    v_reg = v + eps * mean(v).

`||P g|| = ||g||` exactly, so Axiom N holds; near the optimum the step shrinks
with the raw gradient like the baseline does; the only change is the
direction. Candidate `precond_diag`. Expected: recovers the part of the 2D
gain that came from direction equalisation, none of the part that came from
magnitude flooring -- the experiment cleanly separates the two explanations.
Failure mode: at SNR<1, `v` estimates mostly the noise variance per
coordinate; if the noise is isotropic (plausible: the CM head mixes
coordinates) the rule is the identity plus estimation jitter. Rejection: no
p<=0.05 win anywhere, or any significant loss (esp. 10D -- a 10D loss would
mean even the *direction* part of Adam is harmful).

## (b) Full-covariance / low-rank whitening (Shampoo / full-matrix AdaGrad)

With d<=9 the full second-moment matrix is trivially affordable:

    C_t = beta C_{t-1} + (1-beta) g_t g_t^T          (d x d, EMA, causal),
    C_reg = C_t + eps * (tr C_t / d) * I,
    P_cov g = ||g|| * w / ||w||,   w = C_reg^{-1/2} g   (eigh, float64).

This is full-matrix AdaGrad with an EMA window, i.e. the whitening direction
Shampoo approximates by Kronecker factors; at d<=9 no approximation is needed
(eigh of a 9x9 is microseconds). Statistical reading: `C_t` estimates
`E[g g^T]` = (empirical, uncentered) covariance of the gradient *sequence*;
whitening equalises exploration across the eigen-directions of the
noise+signal mix -- the correct move when the MMD landscape (through the CM
network) is anisotropic in a non-axis-aligned way, which the diagonal rule
cannot see. Norm restored per Axiom N; identity during a `warmup` period
(EMA of <2 outer products is rank-deficient; the ridge covers it but the
rotation is then pure noise). Candidate `precond_cov`.

Failure modes: (i) at SNR<1, `E[g g^T] ~ Cov(noise)` + small signal outer
product -- whitening *removes* the signal direction preferentially
(the signal contributes an eigenvalue bump exactly along `E g`, which
whitening divides down). This is the classic argument why AdaGrad-style
methods can hurt in high noise; it predicts `precond_cov` <= `precond_diag`.
(ii) One tail event (100-500x median) dominates the EMA for ~1/(1-beta) steps
and freezes the rotation onto a noise direction. Rejection as in (a).

## (c) Schedule-aware preconditioning

Scale `g_t` by powers of the schedule: `P g = s_t^p * g`, with
`s_t = sqrt(1-alphabar_t)` (noise amplitude) or `1/sqrt(alpha_t)` (the C4a
line-9 factor, already applied after the clip and NOT part of the gradient).
Two observations make this a non-candidate on its own:

1. `trust_noise1` already implements the *capped* version:
   `Delta -> Delta * min(1, tau s_t / ||Delta||)`. For any pure rescaling
   `Delta -> c_t Delta` with `c_t <= tau s_t / ||Delta||` the cap dominates it
   pointwise: when the raw step is small the rescaling shrinks a step that was
   not over-shooting (pure loss of signal), when it is large both rules give
   `~tau s_t` in norm. The cap is the rescaling with the "only when needed"
   qualifier -- and it is verified. A multiplicative `s_t^p` with p>0 is an
   annealing *prior* (guide hard early, gently late); the cap is the same
   prior enforced only on the tail.
2. Any deterministic scalar `c_t` commutes with everything else and is
   equivalent to re-shaping `rho_t` (`rho_structure` already exposes
   increase/decrease). Screening it would re-run Experiment 5's rho-structure
   sweep under a new name.

Conclusion: schedule-awareness enters this round only through the trust cap
(sec f), not as a separate arm. (Kept in the config as nothing: no switch.)

## (d) Gauss-Newton / natural gradient for the MMD -- the kernel-metric preconditioner

The Gauss-Newton Hessian of `L = MMD^2(S(x), S_G)` w.r.t. `x` through the
sampler is `J^T H_S J` with `J = dS/dx` (n*d_y x d) and `H_S` the MMD Hessian
in sample space -- intractable per step (n extra Jacobian products through the
CM network and the denoiser). The *empirical Fisher* substitute is standard:
because the biased V-statistic decomposes over rows of X,

    MMD^2 = (1/n^2) sum_{i,j} k(X_i,X_j) - (2/nm) sum_{i,l} k(X_i,Y_l) + c(Y),

define the per-row contribution (fixed kernel matrix rows)

    L_i = (1/n) sum_j k(X_i, X_j)/n ... collected so that L = sum_i L_i,
    g_i = dL_i/dx = (dL/dX_i)^T (dX_i/dx),     g = sum_i g_i,

each `g_i` a d-vector: row i's share of the gradient, propagated through row
i's sampler path only. The empirical Fisher is

    F = sum_i g_i g_i^T   (d x d),      P_F g = (F + eps I)^{-1/2} g,

the natural-gradient metric of the per-sample gradient distribution *within a
single step* -- unlike (b), no temporal mixing, so no staleness and no
tail-contamination across steps. Derivation of cost: the reverse pass computes
`g = sum_i g_i` in one sweep; separating the n contributions needs n VJPs
(`grad_outputs = one-hot row masks`) or d forward-mode JVPs to get the full
`J` -- either way ~n (or d) times the single backward. Cheap in FLOPs at
d<=9/n<=32 but a 3-10x wall-time change to the inner loop and an engine
surface beyond "apply P to the finished gradient" (the loss must expose
per-row contributions).

**Status: derived, deferred.** Screened only if a P-1..P-4 direction rule
shows a win that looks curvature-limited. Predicted failure mode meanwhile:
`F/n` estimates `E[g_i g_i^T]`, which at SNR<1 is again noise-dominated; and
`P_F` shares whitening's signal-suppression problem (i). Its one real
advantage over (b) is within-step freshness.

## (e) Curvature-free direction fixes

**Sign-SGD.** `P g = sign(g) * ||g|| / sqrt(#nonzero)` (norm-preserving per
Axiom N; classic sign-SGD's fixed `rho * sign(g)` is Adam-without-memory and
dies by the same 10D argument). Rationale: at per-coordinate SNR<1 the
magnitude is mostly noise while the sign is right 60-80% of the time (measured
sign-flip probability 0.2-0.4); discarding magnitudes is a max-heavy-tail-
robust move (the influence function of the sign is bounded). In d=1 (the 2D
setting's spatial x) it is the identity -- 2D cells are a plumbing control.
Failure mode: genuine cross-coordinate scale differences are erased; biased
direction whenever `E g` is not sign-aligned with its coordinate-wise medians.
Candidate `precond_sign`.

**Median-of-means across the n per-row gradients.** The estimator: split the
rows into k groups of size n/k, `gbar_b = (k/n) sum_{i in group b} g_i`,
output per-coordinate `median_b(gbar_b)`. With the V-statistic decomposition
of (d) the group gradients are computable by k extra backward passes over
row-masked losses (`L_b = sum_{i in b} L_i` -- note the XX block couples
groups; using `k(X_i, X_j)` with j ranging over ALL rows keeps `sum_b L_b = L`
and each `L_b` a valid 1/k-share). MoM has breakdown point ~1/(2k) of
arbitrary row corruption and gets sub-Gaussian deviation bounds under only a
second moment -- exactly the medicine for the documented heavy tail. Cost:
k extra backward passes per step (k=4: ~4x autograd wall time, conditional
calls unchanged). **Deferred with (d)** for the same engine-surface reason.

**The temporal proxy implemented instead** (`precond_median`): per-coordinate
median over a sliding window of the last w=5 raw gradients (current
included). The window entries are i.i.d.-ish draws of the same noisy
estimator at slowly-moving `x_t` (the trajectory moves ~||Delta|| per step,
small vs the noise sd 0.2-0.5), so the median-of-window is a MoM across time:
breakdown 40% at w=5 (a 500x tail step is simply ignored -- EMA momentum only
dilutes it by 1-beta), and averaging ~w draws raises effective SNR by
~sqrt(w) ~ 2.2, comparable to Adam's beta1 averaging but robust and with no
magnitude floor. The output's norm is the coordinate-wise median's own norm
(naturally tail-robust; not restored to ||g_t||, which would re-import the
tail). Failure modes: w-step lag biases the direction where the true gradient
rotates fast (late steps, small t); per-coordinate median of a skewed
distribution under-estimates the drift; at n=32 the raw estimator is already
decent and the lag may cost more than the robustness buys.

## (f) Combination with the trust cap

Every candidate ships in two arms: alone, and `_trust` = composed with the
verified champion (`step_clip="noise"`, `step_tau=1`), order

    g -> P g -> * rho_t -> trust cap on ||Delta|| -> /sqrt(alpha_t) -> apply.

Logic: P fixes the *direction* (and, for the median, the estimator), the cap
fixes the *size tail*. These are orthogonal failure modes of the baseline, so
gains should be roughly additive; `P_median + cap` is the pre-registered best
guess (robust estimator + annealed bound). Decision rule per candidate pair:
if `P+trust` is not better than trust_noise1 alone anywhere, P adds nothing
over the champion and is rejected regardless of its stand-alone wins (the
stand-alone comparison to baseline mainly serves mechanism attribution).

## Summary table

| rule | state | norm policy | expected gain | main risk | screened |
|---|---|---|---|---|---|
| (a) diag RMS, norm-preserving | EMA v (causal) | = ||g|| | separates Adam's direction vs magnitude effect; 2D n<=8 | noise isotropy -> identity + jitter | yes (`precond_diag[_trust]`) |
| (b) full-cov whitening | EMA C (causal) | = ||g|| | non-axis-aligned anisotropy, 5D/10D | whitening suppresses the signal direction; tail owns the EMA | yes (`precond_cov[_trust]`) |
| (c) schedule power | none | scalar | none beyond trust cap | dominated by the cap | no (subsumed) |
| (d) empirical Fisher (per-row) | within-step | (F+eps)^{-1/2} | fresh within-step metric | n extra VJPs; same signal-suppression | derived, deferred |
| (e1) sign-SGD, norm-preserving | none | = ||g|| | per-coordinate tail robustness | erases true scale; identity in d=1 | yes (`precond_sign[_trust]`) |
| (e2) MoM over rows | within-step | median norm | sub-Gaussian estimator under heavy tail | k extra backwards | derived, deferred |
| (e2') median over time window | window w=5 | median norm | tail kill + sqrt(w) SNR gain, no floor | w-step lag, skew bias | yes (`precond_median[_trust]`) |
| (f) any + trust cap | -- | capped | additive direction+size fix | none new | yes (`*_trust`) |

Global rejection criteria (pre-registered in `hypotheses/agentP.yaml`):
matched conditional calls always; a candidate must win vs baseline with
p <= 0.05 somewhere, never lose significantly anywhere, and its `_trust` form
must not be dominated by trust_noise1 -- otherwise it joins the graveyard.
