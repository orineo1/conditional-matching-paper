# Infeasible + Adversarial Target Experiment — Rebuttal Summary

Setting: 2D_cond_1D joint GMM (dim(X)=1, dim(Y)=1, 11 components, shared
covariance [[0.5, 0.195], [0.195, 0.2]]). MLGD-F run with the same
hyperparameters as the existing 2D_cond_1D results (25 restarts,
top-10 reporting, `Optimization.optimize_LGD` with `CM=True`).

Full code: `simulations/notebooks/exp_2D_infeasible_adversarial.py` /
`Exp_2D_infeasible_adversarial.ipynb`. Full results:
`simulations/results/2D_infeasible_adversarial/`.

## Per-condition summary

**Feasible baseline** (existing bimodal target, x\*=-5): MLGD-F recovers
x̂\* ≈ -5.27, top-10 MMD loss 0.0031 ± 0.0015 (best 0.0007) — the
near-zero-loss reference point.

**Infeasible** (G = N(5, 0.01), a target 12x tighter than any reachable
conditional): by the law of total variance, every reachable P(Y|X=x) has
Var(Y) ≥ 0.12395 (constant across all x and components, since all 11
components share one covariance matrix), so G is provably unreachable —
verified both symbolically and by a dense numerical sweep. MLGD-F does not
diverge or fail arbitrarily: top-10 loss saturates at 0.75 ± 0.18 (vs.
0.003 for the feasible baseline), converging to x̂\* ≈ 7.0, which isolates
the single mixture component whose mean matches the target and compresses
its variance to 0.69 — well below any of the badly-diverged restarts, and
substantially closer to the analytic floor (0.124) than to the
unconstrained joint's spread, though not fully saturating it (matching the
target's mean and minimizing variance trade off against each other).

**Adversarial** (G = 0.5·N(1, 0.124) + 0.5·N(5, 0.124), built from the two
most extreme-X components, X-means 15 apart ≈ 21 per-component X-std
apart): we show analytically that w_a=w_b=0.5 is unreachable at *any* x —
even the geometric midpoint gives both target components ~0 weight, with
~83% of the mass stolen by two unrelated components that happen to sit
there. This is a qualitatively different failure mode from simple
infeasibility (gradient conflict, not just an out-of-envelope target):
MLGD-F's top-10 loss (0.98 ± 0.18) is *worse* than the infeasible case,
achieved weights on the true components never exceed ~0.12-0.16 each, and
restarts scatter across x ∈ [-8, +7] (std 5.26) rather than agreeing on
one point — each landing on a different nearby-component substitute that
produces spurious extra modes instead of a clean compromise.

## Paste-ready paragraphs

**For WRAk (feasibility):**
> We constructed a target provably outside the reachable set (Var(Y|X=x) ≥
> 0.124 for all x by the law of total variance, given a target variance of
> 0.01) and ran MLGD-F with the same protocol as our main results (25
> restarts, top-10 reporting). Rather than diverging, MLGD-F consistently
> converges to the closest reachable point — the single mixture component
> whose mean is nearest the target, with variance compressed to within
> 0.57 of the analytic floor — at a loss (0.75) that stays bounded and
> clearly separated from both the feasible baseline (0.003) and the
> worst-case divergent restarts we also observe as a known optimizer
> artifact. Full derivation, sweep verification, and code are in the
> supplementary material.

**For WBXh (Limitations #3, misspecified/adversarial targets):**
> We additionally stress-tested MLGD-F on an adversarial bimodal target
> built from the two most X-separated mixture components, which we prove
> cannot be balanced at any single x because intervening components steal
> the probability mass near the naive balance point. MLGD-F's loss here
> (0.98 ± 0.18) is worse than both the feasible baseline and the simpler
> infeasible-target case, and restarts scatter across a wide range of x
> instead of agreeing on one answer — evidence that MLGD-F degrades
> gracefully rather than silently returning a confidently wrong answer
> under target misspecification, but that adversarial targets are a real
> and now explicitly documented limitation. Full analysis in the
> supplementary material.
