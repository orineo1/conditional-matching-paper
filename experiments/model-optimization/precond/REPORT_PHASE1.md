# Agent P -- preconditioning: Phase-1 report (theory, implementation, smoke)

Round 3 of the CDM/TFG performance campaign, 2026-08-24, branch
`tfg-generalization-v2`. Pre-registration: `../hypotheses/agentP.yaml`
(written before any run). Theory survey: `THEORY.md`. Nothing has been
submitted to the cluster; the smoke cells below are the only runs.

## 1. What was implemented

New module `simulations/src/tfg/precond.py` (`GuidancePreconditioner`,
`make_preconditioner`), configured by the new
`tfg.config.PrecondConfig` dataclass exposed as `TemporalConfig.precond`
(mode `none` by default -> `make_preconditioner` returns `None` and the engine
path is byte-identical to the frozen reference; the pre-existing equivalence
tests plus `test_precond.py::test_engine_default_path_identical_with_precond_field_present`
enforce it). Engine hook: one call site in `GeneralizedTFG` between the
`grad_norm` pre-processing and the temporal operator; `rho_t`, `step_clip` and
the line-9 `/sqrt(alpha_t)` are untouched. All four modes are deterministic,
causal, float32-safe (statistics in float64, output cast back), O(d)-O(d^3)
with d <= 9, and cost ZERO extra conditional calls.

| candidate | mode | rule | norm policy |
|---|---|---|---|
| `precond_cov[_trust]` | `whiten` | EMA(0.9) of past `g g^T`, ridge `1e-6 tr(C)/d`, direction `C^{-1/2} g`; identity for the first 5 steps | rescaled to raw `||g||` |
| `precond_diag[_trust]` | `diag` | EMA(0.9) of past `g^2`, direction `g/sqrt(v_reg)`; identity for 5 steps | rescaled to raw `||g||` |
| `precond_sign[_trust]` | `sign` | `sign(g) * ||g||/sqrt(#nonzero)`, stateless | exactly `||g||` |
| `precond_median[_trust]` | `median` | per-coordinate median of the last 5 raw gradients (current incl.) | the median's own (tail-robust) norm |

`_trust` composes with the verified champion (`step_clip="noise"`,
`step_tau=1.0`). The norm-preserving design implements Axiom N of THEORY.md:
never inflate the step (the verified Adam/norm_only/unit 10D failure); change
only the direction (or, for `median`, the estimator), leave size control to
the raw norm / trust cap. The empirical-Fisher (per-row) and
median-of-means-over-rows preconditioners are derived in THEORY.md (d)/(e)
but deferred: they need n (or k) extra backward passes per step and a larger
engine surface than this round's hook.

Tests: `simulations/tests/test_precond.py` (14) -- known-sequence outputs for
every mode (hand-computed whiten/diag algebra, sign norm identity, median
outlier rejection), causality/warmup, rotation invariance of `whiten` (which
`diag` cannot have), state reset, determinism, float32 round-trip, engine
integration (runs, differs from baseline, identical call counts), and the
default-path identity. Full suite: **366 passed, 1 skipped** (up from 348;
concurrent Agent M additions included).

Screening plumbing: candidate names appended to
`estimator/engine_runner.py::candidate_spec` (`precond_{cov,diag,sign,median}[_trust]`);
driver `precond/cells.py` (list/run/report; 90 cells = 2D/5D/10D x n in
{4,8,32} x 10 candidates incl. `baseline` and `trust_noise1` comparators in
every setting, paired restarts 0..39, runs skipped if present);
`precond/submit_precond.sh` (glacier, one cell per array task, same
header/conda pattern as `estimator/submit_screen.sh`).

## 2. Smoke results (local, 2D n=8 and 5D n=8, 10 restarts, offset 0)

Failure-penalised L2, 10 restarts only -- plumbing proof, NOT evidence
(40-restart seed-noise floor is ~0.08; at 10 restarts it is larger):

| candidate | 2D n=8 | 5D n=8 |
|---|---|---|
| baseline | 0.420 | 0.575 |
| trust_noise1 | 0.146 | 0.603 |
| precond_cov | 0.420 (== baseline, exact) | 0.494 |
| precond_cov_trust | 0.146 (== trust_noise1, exact) | -- |
| precond_diag | 0.420 (== baseline, exact) | 0.555 |
| precond_diag_trust | 0.146 (== trust_noise1, exact) | -- |
| precond_sign | 0.420 (== baseline, exact) | 0.656 |
| precond_sign_trust | 0.146 (== trust_noise1, exact) | -- |
| precond_median | **0.094** | -- |
| precond_median_trust | **0.066** | 0.761 |

All cells: 0 divergences, cm_samples = 792 = baseline exactly, 0.20 s/run
(baseline 0.20 s/run -- overhead unmeasurable).

Two structural findings the smoke run confirms:

1. **In the 2D setting the spatial x is 1-dimensional**, so every
   norm-preserving direction rule (cov/diag/sign) is mathematically the
   identity there -- and the scores are bit-identical to their comparators,
   which is a strong end-to-end correctness check of the whole plumbing. The
   informative settings for those three are 5D (d=4) and 10D (d=9); their 2D
   columns in the screen are controls that must come out exactly 0.000 diff.
2. **`precond_median` is the only rule that acts in d=1, and the smoke signal
   is large**: 0.094 vs baseline 0.420 and vs champion trust_noise1 0.146;
   `precond_median_trust` 0.066 with 100% success. Direction consistent with
   the pre-registered P-4 hypothesis (tail kill + ~sqrt(w) SNR gain, no
   magnitude floor). The 5D smoke values scatter both ways at 10 restarts
   (median_trust 0.76, cov 0.49) -- no conclusions before the 40-restart
   paired screen.

Smoke JSONs: `precond/smoke/*.json` (kept out of `runs/` so the cluster
skip-if-present logic is unaffected).

## 3. What to expect from the screen (pre-registered)

* `precond_median[_trust]`: the primary bet; 2D gains at all n, possibly
  beating trust_noise1; risk at n=32 and late steps (lag bias).
* `precond_diag`: separates Adam's direction effect from its magnitude
  effect; expect small 2D/5D gains, and crucially NO 10D loss (if 10D loses,
  the diagonal direction itself is harmful and the Adam graveyard entry gets
  a sharper epitaph).
* `precond_cov`: <= diag if gradient noise is near-isotropic (the whitening
  signal-suppression argument); any win would indicate real non-axis-aligned
  anisotropy in 5D/10D.
* `precond_sign`: identity in 2D; 5D/10D robustness probe.
* Rejection rules and the trust-domination criterion: `hypotheses/agentP.yaml`.

## 4. How to run the screen (boss submits; nothing submitted yet)

Prerequisite: sync the working tree to the cluster mirror
`/sci/labs/orzuk/shaulytolk/cdm-perf/` (changed files: `simulations/src/tfg/{precond.py,config.py,engine.py}`,
`simulations/tests/test_precond.py`, `experiments/model-optimization/estimator/engine_runner.py`,
`experiments/model-optimization/precond/**`).

```
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
N=$(python ../experiments/model-optimization/precond/cells.py list 2>/dev/null | wc -l)   # 90
sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/precond/submit_precond.sh
# afterwards:
python ../experiments/model-optimization/precond/cells.py report   # -> precond_tables.md, precond_rows.csv
```
