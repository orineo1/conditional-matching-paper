# Importance-selected backpropagation -- Phase 1 (implementation + smoke)

Agent B, round 4, 2026-08-24, branch `tfg-generalization-v2` (working tree on
top of `6af2081`). Pre-registration: `hypotheses/agentB.yaml` (written before
any run). Theory: `backsel/THEORY.md`. Nothing was submitted to the cluster.

## 1. What was built

| item | where |
|---|---|
| mechanism | `simulations/src/tfg/backsel.py`: `output_gradients` (kernel-only `g_i = dL/dy_i` + full-batch value), `select_uniform` (HT weights `n/k`), `select_importance` (`p_i ~ (1-floor)||g_i||/sum + floor/n`, weights `c_i/(k p_i)`, de-duplicated), `select_kcenter` (greedy k-center on `y_i`, cluster-aggregated `g_eff`), `wrap_log_f` (no-grad full forward -> select -> regenerate subset with graphs by replaying the per-sample eta keys -> surrogate whose VALUE is the full-batch `-loss` and whose gradient is the subset estimator). Composition hook `inner=` lets it sit on top of `tfg.replay.wrap_log_f` (Agent C's cohort/fifo/fill top-up): selection over the fresh rows, `g` from the replay-stacked geometry (exact, since recycled rows are constants). |
| config | `tfg/config.py::BackselConfig` (`enabled=False`, `rule`, `k`, `floor`) + `TFGConfig.backsel` field; `all_extensions_disabled()` untouched (default path byte-identical, tested). |
| runner | `estimator/engine_runner.py` (append-only): candidates `backsel_{uni,is,clust}_k<K>[_cohort<B>|_fifo<B>][_trust]`; opt-in wrapper hook after the replay hook; `diff_samples` (differentiated samples) per run and `diff_samples_mean` per cell next to `cm_samples` (forwards). |
| tests | `simulations/tests/test_backsel.py`, 19 tests; full suite **402 passed, 1 skipped** (was 383). |
| screening | `backsel/cells.py` (102 cells: 2D/5D/10D x n in {8,32} x {baseline, trust_noise1, 6 plain backsel arms, 6 `_trust` arms, `replay_cohort{2n}_trust`, `backsel_{is,clust}_k4_cohort{2n}_trust`}; fast MMD backend; `report` writes `backsel_tables.md` + `backsel_rows.csv` with score, success, divergence, fwd/run, diff_s/run, s/run, peak RSS, paired diff+p vs comparator and vs baseline) and `backsel/submit_backsel.sh` (sbatch pattern of `estimator/submit_screen.sh`). |

## 2. Exactness gates (all pass)

* **Noise replay is bit-identical; row regeneration is round-off-exact, not bit-exact.**
  `test_cmsampler_subset_regeneration_bit_identical`: for the real 2D
  checkpoint sampler, `CMSampler._noise` on a key subset equals the
  corresponding columns of the full-batch noise exactly (`torch.equal`), and
  the regenerated rows equal the no-grad rows to `rtol=atol=1e-5` -- measured
  max |diff| 1.9e-6 (n=8) / 6.2e-6 (n=32) in float32 (rel 1.1e-6), 5e-14 in
  float64. They are NOT bit-identical because CPU BLAS blocking depends on
  the batch dimension (the FULL key set does regenerate bit-identically).
  The surrogate is linear in the regenerated rows with `g_i` fixed, so the
  round-off only moves the evaluation point of `J_i`, far below the
  estimator's own noise (SNR < 1). The toy tape-keyed sampler regenerates
  bit-identically.
* **k = n reproduces the full gradient and value to 1e-12** (float64, both
  MMD backends, all three rules, also `k > n`).
* **Unbiasedness (statistical):** uniform and importance: the mean over 3000
  tape-keyed selection draws of `sum_{i in S} w_i J_i^T g_i` equals the full
  gradient within 4 SE (and < 5% relative). IS weights verified to be
  `c_i/(k p_i)` with `sum c_i = k`; on a synthetic heavy-tailed `g` (one row =
  90% of the norm) IS halves the uniform estimator's variance.
* **k-center** conserves output-gradient mass (`sum g_eff = sum g`), distinct
  centers, deterministic given the tape.
* **Off by default:** `enabled=False` wrapper returns the plain path (value
  equality, `torch.equal` gradients); through the engine, backsel fields set
  but `enabled=False` gives `torch.equal` final `x` and identical
  `cm_samples`; `backsel_uni_k8` at n=8 (k=n) reproduces the baseline
  trajectory to 1e-5 in float32.
* **Cost accounting through the engine:** uni/clust `cm_samples = (n+k) T`,
  `diff_samples = k T`; IS `diff_samples <= k T` (repeated draws collapse) and
  `cm_samples - nT == diff_samples`. The cohort composition runs and counts
  the same way.

## 3. Smoke (local, 2D n=8, restarts 0..9, fast backend, `backsel/smoke/`)

| candidate | score | succ | fwd/run | diff_s/run | s/run | RSS MB | vs comparator (p) | vs baseline (p) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.301 | 30% | 792 | 792 | 0.11 | 310 | - | - |
| trust_noise1 | 0.224 | 70% | 792 | 792 | 0.11 | 313 | +0.077 (0.31) | +0.077 (0.31) |
| backsel_uni_k2_trust | 0.136 | 80% | 990 | 198 | 0.15 | 316 | +0.088 (0.26) | +0.164 (0.03) |
| backsel_is_k2_trust | 0.237 | 80% | 974 | 182 | 0.15 | 318 | -0.012 (0.91) | +0.064 (0.46) |
| backsel_clust_k2_trust | 0.121 | 90% | 990 | 198 | 0.15 | 315 | +0.103 (0.27) | +0.180 (0.03) |
| replay_cohort16_trust (Agent C) | 0.327 | 60% | 792 | 792 | 0.11 | 309 | -0.103 (0.39) | -0.026 (0.82) |
| backsel_is_k4_cohort16_trust | 0.196 | 60% | 1106 | 314 | 0.15 | 316 | +0.132 (0.21) | +0.105 (0.12) |

Reading (10 restarts: nothing is significant against the 0.08 seed-noise
floor; this is a smoke, not a result):

* All arms run, no divergence; the two currencies separate as designed:
  differentiated samples drop 4x (792 -> 198) while forwards rise 25%
  (792 -> 990), exactly `n + k` per step.
* Quality did not collapse at the harshest point (k=2 of n=8, under trust):
  uni/clust are numerically better than trust_noise1 on these 10 seeds and
  IS is level. The pre-registered expectation (B-1) was a measurable LOSS
  here; either the trust region absorbs the selection noise (B-4) or 10
  restarts are not enough -- the 40-restart grid decides. The +0.16/+0.18 vs
  baseline (p=0.03) is NOT a finding: the comparator for `_trust` arms is
  trust_noise1 (p=0.26).
* **Wall time went UP on this benchmark** (0.11 -> 0.15 s/run): the
  conditional model is a tiny MLP, so the saved backward is negligible while
  the mechanism adds a second forward of k rows and a second kernel
  evaluation (`output_gradients`). Peak RSS unchanged (~315 MB, torch
  itself). The memory/backward win exists only where the differentiated
  sampler is expensive (SD sprinter); on the synthetic benchmark the only
  claim available is "quality at k differentiated samples", which is what
  the grid measures.

## 4. What the grid answers (pre-registered in `hypotheses/agentB.yaml`)

B-1 uniform control; B-2 IS vs uniform ordering (heavy tail of `||g_i||`,
estimator/REPORT.md sec 4); B-3 clust vs IS (Jacobian-substitution bias
diagnostic); B-4 trust interaction; cohort composition vs Agent C's
`replay_cohort{2n}_trust`. Rejection criteria: THEORY.md sec 5. Held-out
verification (offset >= 1000) is the verifier's.

## 5. Commands

```
cd simulations && /Users/stolk/miniconda3/bin/python -m pytest tests -q            # 402 passed, 1 skipped
cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/backsel/cells.py run --only trust_noise1 backsel_is_k2_trust --settings 2D --ns 8 --restarts 10 --dir smoke
# the grid (cluster, NOT submitted) -- sync the round-4 working tree to cdm-perf/ first
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
N=$(python ../experiments/model-optimization/backsel/cells.py list 2>/dev/null | wc -l)   # 102
sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/backsel/submit_backsel.sh
python ../experiments/model-optimization/backsel/cells.py report
```
