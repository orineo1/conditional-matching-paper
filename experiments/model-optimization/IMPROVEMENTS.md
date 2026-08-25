# Verified improvements (campaign of 2026-08-23)

> **Protocol caveat (2026-08-24):** the ACCURACY result (section 1) and every
> quality verdict below were obtained on the legacy protocol (`x_T = 0`, `zeta = 1`),
> since shown defective (see `README.md` protocol note). The SPEED result (section 2)
> is protocol-independent. Section 1 was RE-ESTABLISHED under the corrected protocol in round 5
> (two independent seed sets, verifier FINAL PASS) — see the Round 5 section.

Authority: `VERIFICATION.md` (independent verifier, held-out seeds offset 1000, 100
paired restarts). Full story: `FINAL_REPORT.md`.

## 1. Accuracy: noise-level trust region on the guidance step

`TemporalConfig.step_clip = "noise"`, `step_tau = 1.0` in `simulations/src/tfg`
(opt-in, off by default). After the temporal operator and rho_t scaling, the
guidance step is rescaled so that `||Delta_t|| <= tau * sqrt(1 - alphabar_t)` —
the guidance may never move the sample further than the current noise amplitude.
No tuned constant, no extra conditional calls, O(d) cost.

Held-out numbers (failure-penalised exact GMM L2, lower is better):
- 2D n=8: 0.418 -> 0.167 (95% CI on the paired diff [+0.18, +0.32], p<0.001),
  success 40% -> 80%; beats the un-regularised baseline at n=96 and LGD at n=32,
  i.e. 3-12x fewer conditional samples at better quality.
- 10D: +0.05..+0.12 at n<=16 (p<=0.04); null at n=32.
- 5D: small positive, mostly not significant. No significant regression in any
  of the 12 held-out cells.

Mechanism: the per-step MMD gradient is noise-dominated (SNR < 1 at every n);
uncapped steps overshoot on the heavy tail of the gradient-norm distribution.

## 2. Speed: exact cached-target MMD

`simulations/src/tfg/fast_mmd.py::MMDFixedTarget` (opt-in via
`DistributionalLoss(backend="fast")` / `engine_runner --loss fast`). The target
set is fixed for a whole run, so the target-target kernel block (80-94% of all
kernel entries at n<=32, m=250) is computed once; per step only the X-X and X-Y
blocks are evaluated. Mathematically identical (float64 agreement <= 1.3e-14 on
loss AND input gradient, incl. adaptive bandwidth); 4-7x on the MMD
forward+backward, 1.7-2.0x on the whole synthetic guided loop at identical
conditional-model calls.

## Verified but secondary

- Batched LGD perturbations (one conditional call on 3n rows): exact (0.0), 1.4x.
- Batched independent restarts: ~10-25x throughput, statistically equivalent
  only (float32 chaos) — use for screening, not for reproductions.

## Round 3 (2026-08-24): both tracks rejected

- **Preconditioning** (full-covariance whitening, diagonal RMS, sign-SGD, temporal
  median; alone and + trust region): nothing beats or matches trust_noise1 on 2+
  scales. Mechanism attribution: the gradient's DIRECTION is not the constraint —
  the step size is. (`precond/REPORT.md`)
- **Sample-replay MMD** (Ori's reuse_frac generalised to geometric decay; found on
  branch claude/hybrid-sampling-optimization-55fv3b, never previously evaluated):
  screening suggested matched quality at 3-4x fewer conditional calls, but the
  held-out verification (offset 2000, R=100) **FAILED** — significantly worse at
  the call-matched pairings in all dims; replay bias is real (mmd2_eval worse in
  5D/10D everywhere); only a ~1.33x saving survives. Lesson recorded: never promote
  on non-inferiority-by-nonsignificance at R=40. (`replay/REPORT.md`,
  `VERIFICATION.md` round-3 section)
- **M-8 control (the decisive test)**: at EQUAL fresh cost (f fresh alone vs f fresh
  + 30% recycled, f in {7,14,28}, R=100, offset 3000, pre-registered two-sided):
  recycling never helps (one isolated 5D cell aside) and with the trust region it
  significantly HURTS in 10D (-0.049/-0.074, p<=0.003). Recycled samples add
  nothing over the fresh samples already paid for. (`replay/m8_tables.md`)

## Round 5 — corrected protocol (x_T ~ N(0,I), per-arm calibrated zeta)

- Calibration (`protocol/zeta_star.md`): with the trust region every dim is divergence-free up to
  zeta=32 and calibrates at zeta*=16/8/4 (2D/5D/10D); without it the no-trust arm cannot exceed
  2/0.25/1. **The trust region's main value is that it makes the correct step scale usable.**
- Trust vs no-trust at each arm's own zeta* (`protocol/r5_tables.md`, R=100, offset 6000): 2D
  +0.59..+0.09 (all n, p<=0.008), 10D +0.13..+0.06 (n<=16 p<=0.006), 5D null never negative.
  Independent confirmation (offset 7000, same-node pairs, `verification/heldout_r5_tables.md`):
  2D +0.53/+0.36/+0.19/+0.06 (all p<=0.016), 5D +0.07/+0.08/+0.06/+0.06 (all p<=0.045),
  10D +0.07/+0.12 at n=4/8 (p<=0.024), null at n>=16; no significant loss in 24 cells over
  two seed sets; robust to the zeta-rule choice (2D zeta=8 vs 16 indistinguishable).
  **VERIFIER: FINAL PASS under the corrected protocol.**
- Replay under the corrected protocol (M-11): 2D-only win (f=4 fifo16 beats fresh n=8 at half the
  calls), 5D significantly worse, 10D null -> not promoted; closed.

## Main negatives (do not retry without a new idea)

RFF/Nystrom/FFT approximations of the MMD; adaptive n_t schedules; stale-gradient
reuse; CRN / antithetic conditional noise; absolute gradient clipping (threshold
does not transfer across dims); Adam in 5D/10D; torch.compile / MPS for this loop.
10D at zeta=1 is mis-calibrated (unguided beats guided at small n) — recalibrate
zeta_d (exp5b) before any 10D claim.
