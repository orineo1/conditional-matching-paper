# Debug: why the guidance objective never descends (Scenario B, mmd arm)

*2026-09-10. Data: `output/pvd_B/mmd/{metrics.json,final_latents.pt,npy/}` (the completed
125/125 run, job 46124426; base_zeta 4, n_cond 100, cn 0.5, trust off, backsel off).
No code was changed and nothing was re-run for this analysis.*

## Verdict up front — RESOLVED by the probe (job 46134134, 2026-09-10)

**Checks (1)-(4) all PASS**: sign, coupling, step scale and the MMD estimator are correct.
The probe then settled the two remaining candidates:

* **(B) fp16 underflow: RULED OUT.** Gradients at `loss_scale` 1 / 1e2 / 1e4 agree to
  cosine **0.998** with ~30 exactly-zero elements in all three (P3 below).
* **(A) noise-dominated gradient: CONFIRMED** — but with a decisive twist. Four gradients
  at the SAME latent with different sprinter seeds are **mutually orthogonal** (mean
  pairwise cos **-0.027**, SNR **0.391** vs 0.5 for pure noise), yet a single step along
  one of them **does reduce the loss substantially** (0.575 -> 0.446), and common random
  numbers change nothing (0.4455 CRN vs 0.4435 fresh), so this is *not* a
  variance-reduction problem.

**The fault is step size.** Descent exists in a narrow band of step norms (~6-24 at this
latent); the production run's median correction was **1.84**, i.e. **3.2x too small** to
reach even the bottom of that band, so 125 mutually-orthogonal, under-sized steps
random-walked instead of accumulating. Corrected configuration pre-registered in
HYPOTHESIS.md (amendment of 2026-09-10): `base_zeta` 4 -> 12 and the trust region turned
on at `--trust_noise 0.2`.

## (1) SIGN — correct

`run_mlgd_f.py:931` `correction = -zeta_i * grad`; `grad` is
`torch.autograd.grad(loss_for_grad, latents_step)` (`generation.py:434-441`, i.e.
+d(loss)/d(latent)); `denoise_step` (`generation.py:141-146`) applies
`x_{t-1} = DDIM(x_t) + correction`. So the update is **descent** on the MMD, matching the
DPS convention. (`--trust_noise` only ever shrinks that vector; it was off here.)

## (2) DECOUPLING — correct, end to end

- The differentiated tensor feeds the next step: `latents_step` -> `noise_pred`/`pred_x0`
  -> `pixel_x0_norm` (ControlNet image) -> sprinter, and `latents = denoise_step(...,
  latents_step, correction)` (`run_mlgd_f.py:1010-1012`).
- The effect survives into the artefacts: **||x_guided - x_unguided|| = 45.79 = 25.7 %**
  of the latent norm (178.3); the saved `final_scribble_mlgd_f.png` differs from
  `final_scribble_regular.png` (mean |diff| 7.1/255, max 255), and `eval_source.scribbles
  = png` confirms the eval consumed that optimized PNG.
- Fraction of the applied guidance that survived the remaining denoising:
  `||Δx_final|| / Σ||correction|| = 45.8 / 283.7 = 0.16` — normal prior/guidance tug of
  war, not a decoupling.

## (3) SCALE — small but not degenerate

Per-step `||correction|| / ||x_t||`: median **1.03 %**, max 5.74 % (16384-element latent).
Cumulative 25.7 %. That is a real, sizeable move — the failure is not "the steps are
invisible". For reference the point arm's steps were 5x larger (median ||corr|| 9.4,
max 103) and did *worse*, so simply enlarging zeta is not obviously the fix.

## (4) OBJECTIVE — the estimator is sound (my main suspect, cleared)

`compute_mmd` recomputes the median-heuristic bandwidth per call from the current batch
(`metrics.py:52-64`, `ss = min(1000, n, m) = 100` here), so the loss VALUE is measured with
a slightly different kernel each step. I tested whether that flattens the objective, using
the run's own eval embeddings and the target set:

| interpolate batch toward the target (a) | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|---|---|
| adaptive-bandwidth MMD (what guidance sees) | 0.637 | 0.614 | 0.571 | 0.456 | 0.242 | **0.014** |
| bandwidth | 0.733 | 0.680 | 0.603 | 0.507 | 0.479 | 0.493 |

and under collapse (shrink toward the batch mean) the loss *rises* 0.637 -> 0.878. The
objective is monotone, well-scaled and correctly punishes both mismatch and collapse.
**Not the fault.** (It does mean per-step values are not strictly comparable across steps;
worth fixing for reporting — see "minimal fix" — but it is not why nothing descends.)

## What the trace actually says

- Loss: 0.570 (step 1) -> 0.659 (step 125), min 0.494 @56, sd 0.037, slope −0.003 over the
  whole run, corr(step, loss) = −0.03. **Flat.**
- Sampling noise of the same statistic at n_cond = 100 with a FIXED scribble (bootstrap,
  300 draws from the run's 2000 eval samples): sd **0.0075**, 5-95 % [0.631, 0.656]. The
  trace's sd (0.037) is 5x larger, so the loss *is* responding to the changing scribble —
  it simply never trends down.
- `corr(||correction||_i, loss_{i+1} − loss_i) = −0.042` — no relationship between how hard
  we push and what happens next.
- **The decisive number:** median ||grad|| = 0.273 per unit of latent movement; total
  latent movement 45.8. A gradient valid over even 1 % of that path predicts ≈0.12 of
  descent; the linear prediction is 12.5. Observed: **+0.006**. The directional derivative
  along the realised path is ~0 while the per-step gradient norm is not — i.e. successive
  gradients are nearly orthogonal to any persistent improving direction.

## The two remaining candidates (need one GPU job to separate)

**(A) Noise-dominated gradient.** Variation seeds are refreshed every step
(`run_mlgd_f.py:846-848`: `base = seed*1_000_003 + (i+1)*10_000`), so every step's gradient
is an independent 100-sample estimate. If the systematic component is small relative to the
sampling component, the corrections random-walk (exactly the campaign's synthetic finding:
per-step SNR < 1, `experiments/model-optimization/IMPROVEMENTS.md` §1) and 125 steps of
1 % moves cancel. Predicted signature: low cosine between gradients computed at the SAME
latent with different sprinter seeds.

**(B) fp16 backward underflow.** `latents` are fp16 (`run_mlgd_f.py:721`), the sprinter
UNet/ControlNet are fp16, the chain is 2 sprinter steps + VAE + CLIP with nested
checkpointing, and **`--loss_scale` was 1.0** — no loss scaling anywhere, although the flag
exists precisely to "amplify weak gradients". Per-element latent gradients are ~2e-3, but
intermediate activations' gradients deep in the sprinter backward can underflow to zero,
which would produce a non-zero but *biased and partly-zeroed* latent gradient — large norm,
wrong direction. Predicted signature: gradient direction changes materially with
`--loss_scale`.

## The probe (written, not run): `debug_gradient.py` + `submit_debug.sh`

One salmon job, ~15-25 min, one fixed latent (the SDEdit init at t_start = 125/250), no
trajectory, nothing written outside `debug/B/`:

- **P1 gradient SNR** — 4 gradients at the same latent, different sprinter seed sets:
  pairwise cosines and SNR = ||mean g|| / mean ||g||. *Noise-dominated (A)* <=> mean cos ≈ 0
  and SNR ≈ 1/√4 = 0.5. A healthy gradient gives cos ≳ 0.5.
- **P2 paired line search** — loss at `x − λ·zeta·g` for λ ∈ {0, 0.5, 1, 2, 5, 20, 100},
  evaluated twice: with the **same** sprinter seeds as the gradient (common random numbers,
  isolating true local descent) and with **fresh** seeds (what the run experiences). If the
  CRN curve descends and the fresh one does not, the fix is variance reduction; if neither
  descends at any λ, the direction itself is wrong (points at B, or at a genuinely flat
  landscape).
- **P3 fp16 underflow** — the same gradient at `loss_scale` ∈ {1, 1e2, 1e4}, rescaled;
  report cosines and the count of exactly-zero gradient elements. Directions must agree;
  if cos(1, 1e4) < 0.9 or the zero-count drops sharply with scale, **(B) is confirmed** and
  `--loss_scale` becomes mandatory, not optional.

```bash
cd /sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd
sbatch submit_debug.sh                # full probe (n_cond 100, 4 repeats)
N_COND=25 REPEATS=4 sbatch submit_debug.sh   # ~5 min variant
```

## Minimal fixes, mapped to the outcome (none applied)

| probe outcome | minimal fix | where |
|---|---|---|
| P3 shows direction changes with loss_scale | run with `--loss_scale 1e3` (already implemented, one flag) and/or keep the sprinter backward in fp32/bf16 | `run_mlgd_f.py --loss_scale`; `models.py` dtypes |
| P1 cos ≈ 0 / P2 CRN descends but fresh does not | **common random numbers**: reuse the SAME variation seeds across steps (drop the `(i+1)*10_000` term — a one-line change, mirrors `n_schedule.eta_keying="frozen"` in the shared engine) and/or raise n_cond | `run_mlgd_f.py:846-848` |
| P2 descends nowhere at any λ | the landscape is flat at this t: guide from an earlier `start_step`, or reconsider the target's reachability (Scenario B's separate defect, see RESULTS.md) | config |
| reporting only | log a FIXED-bandwidth MMD alongside the adaptive one so the per-step trace is comparable across steps | `metrics.compute_mmd(bandwidth=...)`, already supported |

Note the trust region and back-selection were both OFF in this run, so neither is implicated.


---

# PROBE RESULTS (job 46134134, 30 min, `debug/B/debug_gradient.json`)

Probe latent: the SDEdit init at step 125/250 (t = 497), n_cond = 100, base_zeta 4.0,
cn 0.5 — i.e. the production configuration at the first guided step.

## P1 — gradient SNR: noise-dominated (candidate A confirmed)

| repeat | loss | \|\|g\|\| |
|---|---|---|
| 0 | 0.5749 | 1.691 |
| 1 | 0.5590 | 0.391 |
| 2 | 0.5737 | 1.974 |
| 3 | 0.5757 | 0.528 |

Pairwise cosines: +0.174, -0.630, +0.592, -0.121, +0.297, -0.474 -> **mean -0.027**.
SNR = ||mean g|| / mean ||g|| = **0.391** (pure noise at R=4 would be 0.500).
Consecutive per-step gradient estimates carry essentially no common direction, which is
exactly why 125 steps did not accumulate. Note also the 5x spread in gradient NORM
(0.39-1.97) at a fixed latent — the adaptive rule `zeta = base_zeta / L` normalises by the
loss, not by the gradient, so the realised step size inherits that spread.

## P2 — paired line search: the direction IS useful, the run under-stepped

`||zeta * g|| = 11.77` at this latent (zeta = 6.958, ||g|| = 1.691).

| lambda | \|\|step\|\| | loss (CRN, same seeds) | loss (fresh seeds) |
|---|---|---|---|
| 0.0 | 0.0 | 0.5749 | 0.5815 |
| **0.5** | **5.9** | **0.4455** | **0.4435** |
| 1.0 | 11.8 | 0.4569 | 0.4498 |
| 2.0 | 23.5 | 0.4625 | 0.4588 |
| 5.0 | 58.8 | 0.5157 | 0.5093 |
| 20.0 | 235.3 | 0.6629 | 0.6637 |
| 100.0 | 1176.6 | 0.6503 | 0.6489 |

Three readings: (i) a single step buys **-0.13 of loss** — the direction is real and
one-shot useful; (ii) CRN and fresh curves are within 0.007 of each other everywhere, so
the estimator's sampling noise is NOT what limits progress; (iii) the useful band is
**||step|| ~ 6-24**, with clear degradation from ||step|| ~ 59 upward.

## The step-size comparison (why the run failed)

| quantity | value |
|---|---|
| useful band (probe) | ||step|| 5.9 - 23.5 |
| production `correction_norm_raw` | p10 0.88, **median 1.84**, p90 4.00, max 10.23 |
| factor to put the median at the band's bottom (5.9) | **3.19x** |
| factor to put the median at lambda = 1.0 (11.8) | 6.38x |
| factor to put the median at the band's top (23.5) | 12.76x |

At `base_zeta = 12` (3x) the whole distribution lands inside the band: median 5.53
(≈ the lambda = 0.5 optimum), p90 12.0 (≈ lambda = 1.0), max 30.7 (just above the top,
which is what the trust cap is for). At 16 (4x) the tail runs well past the degradation
threshold (max 40.9).

## P3 — fp16 underflow: ruled out

| loss_scale | \|\|g\|\| (rescaled) | cos to loss_scale=1 | exactly-zero elements |
|---|---|---|---|
| 1 | 1.767 | 1.000 | 30 |
| 1e2 | 2.012 | 0.998 | 26 |
| 1e4 | 2.011 | 0.998 | 30 |

Directions agree to 0.998 and the zero-count does not change with scale: **no underflow;
`--loss_scale` is not the issue** (the norm difference between 1 and 1e2 is the same
sampling spread P1 measured, not a precision effect).

## What the probe does NOT establish (open risk, stated before the rerun)

The line search measured descent from **one** step at **one** latent. P1 says successive
gradients are mutually orthogonal, so it remains an open question whether 125 correctly-sized
steps accumulate into a trajectory-level improvement or merely random-walk with a larger
stride. That is precisely the falsifiable prediction registered in HYPOTHESIS.md
(amendment 2026-09-10) and tested by the cheap 20-step gate before the 4.5 h arm.

---

# ROUND 2 DIAGNOSIS — why the corrected FULL run failed while the 20-step gate passed
*2026-09-13, from the landed runs (46147857 full, gate, drift control 46150188). No tuning, no relaunch.*

## The headline fact the diagnosis has to explain

| run | schedule | P-C1 | final MMD (fresh) | own ref | verdict |
|---|---|---|---|---|---|
| mmd zeta12 tau0.2 **gate** | 40/20 (20 guided steps) | **PASS** (-0.0388, corr -0.567) | **0.4911** (n=256) | 0.5938 | **beats baseline by +0.103** |
| mmd zeta12 tau0.2 **full** | 250/125 (125 guided steps) | FAIL (+0.0043, corr -0.031, 6.3 SE — well powered) | 0.6358 (n=2000) | 0.6000 | worse than baseline |

The two schedules have essentially the SAME unguided baseline (0.5938 vs 0.6000), so the
difference is not the schedule's own image quality — it is what guidance achieves on it.

## (a) Cap-throttling — REFUTED for 4 of 5 quintiles

Applied vs raw correction, full run: the trust cap binds on **4 %, 0 %, 0 %, 0 %, 52 %** of
steps by quintile. For Q1-Q4 `applied == raw`: the cap is not touching them. The steps are
small because the RAW step is small (median raw 4.23, 2.66, 2.45, 2.20), not because the cap
cut them. The gate clips even harder at the end (75 % in Q5) and still succeeds. So
"tau throttles the long schedule" is not the explanation.

What IS true is that the full run's steps sit mostly BELOW the measured descent band
(5.9-23.5): in-band 28 %, 8 %, 4 %, 8 %, 20 % by quintile, because `zeta = 12/L` with
L ~ 0.63 gives zeta ~ 19 while `||grad||` falls to 0.12-0.22 mid-trajectory. The gate's are
in-band 50 %/25 % in Q1-Q2. Under-stepping relative to the band therefore persists at
zeta = 12 for most timesteps — the probe at t = 497 did not transfer.

## (b) No early descent that later reverses

Full run, every window: first10 0.6167 / last10 0.6210 (+0.0043); first20 vs last20 -0.0024;
first30 -0.0021; first60 -0.0013. Min 0.5698 at step 1, max 0.6505 at step 2. The trace is
**flat from the first step** — there is no "it worked while steps were big, then reversed".

## (c) The mechanism: the prior undoes fine-grained corrections

Fraction of the applied guidance that survives into the final latent,
`||x_guided - x_unguided|| / sum ||correction||`:

| run | steps | sum ‖corr‖ | final displacement | **survived** |
|---|---|---|---|---|
| mmd zeta12 tau0.2 **gate** | 20 | 90.8 | 44.94 | **0.495** |
| mmd zeta12 tau0.2 full | 125 | 452.0 | 66.30 | **0.147** |
| mmd zeta4 (original) full | 125 | 283.7 | 45.79 | 0.161 |
| point zeta14 tau0.4 full | 125 | 2976.0 | 246.21 | 0.083 |

The gate reaches nearly the same final displacement (44.9 vs 66.3) from **5x less** total
correction. Every correction is followed by the remaining denoising steps, which re-project
the latent toward the model's manifold; with 125 steps a correction has 124 subsequent
opportunities to be undone, with 20 steps only 19. This is consistent with the flat trace,
with the near-orthogonal per-step gradients (P1: mean cos -0.027), and with the fact that
the full run applies 5x more total correction for no benefit.

**Verdict on (c):** not cap-throttling; partly sub-band stepping (the t = 497 probe does not
transfer across the schedule); and primarily **non-accumulation — the prior absorbs
fine-grained guidance**. The gate's success is therefore a *short-horizon / coarse-step*
effect, and saying so IS the finding, not a caveat to it.

## (d) Is P-C2 moot?

**Yes, for the run it was registered on.** P-C2 was explicitly conditional ("*If P-C1
holds…*") on the full 250/125 arm-3 run, which failed P-C1 at 6.3 SE; its antecedent is
false, so P-C2 is not evaluated there. The gate *does* satisfy both the trend criterion and
the "beats its own reference" content of P-C2 (+0.103), but it is a different schedule and a
256-sample eval, so it **cannot discharge P-C2 as registered** — it is suggestive evidence
for a claim that was never pre-registered, and is reported as such.

## What would test the mechanism (NOT run, not tuned)

1. A schedule sweep at fixed zeta/tau — 250/125, 100/50, 40/20, 20/10 — measuring survival
   fraction and final MMD. The prediction from the table above is monotone: survival and
   guidance benefit both rise as the schedule coarsens.
2. Per-timestep probes (t ~ 497, 300, 100) to see how the descent band moves with t, since
   the single t = 497 probe demonstrably did not transfer.
Both are new pre-registrations, not adjustments to this one.
