# Pre-registered hypothesis — point vs distributional targets on the SD pipeline

*Registered 2026-09-06, BEFORE any run (smoke included). Agent S, branch
`tfg-generalization-v2`. Spec: boss message of 2026-09-06 (user's design, "paper-critical").
This file is written first and is not edited after runs start; deviations are recorded
in README.md under "Deviations".*

## Question

Two confounded ingredients distinguish MLGD-F from adversarial-style input
optimization: (a) the **diffusion prior** on the optimization variable x (the
scribble), and (b) the **distributional target** (a set S_G matched by MMD rather
than a single CLIP vector y* matched by squared distance). Three arms unconfound them:

| arm | prior on x | target | change isolated |
|---|---|---|---|
| 1. AdvI2I-style PGD | none (eps-ball around x0, Adam on pixels) | point y* | — |
| 2. MLGD-F point | diffusion (SDXL-base DDIM, SDEdit t_start) | point y* (n_cond=1, loss ‖CLIP(y)−y*‖²) | 1→2: the prior |
| 3. MLGD-F (ours) | diffusion | distributional (n_cond=100, MMD to S_G) | 2→3: the distributional term |

## The mathematical mechanism (what makes Scenario B decisive)

For a point target a with one stochastic observation y ~ f_phi(x,·),

    E‖CLIP(y) − a‖² = Var[CLIP(y)]  +  ‖E CLIP(y) − a‖².

When a = the CLIP centroid of a balanced male/female target set, the second term is
minimized by moving the conditional MEAN to the androgynous midpoint, and the first
term additionally REWARDS COLLAPSING the conditional variance. Nothing in the point
loss rewards reproducing the target's spread — the optimum of arms 1-2 is a
low-variance androgynous mode, not a 50/50 mixture. (Toy analogue already measured:
`experiments/witness_unimodal/POINTWISE.md` — the squared point loss is
variance-seeking; its (y−0)² arm landed in the low-variance basin in 97% of restarts
regardless of init.) The MMD arm matches all moments the kernel sees and should
reproduce the two modes.

## Predictions

### Scenario B (G = G_bal, 50/50 male/female; y* = 0.5·mu_male + 0.5·mu_female) — decisive

- **P-B1 (the headline):** arms 1 and 2 collapse to a single androgynous mode:
  p(male) ≈ 50% (an androgynous face is ambiguous to the zero-shot classifier, CI
  overlapping 0.5 or wide), while the **CLIP-PC1 variance of their 2000 eval samples
  is FAR below the target's** — pre-registered threshold: PC1-variance ratio
  (gen/target) **< 0.5** for arms 1-2. The PC1 histogram is unimodal near the
  midpoint of the two target modes.
- **P-B2:** arm 3 splits into two modes: bimodal PC1 histogram aligned with the
  target's two clusters, PC1-variance ratio in **[0.5, 1.5]**, p(male) ≈ 50% *for the
  right reason* (a mixture of confident males and confident females — the per-sample
  p(male) distribution is bimodal at ~0 and ~1, unlike arms 1-2 where it concentrates
  near 0.5).
- **P-B3 (ordering):** MMD-to-G_bal of arm 3 < arm 2 < / ≈ arm 1. Arms 1-2 cannot beat
  arm 3 on MMD because a collapsed mode pays the missing-mass penalty on both target
  clusters. Between 1 and 2 we predict arm 2 ≥ arm 1 in image plausibility (the prior
  keeps x on the scribble manifold) but make no strong MMD ordering claim — the prior
  isolates *where* the collapse happens, not whether it happens.
- **P-B4 (mean is matched):** all three arms land the PC1 MEAN near the target mean
  (|gen mean − target mean| small vs the inter-mode distance) — the failure of arms
  1-2 is invisible to any mean-based metric. This is the paper's point.

### Scenario A (oracle recovery: y* = CLIP(y_f), S_G = 100 samples of f_phi(x_f,·))

- **P-A1:** all arms are reasonable: the target set is near-unimodal (one scribble,
  one prompt), so the variance term costs little; PC1-variance ratios closer to 1
  for every arm; p(male) low (female portrait) for all arms.
- **P-A2:** the oracle floor exists by construction: L(x_f) (mean point loss of fresh
  f_phi(x_f) samples) is reported; arms 1-2 should approach it. ‖x* − x_f‖ (pixel and
  VAE-latent) is reported per arm; we predict arm 1 (eps-ball around the MALE x0)
  stays far from x_f in pixel distance yet reaches a comparable loss — evidence for
  loss/solution non-identifiability — while arms 2-3 may move x further toward
  female-consistent scribble structure. No pre-registered threshold here; these
  numbers are descriptive.

### Honesty clause

If the PC1 variance gap of P-B1 does **not** appear (ratio ≥ 0.5 for arms 1-2, or
arm 3 not separated from them), we report exactly that in the deliverable table and
README — **no tuning of eps / zeta / steps / prompts until the negative result is
recorded**. Any subsequent tuning is a new, separately-registered experiment.

## Pre-registered analysis details

- PC1 = first principal component fitted on the TARGET CLIP embeddings of the run's
  own scenario (Scenario B: fitted on the two anchor groups exactly as the pipeline's
  `build_targets_gender` PCA; Scenario A: fitted on S_G). Gen samples are projected
  with that fixed PCA. Variance ratio = Var(PC1 of 2000 gen) / Var(PC1 of target set).
- p(male): zero-shot CLIP softmax over ("a superrealistic portrait photograph of a
  man, studio lighting", "... a woman, studio lighting") at logit scale 100 (the
  repo's `compute_clip_softmax` convention); p(male) = mean over the 2000 eval
  samples of the per-sample male probability; 95% CI = normal approx on the mean of
  the per-sample probabilities (and we also report the fraction with p>0.5 with a
  Wilson interval).
- MMD: `compute_mmd` (unbiased U-statistic, single median-heuristic bandwidth,
  kernel_alpha=1) of the 2000 eval embeddings vs the scenario target set.
- Eval: 2000 fresh sprinter samples, neutral prompt ("a superrealistic professional
  photograph of"), eval seed of photo i = 42·1000003 + 7000000 + i — derived from the
  common seed 42, indexed by i, IDENTICAL across arms and scenarios (the pipeline's
  `--seeded_rng` convention).
- Arms 2-3 run the unmodified Algorithm 1 loop (`run_mlgd_f.py --engine legacy`,
  no backsel, no trust); the ONLY difference between them is the loss/target
  (`--loss_fn point --point_target_pt` vs `--loss_fn mmd`) and n_cond
  (`--num_variations 1` vs `100`). Arm 1 shares x0, y*, sprinter, CLIP, eval.

## Spec-vs-pipeline audit (documented mismatches, NOT silently changed)

Appendix-E numbers per the spec vs `run_mlgd_f.py` on this branch (the paper appendix
itself is not in the repo — `literature/` holds external papers only — so the spec
message is the authority):

| item | spec | pipeline | action |
|---|---|---|---|
| outer prior | SDXL-base DDIM T=250, t_start=125 | defaults 30/15, configurable | pass `--n_steps 250 --start_step 125` |
| zeta | 4.0 | default 1.0 (perf campaign used 5.0) | pass `--base_zeta 4.0` |
| n_MC | 3 | **NOT IMPLEMENTED** — the SD loop has no LGD/logsumexp smoothing draws (one loss evaluation per step; `n_mc` exists only in the synthetic engine) | **mismatch documented; arms run with the pipeline's single-evaluation step.** Adding n_MC would change Algorithm-1-as-implemented for ALL arms equally; not done without a separate decision |
| f_phi ControlNet scale | 0.5 | guidance/eval paths HARD-CODE 0.8 (`generation.py` variation call, `metrics.py` eval); `--controlnet_scale 0.5` affects target generation only | new opt-in `--variation_cn_scale` (default 0.8 = old behaviour); this experiment passes 0.5 everywhere and records it |
| f_phi CFG / steps | 0.0 / 2 | 0.0 / 2 hard-coded | match |
| sprinter prompt | neutral | default "a superrealistic professional photograph of" | match |
| eval N | 2000 | default 10, `--eval_n` exists | pass 2000 |
| seeds | 42+i identical across arms | `--seeded_rng` derives all noise from `--seed` | `--seed 42` everywhere |
| MMD estimator | U-statistic, ViT-L/14, median heuristic | exactly `compute_mmd` | match |
| G_bal | "Appendix E.2 config in the repo" | the repo's balanced groups: 50 "Man: a superrealistic portrait photograph of a man, studio lighting" + 50 "Woman: … woman …" (the `--target_prompts` defaults scaled to 50+50, = the perf-campaign dev config and `experiments/BalancedTarget`) | use 50+50 |

## Arm-1 design decision (pre-registered)

PGD operates on the **pixel-space scribble** (512², [0,1]) with an L_inf eps-ball
around x0 and Adam steps, NOT on the VAE latent. Justification: (i) the scribble
pixels are the variable the sprinter actually consumes (ControlNet conditioning
image) and the variable MLGD-F's gradient ultimately shapes; (ii) an eps-ball in the
architect's VAE latent would smuggle in a piece of the diffusion model (its decoder)
and blur the "no prior" contrast that arm 1 exists to provide; (iii) AdvI2I-style
attacks are defined in image space. One inner sample per step, seeded per step;
Adam lr = eps/10; 200 steps; eps sweep = AdvI2I's grid {32/255, 64/255, 128/255}
(L_inf); the reported arm = the eps with the best final point loss (stated in the
table; the sweep is cheap).

*Amendment (2026-09-06, pre-run, before any smoke):* the eps grid was aligned to
AdvI2I (arXiv 2410.21471) after reading the paper: they constrain
||g_psi(x) - x||_inf <= eps with eps in {32, 64, 128}/255 and clipping after each
update. Our arm 1 is an ADAPTATION of AdvI2I, not a reproduction: they optimize a
pre-trained-VAE generator g_psi and match UNet latent features toward a
concept-shifted prompt embedding on InstructPix2Pix / SD-inpainting; we optimize the
scribble directly (no generator) and match the CLIP image embedding of our ControlNet
f_phi's output to y*. The full mapping table is in README.md. No other change.

---

# CLOSE-OUT — Scenario B, 2026-09-10

*Scored after the five full arms completed (seed 42, 2000-sample evals, identical eval
seeds / targets / cn_scale 0.5, verified in RESULTS.md "Verification"). Numbers:
RESULTS.md. Nothing below was tuned after seeing results; the honesty clause is invoked.*

| prediction | outcome | evidence |
|---|---|---|
| **P-B1** arms 1-2 collapse: p(male) ~ 50 %, PC1 var-ratio < 0.5 | **PARTLY CONFIRMED — variance half YES, p(male) half NO** | var-ratio pgd 0.005/0.007/0.007, point 0.009 — an order of magnitude below the 0.5 threshold, so the collapse is emphatic. But p(male) is 0.17-0.29, not ~0.5: the collapse landed on a female-leaning mode, not an androgynous one (y\* is the CLIP centroid, and the reachable set under a neutral prompt is already female-leaning: the reference f_phi(x0) has p(male) = 0.317). The mechanism (variance-seeking point loss) is confirmed; the "androgynous midpoint" reading of it is not. |
| **P-B2** arm 3 splits: bimodal PC1, ratio in [0.5, 1.5], bimodal per-sample p(male) | **FALSIFIED** | mmd var-ratio **0.022** — collapsed like the point arms, nowhere near [0.5, 1.5]; frac of samples with p(male) < 0.2 or > 0.8 is 0.315, no cleaner than pgd_eps32 (0.328). The distributional arm did NOT reproduce the target's spread in this configuration. |
| **P-B3** MMD ordering arm 3 < arm 2 <= arm 1 | **CONFIRMED in ordering, VOID in substance** | 0.643 (mmd) < 0.760 (point) < 0.700/0.745/0.801 (pgd; mmd beats all three) — the predicted ordering holds, but every arm is worse than the do-nothing reference 0.3776, so the ordering ranks degrees of damage, not degrees of success. |
| **P-B4** all arms match the PC1 mean | **CONFIRMED** | gen PC1 means +0.081 … +0.101 against a target mean of 0.000, i.e. all arms sit within ~0.10 of the target mean while the inter-mode gap is 0.64 — a mean-based metric cannot distinguish any of these arms, which is exactly the paper's point. The variance metric separates them from the target by 40-200x. |

## Honesty clause, invoked

The pre-registered variance gap **between** the point arms and the MMD arm did **not**
appear: all five arms collapsed (ratios 0.005-0.022). Reported as-is; no parameter has
been changed and no arm re-run to chase it.

## Why (measured, not assumed)

The dominant cause is a **feasibility defect in the experiment's own design**, found only
by running it: G_bal's PC1 variance is 97 % between-group, created by the two generating
PROMPTS (man / woman), while guidance and evaluation both use ONE neutral prompt. The
best PC1 variance ratio anything achieved is the untouched reference f_phi(x0) = 0.300;
the unguided DDIM trajectory alone drops it to 0.046 before guidance acts. On top of that
the guidance objective never descended (mmd flat 0.570 -> 0.61; point rising with
corrections up to 103 in norm, 6.4 % of steps above the noise-level cap). Full analysis
and the four candidate fixes (reachable target, zeta / trust region, start step, arm-1
optimizer) are in RESULTS.md "Findings"; they are future work, deliberately not applied.

## Status of the design question

**Unresolved by Scenario B.** The prior-vs-point contrast (arm 1 -> 2) and the
point-vs-distributional contrast (arm 2 -> 3) cannot be read off a configuration in which
the target is unreachable and no arm improves on doing nothing. Scenario A (oracle
recovery, target = f_phi(x_f, neutral) — reachable BY CONSTRUCTION, since x_f is a
zero-loss solution) is the correct next test and is unaffected by the defect found here:
its target is generated with the same neutral prompt used for guidance and evaluation.

---

# AMENDMENT — corrected step-size configuration (registered 2026-09-10, BEFORE the rerun)

*Written after the diagnostic probe (job 46134134, `DEBUG_B.md`) and BEFORE any run at the
new settings. The Scenario-B close-out above stands as recorded; this amendment opens a
NEW, separately-scored run of arm 3 (and, conditionally, arm 2). Nothing here was chosen
by trying configurations — every number is derived from the probe's line search and the
failed run's own logged distributions.*

## Why a change is licensed at all

The close-out's honesty clause forbids tuning to chase a predicted effect. What licenses
this amendment is different: the probe established that the failed run was **not a
failure of the method under test** but of one operating parameter — the guidance direction
reduces the loss by 0.13 in a single step (0.575 -> 0.446), while the run's median step
was 3.2x too small to reach the descent band at all. Reporting "distributional guidance
does not work" from a run whose steps never entered the working range would be the
dishonest option. The corrected run is registered here, in advance, with a falsifiable
prediction and a pre-committed negative.

## The corrected configuration (arm 3, `mmd`)

| parameter | old | new | derivation |
|---|---|---|---|
| `--base_zeta` | 4.0 | **12.0** | Probe useful band = ||step|| 5.9-23.5. Failed run's `correction_norm_raw`: median 1.84, p90 4.00, max 10.23. 3x puts median at 5.53 (the lambda=0.5 optimum), p90 at 12.0 (lambda=1.0), max at 30.7. 4x (base_zeta 16) pushes the tail to 40.9, past the observed degradation onset (~59 is clearly bad, 23.5 still good). |
| `--trust_noise` | 0.0 (OFF) | **0.2** | Cap = TAU * sqrt(1-abar_prev) * sqrt(16384); logged `trust_cap_tau1` runs 108.2 (step 1) -> 72.2 (mid) -> 3.7 (end). TAU = 0.2 gives a cap of 21.6 early / 14.4 mid — i.e. it sits at the TOP of the probe's useful band and shrinks with the remaining noise. At base_zeta 12 it clips **17 %** of steps, leaving median applied 5.23 and max applied 18.8: the tail is prevented from entering the lambda >= 5 degradation regime while the median is untouched. |

Everything else is unchanged from Appendix E (n_steps 250, start_step 125, n_cond 100,
cn 0.5, seed 42, neutral prompt, eval 2000 with the same eval seeds, backsel off).

This is the campaign-verified **"noise-dominated guidance needs step-size control"** fix:
the synthetic study reached the same conclusion (`experiments/model-optimization/IMPROVEMENTS.md`
§1, `protocol/zeta_star.md`) — per-step SNR < 1, the binding constraint is the step size,
and the trust region's main value is that *it makes the correct step scale usable*
(without it the calibrated zeta diverges; with it every dimension was divergence-free up
to zeta = 32). Scenario B is the first test of that mechanism on the real SD pipeline.

## Falsifiable prediction (P-C1) and the pre-committed negative

**P-C1.** With base_zeta 12 + trust_noise 0.2, the **guidance MMD trace decreases over the
run**: mean(last 10 steps) - mean(first 10 steps) <= **-0.03** and corr(step, loss) <=
**-0.30**. (The failed run, under this k=10 rule: -0.0070 and corr -0.025 — flat. The n_cond=100 sampling noise of the
statistic is sd 0.0075, and a 10-step mean has sd ~0.012 from that source, so -0.03 is
>2 sd.)

**P-C1 scope and exact window (clarified 2026-09-12, no change of threshold).** P-C1 is the
criterion for ANY corrected diffusion arm (arm 3 and, per Amendment 2, arm 2), scored
identically. The window is fixed by this registration: **k = 10** (mean of the last 10 steps
minus the mean of the first 10) for any run of >= 20 steps; runs shorter than 20 steps, i.e.
only the smoke, use k = max(3, n//6) because first-10/last-10 would otherwise overlap. The
gate configuration is `n_steps 40 / start_step 20` = 20 guided steps at the same SDEdit
strength 0.5, eval 256. *Implementation note: `gate_check.py`/`build_report.py` initially
used k = n//6, which reported the 20-step gate as delta -0.0595 (k=3) and the failed mmd run
as -0.0253 (k=20); under the registered k=10 rule those become -0.0388 and -0.0070. Both
verdicts are unchanged (gate PASS, failed run FAIL) and the code now implements k=10.*

**P-C2.** If P-C1 holds, the final 2000-sample MMD of arm 3 beats **both** its own
unguided reference (0.6000) and, ideally, the common reference f_phi(x0) (0.3776). Beating
the own reference is the necessary condition; beating f_phi(x0) is not expected given the
separate reachability defect recorded in the close-out (G_bal's spread is prompt-induced),
and failing to do so is NOT counted against P-C1.

**Honesty clause (restated, binding).** If the corrected run still does not descend
(P-C1 fails at the 20-step gate or over the full run), we report that as the result:
"distributional guidance does not descend on this pipeline even with the step size
calibrated to a measured descent band". No further parameter search follows without a new,
separately registered amendment. Both the gate and the full run are scored on the criteria
written above, fixed now.

## Fairness across arms (registered now, so it cannot be chosen after the fact)

* **Arm 2 (`point`) — needs the same treatment, but NOT the same number.**
  **[SUPERSEDED 2026-09-12 by Amendment 2 — the reasoning below was falsified by arm 2's
  own probe: it WAS under-stepping (its band is 22-45, its median step 9.45). Kept
  verbatim as the pre-registration record; do not read it as the current plan.]** It shares the
  diffusion loop and the `zeta = base_zeta / L` rule, so leaving it at the old scale while
  arm 3 is corrected would confound "distributional vs point" with "calibrated vs
  uncalibrated". But it was not under-stepping: its corrections were median 9.45, p90 30.2,
  max 103 — already at or above arm 3's corrected band, and 41.6 % of its steps exceed the
  TAU=0.2 cap. Multiplying its zeta by 3 would drive it deep into the degradation regime and
  would be a straw man. The registered fair treatment is therefore: **same principle, same
  trust region (TAU = 0.2), zeta set by its own probe** — run `debug_gradient.py` with
  `--loss_fn point` semantics (one line: the point loss and n_cond = 1) to measure ITS
  useful band, then set its base_zeta so its median lands there. If that probe shows its
  steps were already in-band, arm 2 is rerun with `base_zeta 4 + trust_noise 0.2` only.
* **Arm 1 (`pgd`) — separate optimizer, keeps its own sweep, but gets an equivalent
  courtesy.** It is not zeta-based (Adam, lr = eps/10) and its eps sweep is its registered
  free dimension, so the zeta correction does not apply. However, arm 3 having its step
  size calibrated while arm 1 keeps an lr that never descended would be "tuned vs untuned".
  Registered fair treatment: a **3-point lr sweep at the best eps (32/255)**, lr in
  {eps/100, eps/10, eps/3}, 200 steps each (~15 min total), reporting the best final point
  loss — the same courtesy, in arm 1's own currency. If none descends below L(x0) = 0.4218,
  the reported finding ("the AdvI2I-style attack fails to improve its own objective through
  this stochastic generator") stands with the sweep as evidence.
  **[MEASUREMENT-DEFINITION CORRECTION, 2026-09-12: "descends below L(x0) = 0.4218" as written
  is not directly measurable on the sweep's output. The training curve is ONE sprinter sample
  per step (sd ~0.15), while L(x0) = 0.4218 is a 2000-sample fresh average — comparing them
  scored a non-descending run as a success in the first cut of `pull_results.sh`. The
  operational criterion, fixed now and applied to all three lrs equally, is two-stage:
  (i) did the optimiser descend on its OWN noisy objective, mean(last 10 steps) <
  mean(first 10 steps); (ii) ONLY for an lr passing (i), a fresh 2000-sample eval of x*
  decides whether it beats L(x0) = 0.4218. An lr failing (i) cannot beat L(x0) and is not
  fresh-evaluated. This changes how the criterion is measured, not what it claims.]**
* Any arm rerun at corrected settings is reported in a SEPARATE table row (e.g.
  `mmd_zeta12_tau0.2`) next to the original; the original rows are never overwritten.

---

# AMENDMENT 2 — point-arm calibration (registered 2026-09-12, after its probe, before its rerun lands)

*Probe job 46147886 (`debug/B_point/debug_gradient.json`): gradient from the arm's own
n_cond = 1, loss readout along the line search averaged over 32 samples, same latent as the
mmd probe.*

## My registered expectation was WRONG, and its own measurement is what overturned it

Amendment 1 registered: *"it was not under-stepping: its corrections were median 9.45 ...
already at or above arm 3's corrected band ... multiplying its zeta by 3 would drive it
deep into the degradation regime and would be a straw man."* That inference — made from
arm 3's band, not arm 2's — is **falsified**. The point arm has its own, much higher band:

| lambda | \|\|step\|\| | loss (CRN) | loss (fresh) |
|---|---|---|---|
| 0.0 | 0.0 | 0.6984 | — |
| **0.5** | **22.3** | **0.4832** | **0.5131** |
| 1.0 | 44.6 | 0.6921 | — |

Only `lambda = 0.5` improves; degradation has already set in by `lambda = 1.0`. So the
point arm's useful band is **||step|| ~ 22-45**, and its failed-run median correction of
**9.45 sits BELOW it** — the point arm was under-stepping too, by a factor of
**3.5x** (9.45 -> 33.1, mid-band). It was never the "already in-band" case I assumed;
both diffusion arms failed for the same reason, each at its own scale.

**Registered correction (same derivation as arm 3's, from its own probe):**
`--base_zeta 4 -> 14` (3.5x) for the point arm. Launched as job 46148094.

## Open issue flagged at registration time: TAU = 0.2 contradicts the TAU principle here

TAU was not chosen as a number but by a principle: *the cap sits at the top of the arm's
useful band*. For arm 3 (band 5.9-23.5) that gave TAU = 0.2 (cap 21.6 at step 1). Applying
the SAME NUMBER to arm 2, whose band is 4x higher, breaks that principle:

| TAU | cap step1 / mid / last | steps clipped (at zeta 14) | median applied | steps landing INSIDE the 22-45 band |
|---|---|---|---|---|
| 0.2 (as launched) | 21.6 / 14.4 / 0.75 | 87 % | 13.1 | **0 %** |
| 0.3 | 32.5 / 21.7 / 1.12 | 74 % | 17.4 | 30 % |
| **0.4** | 43.3 / 28.9 / 1.49 | 66 % | 21.3 | **46 %** |

TAU matching the top of arm 2's band is **0.41**. At TAU = 0.2 the cap is *below the band's
floor at every step*, so the trust region would cancel the zeta correction and the arm
would under-step again — the exact failure being fixed. **Pre-registered reading of job
46148094 (ZETA 14, TAU 0.2): if it fails P-C1, that outcome does NOT license the conclusion
"the point arm cannot descend"**, because its applied steps are 0 % in-band by construction.
The principled companion run is `ZETA=14 TRUST_TAU=0.4` (band-matched cap), and it is
registered here, in advance, as the arm-2 configuration that carries the fair-treatment
claim. Both are cheap (~4 min guidance + ~25 min eval) and both will be reported.

## Incidental finding worth recording: the mmd arm's noise is not a sample-count problem

| arm | n_cond per gradient | mean pairwise cos | SNR (pure noise = 0.5) |
|---|---|---|---|
| point | **1** | **+0.013** | **0.593** |
| mmd | **100** | -0.027 | 0.391 |

The point arm's *single-sample* gradient is at least as directionally consistent as the mmd
arm's *hundred-sample* gradient. So arm 3's mutual orthogonality cannot be explained by
"too few conditional samples", and averaging more of them is not the lever. Interpretation
(flagged as such, not measured): the point gradient is one `J^T(e - y*)` with a systematic
pull toward a fixed target, whereas the MMD gradient is a sum of `J_i^T g_i` whose `g_i`
depend on the whole batch's kernel geometry — and with the median-heuristic bandwidth
recomputed per draw, the effective objective shifts slightly between draws. This is the
mechanism worth probing if arm 3's corrected run also fails.

---

# NOTE ON P-C1's POWER (recorded 2026-09-12, before the final close-out is written)

P-C1's threshold (delta <= -0.03) was derived from the **mmd** arm's statistic: a
100-sample MMD with per-step sd ~0.0075. The **point** arm logs a **1-sample** loss
(n_cond = 1), whose per-step sd is ~0.09 — 12x larger. Measured from the runs' own traces
(detrended, sd = std(diff)/sqrt(2)):

| arm | per-step sd | SE(10-step delta) | -0.03 expressed in SE | verdict quality |
|---|---|---|---|---|
| mmd (n_cond 100) | 0.0169 | 0.0076 | **4.0 SE** | well powered |
| point (n_cond 1) | 0.084-0.100 | 0.037-0.045 | **0.7-0.8 SE** | **UNDERPOWERED** |

Consequence, stated before interpreting any result: **a P-C1 "FAIL" for the point arm is
not evidence that the point arm does not descend** — the test cannot resolve a -0.03 change
in that arm. What IS resolvable there is the observed *ascent* (+0.077 = 2.1 SE, +0.113 =
2.5 SE). So the honest reading of arm 2 is "its objective rose (marginally significant)",
not "it failed to descend by a criterion it could never have passed". This is a defect in
my registration, not in the runs: the threshold should have been set per-arm from each
statistic's noise. It is recorded rather than silently re-set, and the tables now print
`-0.03 in SE` and mark underpowered rows. Any future point-arm run should log the loss with
`eval_n_cond`-style averaging (e.g. 32 samples) so its trace is comparable.

---

# AMENDMENT 3 — the zeta = 0 drift control (registered 2026-09-12, BEFORE the run and BEFORE looking at it)

## What is being controlled for

Arm 2's point loss ROSE over the trajectory (+0.077 at tau 0.2, +0.113 at tau 0.4; 2.1 and
2.5 SE). Two explanations are observationally equivalent so far: **(D) drift** — the
point-loss statistic rises along the trajectory anyway as pred_x0 sharpens with t -> 0,
guidance or no guidance; **(G) guidance-driven** — the corrections actively push the loss
up. The quintile evidence leans to (D) (the loss rises fastest where the applied steps are
SMALLEST, corr(quintile step, quintile loss) = -0.865), but that is indirect.

## The control

`zeta = 0` (`--base_zeta 0` => `zeta_i = 0/L = 0` => correction identically zero, so the
guided trajectory coincides with the unguided DDIM trajectory), everything else identical:
same seed 42, same SDEdit init, same schedule (250/125), same targets, same y*, same
cn 0.5. It logs the same per-step point-loss statistic at the same pred_x0 points.

**Readout differs deliberately:** the control uses **32 conditional samples per step**
(`--num_variations 32`, with `--backsel 1` so only one is differentiated and the cost stays
~17 min instead of ~1.4 h — the logged loss VALUE is the full 32-sample statistic by
construction). The arm itself logs a **1-sample** estimate (n_cond = 1, its actual
configuration). Both are unbiased for the same expected loss, so their **deltas are
comparable in expectation**; only the noise differs (control SE ~0.016 vs arm ~0.045). The
control is therefore the better-resolved measurement of the drift, and is not a re-run of
the arm.

## Pre-registered reading (fixed before the run finishes)

Let `delta_ctrl = mean(last 10) - mean(first 10)` of the control's trace, same k = 10 rule.

* **delta_ctrl >= +0.05** (i.e. comparable to the arm's +0.077/+0.113): the rise is
  **trajectory drift**. The claim about arm 2 then reduces to *"the point arm does not
  improve on the drift baseline"* — NOT "guidance makes it worse". The paper sentence
  becomes "the distributional arm descends; the pointwise arm merely tracks the drift".
* **|delta_ctrl| < 0.03 and corr(step, loss) within +/-0.2**: the drift is flat, so the
  arm's rise is **attributable to the guidance**, and "the pointwise objective is actively
  worsened by its own guidance" is licensed.
* **anything in between (0.03 <= delta_ctrl < 0.05)**: report the arm's rise MINUS the
  control's as the guidance-attributable part, with both SEs, and make no qualitative claim.

A sanity check is built in: with zeta = 0 the guided and unguided final latents must be
identical, so `final_scribble_mlgd_f.png` must equal `final_scribble_regular.png`. If they
differ, the control is invalid and nothing is read from it.

---

# CLOSE-OUT 3 — final scoring, Scenario B (2026-09-13, all arms landed)

*All runs complete: 5 PGD configs, 3 point configs, 1 drift control, 2 mmd configs + gate.
Verification (identical eval seeds 49000126, cn 0.5, target hash 8320eedea065, eval n=2000
except the gate) in RESULTS.md. Scoring rules are those registered above; none re-negotiated.*

## P-C1 (guidance objective descends) — FAILED on the full schedule, PASSED on the gate

| arm | delta (k=10) | corr | -0.03 in SE | verdict |
|---|---|---|---|---|
| mmd zeta12 tau0.2, 250/125 | +0.0043 | -0.031 | 6.3 SE (well powered) | **FAIL** |
| mmd zeta12 tau0.2, **40/20 gate** | -0.0388 | -0.567 | 3.8 SE | **PASS** |
| point zeta14 tau0.2 | +0.0767 | +0.066 | 0.8 SE | FAIL *(underpowered)* |
| point zeta14 tau0.4 | +0.1128 | +0.421 | 0.7 SE | FAIL *(underpowered)* |

## P-C2 — MOOT as registered

Conditional on P-C1 holding for the full arm-3 run; that antecedent is false (FAIL at
6.3 SE). The gate satisfies P-C2's content (final 0.4911 vs its own reference 0.5938,
+0.103) but on a different schedule with a 256-sample eval, so it does not discharge P-C2.
Reported as suggestive, not as a pre-registered result.

## AMENDMENT 3's control resolves the point arm — the rise is DRIFT

Control (zeta = 0, corrections identically zero, guided == unguided == 0.6000: validity
check passes exactly): **delta_ctrl = +0.0556 +- 0.0140**, corr +0.408. This clears the
pre-registered `>= +0.05` threshold, so **the rise is trajectory drift**. Residuals:

| arm | delta | minus control | SE | z | resolvable? |
|---|---|---|---|---|---|
| point zeta14 tau0.4 | +0.1128 | **+0.0572** | 0.0470 | +1.22 | **no** |
| point zeta14 tau0.2 | +0.0767 | **+0.0212** | 0.0399 | +0.53 | **no** |

The guidance-attributable residual is not resolvable for either configuration. Per the
pre-registered reading, the claim about arm 2 reduces to **"the point arm does not improve
on the drift baseline"** — NOT "guidance actively worsens it". The control's trace shape
(corr +0.408) also matches the arm's (+0.421), as drift predicts.

## Re-scoring P-B1..P-B4 with everything in

| prediction | verdict | evidence |
|---|---|---|
| **P-B1** point arms collapse: p(male)~50 %, PC1 var-ratio < 0.5 | **PARTLY CONFIRMED** (unchanged by calibration) | ratios 0.005-0.009 uncalibrated; the variance half is emphatic, the "androgynous p(male)~0.5" half is wrong (0.17-0.29 — a female-leaning mode, and the reachable set is already female-leaning: reference p(male) 0.317) |
| **P-B2** mmd arm bimodal, ratio in [0.5, 1.5] | **FALSIFIED** | 0.022 at zeta 4; the calibrated full run does not rescue it (see RESULTS table) |
| **P-B3** ordering mmd < point <= pgd | **CONFIRMED in order, VOID in substance** | mmd 0.636 < point 0.700-0.760 < pgd 0.700-0.801, but every full-schedule arm is worse than its own reference, so the ordering ranks degrees of damage |
| **P-B4** all arms match the PC1 mean | **CONFIRMED** | gen PC1 means +0.08..+0.10 vs target 0.000 against an inter-mode gap of 0.64 — no mean-based metric separates these arms; this is the paper's point and it survives every configuration change |

## The honest headline

**No configuration we found makes SD-scale distributional guidance improve on the unguided
baseline over the full 250-step schedule; the 20-step (40/20) schedule does (+0.103), and
that is the only configuration that works.** The mechanism is measured, not assumed: only
15 % of the applied guidance survives to the final latent on the fine schedule versus 49 %
on the coarse one (DEBUG_B.md round 2), so the prior absorbs fine-grained corrections. Two
further defects, both recorded rather than repaired: the target is essentially unreachable
under the neutral eval prompt (its spread is prompt-induced), and P-C1's threshold was
calibrated on one arm's noise and is underpowered for the other.

## Honesty clause — invoked, and nothing was tuned after the fact

Arm 3's corrected full run failed its pre-registered criterion. Per the clause: reported as
the result, with no further parameter search. The step-size correction that preceded it was
licensed by a measured descent band, scored against a criterion fixed in advance, and is
reported whether it worked (gate) or not (full run).

---

# AMENDMENT 4 — the premise was wrong: Scenario B tested a configuration WITHOUT witness back-selection (2026-09-13)

## What was wrong

Ori's working method uses **MMD-witness back-selection**, which exists on
`upstream/main` and **does not exist on this branch**: main's `run_mlgd_f.py` exposes
`--backsel_k / --backsel_rule {uniform,witness} / --witness_floor / --witness_temperature /
--witness_replacement`, implemented in main's `src/generation.py`
(`_witness_select_indices`, `_log_witness_selection`, `_expand_rows_by_count`) and
`src/metrics.py` (`compute_witness_scores`). This branch's back-selection is a different
mechanism (uniform / is / kcenter / strat + soft weighting), built here for the
performance campaign. The two files differ by ~525 diff lines.

Ori's ACTUAL working gender configuration (his words; the committed example showing
30/15/6/uniform/k20 is stale and was the source of my earlier mistaken inference):

```
--n_steps 250 --start_step 125 --num_variations 100 --backsel_k 50 --backsel_rule witness
--witness_floor 0.0 --witness_temperature 0.3 --base_zeta 5.0 --guidance_scale 0.0
--controlnet_scale 0.5 --loss_fn mmd --mode gender --seed 1
--target_prompts "Man:...:10" "Woman:...:10"
```

Every earlier Scenario-B arm ran **backsel OFF, no witness**, `base_zeta` 4 (then 12),
**50+50** targets, seed 42, and our own trust region. So the schedule was never the
problem — my "the working config is a short schedule" inference is **WITHDRAWN** (it came
from the stale committed example, and CLOSE-OUT 3's headline rested on it).

## Status of everything previously concluded about Scenario B

**All Scenario-B conclusions are re-labelled "our-configuration (no witness back-selection)"
results and are SUPERSEDED pending the re-run.** They do not test Ori's method. Concretely:

* The headline "no configuration makes SD-scale distributional guidance descend over the
  full schedule, though a 20-step version does" — **withdrawn as a claim about the method**;
  it stands only as a statement about the no-backsel configuration.
* P-C1/P-C2 verdicts, the zeta 4 -> 12 calibration, the trust-region work, and the
  band/step-size analysis — all conditional on that configuration.

**What survives unchanged**, because it characterises the runs that were actually made and
does not depend on which selection rule is used:

* the **step-survival** measurement (0.147 on the 125-step schedule vs 0.495 on 20 steps) —
  a property of applying corrections along a fine vs coarse DDIM trajectory;
* the **drift control** (zeta = 0 rises +0.0556 +- 0.0140, so the point arm's rise is drift,
  residual not resolvable) — it contains no back-selection at all;
* the **reachability defect** (G_bal's spread is prompt-induced; unreachable under a neutral
  eval prompt) — a property of the target construction;
* the **P-C1 power defect** (a threshold calibrated on one arm's noise cannot adjudicate the
  other) — a methodological finding;
* the PGD arm's failure across six runs / five configurations — it never used back-selection
  by design.

The old numbers, tables and the debugging narrative are kept in RESULTS.md / DEBUG_B.md
under that label. Nothing is deleted.

## The port (so all arms share one code path)

`compute_witness_scores` (+ main's module-level `rbf_kernel` / `estimate_bandwidth`) is
ported into our `src/metrics.py`; `_witness_select_indices`, `_log_witness_selection`,
`_expand_rows_by_count` and the witness branch into our `src/generation.py`; flags
`--backsel_rule witness --witness_floor --witness_temperature --witness_bandwidth_scale
--witness_kernel_alpha --witness_replacement` and a `--backsel_k` alias (so Ori's command
line runs verbatim) into `run_mlgd_f.py`. Our existing flags are unchanged.
**Equivalence is asserted, not assumed:** `tests/test_witness_port.py` carries main's
functions verbatim as the reference and checks bitwise-identical selections and
probabilities across k in {1,5,20,40} x floor in {0,0.3,1} x T in {0.3,1,2} x
replacement in {False,True}, plus Ori's exact (k=50 of 100, floor 0.0, T 0.3) case
(88 tests). That equivalence is what makes the comparison legitimate.

## Re-spec: shared backbone for every arm

Backbone (all arms): `n_steps 250, start_step 125, num_variations 100, base_zeta 5.0,
guidance_scale 0, controlnet_scale 0.5, seed 1, 10 targets per group, trust region OFF`.

* **Arm 3 (ours / full CDM)** — `--loss_fn mmd --backsel_k 50 --backsel_rule witness
  --witness_floor 0.0 --witness_temperature 0.3`. Exactly Ori's configuration.
* **Arm 2 (point)** — same backbone and the same witness back-selection, with the point
  loss. **How back-selection applies:** at the backbone's `num_variations = 100` it applies
  normally — the loss is `mean_i ||e_i - y*||^2` over 100 samples and witness selection
  chooses which 50 to backprop through. One coherence fix was required and is registered
  here: the witness must be scored against **y\*** (a 1-row target), not against a
  distributional target set the point loss never sees, otherwise selection and objective
  disagree; implemented as `witness_target` in `variation_objective` and used automatically
  when `--loss_fn point`. With a single target row the score reduces to
  `mean_i k(x_l, x_i) - k(x_l, y*)`, which is the natural analogue. **Where it IS
  meaningless:** the ORIGINAL arm-2 definition (`G = delta_{y*}`, `n_cond = 1`) — with one
  sample there is nothing to select and `k >= n` is the identity. That variant is retained
  as a labelled secondary arm (`N_COND=1`, dir `point_n1_nobacksel`) so the earlier
  registration is still represented; the backbone version carries the arm-2 claim.
* **Arm 1 (PGD)** — unchanged, with its own eps sweep; it has no conditional batch, so
  back-selection does not apply.
* **Secondary (labelled, optional):** `ZETA=12 TRUST_TAU=0.2 BACKSEL_K=0` reproduces the
  superseded configuration for continuity. Not part of the main comparison.

## Run order

**Scenario A runs FIRST** (user's instruction, 2026-09-13): its target is
`S_G = 100 samples of f_phi(x_f, neutral)` with `y* = CLIP(y_f)` and `x_f = HED(y_f)` a
zero-loss solution by construction — i.e. reachable, which is exactly the defect Scenario B
has. Scenario B follows.

## No prediction is re-registered here

This amendment fixes the code path and the specification. Predictions for the re-run
(the analogues of P-B1..P-B4 and P-C1) will be registered in a separate dated section
BEFORE the first re-run result is looked at.

---

# AMENDMENT 5 — P-C1 is the wrong PRIMARY criterion; final fresh-sample MMD is (2026-09-15)

*Registered after Scenario A landed at Ori's backbone and BEFORE Scenario B is read. This
changes WHAT WE REPORT AS PRIMARY. It is not a re-scoring: every P-C1 verdict already
computed stays printed exactly as it was, including the two that disagree with the outcome.*

## What Scenario A showed

| arm | final MMD (2000 fresh) | vs common unguided ref 0.2059 | P-C1 |
|---|---|---|---|
| mmd + witness k=50 | **0.1557** | **+0.0502 (wins)** | **FAIL** (delta -0.0008, corr -0.266, 2.8 SE) |
| point + witness k=50 | 0.2408 | -0.0349 (loses) | **PASS** (delta -0.0610, corr -0.832, 6.2 SE) |
| point n_cond=1 | 0.5994 | -0.3935 | FAIL (underpowered) |

## Why each disagreement happens — they are NOT the same phenomenon

**The mmd arm: a genuine window artefact in P-C1.** Its trace is not flat. Per-10-step
means run `0.2378, 0.1944, 0.1596, 0.1486, 0.1263, 0.1131, 0.1189, 0.1383, 0.1181, 0.1256,
0.1124,` then `0.1559, 0.2696`. From steps 1-10 to steps 106-115 the loss falls **-0.1349
= 12.6 SE**. What kills the P-C1 number is a single late step: **step 116 applies
||correction|| = 18.6** (run median 2.47, i.e. 7.5x; ratio to the tau=1 noise cap 0.824 vs
a median ratio of 0.038) and the trace never recovers (0.259, 0.257, 0.159, 0.221, 0.253,
0.301, 0.350, 0.275, 0.170). P-C1 compares first-10 with last-10 and so straddles exactly
that blow-up, reporting -0.0008. **P-C1 measured the endpoints of a U-shaped trace and
called a 12.6-SE descent "flat".**

**The point arm: NOT a defect — it is the thesis.** Its trace measures its OWN objective
(the point loss ||e - y*||^2), while the final metric measures MMD to S_G. It descends its
own objective monotonically (0.4084 -> 0.3410) and thereby gets WORSE on the distributional
metric (0.2408 vs 0.2059 unguided). An arm that successfully optimises a point target and
is thereby worse at matching a distribution is exactly what P-B1/P-B2 predict. P-C1 is
working correctly there; it is simply answering a different question from the one the paper
asks.

So the criterion is not "broken in both directions". It is (i) fragile to trace shape for
the arm whose trace and metric coincide, and (ii) arm-relative by construction — each arm's
trace is its own loss, so traces are not comparable ACROSS arms at all.

## The primary criterion, stated for the paper

**PRIMARY: the final fresh-sample MMD (2000 samples, fixed eval seeds) against the ONE
common unguided reference.** It is the same estimand for every arm, it is the quantity the
method claims to improve, it was pre-registered from the start (P-C2's content and the
Scenario-B main table), and its measurement noise is known and tiny (half-split spreads
0.0002-0.0013 in Scenario A).

**DIAGNOSTIC ONLY: the guidance-loss trace and P-C1.** Retained and reported, with per-arm
SE and the underpowered flag, because it is what exposes mechanism (it is how the step-116
blow-up was found). It is not a pass/fail gate on the method.

Two reporting rules follow, fixed now:
1. Trace deltas are reported with the full per-10-step profile, never as a single
   first-vs-last number, so a U-shape cannot masquerade as flatness.
2. A trace is compared only against the SAME arm's own trace, never across arms.

## Also fixed now (bugs, not scoring)

* **`cache/A/oracle.pt` was never written.** Setup job 46160178 exited 1: the oracle block
  fed fp16 latents to the fp32 sprinter VAE (`Input type (c10::Half) and bias type (float)`)
  — the cast that `generate_and_store_cs` / `evaluate_distribution_mmd` both perform was
  missing. Everything else in `cache/A` was written before the crash, so **the completed A
  arms are unaffected**; only the oracle floor L(x_f) and the ||x* - x_f|| columns are
  missing. Fixed, plus `--oracle_only` / `ORACLE_ONLY=1` to compute it from the EXISTING
  cache without rewriting anything the finished runs used.
* **The Scenario-A `pgd` directories are mislabelled `_witness50`.** They used no
  back-selection (arm 1 has no conditional batch); an earlier version of the tag logic
  appended the suffix to every arm. Fixed for future runs; the existing A directories keep
  their names and are labelled in the write-up.

## The trust region, as a testable follow-up (NOT a change to the backbone)

8 of 125 steps exceeded a tau = 0.2 noise cap, and the step-116 blow-up (ratio 0.824) is
exactly the heavy-tail event the noise-level trust region was built to clip. That makes
"`--trust_noise 0.2` on top of Ori's backbone prevents the late rebound and improves the
final MMD beyond 0.1557" a concrete, falsifiable follow-up. It is NOT run here and is not
part of the backbone; Ori's configuration is reported as-is.
