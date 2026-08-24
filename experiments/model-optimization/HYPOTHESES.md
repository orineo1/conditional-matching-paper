# HYPOTHESES -- every method considered in the CDM performance campaign (Agent 7)

Merged from `hypotheses/agent{1..5}.yaml` (machine-readable: `hypotheses.yaml`, 49 pre-registered entries). Verdicts follow `VERIFICATION.md` wherever the verifier (Agent 6) checked the item; other rows carry the implementing agent's own numbers. Screening numbers are paired diffs `base - cand` of the failure-penalised exact GMM L2 (+ = candidate better; `*` = p <= 0.05; 40 restarts, offset 0); held-out numbers are the same on 100 restarts at offset 1000 (VERIFICATION.md 5).

Verification column: **V** = verifier-confirmed; **I** = implementer-reported, not independently verified; **S** = static analysis only, not run.

| group | method | agent | verdict | verif. | reason (numbers) | source |
|---|---|---|---|---|---|---|
| exact loss | cached target-target block (XX/XY only) -- fast_mmd.MMDFixedTarget / BatchedMMD | 1,2,5 | **promoted** | V | exact to 1.3e-14 (f64); MMD fwd+bwd 1.96->0.42 ms (n=8, 4.6x); whole loop 1.66-2.02x; same cond calls | VERIFICATION.md 4.1; exact_loss/REPORT.md; end_to_end_results.csv |
| exact loss | matmul distances (dist='mm') | 2 | **promoted** | V | exact (verifier grid); 6.3x f32 / 10.2x f64 micro-bench geo-mean vs reference | exact_loss/bench_summary_small.md |
| exact loss | powchain kernel E+E^2+E^4+E^8+E^16 | 2 | **promoted** | V | exact (few ulp); 0.28 ms at n=8 (7.0x); best end-to-end 1.77-2.02x | VERIFICATION.md 4.1; end_to_end_results.csv |
| exact loss | chunked fused XY (autograd.Function) | 2 | **conditional** | V | exact 1st order; speed-neutral; memory only (m >> 2000) | exact_loss/REPORT.md 1 |
| exact loss | batched sets (B sample sets, one Y) | 2 | **conditional** | V | exact; 12.6x vs 3 reference calls, ~2x vs 3 fixed_mm calls; needs concatenated sampler outputs | exact_loss/REPORT.md 2 |
| exact loss | population-GMM target (closed-form KME) | 3 | not-run | I | cos 0.989-0.998 vs empirical gradient, 0.02-0.55x cost, deterministic; changes the objective; Stage-1 screen never run | approx_loss/REPORT.md |
| exact loss | tabulated KME (d=1) | 3 | rejected | I | cos 0.999 at 0.04-0.12x but dominated by population-GMM; d=1 only | approx_loss/THEORY.md FFT section |
| approx loss | RFF / ORF random features | 3 | rejected | I | D=256 costs 1.2-2.2x MORE than cached-exact at m=250 with cos 0.82-0.90 (d=8,16); d=768 needs D~2^16; alpha=2 kernel not PD | approx_loss/THEORY.md 2a-b, DIAGNOSTICS.md |
| approx loss | Nystrom (target landmarks) | 3 | rejected | I | gradient collapses off-target (cos 0.25, d=768); d=8,16 cos 0.93-0.94 at 4-6x the exact cost | approx_loss/THEORY.md 2c |
| approx loss | linear-time / block estimator = target subsample (fixed subset) | 3 | not-run | I | 0.28-0.34x cost at B=64, cos 0.96-0.98 (d<=16), 0.74 (d=768); kept as a knob; no L2 screen | approx_loss/REPORT.md |
| approx loss | sliced W2 (fixed projections) | 3 | rejected | I | different objective: cos 0.3-0.9 (d<=16), 0.1 (d=768); MNIST must not evaluate with the guidance statistic | approx_loss/THEORY.md 2e |
| approx loss | FFT / NUFFT kernel mean embedding | 3 | rejected | I | no grid in d>=3; one-off d=1 table fill is 0.2 ms by direct sum; no candidate proceeds | approx_loss/THEORY.md |
| estimator | Adam temporal arm (repo baseline, rho=0.4) | 4 | rejected | I | worse than plain guidance in 5D/10D (5D n=8 0.61 vs 0.50; 10D n=32 0.57 vs 0.49); regime flip at n=32 in 2D; plain+trust_noise1 beats Adam in every dim | estimator/REPORT.md 3.3, 4 |
| estimator | norm-only (Adam beta1=0) | 4 | rejected | I | 2D n=4 +0.213*, n=8 +0.126*; 10D -0.12..-0.18* at every n | estimator/round2_matrix.md |
| estimator | absolute clip 0.5 | 4 | rejected | V | 2D n=4/8 +0.36/+0.19 (held-out); 10D n=32 -0.075* held-out FAIL; 5D inconclusive -> scale-dependent, one scale | VERIFICATION.md 5.4 |
| estimator | absolute clip 0.1 | 4 | rejected | I | 10D n=4/8 +0.09/+0.12*; 2D n=8/16/32 -0.13/-0.23/-0.36* (over-clips) | estimator/round2_matrix.md |
| estimator | unit-norm gradient (0.4 / 0.1) | 4 | rejected | I | unit0.4 2D +0.31/+0.12 but 10D n=32 -0.093*; unit0.1 2D -0.18..-0.21* | estimator/round2_matrix.md |
| estimator | relclip2 (2 x running median) | 4 | rejected | V | best 2D rule (held-out 0.153/0.139 at n=4/8, +0.44/+0.28*), 5D inconclusive, 10D n=4 -0.047* -> not promoted (one scale) | VERIFICATION.md 5.4 |
| estimator | relclip_ema2 (2 x EMA) | 4 | rejected | V | 2D PASS (4/4), never significantly negative, 5D/10D null -> safe but 2D-only | VERIFICATION.md 5.4 |
| estimator | relclip1 == qclip0.5 (1 x median) | 4 | rejected | V | 2D n<=16 wins; 10D n=32 -0.076* (held-out); 2D n=32 mmd2_eval -0.049*; duplicate rule counted once | VERIFICATION.md 5.4, red flag 3 |
| estimator | relclip0.5 / relclip_ema0.5 / relclip_ema1 / qclip0.75 | 4 | rejected | I | 0.5 variants 2D n=32 -0.23/-0.25*; ema1 10D n=32 -0.037 n.s. but no 5D/10D win; qclip0.75 2D-only | estimator/round2_matrix.md |
| estimator | trust_noise1: ||Delta_t|| <= 1.0*sqrt(1-alphabar_t)  (step_clip='noise', step_tau=1) | 4 | **promoted** | V | held-out PASS 2D (+0.40/+0.25/+0.09/+0.02*) and 10D (+0.05/+0.12/+0.08* at n=4/8/16), 5D all + (n=8 +0.036*), no significant regression in 12 cells; 10D frontier at every n<=32 | VERIFICATION.md 5.1-5.4 |
| estimator | trust_noise0.3 / trust_noise0.1 | 4 | rejected | I | 0.3: 2D n=32 -0.037 n.s., otherwise wins at n<=8 in all dims; 0.1: 2D -0.17..-0.38* | estimator/round2_matrix.md |
| estimator | trust_ddim (tau 0.1/0.3/1): ||Delta_t|| <= tau ||x_ddim - x_t|| | 4 | rejected | I | 10D wins +0.08..+0.19* but 2D -0.21..-0.46*; tau=0.1 equals the UNGUIDED score in 2D/5D (guidance off) | estimator/REPORT.md 3.2 |
| estimator | sqrt_floor loss transform | 4 | **conditional** | V | held-out PASS 2D (+0.23/+0.12/+0.05/+0.04) + 5D (n=8 +0.049*, n=32 +0.023*), FAIL 10D n=32 (-0.038*) -> 2D/5D only | VERIFICATION.md 5.4 |
| estimator | sqrtfloor_clip0.5 | 4 | **conditional** | V | PASS 2D (4/4) + 5D (+0.059*/+0.035*; 5D frontier n=4/8/32), FAIL 10D n=32 (-0.088*) | VERIFICATION.md 5.4 |
| estimator | sqrtfloor_clip0.1 / sqrtfloor_relclip1 / sqrt_abs_eps (SD transform) | 4 | rejected | I | clip0.1 combo 2D -0.22..-0.33*; relclip1 combo worse than its parts (10D n=32 -0.068*); sqrt_abs_eps null | estimator/round2_matrix.md |
| estimator | bandwidth policies (pooled / pooled_floor vs fixed) | 4 | rejected | I | null everywhere (|diff| <= 0.08, p > .05); pooled collapses only with tiny targets (unit test) | estimator/round2_matrix.md; REPORT.md 2 |
| estimator | adaptive n_t (agreement 0.5/0.8, improvement; equal total calls) | 4 | rejected | I | 2D n=8 -0.20..-0.34*, 10D n=32 -0.08..-0.14*; binding constraint is the step rule, not the per-step budget | estimator/round2_matrix.md; REPORT.md 4 |
| estimator | adaptive recurrence v1 (<= 2 recurrences) | 4 | rejected | I | null at up to 2x calls; 5D n=32 -0.111*, 10D n=4 -0.088* | estimator/round2_matrix.md |
| estimator | stale gradient cache (refresh every 2/3) | 4 | rejected | I | 1/k calls but 2D n=8 -0.23/-0.32*, 5D n=4 -0.34/-0.67*, 10D n=32 -0.20/-0.24* | estimator/round2_matrix.md |
| estimator | common random numbers (frozen conditional noise; approximate) | 4 | rejected | I | null; 2D/adam n=8 -0.133*, 10D n=32 -0.065* | estimator/round2_matrix.md |
| estimator | antithetic conditional noise | 4 | rejected | I | null; 2D/adam n=32 -0.043* | estimator/round2_matrix.md |
| estimator | clip + Adam / clip + LGD combinations | 4 | rejected | I | clip before Adam: 10D n=16/32 -0.075..-0.122*; clip+LGD 10D n=32 -0.154*; clipped no-LGD n=8 beats LGD n=8 and n=32 in 2D only | estimator/REPORT.md 3.3 |
| systems | batched restarts (B chains in one batch) | 5 | **conditional** | V | ~10x throughput at B>=8 (verifier 4-15x loaded Mac); per-step grad jumps up to 8% rel at ReLU boundaries -> statistical equivalence only; screening use, not reproduction | VERIFICATION.md 4.2 |
| systems | batched LGD perturbations (3n rows, pre-drawn noise) | 5 | **promoted** | V | EXACT 0.0; 1.4x on LGD cells | systems/BENCH.md; check_systems.log |
| systems | cached YY / batched MMD (BatchedMMD) | 5 | **promoted** | V | REORDER 1e-6 per step; 1.5x (no-LGD) to 4.5x (LGD) per restart | systems/BENCH.md |
| systems | torch.Generator instead of global manual_seed | 5 | **promoted** | V | EXACT 0.0; 99 us x M per step saved | systems/AUDIT.md 1.3 |
| systems | requires_grad_(False) on frozen models | 5 | **conditional** | V | EXACT; wall within noise on the synthetic MLPs; hygiene (MNIST/SD memory) | systems/BENCH.md |
| systems | lean DDIM step / cached time embedding / torch.full | 5 | **conditional** | V | EXACT; +1-5% | systems/AUDIT.md 1.3 |
| systems | micro-batched MMD gradient (chunked VJP) | 5 | **conditional** | I | exact identity 3.4e-16; no gain at n<=256; for n>>1e3 / 768-d | systems/microbatch_mmd.log |
| systems | torch.compile (loop / MMD) | 2,5 | rejected | I | loop 0.84x +16 s compile; MMD 1.4-2.5x steady but 2-7 s per shape, 26k-58k-call break-even, macOS inductor hangs | systems/BENCH.md; exact_loss/bench_compile_cpu.md |
| systems | MPS device | 5 | rejected | I | 5x slower at B=1; 24 r/s at B=32 vs 70-90 CPU | systems/BENCH.md |
| systems | float64 throughout | 5 | rejected | I | diagnostic: 0.5-0.9x; reveals float32 chaos (|dx| 0.04-1.1, mode flips) | systems/AUDIT.md 1.0 |
| systems (MNIST, static) | cache cond_embed across the CM ladder; batch perturbations + seeds; drop retain_graph; host-sync removal; skip uniform-target inner loop | 1,5 | not-run | S | 5->1 encoder evals (EXACT); batching est. 5-15x; ~25% of uniform runs wasted after step_size=0 | systems/AUDIT.md 2; profiling/baseline_profile.md 4b |
| systems (SD, static) | gate visualize_step / eval; gs==0 single-batch architect UNet; un-nest checkpointing; batch variations with per-sample generators; cache prompt embeds + freeze text encoders; keep CLIP on GPU, stop VAE dtype toggles, drop gc/empty_cache | 1,5 | not-run | S | est. 25-35% (vis) + 15-25% (ckpt) + 1.5-2x sprinter path; all EXACT for the guided path | systems/AUDIT.md 3; profiling/baseline_profile.md 4c |
| systems (SD, static) | fp16-fix VAE / bf16 CLIP / truncated sprinter backprop / 1-step sprinter | 5 | not-run | S | CHANGES NUMERICS; est. ~2x on VAE/CLIP; needs grad-cosine + delta-MMD validation | systems/AUDIT.md 3.4, 3.7 |

## Summary

* **Promoted (verifier-confirmed):** cached-target exact MMD (`exact_loss/fast_mmd.py`, drop-in, 4-7x on the loss, ~1.7-2x on the loop); batched LGD perturbations, generator seeding (exact); and ONE estimator rule, `trust_noise1` (`step_clip='noise', step_tau=1.0`), the only rule that is a credible Pareto improvement at two task scales (2D, 10D) with no significant regression at any of the 12 held-out cells.
* **Conditional:** `sqrt_floor` and `sqrtfloor_clip0.5` (2D/5D only; regress at 10D n=32); batched restarts (statistical equivalence only, ~10x throughput); hygiene items (requires_grad_(False), lean DDIM); chunked / batched / micro-batched MMD forms (exact, no gain at the paper's sizes).
* **Rejected:** every approximate loss (RFF/ORF, Nystrom, sliced, FFT, subsample-as-approximation); absolute clipping (scale-dependent), relclip family (2D-only or 10D regressions), unit/norm-only, trust_ddim, adaptive n_t, adaptive recurrence, stale gradients, CRN, antithetic, bandwidth policies, Adam (+clip) in 5D/10D, torch.compile, MPS, float64-as-production.
* **Not run:** population-GMM target (promising diagnostics, different objective), target subsample knob, all MNIST and SD static recommendations (no GPU / checkpoints locally; cluster SD/MNIST stages were out of scope).
