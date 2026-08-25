# SD perf campaign — seed-1 report (2026-08-24)

Task: 50/50 gender targets (50 man + 50 woman, HED scribble), SDXL-base architect,
DDIM 100 steps from 50 (50 guided steps), base_zeta 5, N = 32 variations, sprinter
SDXL-Turbo + ControlNet-Scribble, CLIP MMD. All arms `--seeded_rng --profile --seed 1`,
shared target cache, final eval 2000 fresh photos. Jobs 45939271-76, salmon L40S.
Full tables: `RESULTS.md` (regenerate with `analyze_sd.py`).

## Quality (single seed — see noise floor below)

| arm | flags | final MMD | unguided | delta |
|---|---|---|---|---|
| baseline | vis on, CFG double batch | 0.400 | 0.520 | +0.120 |
| novis (control) | `--no_vis --arch_single_batch` | 0.431 | 0.521 | +0.090 |
| trust tau=0.25 | novis + `--trust_noise 0.25` | **0.375** | 0.521 | +0.146 |
| backsel k=8 kcenter | novis + `--backsel 8 kcenter` | 0.451 | 0.521 | +0.070 |
| backsel k=8 uniform | novis + `--backsel 8 uniform` | 0.423 | 0.521 | +0.097 |
| trust_backsel | trust + backsel kcenter | 0.460 | 0.521 | +0.061 |

## Cost (profile.json, mean seconds per guided step; CUDA-synced sections)

| arm | total | backward (VJP incl. ckpt recompute) | sprinter fwd (graph) | VAE+CLIP (graph) | no-grad pass (sprinter+VAE+CLIP) | vis | peak VRAM GB | fwd / diff. variations |
|---|---|---|---|---|---|---|---|---|
| baseline | 46.3 | 43.5 | 11.6 | 2.2 | — | 3.2 | 34.3 | 32 / 32 |
| novis | 43.1 | 43.6 | 11.5 | 2.1 | — | — | 34.2 | 32 / 32 |
| trust | 42.8 | 43.1 | 11.5 | 2.1 | — | — | 34.2 | 32 / 32 |
| backsel k=8 kcenter | **16.9** | 10.2 | 2.6 | 0.5 | 6.5 | — | **24.9** | 40 / 8 |
| backsel k=8 uniform | 17.2 | 10.4 | 2.6 | 0.5 | 6.6 | — | 24.9 | 40 / 8 |
| trust_backsel | 17.2 | 10.4 | 2.6 | 0.5 | 6.6 | — | 24.9 | 40 / 8 |

(Section sums exceed the total slightly because the closure's forward sections are
nested inside the checkpointed call that the `backward` section also times during
recompute; architect ~0.2 s, MMD/select/denoise < 0.03 s, intermediate eval 0.15 s
amortised.) Optimisation wall time: 62-65 min (full backprop arms) vs 40 min
(backsel arms) for 50 steps, plus ~8 min of target generation and 2000-sample eval.

Reading the split: one in-graph variation costs (11.5 + 2.1 + 43.6)/32 = **1.8 s**;
one no-grad variation costs 6.5/32 = **0.2 s** — a 9:1 ratio (the audit's accounting
assumed 5:1; the extra comes from the fp32 VAE decode + fp32 CLIP being recomputed
inside the checkpoint). Backsel k of N therefore costs `0.2 N + 1.8 k` s/step:
k=8 -> 16.9 s (2.6x), k=16 -> ~31 s (1.4x), k=4 -> ~11 s (3.9x). Peak VRAM drops
34 -> 25 GB because the k-row graph pass at `variation_batch_size=1` replaces the
32-row one AND the no-grad pass runs without the checkpoint's recompute workspace.
`regen_max_abs_err = 0.0` on every step of every backsel arm: the seeded replay is
bit-identical (deterministic sprinter kernels at batch 1).

## The noise floor — why seed 1 cannot rank the arms

1. `baseline` vs `novis` differ ONLY by `--arch_single_batch` (vis does not touch
   the guidance RNG under `--seeded_rng`: variation seeds are explicit and the vis
   sprinter calls draw from the global RNG). Step-1 MMD: 0.5256 (baseline, CFG
   batch 2) vs **0.5259 in every single-batch arm** — the fp16 batch-dimension
   round-off of the architect UNet is the only difference at step 1 — and by step 2
   the trajectories have diverged (0.461 vs 0.475). Chaotic sensitivity, not RNG.
2. Stronger: `novis`, `trust` and the two backsel arms all compute IDENTICAL step-1
   values (0.5259) and `trust`'s cap never binds until much later (3/50 steps capped,
   first at step > 5), yet `novis` and `trust` already differ at step 2 (0.4753 vs
   0.4770). Those two arms run the same computation at step 2 — the divergence is
   **GPU nondeterminism** (checkpoint recompute of fp16 UNet blocks / non-deterministic
   atomics in the backward) amplified by the chaotic loop. Their final MMDs, 0.431 vs
   0.375, therefore bound the single-run noise floor at **~0.05** in final MMD.
3. So none of the seed-1 gaps (trust -0.056, backsel +0.02/-0.007, trust_backsel
   +0.03) is interpretable. Seeds 2-4 are running; with 4 paired seeds the SE of a
   mean difference is ~0.05/sqrt(4) ~ 0.025 — enough to see a 0.07 effect, not 0.03.

## Trust-region calibration numbers (from the always-on cap logging)

`correction_norm_raw / cap_tau1` per step: median 0.03, but a heavy tail — max 1.0-1.4
(30-45x the median) in every arm. tau = 0.25 binds on 3/50 steps (trust) and 2/50
(trust_backsel): it clips exactly the tail steps and nothing else, which is the
synthetic mechanism (IMPROVEMENTS.md sec 1). tau = 1.0 would bind on ~1 step.

## What is solid after seed 1

* Cost: backsel k=8 is 2.6x per step and -9 GB peak VRAM with bit-exact replay; the
  trust region and single-batch architect are free.
* Quality: undetermined; guided beats unguided in every arm (delta +0.06..+0.15).
