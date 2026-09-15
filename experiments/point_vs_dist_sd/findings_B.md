## Findings (hand-written; `findings_<scenario>.md`, appended by build_report.py)

### Headline (UNCALIBRATED configuration: zeta 4, trust region OFF — the original registration)
### NO arm beat its reference, and the target is barely reachable

> **Scope.** Everything in this section describes the original zeta = 4 / no-trust runs.
> Its headline survived calibration: at zeta 12 + tau 0.2 the full-schedule arm is still
> worse than its baseline (0.6358 vs 0.6000). What changed is that a 20-step schedule DOES
> beat its baseline (0.4911 vs 0.5938) — see CLOSE-OUT at the end.
> The step-size diagnosis (DEBUG_B.md) showed those runs never entered the measured
> descent band; results at the calibrated step size are scored separately in
> **CLOSE-OUT 2** at the end of this document. Do not read the numbers below as the
> method's capability — read them as the uncalibrated baseline.

Every arm's 2000-sample MMD is **worse** than both the common reference f_phi(x0) = 0.3776
and its own reference. The ordering predicted by P-B3 does hold (mmd 0.643 < point 0.760 <
pgd 0.700/0.745/0.801 — mmd best among the optimized arms), but the improvement the
experiment was designed to show does not exist here. Three measurements explain why.

**1. The target's spread is prompt-induced, and the eval prompt is neutral.**
G_bal's PC1 variance (0.1056) is almost entirely BETWEEN groups: Man mean −0.319
(within-var 0.0039), Woman +0.319 (within-var 0.0018) — 0.319² ≈ 0.102 of the 0.1056 is
the gap between the two prompt-conditioned clusters. The targets were generated with two
different prompts; every arm is evaluated (and guided) with ONE neutral prompt. Matching
G_bal therefore requires the scribble alone to make f_phi produce a 50/50 male/female
mixture. The best set anyone achieved on that axis is the do-nothing reference
(f_phi(x0), ratio 0.300); every optimized arm is far below it. **The target is close to
outside the reachable family {f_phi(x, neutral) : x}** — a feasibility problem in the
experiment's design, not (only) a failure of the optimizers.

**2. There is an MMD floor near 0.36–0.38.** The target's own Man half scores
MMD = 0.3621 against the full G, the Woman half 0.3656; f_phi(x0) scores 0.3776. So
"cover one mode well" already costs ~0.36, and the head-room between the reference and a
perfect match is small compared to the damage any arm did (0.64–0.80).

**3. The guidance objective itself never decreased.** Over the mmd arm's 125 steps the
guidance MMD went 0.570 → ~0.61 (min 0.494 at step 56, slope −0.003 across the run,
corr(step, loss) = −0.03: flat). The point arm went 0.486 → ~0.88 (rising), with
correction norms median 9.4 and max 103 in a 16384-element latent — 6.4 % of its steps
exceeded the noise-level trust cap (max 2.74x) while the mmd arm never did (max 0.70x).
So arm 2's steps are destructive-large, and arm 3's guidance is not making progress on
its own objective at zeta = 4.0 with this step rule.

### Anomaly (a): the two "unguided references" are different objects — deltas were not comparable

| arm | its "regular" path | MMD -> G |
|---|---|---|
| pgd_eps* | f_phi(**x0**) — the source scribble, no diffusion at all | 0.3776 |
| point, mmd | f_phi(**unguided DDIM scribble**) — SDEdit init + 125 unguided DDIM steps | 0.6000 |

The saved embeddings confirm they are distinct sets (MMD between them 0.4772) and that
each group of arms shares its own reference byte-for-byte. The gap is itself a result:
**the unguided diffusion trajectory destroys most of the conditional diversity** — PC1
variance ratio falls from 0.300 (x0) to 0.046 (unguided DDIM) — so the diffusion arms
start 0.22 MMD in the hole before any guidance is applied. All five arms are therefore
re-scored in the main table against the ONE common reference f_phi(x0), the do-nothing
baseline every arm shares; "Δ vs own ref" is kept for completeness.

### Anomaly (b): the intermediate "improvement" was never a like-for-like comparison

`run_mlgd_f.py`'s per-step `[eval]` line prints `mlgd_f = the guidance objective` against
`unguided = a fresh-sample eval`. These differ in four ways: (i) the guidance number is
the MMD of **N in-graph variations conditioned on the GUIDED pred_x0 float pixels**
(N = 100 here, 8 in the smoke) while the eval number decodes the **UNGUIDED** latent to an
8-bit PIL scribble and draws `eval_n_intermediate` (8–10) fresh samples; (ii) the two use
different sample sizes, so the median-heuristic bandwidth differs; (iii) different sprinter
seeds; (iv) guided vs unguided conditioning. The smoke's apparent 0.526 → 0.448 "descent"
was a small-n comparison of the guidance objective against an unrelated unguided eval.
In the full run the same deltas oscillate around zero (+0.038, −0.020, −0.039, −0.004,
+0.059), consistent with the flat guidance trace and with the final 2000-sample eval.
**No contradiction: the final eval is the trustworthy number, and nothing was improving.**

### What would need to change (future work — NOT tuned now, per the honesty clause)

1. **Make the target reachable**: evaluate/guide with the same conditioning that generated
   the targets (per-sample prompt sampled from {man, woman}), or build G_bal itself with
   the neutral prompt so the spread is scribble-attainable. This is the design fix the
   feasibility measurement points to, and it applies to all three arms equally.
2. **Step-size regime**: zeta = 4.0 with `zeta = base_zeta / L` produced flat (mmd) or
   destructive (point, ||corr|| up to 103) steps. A zeta sweep, or the noise-level trust
   region (`--trust_noise`, already implemented and off here), is the obvious next control.
3. **Start step / horizon**: guidance from t_start = 125 of 250 fights an unguided
   trajectory that already collapses diversity (ratio 0.300 → 0.046); earlier start or
   more steps would test whether the collapse is recoverable.
4. **Arm 1 optimizer**: Adam at lr = eps/10 with ONE stochastic sample per step never
   descended its own objective at any eps (below); a larger inner batch or a smaller lr
   would test whether that is the estimator or the landscape.

These are recorded as future work. No parameter was changed after seeing these results.

## What went wrong and how we found it (reproducibility note)

This experiment produced a null result, then a diagnosis, then a fix. The path matters
because three of the four failure modes were silent — they would not have surfaced from
the headline numbers alone.

**1. A "COMPLETED" job that had not completed.** The first full `mmd` run (job 46103120)
was reported by SLURM as `COMPLETED 0:0` after 3 h 27, but its log stopped mid-step 98 of
125, `profile.json` was left 0 bytes, and no eval ran; `runs/B/mmd/` still held the earlier
SMOKE results (16 targets, 64 eval samples). The trigger was a full `/sci/home` (96 %,
233 MB free — visible only as a wandb "no space left on device" line in the `.err`).
Aggregating that directory would have put 8-variation / 64-sample smoke numbers into a
paper table under the label of a 100-variation / 2000-sample run. *Guard added:*
`build_report.py` now refuses any run whose target-set hash differs from the scenario
cache or whose eval count differs from the modal one, and lists it as excluded with the
reason. The partial run is archived as `runs/B/mmd_incomplete_46103120`.

**2. Two different "unguided references".** The PGD arm's reference is f_phi(x0) (MMD 0.3776)
and the diffusion arms' is f_phi(unguided DDIM scribble) (0.6000) — different sample sets
(MMD 0.4772 apart), so the per-arm deltas printed by the pipeline were not comparable.
*Fix:* every arm is additionally scored against one common reference, f_phi(x0).

**3. A per-step "[eval]" line that compared unlike things.** `run_mlgd_f.py` prints
`mlgd_f =` the guidance objective (N in-graph variations on the GUIDED pred_x0) next to
`unguided =` a fresh 8-10-sample eval of the UNGUIDED latent. In the 5-step smoke this
looked like descent (0.526 -> 0.448 vs 0.551) and was cited as evidence the run was
working; in the full run the same deltas oscillate around zero. It is not a like-for-like
comparison and should not be read as a learning curve.

**4. The real fault: step size, found only by probing.** The completed run's guidance MMD
was flat (0.570 -> 0.659, corr(step,loss) = -0.03) although sign, coupling, step scale and
the MMD estimator all checked out (DEBUG_B.md §1-4). The decisive clue was quantitative:
median ||grad|| 0.273 with 45.8 of realised latent movement predicts ~12 loss units of
available descent; observed +0.006. A 30-minute probe at one latent (`debug_gradient.py`,
job 46134134) then separated the candidates: fp16 underflow ruled out (cos 0.998 across
loss_scale 1..1e4); gradients mutually orthogonal (mean cos -0.027, SNR 0.391) BUT a single
step along one of them cut the loss 0.575 -> 0.446, with CRN and fresh seeds identical.
So the direction was usable and the run was simply under-stepping: the useful band is
||step|| ~ 6-24 and the run's median step was 1.84, **3.2x too small**. Raising base_zeta
4 -> 12 (median step 5.5, inside the band) and enabling the noise-level trust region
(TAU = 0.2, cap 21.6 early, clipping 17 % of steps) was pre-registered with a falsifiable
criterion (P-C1) before rerunning; the 20-step gate then passed (0.4805 -> 0.4210,
corr -0.567).

**Transferable lessons.** (i) Never trust a scheduler's exit status alone — verify a run
from its own artefacts (step count, target hash, eval size). (ii) A flat loss trace is not
evidence of a broken gradient; measure the descent band before concluding anything.
(iii) The per-step guidance objective and the fresh-sample eval are different statistics
on different conditioning — only the fresh-sample eval is the result. (iv) On a
noise-dominated guidance signal, step-size control (calibrated zeta + trust region) is the
intervention that matters — the same conclusion the synthetic campaign reached
independently (`experiments/model-optimization/IMPROVEMENTS.md` §1).
