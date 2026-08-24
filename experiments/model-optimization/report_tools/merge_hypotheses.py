"""Merge hypotheses/agent{1..5}.yaml into hypotheses.yaml (one normalised list) and write
HYPOTHESES.md (one row per method considered, with verdict + one-line evidence).

Verdicts: promoted / conditional / rejected / not-run. Verification status is taken from
VERIFICATION.md where the verifier (Agent 6) checked the item; everything else is labelled
"implementer-reported, not independently verified".

Run: /Users/stolk/miniconda3/bin/python experiments/model-optimization/report_tools/merge_hypotheses.py
"""
import os, yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HYP = os.path.join(ROOT, "hypotheses")
FIELDS = ["id", "agent", "name", "mechanism", "expected_benefit", "possible_bias", "affected_tasks",
          "primary_metric", "compute_budget", "rejection_criterion", "verdict", "verification", "evidence", "source_files"]
VER = "verifier-confirmed (VERIFICATION.md)"
IMPL = "implementer-reported, not independently verified"
NOTRUN = "not run (static analysis only)"


def clean(s):
    if s is None:
        return ""
    if isinstance(s, list):
        return ", ".join(str(x) for x in s)
    return " ".join(str(s).split())


# ----------------------------------------------------------------------------- curated verdicts
# key -> (verdict, verification, evidence)   (evidence cites the source file)
V = {
 # Agent 1
 "A1-cache_target_target_kernel_and_split_blocks": ("promoted", VER,
   "Same mechanism as A2-fixed_target_cache / A5-sys_mmd_blocks_cached_yy. Verifier: exact to 1.3e-14 (float64), "
   "4.6-7.0x on the MMD fwd+bwd, 1.6-2.6x per restart (VERIFICATION.md 4.1; profiling/baseline_profile.md: YY block = 94% of kernel entries at n=8)."),
 "A1-batch_lgd_perturbations_into_one_sampler_call": ("promoted", VER,
   "= A5-sys_batched_lgd_perturbations: EXACT (0.0 end-to-end, verifier check_systems.log), 0.74->0.54 s on LGD n=8 (systems/BENCH.md)."),
 "A1-freeze_parameters_and_trim_autograd_graph": ("conditional", VER,
   "requires_grad_(False): EXACT (0.0) but ~0% wall on the synthetic MLPs (systems/BENCH.md; verifier confirmed exactness). "
   "Keep as hygiene; the predicted 10-20% did not materialise. Lean DDIM step / cached time embedding: +1-5% (A5-sys_lean_ddim_step)."),
 "A1-reduce_per_step_framework_overhead (fuse kernel ops / compile)": ("rejected", IMPL,
   "torch.compile: 0.84x steady state +16 s compile on the loop (systems/BENCH.md); on the MMD alone 1.4-2.5x steady state "
   "but 2-7 s compile per shape and 26k-58k-call break-even (exact_loss/REPORT.md 4). The torch.Generator part is promoted as A5-sys_generator_seeding (EXACT)."),
 "A1-sd_remove_non_guidance_sprinter_work": ("not-run", NOTRUN,
   "Static: 5 of 11 sprinter forwards per step are visualisation, second CFG UNet pass, VAE dtype casts, CLIP CPU<->GPU moves "
   "(profiling/baseline_profile.md 4c; systems/AUDIT.md 3). Estimated 25-45% wall/step; no SD run in this campaign."),
 "A1-mnist_cache_image_encoder_and_fix_target_set": ("not-run", NOTRUN,
   "Static: image encoder recomputed 5x per sampler call (15x/step), targets + SWD projections resampled per call, retain_graph=True "
   "(profiling/baseline_profile.md 4b; systems/AUDIT.md 2). Fixed targets/projections is an estimator change -- would need its own screen."),
 # Agent 2
 "A2-fixed_target_cache": ("promoted", VER,
   "Verifier: value + dX gradient <= 1.3e-14 relative vs LossFunctions.MMDLoss over 7 (n,m,d) x 2 seeds, fixed and adaptive bw; 236 author tests pass; "
   "n=8: 1.96 -> 0.42 ms (4.6x); end-to-end 1.66-1.93x (exact_loss/end_to_end_results.csv). float32: reorder-only, trajectories move up to 0.18 (chaos, not error)."),
 "A2-fixed_target_mm": ("promoted", VER,
   "Covered by the verifier's cdist/mm x exp/powchain/loop grid (<= 1.3e-14). 6.3x (f32) / 10.2x (f64) geometric-mean micro-benchmark speedup vs reference (exact_loss/bench_summary_small.md)."),
 "A2-powchain_kernel": ("promoted", VER,
   "Exact (few ulp; verifier grid). fixed_mm_powchain: 0.28 ms at n=8 (7.0x, verifier), best end-to-end variant 1.77-2.02x (exact_loss/end_to_end_results.csv)."),
 "A2-chunked_fused_xy": ("conditional", VER,
   "Exact to first order (verifier chunk 64/7 check <= 1.3e-14; gradcheck). Speed-neutral, memory only -- not needed at m=250; relevant for m >> 2000 on a memory-limited GPU (exact_loss/REPORT.md)."),
 "A2-batched_sets": ("conditional", VER,
   "Exact (== B reference calls, verifier batched() check). 12.6x f32 vs 3 reference calls but only ~2x over 3 fixed_mm calls; useful only once the M=3 sampler outputs are concatenated (A5 batched LGD)."),
 "A2-torch_compile_powchain": ("rejected", IMPL,
   "Steady state 1.4-2.5x over eager, exact, but 2-7 s compile per shape (dynamic=False; n_t and dtype are shapes), break-even 26k-58k calls vs 99-297 per restart; inductor hung 9/12 attempts on macOS (exact_loss/bench_compile_cpu.md)."),
 # Agent 3
 "A3-pop-gmm-target": ("not-run", IMPL,
   "Diagnostics only (approx_loss/REPORT.md): gradient cosine 0.989-0.998 to the empirical-target gradient, cost 0.02-0.12x (d=1) / 0.2-0.55x (d=8,16) of the cached-exact loss, deterministic. "
   "Stage-1 screen proposed, never run; also a change of objective (population vs S_G target), synthetic/MNIST only."),
 "A3-subsample-target-fixed": ("not-run", IMPL,
   "Cost 0.28-0.34x at B=64, grad cos 0.96-0.98 (d<=16), 0.74 at d=768 (approx_loss/DIAGNOSTICS.md). Documented as a cost-accuracy knob only; no L2 screen run."),
 "A3-rff-multibandwidth": ("rejected", IMPL,
   "At m=250, D=256 costs 1.2-2.2x MORE than the cached-exact loss with cos 0.82-0.90 (d=8,16); d=768 needs D~2^16 (cos 0.23 at D=256, 0.97 at 65536); ORF does not change it; alpha=2 SD kernel not PD (approx_loss/THEORY.md 2a-b)."),
 "A3-nystrom-target-landmarks": ("rejected", IMPL,
   "Gradient collapses when X is off-target (cos 0.25 at d=768, L=64/120); d=8,16: cos 0.93-0.94 at 4-6x the exact cost; d=1 exact-ish but no cheaper (approx_loss/THEORY.md 2c)."),
 "A3-sliced-w2-fixed-projections": ("rejected", IMPL,
   "Different objective, not an MMD approximation: cos 0.3-0.9 (d<=16), ~0.1 at d=768 with P=32 (approx_loss/DIAGNOSTICS.md). Note for MNIST: guidance and evaluation must not share the sliced statistic."),
 "A3-fft-kme": ("rejected", IMPL,
   "No grid structure for point clouds in d>=3 / CLIP-768; alpha=2 kernel not PD; in d=1 the table fill is a 0.2 ms one-off direct sum, so the FFT saves nothing (approx_loss/THEORY.md FFT section)."),
 # Agent 4 (pre-registered families; per-rule rows below)
 "A4-1a": ("rejected", IMPL,
   "norm_only: 2D n=4 +0.213 (p=.008), n=8 +0.126; 10D significantly worse at every n (-0.12..-0.18, p<=.004) (estimator/round2_matrix.md)."),
 "A4-1b": ("rejected", VER,
   "Absolute clip is scale-dependent: clip0.5 2D n=4/8 +0.36/+0.19 (held-out, p<.001) but 10D n=32 -0.075* (held-out FAIL); clip0.1 the reverse (2D n=32 -0.356, 10D n=8 +0.122) (VERIFICATION.md 5.1/5.4; estimator/REPORT.md 3.2)."),
 "A4-1c": ("rejected", IMPL,
   "unit0.4: 2D +0.311/+0.120 at n=4/8, 10D n=32 -0.093*; unit0.1 catastrophic in 2D (-0.18..-0.21) (estimator/round2_matrix.md)."),
 "A4-2": ("rejected", IMPL,
   "adaptive n_t (agreement 0.5/0.8, improvement) null or negative at equal calls: 2D n=8 -0.27..-0.34*, 10D n=32 -0.08..-0.14* (estimator/round2_matrix.md)."),
 "A4-3": ("rejected", IMPL,
   "recur2 (early-stopped 2nd recurrence) null at up to 2x calls (2D n=4 +0.023 p=.81 for 792 vs 396 calls); 5D n=32 -0.111*, 10D n=4 -0.088* (estimator/round2_matrix.md)."),
 "A4-4a": ("rejected", IMPL,
   "CRN (frozen conditional noise, approximate/biased) null: 2D/adam n=8 -0.133*, 10D n=32 -0.065* (estimator/round2_matrix.md)."),
 "A4-4b": ("rejected", IMPL,
   "antithetic pairs null everywhere (2D/adam n=32 -0.043*, the one positive 10D/adam n=32 +0.069 p=.052) (estimator/round2_matrix.md)."),
 "A4-5": ("conditional", VER,
   "bw_pooled / bw_pooled_floor / sqrt_abs_eps null (screening). sqrt_floor: held-out PASS 2D (+0.23/+0.12/+0.05/+0.04) and 5D (n=8 +0.049*, n=32 +0.023*), FAIL 10D n=32 (-0.038*, p=.024) -> conditional: 2D/5D only (VERIFICATION.md 5.4)."),
 "A4-6": ("rejected", IMPL,
   "stale2/3 (gradient reused k steps, 1/k calls) worse wherever significant: 2D n=8 -0.227*, 5D n=4 -0.338*/-0.669*, 10D n=32 -0.195*/-0.241* (estimator/round2_matrix.md)."),
 "A4-7a": ("rejected", VER,
   "relclip2 held-out: 2D PASS (+0.444/+0.279/+0.114/+0.026), 5D inconclusive, 10D FAIL (n=4 -0.047*, p=.048) -> not promoted (one scale). relclip_ema2: 2D PASS, never negative, 5D/10D null -> safe but 2D-only. relclip1 (=qclip0.5): 10D n=32 -0.076*. relclip0.5 / ema0.5: 2D n=32 -0.23/-0.25* (VERIFICATION.md 5.4; estimator/round2_matrix.md)."),
 "A4-7b": ("rejected", VER,
   "qclip0.5 == relclip1 (identical in every cell; count once, VERIFICATION red flag 3): 10D n=32 -0.076* held-out. qclip0.75: 2D wins only (+0.39/+0.18/+0.08), no transfer (screening)."),
 "A4-7c": ("promoted", VER,
   "trust_noise1 (||Delta_t|| <= sqrt(1-alphabar_t)): held-out PASS 2D (+0.401/+0.250/+0.090/+0.024, all p<=.05) and 10D (+0.053/+0.123/+0.075 at n=4/8/16, n=32 +0.019 n.s.), 5D all positive (n=8 +0.036*), no significant regression in any of 12 cells -> the only rule passing the promotion rule (VERIFICATION.md 5.4). "
   "trust_noise0.3: 2D n=32 -0.037 n.s.; trust_noise0.1 and trust_ddim{0.1,0.3,1}: 10D wins (+0.08..+0.19) but catastrophic in 2D (-0.17..-0.46); trust_ddim0.1 equals the UNGUIDED score in 2D/5D (estimator/REPORT.md 3.2) -> rejected."),
 "A4-7d": ("conditional", VER,
   "sqrtfloor_clip0.5: held-out PASS 2D (4/4) and 5D (n=8 +0.059*, n=32 +0.035*), FAIL 10D n=32 (-0.088*) -> conditional 2D/5D (5D frontier holder at n=4/8/32). sqrtfloor_clip0.1 as clip0.1 (2D -0.22..-0.33); sqrtfloor_relclip1 worse than its parts (VERIFICATION.md 5.4; estimator/round2_matrix.md)."),
 "A4-8": ("rejected", IMPL,
   "Clip before Adam never significantly positive, 10D n=16/32 -0.075/-0.083 (clip0.5), -0.106/-0.122 (clip0.1); clip + LGD adds little (5D n=32 +0.017/+0.018*) and 10D n=32 -0.154*. Adam itself worse than plain guidance in 5D/10D; plain+trust_noise1 beats Adam in every dim (2D n=8 0.181 vs 0.303) (estimator/REPORT.md 3.3)."),
 "A4-9": ("promoted", VER,
   "Pareto support cells (baseline n in {2..96}); held-out confirms: 2D trust_noise1@n=8 (792 calls, 0.167) beats baseline@n=96 (9504, 0.259) and LGD/none@n=8/32; 10D trust_noise1@n=32 (3168, 0.457) = baseline@n=64 (6336, 0.456) (pareto.md; VERIFICATION.md 5.3)."),
 # Agent 5
 "A5-sys_batched_restarts": ("conditional", VER,
   "~10x throughput at B>=8 (verifier: B=8 4-15x per restart on a loaded Mac; implementer 14-25x at B=32) but per-step gradient jumps up to 2.7e-3 abs = 8% rel at ReLU-boundary steps (verifier) -- statistical equivalence only; report batched runs as re-runs, not reproductions (VERIFICATION.md 4.2)."),
 "A5-sys_mmd_blocks_cached_yy": ("promoted", VER,
   "Per-step teacher-forced |dg| ~1e-6 (REORDER), 1.5x (no-LGD) to 4.5x (LGD) per restart (systems/BENCH.md; verifier check_systems.log). Same mechanism as A2-fixed_target_cache."),
 "A5-sys_batched_lgd_perturbations": ("promoted", VER,
   "EXACT (0.0 end-to-end, verifier), 1.4x on LGD cells (systems/BENCH.md)."),
 "A5-sys_generator_seeding": ("promoted", VER,
   "EXACT (0.0, verifier); removes 99 us x M global reseeds per step (~3-8%); prerequisite for batching (systems/AUDIT.md 1.3)."),
 "A5-sys_requires_grad_false": ("conditional", VER,
   "EXACT (0.0, verifier); wall within noise on the synthetic MLPs; hygiene, matters for MNIST/SD saved activations (systems/BENCH.md)."),
 "A5-sys_lean_ddim_step": ("conditional", VER,
   "EXACT relative to the batched runner (verifier); +1-5%; 0.27 -> 0.18 ms per DDIM step (systems/AUDIT.md 1.3). Hygiene."),
 "A5-sys_microbatched_mmd_gradient": ("conditional", IMPL,
   "Exact identity (3.4e-16 rel float64 vs exact kernel; systems/microbatch_mmd.log); no speed gain at n<=256; enables n>>1e3 / 768-d embeddings without the (n+m)^2 slab. Not verifier-checked."),
 "A5-sys_torch_compile": ("rejected", IMPL,
   "0.84x steady state, +16 s compile, +270 MB RSS; tiny kernels + autograd dominate (systems/BENCH.md)."),
 "A5-sys_mps_device": ("rejected", IMPL,
   "5x slower than CPU at B=1 (1.0 s/run), 24 r/s at B=32 vs 70-90 on CPU; launch/sync bound; end-to-end diffs O(1-10) (systems/BENCH.md)."),
 "A5-sys_float64_throughout": ("rejected", IMPL,
   "Diagnostic only: 0.5-0.9x speed; the same algorithm in float64 moves x_hat by 0.04-1.1 and flips modes -> the float32 loop is chaotic per trajectory (systems/AUDIT.md 1.0). Not a production candidate."),
 "A5-sys_mnist_cache_cond_embed": ("not-run", NOTRUN,
   "Static: cond_embed re-run at each of 5 ladder steps (cond_model.py:176,230-233); 5->1 encoder fwd+bwd per sample() (~80% of encoder cost); EXACT (systems/AUDIT.md 2.2)."),
 "A5-sys_mnist_batch_perturbations_and_seeds": ("not-run", NOTRUN,
   "Static: num_x_t perturbations and 15 seeds sequential at batch 1 with retain_graph=True; batching estimated 5-15x wall, changes RNG stream (systems/AUDIT.md 2.6, patch list)."),
 "A5-sys_sd_trim_visualize_and_eval": ("not-run", NOTRUN,
   "Static: visualize_step every step = 5 VAE decodes + 5 sprinter gens + 2 full-VAE dtype casts + savefig; CLIP moved GPU<->CPU every eval; est. 25-35% wall at N=6, gradient-neutral (systems/AUDIT.md 3.8)."),
 "A5-sys_sd_single_batch_when_gs0": ("not-run", NOTRUN,
   "Static: predict_noise_cfg runs the architect UNet on cat([latents]*2) inside the graph although guidance_scale defaults to 0.0; halves architect UNet fwd+recompute+bwd (systems/AUDIT.md 3.2)."),
 "A5-sys_sd_unnest_checkpointing_and_batch_variations": ("not-run", NOTRUN,
   "Static: nested checkpointing runs each sprinter UNet/CN block 3x; variation_batch_size=1 hard-coded; sprinter prompt re-encoded per call with unfrozen text encoders; est. 15-25% + 1.5-2x on the sprinter path (systems/AUDIT.md 3.5-3.6)."),
 "A5-sys_sd_mixed_precision_vae_clip": ("not-run", NOTRUN,
   "Static, CHANGES NUMERICS: fp16-fix VAE + bf16 CLIP, est. ~2x on VAE/CLIP; needs grad-cosine and delta-MMD validation (systems/AUDIT.md 3.4)."),
}


import re as _re
def load_yaml(path):
    """safe_load with a fallback that quotes plain scalars containing ': ' (agent1/agent4 files)."""
    txt = open(path).read()
    try:
        return yaml.safe_load(txt)
    except yaml.YAMLError:
        out = []
        pat = _re.compile(r'^(\s*(?:- )?[^:\s#][^:]*?):\s+(.*)$')
        for line in txt.splitlines():
            m = pat.match(line)
            if m:
                key, val = m.groups()
                if val and val[0] not in '>|[{"\'' and (': ' in val or val.endswith(':')) and not val.startswith('#'):
                    line = f'{key}: "{val.replace(chr(34), chr(39))}"'
            out.append(line)
        return yaml.safe_load("\n".join(out))

entries = []


def add(id_, agent, name, mech, ben, bias, tasks, metric, budget, rej, files):
    verdict, verification, evidence = V[id_]
    entries.append({
        "id": id_, "agent": agent, "name": clean(name), "mechanism": clean(mech), "expected_benefit": clean(ben),
        "possible_bias": clean(bias), "affected_tasks": clean(tasks), "primary_metric": clean(metric),
        "compute_budget": clean(budget), "rejection_criterion": clean(rej), "verdict": verdict,
        "verification": verification, "evidence": evidence, "source_files": files})


# Agent 1
a1 = load_yaml(os.path.join(HYP, "agent1.yaml"))
for h in a1["hypotheses"]:
    add("A1-" + h["name"], "agent1 (baseline/profiler)", h["name"], h["mechanism"], h["expected_benefit"],
        h["possible_bias"], h["affected_tasks"], h["primary_metric"], "", h["rejection_criterion"],
        "hypotheses/agent1.yaml; profiling/baseline_profile.md")
# Agent 2
a2 = load_yaml(os.path.join(HYP, "agent2.yaml"))
for h in a2["candidates"]:
    add("A2-" + h["name"], "agent2 (exact loss)", h["name"], h["mechanism"], h["expected_benefit"],
        h["possible_bias"], h["affected_tasks"], h["primary_metric"], "", h["rejection_criterion"],
        "hypotheses/agent2.yaml; exact_loss/REPORT.md")
# Agent 3
a3 = load_yaml(os.path.join(HYP, "agent3.yaml"))
for h in a3:
    add(h["name"], "agent3 (approximate loss)", h["name"], h["mechanism"], h["expected_benefit"],
        h["possible_bias"], h["affected_tasks"], h["primary_metric"], h.get("compute_budget", ""),
        h["rejection_criterion"], "hypotheses/agent3.yaml; approx_loss/REPORT.md")
# Agent 4
a4 = load_yaml(os.path.join(HYP, "agent4.yaml"))
for h in a4["candidates"] + a4["round2_candidates"]:
    add(h["id"], "agent4 (estimator/update rules)", h["name"], h.get("mechanism", "") + " switch: " + clean(h.get("switch", h.get("cells", ""))),
        h.get("expected_benefit", h.get("expected", h.get("question", ""))), h.get("possible_bias", ""),
        h.get("affected_tasks", "synthetic 2D/5D/10D"), h.get("primary_metric", "failure-penalised exact GMM L2, paired"),
        h.get("compute_budget", "matched conditional calls (n*T)"), h.get("rejection", ""),
        "hypotheses/agent4.yaml; estimator/REPORT.md; VERIFICATION.md")
# Agent 5
a5 = load_yaml(os.path.join(HYP, "agent5.yaml"))
for h in a5["entries"]:
    add("A5-" + h["name"], "agent5 (systems)", h["name"], h["mechanism"] + " exactness: " + clean(h.get("exactness", "")),
        h["expected_benefit"], h["possible_bias"], h["affected_tasks"], h["primary_metric"], "",
        h["rejection_criterion"], "hypotheses/agent5.yaml; systems/AUDIT.md; systems/BENCH.md")

missing = [k for k in V if k not in {e["id"] for e in entries}]
assert not missing, missing

with open(os.path.join(ROOT, "hypotheses.yaml"), "w") as f:
    f.write("# Merged hypotheses of the CDM performance campaign (Agent 7). One entry per pre-registered\n"
            "# candidate; verdicts: promoted / conditional / rejected / not-run. 'verification' says whether the\n"
            "# verdict rests on Agent 6's independent checks (VERIFICATION.md) or only on the implementer's numbers.\n"
            "# Generated by report_tools/merge_hypotheses.py from hypotheses/agent{1..5}.yaml.\n")
    yaml.safe_dump({"commit": "6af2081", "hypotheses": entries}, f, sort_keys=False, width=110, allow_unicode=True)
print("wrote hypotheses.yaml:", len(entries), "entries")

# --------------------------------------------------------------------------- HYPOTHESES.md
# one row per concrete method (finer than the yaml where one family has several verdicts)
ROWS = [
 # (group, method, agent, verdict, verification, one-line reason, source)
 ("exact loss", "cached target-target block (XX/XY only) -- fast_mmd.MMDFixedTarget / BatchedMMD", "1,2,5", "promoted", "V",
  "exact to 1.3e-14 (f64); MMD fwd+bwd 1.96->0.42 ms (n=8, 4.6x); whole loop 1.66-2.02x; same cond calls", "VERIFICATION.md 4.1; exact_loss/REPORT.md; end_to_end_results.csv"),
 ("exact loss", "matmul distances (dist='mm')", "2", "promoted", "V", "exact (verifier grid); 6.3x f32 / 10.2x f64 micro-bench geo-mean vs reference", "exact_loss/bench_summary_small.md"),
 ("exact loss", "powchain kernel E+E^2+E^4+E^8+E^16", "2", "promoted", "V", "exact (few ulp); 0.28 ms at n=8 (7.0x); best end-to-end 1.77-2.02x", "VERIFICATION.md 4.1; end_to_end_results.csv"),
 ("exact loss", "chunked fused XY (autograd.Function)", "2", "conditional", "V", "exact 1st order; speed-neutral; memory only (m >> 2000)", "exact_loss/REPORT.md 1"),
 ("exact loss", "batched sets (B sample sets, one Y)", "2", "conditional", "V", "exact; 12.6x vs 3 reference calls, ~2x vs 3 fixed_mm calls; needs concatenated sampler outputs", "exact_loss/REPORT.md 2"),
 ("exact loss", "population-GMM target (closed-form KME)", "3", "not-run", "I", "cos 0.989-0.998 vs empirical gradient, 0.02-0.55x cost, deterministic; changes the objective; Stage-1 screen never run", "approx_loss/REPORT.md"),
 ("exact loss", "tabulated KME (d=1)", "3", "rejected", "I", "cos 0.999 at 0.04-0.12x but dominated by population-GMM; d=1 only", "approx_loss/THEORY.md FFT section"),
 ("approx loss", "RFF / ORF random features", "3", "rejected", "I", "D=256 costs 1.2-2.2x MORE than cached-exact at m=250 with cos 0.82-0.90 (d=8,16); d=768 needs D~2^16; alpha=2 kernel not PD", "approx_loss/THEORY.md 2a-b, DIAGNOSTICS.md"),
 ("approx loss", "Nystrom (target landmarks)", "3", "rejected", "I", "gradient collapses off-target (cos 0.25, d=768); d=8,16 cos 0.93-0.94 at 4-6x the exact cost", "approx_loss/THEORY.md 2c"),
 ("approx loss", "linear-time / block estimator = target subsample (fixed subset)", "3", "not-run", "I", "0.28-0.34x cost at B=64, cos 0.96-0.98 (d<=16), 0.74 (d=768); kept as a knob; no L2 screen", "approx_loss/REPORT.md"),
 ("approx loss", "sliced W2 (fixed projections)", "3", "rejected", "I", "different objective: cos 0.3-0.9 (d<=16), 0.1 (d=768); MNIST must not evaluate with the guidance statistic", "approx_loss/THEORY.md 2e"),
 ("approx loss", "FFT / NUFFT kernel mean embedding", "3", "rejected", "I", "no grid in d>=3; one-off d=1 table fill is 0.2 ms by direct sum; no candidate proceeds", "approx_loss/THEORY.md"),
 ("estimator", "Adam temporal arm (repo baseline, rho=0.4)", "4", "rejected", "I", "worse than plain guidance in 5D/10D (5D n=8 0.61 vs 0.50; 10D n=32 0.57 vs 0.49); regime flip at n=32 in 2D; plain+trust_noise1 beats Adam in every dim", "estimator/REPORT.md 3.3, 4"),
 ("estimator", "norm-only (Adam beta1=0)", "4", "rejected", "I", "2D n=4 +0.213*, n=8 +0.126*; 10D -0.12..-0.18* at every n", "estimator/round2_matrix.md"),
 ("estimator", "absolute clip 0.5", "4", "rejected", "V", "2D n=4/8 +0.36/+0.19 (held-out); 10D n=32 -0.075* held-out FAIL; 5D inconclusive -> scale-dependent, one scale", "VERIFICATION.md 5.4"),
 ("estimator", "absolute clip 0.1", "4", "rejected", "I", "10D n=4/8 +0.09/+0.12*; 2D n=8/16/32 -0.13/-0.23/-0.36* (over-clips)", "estimator/round2_matrix.md"),
 ("estimator", "unit-norm gradient (0.4 / 0.1)", "4", "rejected", "I", "unit0.4 2D +0.31/+0.12 but 10D n=32 -0.093*; unit0.1 2D -0.18..-0.21*", "estimator/round2_matrix.md"),
 ("estimator", "relclip2 (2 x running median)", "4", "rejected", "V", "best 2D rule (held-out 0.153/0.139 at n=4/8, +0.44/+0.28*), 5D inconclusive, 10D n=4 -0.047* -> not promoted (one scale)", "VERIFICATION.md 5.4"),
 ("estimator", "relclip_ema2 (2 x EMA)", "4", "rejected", "V", "2D PASS (4/4), never significantly negative, 5D/10D null -> safe but 2D-only", "VERIFICATION.md 5.4"),
 ("estimator", "relclip1 == qclip0.5 (1 x median)", "4", "rejected", "V", "2D n<=16 wins; 10D n=32 -0.076* (held-out); 2D n=32 mmd2_eval -0.049*; duplicate rule counted once", "VERIFICATION.md 5.4, red flag 3"),
 ("estimator", "relclip0.5 / relclip_ema0.5 / relclip_ema1 / qclip0.75", "4", "rejected", "I", "0.5 variants 2D n=32 -0.23/-0.25*; ema1 10D n=32 -0.037 n.s. but no 5D/10D win; qclip0.75 2D-only", "estimator/round2_matrix.md"),
 ("estimator", "trust_noise1: ||Delta_t|| <= 1.0*sqrt(1-alphabar_t)  (step_clip='noise', step_tau=1)", "4", "promoted", "V", "held-out PASS 2D (+0.40/+0.25/+0.09/+0.02*) and 10D (+0.05/+0.12/+0.08* at n=4/8/16), 5D all + (n=8 +0.036*), no significant regression in 12 cells; 10D frontier at every n<=32", "VERIFICATION.md 5.1-5.4"),
 ("estimator", "trust_noise0.3 / trust_noise0.1", "4", "rejected", "I", "0.3: 2D n=32 -0.037 n.s., otherwise wins at n<=8 in all dims; 0.1: 2D -0.17..-0.38*", "estimator/round2_matrix.md"),
 ("estimator", "trust_ddim (tau 0.1/0.3/1): ||Delta_t|| <= tau ||x_ddim - x_t||", "4", "rejected", "I", "10D wins +0.08..+0.19* but 2D -0.21..-0.46*; tau=0.1 equals the UNGUIDED score in 2D/5D (guidance off)", "estimator/REPORT.md 3.2"),
 ("estimator", "sqrt_floor loss transform", "4", "conditional", "V", "held-out PASS 2D (+0.23/+0.12/+0.05/+0.04) + 5D (n=8 +0.049*, n=32 +0.023*), FAIL 10D n=32 (-0.038*) -> 2D/5D only", "VERIFICATION.md 5.4"),
 ("estimator", "sqrtfloor_clip0.5", "4", "conditional", "V", "PASS 2D (4/4) + 5D (+0.059*/+0.035*; 5D frontier n=4/8/32), FAIL 10D n=32 (-0.088*)", "VERIFICATION.md 5.4"),
 ("estimator", "sqrtfloor_clip0.1 / sqrtfloor_relclip1 / sqrt_abs_eps (SD transform)", "4", "rejected", "I", "clip0.1 combo 2D -0.22..-0.33*; relclip1 combo worse than its parts (10D n=32 -0.068*); sqrt_abs_eps null", "estimator/round2_matrix.md"),
 ("estimator", "bandwidth policies (pooled / pooled_floor vs fixed)", "4", "rejected", "I", "null everywhere (|diff| <= 0.08, p > .05); pooled collapses only with tiny targets (unit test)", "estimator/round2_matrix.md; REPORT.md 2"),
 ("estimator", "adaptive n_t (agreement 0.5/0.8, improvement; equal total calls)", "4", "rejected", "I", "2D n=8 -0.20..-0.34*, 10D n=32 -0.08..-0.14*; binding constraint is the step rule, not the per-step budget", "estimator/round2_matrix.md; REPORT.md 4"),
 ("estimator", "adaptive recurrence v1 (<= 2 recurrences)", "4", "rejected", "I", "null at up to 2x calls; 5D n=32 -0.111*, 10D n=4 -0.088*", "estimator/round2_matrix.md"),
 ("estimator", "stale gradient cache (refresh every 2/3)", "4", "rejected", "I", "1/k calls but 2D n=8 -0.23/-0.32*, 5D n=4 -0.34/-0.67*, 10D n=32 -0.20/-0.24*", "estimator/round2_matrix.md"),
 ("estimator", "common random numbers (frozen conditional noise; approximate)", "4", "rejected", "I", "null; 2D/adam n=8 -0.133*, 10D n=32 -0.065*", "estimator/round2_matrix.md"),
 ("estimator", "antithetic conditional noise", "4", "rejected", "I", "null; 2D/adam n=32 -0.043*", "estimator/round2_matrix.md"),
 ("estimator", "clip + Adam / clip + LGD combinations", "4", "rejected", "I", "clip before Adam: 10D n=16/32 -0.075..-0.122*; clip+LGD 10D n=32 -0.154*; clipped no-LGD n=8 beats LGD n=8 and n=32 in 2D only", "estimator/REPORT.md 3.3"),
 ("systems", "batched restarts (B chains in one batch)", "5", "conditional", "V", "~10x throughput at B>=8 (verifier 4-15x loaded Mac); per-step grad jumps up to 8% rel at ReLU boundaries -> statistical equivalence only; screening use, not reproduction", "VERIFICATION.md 4.2"),
 ("systems", "batched LGD perturbations (3n rows, pre-drawn noise)", "5", "promoted", "V", "EXACT 0.0; 1.4x on LGD cells", "systems/BENCH.md; check_systems.log"),
 ("systems", "cached YY / batched MMD (BatchedMMD)", "5", "promoted", "V", "REORDER 1e-6 per step; 1.5x (no-LGD) to 4.5x (LGD) per restart", "systems/BENCH.md"),
 ("systems", "torch.Generator instead of global manual_seed", "5", "promoted", "V", "EXACT 0.0; 99 us x M per step saved", "systems/AUDIT.md 1.3"),
 ("systems", "requires_grad_(False) on frozen models", "5", "conditional", "V", "EXACT; wall within noise on the synthetic MLPs; hygiene (MNIST/SD memory)", "systems/BENCH.md"),
 ("systems", "lean DDIM step / cached time embedding / torch.full", "5", "conditional", "V", "EXACT; +1-5%", "systems/AUDIT.md 1.3"),
 ("systems", "micro-batched MMD gradient (chunked VJP)", "5", "conditional", "I", "exact identity 3.4e-16; no gain at n<=256; for n>>1e3 / 768-d", "systems/microbatch_mmd.log"),
 ("systems", "torch.compile (loop / MMD)", "2,5", "rejected", "I", "loop 0.84x +16 s compile; MMD 1.4-2.5x steady but 2-7 s per shape, 26k-58k-call break-even, macOS inductor hangs", "systems/BENCH.md; exact_loss/bench_compile_cpu.md"),
 ("systems", "MPS device", "5", "rejected", "I", "5x slower at B=1; 24 r/s at B=32 vs 70-90 CPU", "systems/BENCH.md"),
 ("systems", "float64 throughout", "5", "rejected", "I", "diagnostic: 0.5-0.9x; reveals float32 chaos (|dx| 0.04-1.1, mode flips)", "systems/AUDIT.md 1.0"),
 ("systems (MNIST, static)", "cache cond_embed across the CM ladder; batch perturbations + seeds; drop retain_graph; host-sync removal; skip uniform-target inner loop", "1,5", "not-run", "S", "5->1 encoder evals (EXACT); batching est. 5-15x; ~25% of uniform runs wasted after step_size=0", "systems/AUDIT.md 2; profiling/baseline_profile.md 4b"),
 ("systems (SD, static)", "gate visualize_step / eval; gs==0 single-batch architect UNet; un-nest checkpointing; batch variations with per-sample generators; cache prompt embeds + freeze text encoders; keep CLIP on GPU, stop VAE dtype toggles, drop gc/empty_cache", "1,5", "not-run", "S", "est. 25-35% (vis) + 15-25% (ckpt) + 1.5-2x sprinter path; all EXACT for the guided path", "systems/AUDIT.md 3; profiling/baseline_profile.md 4c"),
 ("systems (SD, static)", "fp16-fix VAE / bf16 CLIP / truncated sprinter backprop / 1-step sprinter", "5", "not-run", "S", "CHANGES NUMERICS; est. ~2x on VAE/CLIP; needs grad-cosine + delta-MMD validation", "systems/AUDIT.md 3.4, 3.7"),
]
LEG = {"V": "verifier-confirmed", "I": "implementer-reported, not independently verified", "S": "static analysis, not run"}
md = ["# HYPOTHESES -- every method considered in the CDM performance campaign (Agent 7)", "",
      "Merged from `hypotheses/agent{1..5}.yaml` (machine-readable: `hypotheses.yaml`, "
      f"{len(entries)} pre-registered entries). Verdicts follow `VERIFICATION.md` wherever the verifier (Agent 6) "
      "checked the item; other rows carry the implementing agent's own numbers. Screening numbers are paired "
      "diffs `base - cand` of the failure-penalised exact GMM L2 (+ = candidate better; `*` = p <= 0.05; 40 restarts, "
      "offset 0); held-out numbers are the same on 100 restarts at offset 1000 (VERIFICATION.md 5).", "",
      "Verification column: **V** = verifier-confirmed; **I** = implementer-reported, not independently verified; "
      "**S** = static analysis only, not run.", "",
      "| group | method | agent | verdict | verif. | reason (numbers) | source |", "|---|---|---|---|---|---|---|"]
for g, m, a, v, ver, why, src in ROWS:
    vv = {"promoted": "**promoted**", "conditional": "**conditional**", "rejected": "rejected", "not-run": "not-run"}[v]
    md.append(f"| {g} | {m} | {a} | {vv} | {ver} | {why} | {src} |")
md += ["", "## Summary", "",
       "* **Promoted (verifier-confirmed):** cached-target exact MMD (`exact_loss/fast_mmd.py`, drop-in, 4-7x on the loss, "
       "~1.7-2x on the loop); batched LGD perturbations, generator seeding (exact); and ONE estimator rule, "
       "`trust_noise1` (`step_clip='noise', step_tau=1.0`), the only rule that is a credible Pareto improvement at two "
       "task scales (2D, 10D) with no significant regression at any of the 12 held-out cells.",
       "* **Conditional:** `sqrt_floor` and `sqrtfloor_clip0.5` (2D/5D only; regress at 10D n=32); batched restarts "
       "(statistical equivalence only, ~10x throughput); hygiene items (requires_grad_(False), lean DDIM); chunked / "
       "batched / micro-batched MMD forms (exact, no gain at the paper's sizes).",
       "* **Rejected:** every approximate loss (RFF/ORF, Nystrom, sliced, FFT, subsample-as-approximation); absolute "
       "clipping (scale-dependent), relclip family (2D-only or 10D regressions), unit/norm-only, trust_ddim, adaptive "
       "n_t, adaptive recurrence, stale gradients, CRN, antithetic, bandwidth policies, Adam (+clip) in 5D/10D, "
       "torch.compile, MPS, float64-as-production.",
       "* **Not run:** population-GMM target (promising diagnostics, different objective), target subsample knob, all "
       "MNIST and SD static recommendations (no GPU / checkpoints locally; cluster SD/MNIST stages were out of scope).",
       ""]
with open(os.path.join(ROOT, "HYPOTHESES.md"), "w") as f:
    f.write("\n".join(md))
print("wrote HYPOTHESES.md:", len(ROWS), "rows")
