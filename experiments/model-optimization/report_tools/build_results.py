"""Merge every agent's row file into experiments/model-optimization/results.csv.

Column set (README): commit,candidate,task,target,seed,config,hardware,dtype,wall_s,
peak_mem_mb,score_calls,cond_calls,cond_samples,opt_loss,eval_metric,status  + source.

Sources
  profiling/baseline_rows.csv          (Agent 1, seed-level, already in schema)
  estimator/screening_rows.csv         (Agent 4, cell-level 40 restarts; hardware relabelled, see note)
  verification/heldout_runs/*.json     (Agent 6, expanded to seed-level: 108 cells x 100 restarts)
  verification/heldout_rows.csv        (Agent 6, cell-level paired statistics)
  systems/bench_rows.csv               (Agent 5, per-variant timing rows)
  exact_loss/end_to_end_results.csv    (Agent 2, _guided.run with fast MMD, 2D restart 0)
  exact_loss/bench_results_small.csv   (Agent 2, MMD micro-benchmark grid)

Run:  /Users/stolk/miniconda3/bin/python experiments/model-optimization/report_tools/build_results.py
"""
import csv, glob, json, os, sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLS = ["commit", "candidate", "task", "target", "seed", "config", "hardware", "dtype",
        "wall_s", "peak_mem_mb", "score_calls", "cond_calls", "cond_samples", "opt_loss",
        "eval_metric", "status"]
OUT_COLS = COLS + ["source"]
COMMIT = "6af2081"
HELDOUT_HW = "x86_64 AMD EPYC 7662 (glacier) cpu 2thr"
SCREEN_HW = "x86_64 cluster cpu (glacier, 2thr; relabelled from 'arm64 cpu' = report host, VERIFICATION red flag 5)"


def row(**kw):
    r = {c: "" for c in OUT_COLS}
    r.update(kw)
    return r


def fmt(x, nd=6):
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


rows = []
counts = Counter()

# 1. baseline (already in schema)
src = "profiling/baseline_rows.csv"
for r in read_csv(os.path.join(ROOT, src)):
    rows.append(row(**{c: r.get(c, "") for c in COLS}, source=src))
    counts[src] += 1

# 2. screening (already in schema; hardware relabelled)
src = "estimator/screening_rows.csv"
for r in read_csv(os.path.join(ROOT, src)):
    d = {c: r.get(c, "") for c in COLS}
    d["hardware"] = SCREEN_HW
    rows.append(row(**d, source=src))
    counts[src] += 1

# 3. held-out runs, seed-level
src = "verification/heldout_runs"
for p in sorted(glob.glob(os.path.join(ROOT, src, "*.json"))):
    d = json.load(open(p))
    cfg = (f"n={d['n']};{d['spatial']}/{d['temporal']}/rng=tape/offset={d['offset']}"
           f"/restarts={d['restarts']};candidate={d['candidate']};n_eval={d.get('n_eval','')}")
    hw = f"{d['verifier'].get('cpu','?')} ({d['verifier'].get('host','?')}) cpu {d['verifier'].get('threads','?')}thr"
    for i, run in enumerate(d["runs"]):
        seed = d["offset"] + i
        rows.append(row(
            commit=COMMIT, candidate=f"A6:{d['candidate']}", task=f"synthetic_{d['setting']}",
            target="S_G_250_seed987654", seed=str(seed), config=cfg, hardware=hw, dtype=d["dtype"],
            wall_s=fmt(run.get("seconds"), 4), peak_mem_mb=fmt(d.get("peak_mem_mb"), 5),
            score_calls="99", cond_calls=fmt(run.get("conditional_calls")),
            cond_samples=fmt(run.get("cm_samples")), opt_loss=fmt(run.get("mmd2_eval")),
            eval_metric=fmt(run.get("L2")),
            status=("diverged" if run.get("diverged") else "ok")
                   + f";abs_err={fmt(run.get('abs_err'),4)};x_hat={run.get('x_hat')}",
            source=src))
        counts[src] += 1

# 4. held-out paired cell rows
src = "verification/heldout_rows.csv"
for r in read_csv(os.path.join(ROOT, src)):
    st = (f"paired_vs_baseline;diff_L2={fmt(r['diff'],4)};ci=[{fmt(r['ci_lo'],4)},{fmt(r['ci_hi'],4)}];"
          f"p={fmt(r['perm_p'],3)};wins={r['wins']}/{r['restarts']};base_score={fmt(r['base_score'],4)};"
          f"success={r['cand_success']};div={r['cand_div']};diff_mmd2_eval={fmt(r['mmd_diff'],4)};"
          f"mmd_p={fmt(r['mmd_p'],3)};screening_diff={fmt(r['screening_diff'],4)};calls_match={r['calls_match']}")
    rows.append(row(
        commit=COMMIT, candidate=f"A6:{r['candidate']}", task=f"synthetic_{r['setting']}",
        target="S_G_250_seed987654",
        seed=f"restarts_{int(r['offset'])}..{int(r['offset'])+int(r['restarts'])-1}",
        config=f"n={r['n']};{r['spatial']}/{r['temporal']}/rng=tape/offset={r['offset']}/restarts={r['restarts']}",
        hardware=HELDOUT_HW, dtype="float32", wall_s=fmt(r["s_per_run"], 4), score_calls="99",
        cond_calls=r["calls"], cond_samples=r["calls"], eval_metric=fmt(r["cand_score"]),
        status=st, source=src))
    counts[src] += 1

# 5. systems bench rows
src = "systems/bench_rows.csv"
for r in read_csv(os.path.join(ROOT, src)):
    st = (f"{r['status']};wall_per_restart_s={r['wall_per_restart_s']};restarts_per_s={r['restarts_per_s']};"
          f"end2end_max_abs_dx={r['end2end_max_abs']};step_grad_max_abs={r['step_grad_max_abs']}")
    rows.append(row(
        commit=r["commit"], candidate=r["candidate"], task=r["task"], target=r["target"], seed=r["seed"],
        config=r["config"], hardware=r["hardware"], dtype=r["dtype"], wall_s=r["wall_s"],
        peak_mem_mb=r["peak_mem_mb"], score_calls=r["score_calls"], cond_calls=r["cond_calls"],
        cond_samples=r["cond_samples"], status=st, source=src))
    counts[src] += 1

# 6. exact_loss end-to-end
src = "exact_loss/end_to_end_results.csv"
for r in read_csv(os.path.join(ROOT, src)):
    rows.append(row(
        commit=COMMIT, candidate=f"A2:{r['variant']}", task=f"synthetic_{r['setting']}",
        target="S_G250_seed987654", seed="0",
        config=f"spatial={r['spatial']};temporal=none;n={r['n']};loop=_guided.run;mmd={r['variant']}",
        hardware="arm64-cpu-4thr (loaded)", dtype="float32", wall_s=fmt(r["fast_wall_s"], 4),
        score_calls="99", cond_calls=str(99 * (3 if r["spatial"] == "lgd" else 1)),
        cond_samples=r["cond_calls"],
        status=f"ok;ref_wall_s={fmt(r['ref_wall_s'],4)};speedup={fmt(r['speedup'],3)};max_abs_dx_vs_ref={fmt(r['max_abs_dx'],3)}",
        source=src))
    counts[src] += 1

# 7. exact_loss micro-benchmark grid
for src, hw in (("exact_loss/bench_results_small.csv", "{dev} arm64 M4 4thr"),
                ("exact_loss/bench_results.csv", "{dev} cluster EPYC-7662-4thr/L4")):
  for r in read_csv(os.path.join(ROOT, src)):
    rows.append(row(
        commit=COMMIT, candidate=f"A2:{r['variant']}", task="mmd_microbench",
        target=f"Y_m{r['n_target']}_d{r['dim']}", seed="",
        config=f"kind={r['kind']};n={r['n_cond']};m={r['n_target']};d={r['dim']};fwd+bwd",
        hardware=hw.format(dev=r['device']), dtype=r["dtype"], wall_s=fmt(r["median_s"], 4),
        peak_mem_mb=fmt(r["rss_mb"], 5), opt_loss=fmt(r["value"]),
        status=f"{r['status']};min_s={fmt(r['min_s'],4)};first_call_s={fmt(r['first_call_s'],4)}",
        source=src))
    counts[src] += 1

out = os.path.join(ROOT, "results.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=OUT_COLS)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}: {len(rows)} rows")
for k, v in counts.items():
    print(f"  {v:6d}  {k}")
