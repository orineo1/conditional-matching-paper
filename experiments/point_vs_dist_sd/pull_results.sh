#!/bin/bash
# pull_results.sh — run LOCALLY once the ssh tunnel is up. One shot: job states,
# calibration verdicts, and pull of any finished arm/calibration outputs.
# Usage: bash experiments/point_vs_dist_sd/pull_results.sh
set -u
SSH="ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10 shaulytolk@localhost"
R=/sci/labs/orzuk/shaulytolk/cdm-perf/experiments/point_vs_dist_sd
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)/output/pvd_B
$SSH 'echo TUNNEL_OK' >/dev/null 2>&1 || { echo "TUNNEL DOWN — re-establish it first"; exit 1; }

echo "=== job states ==="
$SSH "bash -lc 'sacct -j 46147857,46147886,46147945 --format=JobID,JobName%12,Partition%8,State,Elapsed,ExitCode | egrep -v \"batch|extern\"'"
echo; echo "=== queue ==="
$SSH "bash -lc 'squeue -u shaulytolk -o \"%.10i %.12j %.9P %.8T %.10M %R\"'"

echo; echo "=== 1. POINT-ARM PROBE (46147886) ==="
$SSH "cat $R/debug/B_point/debug_gradient.json 2>/dev/null" > /tmp/pvd_point_probe.json 2>/dev/null
if [ -s /tmp/pvd_point_probe.json ]; then
  /Users/stolk/miniconda3/bin/python - <<'PY'
import json
d = json.load(open("/tmp/pvd_point_probe.json"))
p1, p2 = d["P1"], d["P2"]
print(f"P1 mean pairwise cos {p1['mean_cos']:+.4f}  SNR {p1['snr']:.3f} (pure noise {p1['snr_pure_noise']:.2f})")
print(f"P2 ||zeta*g|| = {p2.get('step_unit_norm', float('nan')):.2f}   (loss readout averaged over "
      f"{p2.get('eval_n_cond')} samples, gradient from n_cond=1)")
best = min(range(len(p2["lambdas"])), key=lambda i: p2["crn"][i])
for lam, c, f in zip(p2["lambdas"], p2["crn"], p2["fresh"]):
    star = "  <-- best" if lam == p2["lambdas"][best] else ""
    print(f"  lam {lam:6.1f} ||step|| {lam*p2.get('step_unit_norm',0):8.2f}  CRN {c:.4f}  fresh {f:.4f}{star}")
lo = min(l for l, c in zip(p2["lambdas"], p2["crn"]) if c < p2["crn"][0] and l > 0) if any(
    c < p2["crn"][0] for l, c in zip(p2["lambdas"], p2["crn"]) if l > 0) else None
print(f"\nVERDICT: point arm's failed-run median correction was 9.45.")
if lo is not None:
    band = [l*p2.get('step_unit_norm',0) for l, c in zip(p2["lambdas"], p2["crn"]) if l > 0 and c < p2["crn"][0]]
    print(f"  useful band (steps that beat lambda=0): ||step|| {min(band):.1f} - {max(band):.1f}")
    print(f"  -> median 9.45 {'IS' if min(band) <= 9.45 <= max(band) else 'is NOT'} in band "
          f"=> {'keep base_zeta 4, add TAU=0.2 only' if min(band) <= 9.45 <= max(band) else 'rescale zeta by ' + format((min(band)+max(band))/2/9.45, '.1f') + 'x'}")
else:
    print("  no lambda beat lambda=0 -> the point direction is not useful at this latent (report as-is)")
PY
else
  echo "  (no probe JSON yet)"
fi

echo; echo "=== 2. PGD lr SWEEP (46147945) ==="
# NOTE: analysed LOCALLY. An earlier version ran python -c inside a nested ssh/bash -lc
# quote stack and died with "unexpected EOF while looking for matching quote"; pulling the
# small JSONs and parsing them here removes the nesting entirely.
PGD_TMP=$(mktemp -d)
for DIV in 100 10 3; do
  $SSH "cat $R/runs/B/pgd_eps32_lrdiv${DIV}/metrics_partial.json 2>/dev/null" \
      > "$PGD_TMP/lrdiv${DIV}.json" 2>/dev/null
  [ -s "$PGD_TMP/lrdiv${DIV}.json" ] || rm -f "$PGD_TMP/lrdiv${DIV}.json"
done
if ls "$PGD_TMP"/*.json >/dev/null 2>&1; then
  /Users/stolk/miniconda3/bin/python - "$PGD_TMP" <<'PYEOF'
import glob, json, os, sys
import numpy as np
# The training curve is ONE sprinter sample per step, so single values are very noisy and
# step 0 is the un-optimised starting point evaluated with that same 1-sample noise.
# Criterion: did the optimiser descend on its OWN objective (last-10 mean < first-10 mean)?
# The fresh 2000-sample L(x0) = 0.4218 is NOT comparable to single-sample training values
# and is shown only for scale; the authoritative check is a fresh eval of x*.
L0_FRESH = 0.4218
print(f"  [1-sample training curves; fresh-eval L(x0) = {L0_FRESH:.4f} shown for scale only]")
any_desc = False
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    m = json.load(open(f)); a = m["args"]
    L = np.array([s["loss"] for s in m["steps"]])
    k = min(10, max(1, len(L) // 10))
    first, last = L[:k].mean(), L[-k:].mean()
    desc = last < first
    any_desc |= bool(desc)
    print(f"  {os.path.basename(f):14s} lr {a['lr']:.5f}  step0 {L[0]:.4f}  first{k} {first:.4f}  "
          f"last{k} {last:.4f}  min(steps>=1) {L[1:].min():.4f}  sd {L.std():.3f}  "
          f"descended: {'YES' if desc else 'NO'}")
print(f"  VERDICT: {'some lr descends on its own objective -> fresh-eval it before claiming failure' if any_desc else 'NO lr descends on its own objective -> the registered finding stands'}")
PYEOF
else
  echo "  (no PGD sweep output yet — job still running?)"
fi
rm -rf "$PGD_TMP"

echo; echo "=== 3. pull finished run dirs ==="
for arm in mmd_zeta12_tau0.2 mmd_zeta12_tau0.2_gate point_tau0.2 pgd_eps32_lrdiv100 pgd_eps32_lrdiv10 pgd_eps32_lrdiv3; do
  $SSH "test -f $R/runs/B/$arm/metrics.json" 2>/dev/null || continue
  mkdir -p "$LOCAL/$arm/npy"
  for f in metrics.json metrics_partial.json profile.json target_clip_embeddings.pt; do
    $SSH "cat $R/runs/B/$arm/$f 2>/dev/null" > "$LOCAL/$arm/$f" 2>/dev/null
    [ -s "$LOCAL/$arm/$f" ] || rm -f "$LOCAL/$arm/$f"
  done
  for f in eval_clip_mlgd_f.npy eval_clip_regular.npy; do
    $SSH "cat $R/runs/B/$arm/npy/$f 2>/dev/null" > "$LOCAL/$arm/npy/$f" 2>/dev/null
    [ -s "$LOCAL/$arm/npy/$f" ] || rm -f "$LOCAL/$arm/npy/$f"
  done
  echo "  pulled $arm"
done
echo; echo "Next: python experiments/point_vs_dist_sd/build_report.py --runs output/pvd_B --scenario B"
