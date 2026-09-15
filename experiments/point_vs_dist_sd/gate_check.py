"""gate_check.py — P-C1 gate: does the guidance MMD trace trend DOWN?
Usage: python gate_check.py <run_dir>    (criterion registered in HYPOTHESIS.md amendment)"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import band_for

d = sys.argv[1]
m = json.load(open(d + "/metrics.json"))
L = np.array([s["mmd_loss"] for s in m["steps"]])
# P-C1 window exactly as registered: k=10 for runs >= 20 steps (shorter: max(3, n//6))
k = 10 if len(L) >= 20 else max(3, len(L) // 6)
delta = L[-k:].mean() - L[:k].mean()
r = np.corrcoef(np.arange(len(L)), L)[0, 1]
c = np.array([s["correction_norm_raw"] for s in m["steps"]])
a = np.array([s["correction_norm_applied"] for s in m["steps"]])
print(f"[GATE] n={len(L)} first{k} {L[:k].mean():.4f} last{k} {L[-k:].mean():.4f} "
      f"delta {delta:+.4f} (need <= -0.03) | corr(step,loss) {r:+.3f} (need <= -0.30)")
lo, hi = band_for(m.get("args", {}))          # the ARM's own band, never a global one
in_band = ((a >= lo) & (a <= hi)).mean()
sd_step = np.diff(L).std() / np.sqrt(2)       # per-step noise of THIS run's statistic
se_delta = sd_step * np.sqrt(2.0 / k)
print(f"[GATE] ||corr|| raw median {np.median(c):.2f} | applied median {np.median(a):.2f} "
      f"| arm band {lo}-{hi} | applied in-band {in_band:.0%} | clipped {np.mean(a < c - 1e-6):.0%}")
print(f"[GATE] per-step sd {sd_step:.4f} -> SE(delta) {se_delta:.4f}; the -0.03 threshold is "
      f"{0.03/se_delta:.1f} SE for this arm ({'well powered' if 0.03/se_delta >= 2 else 'UNDERPOWERED: a FAIL here is not evidence of no-descent'})")
print(f"[GATE] final MMD guided {m.get('final_mlgd_f_mmd'):.4f} vs own ref "
      f"{m.get('final_regular_mmd'):.4f}")
print("[GATE] VERDICT:", "PASS" if (delta <= -0.03 and r <= -0.30) else "FAIL (P-C1 not met)")
