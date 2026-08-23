"""Frozen hyperparameter-selection rule for EXP-020.

Selecting on the median alone hides catastrophic failures (EXP-019 PILOT: Adam
had the best median in the table while its success rate fell to 84%). Selecting
on the raw mean is also wrong when runs can diverge, because a diverged run has
no finite L2 to average.

Rule, fixed before looking at any EXP-020 result:

    score = mean over ALL restarts of  min(L2, PENALTY)      with diverged and
                                                             non-finite runs
                                                             assigned PENALTY

    admissible  <=>  success_rate >= baseline_success - SUCCESS_TOLERANCE

Among admissible configurations, pick the lowest score. Median and win rate are
reported as SECONDARY metrics and never used to select.

PENALTY = 2.0 is above the worst finite L2 observed in any pilot arm (~1.7) and
below the scale of a true divergence, so it charges failures without letting a
single outlier dominate.
"""

PENALTY = 2.0
SUCCESS_TOLERANCE = 0.05


def score(runs, penalty=PENALTY):
    """Failure-penalised mean L2 over ALL restarts."""
    vals = []
    for r in runs:
        v = r.get("L2", float("inf"))
        if r.get("diverged") or not (v == v) or v == float("inf"):
            v = penalty
        vals.append(min(v, penalty))
    return sum(vals) / len(vals)


def admissible(summary, baseline_success, tol=SUCCESS_TOLERANCE):
    return summary["success_rate"] >= baseline_success - tol


def select(arms, baseline_success, penalty=PENALTY, tol=SUCCESS_TOLERANCE):
    """Return (best_arm, table). ``arms`` is a list of dicts with runs+summary."""
    table = []
    for a in arms:
        s = score(a["runs"], penalty)
        ok = admissible(a["summary"], baseline_success, tol)
        table.append({**{k: a["summary"].get(k) for k in
                         ("spatial", "temporal", "adam_rho", "lam_smooth",
                          "L2_mean", "L2_median", "success_rate", "diverged")},
                      "penalised_score": s, "admissible": ok})
    ok_rows = [r for r in table if r["admissible"]]
    best = min(ok_rows, key=lambda r: r["penalised_score"]) if ok_rows else None
    return best, table
