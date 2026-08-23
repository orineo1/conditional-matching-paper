"""Passive recording of intermediate states, for equivalence testing.

Design constraints, in order of importance:

  * The tracer must not change numerics.  It stores ``detach().clone()``:
    ``detach`` so no graph is retained, ``clone`` so a later in-place write
    cannot corrupt an already-recorded value.
  * The tracer must not consume randomness.
  * The default must be a no-op, so that a traced run and an untraced run are
    provably identical.

We deliberately do NOT use ``register_forward_hook``.  Hooks fire on every
forward pass including those triggered by double-backward, so the number of
firings is itself configuration-dependent and two engines would record
different things for reasons unrelated to correctness.
"""

import torch


class Tracer:
    """Records named tensors keyed by ``(name, t, r, k)``."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = {}
        self.order = []

    def __call__(self, name, t, r, k, tensor):
        if not self.enabled:
            return
        key = (name, None if t is None else int(t),
               None if r is None else int(r),
               None if k is None else int(k))
        if key in self.records:
            raise KeyError(
                f"trace key {key!r} written twice; keys must be unique so that "
                "the two engines can be compared by key rather than by position"
            )
        value = tensor.detach().clone() if torch.is_tensor(tensor) else tensor
        self.records[key] = value
        self.order.append(key)

    def keys(self):
        return set(self.records)


def compare_traces(tracer_a, tracer_b, atol=0.0, label_a="reference", label_b="engine"):
    """Compare two traces key-by-key.

    Returns ``(ok, report)``.  ``report`` is a dict with:

      ``missing_in_b`` / ``missing_in_a``
          Keys one engine produced and the other did not.  A non-empty set here
          is a structural divergence and is reported *before* any numeric
          comparison, because comparing values would be meaningless.
      ``max_abs_err``
          Largest absolute difference over all shared keys.
      ``first_divergence``
          The first key, in engine A's execution order, whose absolute
          difference exceeds ``atol``; ``None`` if there is none.
      ``per_key``
          Sorted list of ``(key, max_abs_diff)`` for the worst offenders.
    """
    a, b = tracer_a.records, tracer_b.records
    missing_in_b = sorted(set(a) - set(b), key=repr)
    missing_in_a = sorted(set(b) - set(a), key=repr)

    per_key = []
    first_divergence = None
    max_abs_err = 0.0

    for key in tracer_a.order:
        if key not in b:
            continue
        va, vb = a[key], b[key]
        if torch.is_tensor(va) != torch.is_tensor(vb):
            per_key.append((key, float("inf")))
            if first_divergence is None:
                first_divergence = (key, "tensor/non-tensor mismatch")
            max_abs_err = float("inf")
            continue
        if not torch.is_tensor(va):
            continue
        if va.shape != vb.shape:
            per_key.append((key, float("inf")))
            if first_divergence is None:
                first_divergence = (key, "shape mismatch", tuple(va.shape), tuple(vb.shape))
            max_abs_err = float("inf")
            continue
        diff = (va.double() - vb.double()).abs().max().item()
        per_key.append((key, diff))
        max_abs_err = max(max_abs_err, diff)
        if diff > atol and first_divergence is None:
            first_divergence = (key, diff)

    per_key.sort(key=lambda kv: -kv[1])
    # A comparison of zero keys is not agreement, it is an empty harness.
    ok = (len(per_key) > 0
          and not missing_in_a and not missing_in_b
          and max_abs_err <= atol)
    return ok, {
        "label_a": label_a,
        "label_b": label_b,
        "missing_in_b": missing_in_b,
        "missing_in_a": missing_in_a,
        "max_abs_err": max_abs_err,
        "first_divergence": first_divergence,
        "per_key": per_key[:20],
        "n_keys_compared": len(per_key),
    }


def format_report(report):
    lines = [
        f"trace comparison: {report['label_a']} vs {report['label_b']}",
        f"  keys compared : {report['n_keys_compared']}",
        f"  max abs error : {report['max_abs_err']:.3e}",
    ]
    if report["missing_in_b"]:
        lines.append(f"  MISSING in {report['label_b']}: {report['missing_in_b'][:10]}")
    if report["missing_in_a"]:
        lines.append(f"  MISSING in {report['label_a']}: {report['missing_in_a'][:10]}")
    if report["first_divergence"] is not None:
        lines.append(f"  FIRST DIVERGENCE: {report['first_divergence']}")
    return "\n".join(lines)
