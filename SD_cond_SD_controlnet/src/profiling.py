"""
profiling.py — per-step wall-time split and peak memory for the guided loop.

Usage:
    prof = StepProfiler(enabled=True, device="cuda")
    prof.begin_step(i, t)
    with prof.section("architect"): ...
    prof.in_backward = True; ...autograd.grad...; prof.in_backward = False
    prof.end_step(extra={"mmd": ...})
    prof.dump(path)

Every `section` synchronises the CUDA device before and after, so the split
is wall-clock accurate (and adds ~N*2 syncs per step — profile runs are for
the timing split, not the absolute best throughput).  While `in_backward`
is True, any section (e.g. the checkpointed sprinter closure being
recomputed) is attributed to "backward".  When disabled every call is a
no-op context.
"""

import contextlib
import json
import time

import torch


class StepProfiler:
    def __init__(self, enabled=False, device="cuda"):
        self.enabled = enabled
        self.device = device
        self.cuda = enabled and torch.cuda.is_available()
        self.steps = []
        self.current = None
        self.in_backward = False
        self.prefix = ""      # e.g. "nograd_" during the backsel no-grad pass
        self.meta = {}
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def begin_step(self, step, timestep):
        if not self.enabled:
            return
        self._sync()
        self.current = {"step": int(step), "timestep": float(timestep),
                        "sections": {}, "_t0": time.perf_counter()}
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def section(self, name):
        if not self.enabled or self.current is None:
            yield
            return
        name = "backward" if self.in_backward else self.prefix + name
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            secs = self.current["sections"]
            secs[name] = secs.get(name, 0.0) + (time.perf_counter() - t0)

    def end_step(self, extra=None):
        if not self.enabled or self.current is None:
            return
        self._sync()
        cur = self.current
        cur["total_s"] = time.perf_counter() - cur.pop("_t0")
        cur["accounted_s"] = sum(cur["sections"].values())
        if self.cuda:
            cur["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            cur["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        if extra:
            cur.update(extra)
        self.steps.append(cur)
        self.current = None

    def summary(self):
        if not self.steps:
            return {}
        keys = sorted({k for s in self.steps for k in s["sections"]})
        n = len(self.steps)
        out = {"n_steps": n,
               "mean_total_s": sum(s["total_s"] for s in self.steps) / n,
               "mean_sections_s": {k: sum(s["sections"].get(k, 0.0) for s in self.steps) / n
                                   for k in keys}}
        if self.cuda:
            out["max_memory_allocated_mb"] = max(s["max_memory_allocated_mb"] for s in self.steps)
        return out

    def dump(self, path):
        if not self.enabled:
            return
        with open(path, "w") as f:
            json.dump({"meta": self.meta, "summary": self.summary(), "steps": self.steps},
                      f, indent=2)
