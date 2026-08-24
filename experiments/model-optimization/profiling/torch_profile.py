"""Agent 1 -- operator-level torch.profiler view of _guided.run (CPU).

    python experiments/model-optimization/profiling/torch_profile.py
Writes profiling/torch_profile_<setting>_<spatial>_n<n>.txt (key_averages,
sorted by self CPU time) and an aggregated memory-by-op table.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2] / "simulations"
sys.path.insert(0, str(SIM / "src"))
sys.path.insert(0, str(SIM / "experiments"))

import torch                                                   # noqa: E402
from torch.profiler import ProfilerActivity, profile           # noqa: E402
from _common import fixed_bandwidth, load, target_set          # noqa: E402
from _guided import run                                        # noqa: E402
from _models import SEEDS, conditional_model, unconditional_model  # noqa: E402

TAG = {"2D": "", "5D": "_canonical", "10D": "_canonical"}


def main():
    for setting, spatial, n in [("2D", "no_lgd", 8), ("2D", "no_lgd", 32),
                                ("2D", "lgd", 32), ("10D", "no_lgd", 32)]:
        params = load(setting)
        S_G = target_set(params)
        bw = fixed_bandwidth(S_G)
        mc = conditional_model(params, seed=SEEDS[0], tag=TAG[setting])
        mu = unconditional_model(params, seed=SEEDS[0], tag=TAG[setting])
        run(mc, mu, S_G, bw, n, spatial, "none", 0)
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True,
                     record_shapes=False) as prof:
            run(mc, mu, S_G, bw, n, spatial, "none", 0)
        ka = prof.key_averages()
        txt = ka.table(sort_by="self_cpu_time_total", row_limit=40)
        txt2 = ka.table(sort_by="self_cpu_memory_usage", row_limit=20)
        p = HERE / f"torch_profile_{setting}_{spatial}_n{n}.txt"
        p.write_text(txt + "\n\n== sorted by self CPU memory ==\n" + txt2)
        print(p.name); print(txt[:6000]); print(flush=True)


if __name__ == "__main__":
    main()
