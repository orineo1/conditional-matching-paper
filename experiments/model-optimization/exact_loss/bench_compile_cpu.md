# torch.compile (fixed_mm_powchain), device=cpu, torch 2.12.0, Apple M4 (partial: run stopped after 5 cells; remaining cells hung repeatedly in inductor's compile pool on macOS)

| dtype | n | m | d | compile s (first call) | compiled ms | eager ms | reference ms | x vs eager | x vs ref | break-even calls | max err vs eager |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float32 | 1 | 250 | 2 | 4.5 | 0.067 | 0.168 | 0.973 | 2.49 | 14.45 | 44555 | 4.8e-07 |
| float32 | 8 | 250 | 2 | 5.1 | 0.129 | 0.271 | 1.707 | 2.10 | 13.20 | 36158 | 9.5e-07 |
| float32 | 8 | 250 | 10 | 5.8 | 0.123 | 0.261 | 1.756 | 2.11 | 14.23 | 41846 | 4.8e-07 |
| float64 | 1 | 250 | 2 | 6.6 | 0.149 | 0.263 | 3.658 | 1.77 | 24.63 | 57783 | 0.0e+00 |
| float64 | 8 | 250 | 2 | 2.2 | 0.218 | 0.303 | 3.540 | 1.39 | 16.27 | 26134 | 6.9e-17 |
