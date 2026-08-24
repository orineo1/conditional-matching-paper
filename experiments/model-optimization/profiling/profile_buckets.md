## 2D no_lgd/none n=8  total median 228.5 ms/run, 2.31 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0232 | 10.2% | 0.235 | 0.0207 | 0.0335 |
|   uncond_fwd | 0.0123 | 5.4% | 0.124 | 0.0110 | 0.0177 |
| cond_sample | 0.0431 | 18.9% | 0.435 | 0.0398 | 0.0619 |
|   cond_fwd | 0.0398 | 17.4% | 0.402 | 0.0369 | 0.0574 |
| mmd | 0.0441 | 19.3% | 0.445 | 0.0308 | 0.0737 |
|   kernel | 0.0384 | 16.8% | 0.388 | 0.0265 | 0.0618 |
|     cdist | 0.0062 | 2.7% | 0.063 | 0.0051 | 0.0089 |
|     exp | 0.0135 | 5.9% | 0.136 | 0.0088 | 0.0212 |
| backward | 0.0955 | 41.8% | 0.964 | 0.0773 | 0.1835 |
| other | 0.0212 | 9.3% | 0.214 | 0.0186 | 0.0369 |
| total | 0.2285 | 100.0% | 2.308 | 0.1877 | 0.3894 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 99, "cond_network_evals": 495, "conditional_samples": 792, "target_samples": 250, "mmd_evals": 99, "kernel_entries_per_mmd": 332820, "kernel_entries_per_run": 32949180, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 99, "target_target_fraction_of_kernel": 0.9389459768042786, "backward_calls": 99, "adam_steps": 0, "cdist_shapes": {"(258, 1)": 99}}

## 2D no_lgd/none n=32  total median 238.7 ms/run, 2.41 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0230 | 9.6% | 0.232 | 0.0216 | 0.0309 |
|   uncond_fwd | 0.0122 | 5.1% | 0.123 | 0.0115 | 0.0157 |
| cond_sample | 0.0484 | 20.3% | 0.489 | 0.0469 | 0.0601 |
|   cond_fwd | 0.0452 | 18.9% | 0.456 | 0.0438 | 0.0555 |
| mmd | 0.0445 | 18.7% | 0.450 | 0.0386 | 0.0832 |
|   kernel | 0.0395 | 16.6% | 0.399 | 0.0340 | 0.0764 |
|     cdist | 0.0062 | 2.6% | 0.062 | 0.0056 | 0.0091 |
|     exp | 0.0124 | 5.2% | 0.125 | 0.0109 | 0.0210 |
| backward | 0.1028 | 43.1% | 1.039 | 0.0925 | 0.1697 |
| other | 0.0212 | 8.9% | 0.214 | 0.0194 | 0.0346 |
| total | 0.2387 | 100.0% | 2.411 | 0.2192 | 0.3760 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 99, "cond_network_evals": 495, "conditional_samples": 3168, "target_samples": 250, "mmd_evals": 99, "kernel_entries_per_mmd": 397620, "kernel_entries_per_run": 39364380, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 99, "target_target_fraction_of_kernel": 0.7859262612544641, "backward_calls": 99, "adam_steps": 0, "cdist_shapes": {"(282, 1)": 99}}

## 2D lgd/none n=8  total median 924.3 ms/run, 9.34 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0409 | 4.4% | 0.413 | 0.0248 | 0.0496 |
|   uncond_fwd | 0.0209 | 2.3% | 0.211 | 0.0130 | 0.0243 |
| cond_sample | 0.1928 | 20.9% | 1.947 | 0.1242 | 0.2396 |
|   cond_fwd | 0.1781 | 19.3% | 1.799 | 0.1151 | 0.2228 |
| mmd | 0.1900 | 20.6% | 1.919 | 0.1020 | 0.2484 |
|   kernel | 0.1641 | 17.8% | 1.658 | 0.0881 | 0.2102 |
|     cdist | 0.0280 | 3.0% | 0.283 | 0.0160 | 0.0335 |
|     exp | 0.0549 | 5.9% | 0.555 | 0.0301 | 0.0698 |
| backward | 0.4041 | 43.7% | 4.082 | 0.2282 | 0.5767 |
| other | 0.0939 | 10.2% | 0.948 | 0.0556 | 0.1165 |
| total | 0.9243 | 100.0% | 9.336 | 0.5348 | 1.2308 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 297, "cond_network_evals": 1485, "conditional_samples": 2376, "target_samples": 250, "mmd_evals": 297, "kernel_entries_per_mmd": 332820, "kernel_entries_per_run": 98847540, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 297, "target_target_fraction_of_kernel": 0.9389459768042786, "backward_calls": 99, "adam_steps": 0, "cdist_shapes": {"(258, 1)": 297}}

## 2D lgd/none n=32  total median 1811.6 ms/run, 18.30 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0653 | 3.6% | 0.659 | 0.0311 | 0.1180 |
|   uncond_fwd | 0.0332 | 1.8% | 0.335 | 0.0159 | 0.0622 |
| cond_sample | 0.3572 | 19.7% | 3.608 | 0.1648 | 0.8166 |
|   cond_fwd | 0.3320 | 18.3% | 3.354 | 0.1538 | 0.7416 |
| mmd | 0.4042 | 22.3% | 4.083 | 0.1405 | 1.0711 |
|   kernel | 0.3594 | 19.8% | 3.630 | 0.1236 | 0.9062 |
|     cdist | 0.0576 | 3.2% | 0.582 | 0.0196 | 0.2531 |
|     exp | 0.1142 | 6.3% | 1.153 | 0.0396 | 0.2367 |
| backward | 0.8272 | 45.7% | 8.356 | 0.3085 | 1.9205 |
| other | 0.1569 | 8.7% | 1.585 | 0.0685 | 0.3170 |
| total | 1.8116 | 100.0% | 18.299 | 0.7134 | 4.0877 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 297, "cond_network_evals": 1485, "conditional_samples": 9504, "target_samples": 250, "mmd_evals": 297, "kernel_entries_per_mmd": 397620, "kernel_entries_per_run": 118093140, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 297, "target_target_fraction_of_kernel": 0.7859262612544641, "backward_calls": 99, "adam_steps": 0, "cdist_shapes": {"(282, 1)": 297}}

## 2D no_lgd/adam n=32  total median 904.8 ms/run, 9.14 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0820 | 9.1% | 0.828 | 0.0336 | 0.1095 |
|   uncond_fwd | 0.0438 | 4.8% | 0.442 | 0.0176 | 0.0523 |
| cond_sample | 0.1491 | 16.5% | 1.506 | 0.0687 | 0.1761 |
|   cond_fwd | 0.1372 | 15.2% | 1.386 | 0.0639 | 0.1609 |
| mmd | 0.1748 | 19.3% | 1.766 | 0.0648 | 0.2108 |
|   kernel | 0.1472 | 16.3% | 1.487 | 0.0574 | 0.1828 |
|     cdist | 0.0219 | 2.4% | 0.221 | 0.0087 | 0.0326 |
|     exp | 0.0389 | 4.3% | 0.393 | 0.0205 | 0.0437 |
| backward | 0.4007 | 44.3% | 4.047 | 0.1468 | 0.4974 |
| adam | 0.0077 | 0.8% | 0.078 | 0.0027 | 0.0101 |
| other | 0.0809 | 8.9% | 0.817 | 0.0311 | 0.0969 |
| total | 0.9048 | 100.0% | 9.139 | 0.3505 | 1.0906 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 99, "cond_network_evals": 495, "conditional_samples": 3168, "target_samples": 250, "mmd_evals": 99, "kernel_entries_per_mmd": 397620, "kernel_entries_per_run": 39364380, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 99, "target_target_fraction_of_kernel": 0.7859262612544641, "backward_calls": 99, "adam_steps": 99, "cdist_shapes": {"(282, 1)": 99}}

## 10D no_lgd/none n=32  total median 504.6 ms/run, 5.10 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0481 | 9.5% | 0.486 | 0.0394 | 0.0558 |
|   uncond_fwd | 0.0249 | 4.9% | 0.251 | 0.0203 | 0.0290 |
| cond_sample | 0.0956 | 18.9% | 0.966 | 0.0787 | 0.1185 |
|   cond_fwd | 0.0891 | 17.6% | 0.900 | 0.0733 | 0.1102 |
| mmd | 0.0986 | 19.5% | 0.996 | 0.0763 | 0.1114 |
|   kernel | 0.0847 | 16.8% | 0.856 | 0.0673 | 0.0973 |
|     cdist | 0.0133 | 2.6% | 0.134 | 0.0104 | 0.0160 |
|     exp | 0.0268 | 5.3% | 0.271 | 0.0218 | 0.0331 |
| backward | 0.2169 | 43.0% | 2.191 | 0.1744 | 0.2535 |
| other | 0.0464 | 9.2% | 0.469 | 0.0384 | 0.0553 |
| total | 0.5046 | 100.0% | 5.097 | 0.4071 | 0.5934 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 99, "cond_network_evals": 495, "conditional_samples": 3168, "target_samples": 250, "mmd_evals": 99, "kernel_entries_per_mmd": 397620, "kernel_entries_per_run": 39364380, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 99, "target_target_fraction_of_kernel": 0.7859262612544641, "backward_calls": 99, "adam_steps": 0, "cdist_shapes": {"(282, 1)": 99}}

## 10D lgd/none n=32  total median 1066.5 ms/run, 10.77 ms/step
| bucket | median s | % of total | per step ms | min s | max s |
|---|---|---|---|---|---|
| ddim | 0.0417 | 3.9% | 0.422 | 0.0366 | 0.0489 |
|   uncond_fwd | 0.0211 | 2.0% | 0.213 | 0.0186 | 0.0247 |
| cond_sample | 0.2296 | 21.5% | 2.319 | 0.2100 | 0.2635 |
|   cond_fwd | 0.2142 | 20.1% | 2.164 | 0.1958 | 0.2446 |
| mmd | 0.2254 | 21.1% | 2.276 | 0.1974 | 0.2670 |
|   kernel | 0.1990 | 18.7% | 2.010 | 0.1759 | 0.2347 |
|     cdist | 0.0304 | 2.9% | 0.307 | 0.0242 | 0.0370 |
|     exp | 0.0634 | 5.9% | 0.640 | 0.0604 | 0.0740 |
| backward | 0.4670 | 43.8% | 4.718 | 0.3946 | 0.5484 |
| other | 0.0991 | 9.3% | 1.001 | 0.0877 | 0.1180 |
| total | 1.0665 | 100.0% | 10.773 | 0.9263 | 1.2454 |
{"steps": 99, "denoiser_calls": 99, "cond_sampler_calls": 297, "cond_network_evals": 1485, "conditional_samples": 9504, "target_samples": 250, "mmd_evals": 297, "kernel_entries_per_mmd": 397620, "kernel_entries_per_run": 118093140, "target_target_entries_per_mmd": 312500, "target_target_recomputed_per_run": 297, "target_target_fraction_of_kernel": 0.7859262612544641, "backward_calls": 99, "adam_steps": 0, "cdist_shapes": {"(282, 1)": 297}}

