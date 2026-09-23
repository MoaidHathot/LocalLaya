# A/B: fp32 vs wg vs dml (2026-09-23T10:44:05.552Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload both (preset smart-home: 24/65 items from data/smart-home-eval.mjs). Baseline: fp32.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | preset smart-home p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | `webgpu` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 43.1 / 43.8 | 69.7 / 79.3 | 167.0 / 180.1 | 191.4 / 219.9 | 189.3 | baseline | - | - | - | 51/72 |
| wg | `webgpu` @ models/laya-onnx-fp16-optB | - | 24.6 / 33.0 | 33.2 / 35.0 | 83.7 / 84.7 | 81.6 / 85.4 | 81.5 | 0.425 [0.420, 0.429] (-57.5 %) | 134/134 | 0.0415 | 0.0035 | 51/72 |
| dml | `dml` @ models/laya-onnx-fp16-optB | - | 18.2 / 19.4 | 131.3 / 154.6 | 137.4 / 197.0 | 139.0 / 172.7 | 138.5 | 0.737 [0.725, 0.743] (-26.3 %) | 134/134 | 0.0355 | 0.0025 | 51/72 |
