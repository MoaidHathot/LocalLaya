# A/B: fp32 vs wg vs cuda (2026-09-23T11:14:02.148Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload both (preset smart-home: 24/65 items from data/smart-home-eval.mjs). Baseline: fp32.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | preset smart-home p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | `webgpu` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 45.2 / 53.0 | 62.8 / 68.4 | 157.7 / 208.1 | 186.1 / 206.3 | 184.9 | baseline | - | - | - | 51/72 |
| wg | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 28.7 / 32.0 | 31.9 / 32.2 | 83.4 / 90.3 | 79.8 / 82.6 | 79.8 | 0.429 [0.425, 0.433] (-57.1 %) | 134/134 | 0.0415 | 0.0035 | 51/72 |
| cuda | `cuda:fp16` @ models/laya-onnx-fp16 | - | 10.2 / 25.8 | 12.9 / 25.0 | 25.3 / 28.4 | 24.4 / 27.4 | 24.4 | 0.131 [0.130, 0.132] (-86.9 %) | 134/134 | 0.0375 | 0.0016 | 51/72 |
