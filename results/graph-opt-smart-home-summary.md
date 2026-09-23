# A/B: fp32 vs cur vs optA vs optB (2026-09-23T10:37:49.107Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload both (preset smart-home: 24/65 items from data/smart-home-eval.mjs). Baseline: fp32.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | preset smart-home p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | `webgpu` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 37.0 / 42.3 | 60.0 / 67.9 | 153.5 / 160.4 | 179.9 / 197.9 | 178.7 | baseline | - | - | - | 51/72 |
| cur | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 40.3 / 45.7 | 53.2 / 64.1 | 132.8 / 134.9 | 152.3 / 162.1 | 151.9 | 0.846 [0.840, 0.850] (-15.4 %) | 134/134 | 0.0415 | 0.0033 | 51/72 |
| optA | `webgpu` @ models/laya-onnx-fp16-optA | - | 25.5 / 32.2 | 32.1 / 34.3 | 83.2 / 86.8 | 81.2 / 84.3 | 81.2 | 0.454 [0.450, 0.457] (-54.6 %) | 134/134 | 0.0415 | 0.0034 | 51/72 |
| optB | `webgpu` @ models/laya-onnx-fp16-optB | - | 21.1 / 27.3 | 31.7 / 32.4 | 83.9 / 85.4 | 81.0 / 84.2 | 81.0 | 0.453 [0.448, 0.457] (-54.7 %) | 134/134 | 0.0415 | 0.0035 | 51/72 |
