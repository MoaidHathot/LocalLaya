# A/B: fp32 vs wg vs cuda vs cudadyn (2026-09-23T20:45:42.266Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload poc. Baseline: fp32.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | `webgpu` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 36.8 / 38.3 | 59.4 / 62.9 | 156.6 / 166.1 | 59.4 | baseline | - | - | - | - |
| wg | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 22.7 / 27.1 | 31.2 / 32.3 | 84.0 / 86.1 | 31.2 | 0.542 [0.519, 0.592] (-45.8 %) | 14/14 | 0.0322 | 0.0033 | - |
| cuda | `cuda:fp16` @ models/laya-onnx-fp16 | `ran 18 graph` | 5.2 / 6.6 | 9.5 / 10.3 | 23.1 / 24.0 | 9.5 | 0.150 [0.142, 0.155] (-85.0 %) | 14/14 | 0.0014 | 0.0003 | - |
| cudadyn | `cuda:fp16` @ models/laya-onnx-fp16 | `exec {"graph":false}; ran 18 dynamic` | 9.8 / 20.1 | 12.4 / 20.6 | 24.6 / 26.2 | 14.8 | 0.213 [0.161, 0.266] (-78.7 %) | 14/14 | 0.0031 | 0.0004 | - |
