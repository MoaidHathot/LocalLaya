# A/B: fp32 vs wg vs cuda (2026-09-23T11:15:30.699Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload preset (preset dev-request: 40/40 items from W:\Github\LocalLaya\presets\dev-request.eval.json). Baseline: fp32.

| variant | lane | options | preset dev-request p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | `webgpu` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 133.0 / 153.7 | 133.0 | baseline | - | - | - | 64/80 |
| wg | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 56.6 / 60.6 | 56.6 | 0.423 [0.419, 0.425] (-57.7 %) | 118/120 | 0.0525 | 0.0018 | 64/80 |
| cuda | `cuda:fp16` @ models/laya-onnx-fp16 | - | 18.9 / 20.9 | 18.9 | 0.140 [0.138, 0.141] (-86.0 %) | 119/120 | 0.0416 | 0.0007 | 64/80 |
