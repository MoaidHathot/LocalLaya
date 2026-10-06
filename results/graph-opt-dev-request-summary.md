# A/B: fp32 vs cur vs optA vs optB (2026-09-23T10:40:36.230Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload preset (preset dev-request: 40/40 items from presets/dev-request.eval.json). Baseline: fp32.

| variant | lane | options | preset dev-request p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | `webgpu` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 136.2 / 159.0 | 136.2 | baseline | - | - | - | 64/80 |
| cur | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 124.6 / 141.3 | 124.6 | 0.916 [0.908, 0.922] (-8.4 %) | 118/120 | 0.0520 | 0.0019 | 64/80 |
| optA | `webgpu` @ models/laya-onnx-fp16-optA | - | 58.6 / 63.1 | 58.6 | 0.434 [0.428, 0.439] (-56.6 %) | 117/120 | 0.0520 | 0.0019 | 63/80 |
| optB | `webgpu` @ models/laya-onnx-fp16-optB | - | 58.0 / 62.6 | 58.0 | 0.430 [0.423, 0.433] (-57.0 %) | 118/120 | 0.0525 | 0.0018 | 64/80 |
