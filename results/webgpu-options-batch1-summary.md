# A/B: base vs noval vs norobust vs unibucket vs defbucket (2026-09-23T10:21:34.967Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload both (preset smart-home: 24/65 items from data/smart-home-eval.mjs). Baseline: base.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | preset smart-home p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 36.4 / 37.7 | 52.3 / 55.0 | 133.6 / 140.8 | 159.8 / 176.3 | 158.0 | baseline | - | - | - | 51/72 |
| noval | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"validationMode":"disabled"}` | 35.0 / 37.7 | 51.0 / 55.5 | 131.0 / 146.4 | 159.0 / 178.8 | 157.5 | 0.997 [0.983, 1.003] (-0.3 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
| norobust | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"enableRobustness":false}` | 34.8 / 38.8 | 53.4 / 55.7 | 135.1 / 142.7 | 159.3 / 174.6 | 158.0 | 1.003 [0.989, 1.009] (+0.3 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
| unibucket | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"uniformBufferCacheMode":"bucket"}` | 34.6 / 41.2 | 51.7 / 55.2 | 135.7 / 145.0 | 160.3 / 179.1 | 159.2 | 1.002 [0.993, 1.009] (+0.2 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
| defbucket | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"defaultBufferCacheMode":"bucket"}` | 33.2 / 38.3 | 50.9 / 55.3 | 134.2 / 149.0 | 160.0 / 179.1 | 158.0 | 1.001 [0.990, 1.010] (+0.1 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
