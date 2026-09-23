# A/B: base vs stsimple vs stlazy vs nhwc vs combo (2026-09-23T10:24:39.397Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload both (preset smart-home: 24/65 items from data/smart-home-eval.mjs). Baseline: base.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | preset smart-home p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | `webgpu:fp16` @ models/laya-onnx-fp16 | - | 35.5 / 40.4 | 52.5 / 54.2 | 136.9 / 150.4 | 158.1 / 177.1 | 156.2 | baseline | - | - | - | 51/72 |
| stsimple | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"storageBufferCacheMode":"simple"}` | 37.3 / 46.4 | 52.3 / 57.7 | 138.4 / 147.8 | 156.6 / 179.9 | 155.6 | 1.002 [0.987, 1.005] (+0.2 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
| stlazy | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"storageBufferCacheMode":"lazyRelease"}` | 35.7 / 40.2 | 51.2 / 54.7 | 139.9 / 146.8 | 157.9 / 175.5 | 156.8 | 1.001 [0.994, 1.004] (+0.1 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
| nhwc | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"preferredLayout":"NHWC"}` | 37.4 / 41.0 | 53.4 / 55.9 | 136.9 / 154.8 | 157.4 / 176.1 | 156.3 | 0.998 [0.986, 1.003] (-0.2 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
| combo | `webgpu:fp16` @ models/laya-onnx-fp16 | `{"validationMode":"disabled","enableRobustness":false,"uniformBufferCacheMode":"bucket","defaultBufferCacheMode":"bucket"}` | 34.7 / 39.0 | 54.1 / 58.9 | 136.7 / 142.8 | 157.7 / 181.2 | 156.3 | 1.005 [0.995, 1.010] (+0.5 %) | 134/134 | 0.0000 | 0.0000 | 51/72 |
