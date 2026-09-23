# A/B: cpu vs cpuopt vs cpuopt16 (2026-09-23T10:47:40.331Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 6 interleaved rounds; workload poc. Baseline: cpu.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | `cpu:8` @ models/receptron--laya-onnx/68f27dfe5a27a54fb2b1fefc432f43f972e90868 | - | 513.3 / 607.1 | 1069.4 / 1321.2 | 1794.6 / 2419.1 | 1069.4 | baseline | - | - | - | - |
| cpuopt | `cpu:8` @ models/laya-onnx-fp32-opt | - | 514.9 / 882.6 | 1072.6 / 1503.5 | 1937.1 / 2057.4 | 1072.6 | 1.030 [0.953, 1.058] (+3.0 %) | 14/14 | 0.0000 | 0.0000 | - |
| cpuopt16 | `cpu:8` @ models/laya-onnx-fp16-optB | - | 587.8 / 851.7 | 1148.4 / 1357.9 | 1953.3 / 2029.2 | 1148.4 | 1.010 [0.961, 1.214] (+1.0 %) | 14/14 | 0.0012 | 0.0002 | - |
