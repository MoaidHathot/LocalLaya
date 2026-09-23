# A/B: cudadyn vs cuda (2026-09-23T20:46:23.784Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), GPU NVIDIA GeForce RTX 4070; Node v25.3.0. 8 interleaved rounds; workload both (preset dev-request: 40/40 items from W:\Github\LocalLaya\presets\dev-request.eval.json). Baseline: cudadyn.

| variant | lane | options | poc 1q p50/p90 | poc 3q p50/p90 | poc 10q p50/p90 | preset dev-request p50/p90 | all p50 | paired ratio vs base [95 % CI] | arg-max agree | max abs dp | mean abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cudadyn | `cuda:fp16` @ models/laya-onnx-fp16 | `exec {"graph":false}; ran 344 dynamic` | 10.8 / 20.6 | 12.4 / 22.0 | 24.5 / 26.6 | 18.8 / 20.2 | 18.8 | baseline | - | - | - | 64/80 |
| cuda | `cuda:fp16` @ models/laya-onnx-fp16 | `ran 344 graph` | 5.0 / 5.8 | 8.6 / 10.2 | 22.9 / 24.3 | 16.0 / 16.9 | 16.0 | 0.858 [0.848, 0.862] (-14.2 %) | 134/134 | 0.0063 | 0.0003 | 64/80 |
