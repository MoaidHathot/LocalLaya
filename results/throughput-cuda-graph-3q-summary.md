# Laya throughput experiment (2026-09-23T20:56:12.191Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `cuda:fp16` (worker threads). 3 question(s) per call (~255 input tokens). Process affinity: P-cores (0xffff). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (12 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `cuda:fp16` | 1 | 114.27 | 342.8 | 9 | 10 | 9 | 0 | 0 |
| `cuda:fp16` | 2 | 115.13 | 345.4 | 17 | 18 | 8 | 9 | 9 |
| `cuda:fp16` | 4 | 115.29 | 345.9 | 35 | 35 | 8 | 26 | 27 |
| `cuda:fp16` | 8 | 113.88 | 341.6 | 52 | 70 | 9 | 44 | 62 |

