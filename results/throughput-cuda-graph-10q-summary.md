# Laya throughput experiment (2026-09-23T20:56:06.979Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `cuda:fp16` (worker threads). 10 question(s) per call (~850 input tokens). Process affinity: P-cores (0xffff). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (12 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `cuda:fp16` | 1 | 44.69 | 446.9 | 22 | 23 | 22 | 0 | 0 |
| `cuda:fp16` | 2 | 44.26 | 442.6 | 45 | 46 | 22 | 22 | 24 |
| `cuda:fp16` | 4 | 44.08 | 440.8 | 90 | 92 | 23 | 68 | 70 |
| `cuda:fp16` | 8 | 44.24 | 442.4 | 138 | 181 | 22 | 115 | 159 |

