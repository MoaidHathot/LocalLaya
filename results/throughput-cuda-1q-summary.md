# Laya throughput experiment (2026-09-23T11:23:19.536Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `cuda:fp16` (worker threads). 1 question(s) per call (~85 input tokens). Process affinity: P-cores (0xffff). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (30 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `cuda:fp16` | 1 | 96.47 | 96.5 | 9 | 15 | 9 | 0 | 0 |

