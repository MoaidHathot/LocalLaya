# Laya throughput experiment (2026-09-23T11:22:42.110Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `cuda:fp16` (worker threads). 3 question(s) per call (~255 input tokens). Process affinity: P-cores (0xffff). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (24 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `cuda:fp16` | 1 | 65.89 | 197.7 | 14 | 23 | 14 | 0 | 0 |
| `cuda:fp16` | 4 | 82.91 | 248.7 | 48 | 52 | 12 | 36 | 39 |
| `cuda:fp16` | 8 | 86.21 | 258.6 | 92 | 94 | 11 | 80 | 82 |

## D) Open loop: fixed arrival rate (~8 s per rate, min 8 / max 60 calls, 2500 ms idle before each rate)

| set | offered calls/s | achieved calls/s | total p50 (ms) | total p95 | total max | inference p50 | lanes used | note |
|---|---|---|---|---|---|---|---|---|
| `cuda:fp16` | 5.00 | 5.09 | 41 | 61 | 63 | 41 | cuda:fp16 40 |  |
| `cuda:fp16` | 10.00 | 10.11 | 30 | 40 | 174 | 29 | cuda:fp16 60 |  |
| `cuda:fp16` | 20.00 | 20.13 | 20 | 75 | 78 | 20 | cuda:fp16 60 |  |
| `cuda:fp16` | 40.00 | 39.96 | 13 | 79 | 88 | 13 | cuda:fp16 60 |  |
| `cuda:fp16` | 60.00 | 59.19 | 35 | 84 | 89 | 12 | cuda:fp16 60 | latency > interval |

