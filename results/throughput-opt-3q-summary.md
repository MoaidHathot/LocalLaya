# Laya throughput experiment (2026-09-23T10:51:19.482Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `webgpu:fp16` (worker threads). 3 question(s) per call (~255 input tokens). Process affinity: P-cores (0xffff). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (24 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `webgpu:fp16` | 1 | 30.32 | 90.9 | 33 | 35 | 33 | 0 | 0 |
| `webgpu:fp16` | 4 | 30.90 | 92.7 | 129 | 132 | 32 | 97 | 100 |

## D) Open loop: fixed arrival rate (~8 s per rate, min 8 / max 60 calls, 2500 ms idle before each rate)

| set | offered calls/s | achieved calls/s | total p50 (ms) | total p95 | total max | inference p50 | lanes used | note |
|---|---|---|---|---|---|---|---|---|
| `webgpu:fp16` | 5.00 | 5.08 | 70 | 73 | 108 | 70 | webgpu:fp16 40 |  |
| `webgpu:fp16` | 10.00 | 10.08 | 35 | 40 | 244 | 35 | webgpu:fp16 60 |  |
| `webgpu:fp16` | 20.00 | 20.10 | 33 | 92 | 114 | 33 | webgpu:fp16 60 |  |
| `webgpu:fp16` | 30.00 | 28.12 | 196 | 223 | 227 | 32 | webgpu:fp16 60 | latency > interval |

