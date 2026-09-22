# Laya throughput experiment (2026-09-22T22:44:26.151Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `webgpu:fp16`, `cpu:8`. 3 question(s) per call (~255 input tokens). Process affinity: P-cores (0xffff). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (24 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `webgpu:fp16` | 1 | 17.92 | 53.8 | 54 | 67 | 54 | 0 | 0 |
| `webgpu:fp16` | 2 | 19.06 | 57.2 | 105 | 109 | 52 | 52 | 58 |
| `webgpu:fp16` | 4 | 19.28 | 57.9 | 207 | 210 | 52 | 155 | 159 |
| `webgpu:fp16` | 8 | 19.25 | 57.7 | 407 | 433 | 51 | 355 | 384 |
| `cpu:8` | 1 | 2.46 | 7.4 | 375 | 561 | 375 | 0 | 0 |
| `cpu:8` | 2 | 2.90 | 8.7 | 673 | 749 | 338 | 337 | 401 |
| `cpu:8` | 4 | 2.93 | 8.8 | 1343 | 1519 | 326 | 991 | 1212 |
| `cpu:8` | 8 | 2.79 | 8.4 | 2716 | 3160 | 342 | 2379 | 2760 |

## B) All lanes, closed loop (24 calls): policy auto vs prefer-gpu

| policy | k | calls/s | q/s | total p50 (ms) | total p95 | total max | lanes used | median total/predicted |
|---|---|---|---|---|---|---|---|---|
| auto | 1 | 17.65 | 53.0 | 54 | 65 | 71 | webgpu:fp16 24 | 0.84 |
| auto | 2 | 18.90 | 56.7 | 104 | 116 | 116 | webgpu:fp16 24 | 0.87 |
| auto | 4 | 18.90 | 56.7 | 209 | 227 | 228 | webgpu:fp16 24 | 0.85 |
| auto | 8 | 19.30 | 57.9 | 409 | 428 | 428 | webgpu:fp16 24 | 0.87 |
| prefer-gpu | 1 | 19.77 | 59.3 | 50 | 54 | 58 | webgpu:fp16 24 | 0.86 |
| prefer-gpu | 2 | 18.95 | 56.9 | 104 | 114 | 117 | webgpu:fp16 24 | 0.90 |
| prefer-gpu | 4 | 14.27 | 42.8 | 206 | 556 | 647 | webgpu:fp16 24 | 0.82 |
| prefer-gpu | 8 | 19.03 | 57.1 | 417 | 436 | 438 | webgpu:fp16 24 | 0.85 |

## C) Cross-lane interference (sequential calls on one lane while the other lane runs back-to-back in the same process)

| lane | isolated p50 (ms) | isolated p95 | other lane busy p50 | busy p95 | slowdown | other lane |
|---|---|---|---|---|---|---|
| `webgpu:fp16` | 53 | 267 | 98 | 229 | 1.83x | `cpu:8` |
| `cpu:8` | 316 | 492 | 335 | 511 | 1.06x | `webgpu:fp16` |

## D) Open loop: fixed arrival rate (~8 s per rate, min 8 / max 60 calls, 2500 ms idle before each rate)

| set | offered calls/s | achieved calls/s | total p50 (ms) | total p95 | total max | inference p50 | lanes used | note |
|---|---|---|---|---|---|---|---|---|
| auto | 0.33 | 0.37 | 182 | 309 | 309 | 182 | webgpu:fp16 8 |  |
| auto | 1.00 | 1.12 | 116 | 249 | 249 | 116 | webgpu:fp16 8 |  |
| auto | 2.00 | 2.11 | 98 | 266 | 266 | 97 | webgpu:fp16 16 |  |
| auto | 5.00 | 5.07 | 61 | 88 | 284 | 61 | webgpu:fp16 40 |  |
| auto | 10.00 | 10.05 | 60 | 68 | 152 | 60 | webgpu:fp16 60 |  |
| auto | 20.00 | 18.18 | 258 | 317 | 319 | 53 | webgpu:fp16 60 | latency > interval |
| `webgpu:fp16` | 0.33 | 0.37 | 172 | 315 | 315 | 172 | webgpu:fp16 8 |  |
| `webgpu:fp16` | 1.00 | 1.12 | 118 | 267 | 267 | 118 | webgpu:fp16 8 |  |
| `webgpu:fp16` | 2.00 | 2.10 | 100 | 225 | 225 | 100 | webgpu:fp16 16 |  |
| `webgpu:fp16` | 5.00 | 5.07 | 81 | 97 | 292 | 80 | webgpu:fp16 40 |  |
| `webgpu:fp16` | 10.00 | 10.07 | 56 | 73 | 254 | 56 | webgpu:fp16 60 |  |
| `webgpu:fp16` | 20.00 | 17.69 | 284 | 421 | 429 | 55 | webgpu:fp16 60 | latency > interval |
| `cpu:8` | 0.33 | 0.37 | 333 | 410 | 410 | 333 | cpu:8 8 |  |
| `cpu:8` | 1.00 | 1.09 | 364 | 431 | 431 | 364 | cpu:8 8 |  |
| `cpu:8` | 2.00 | 2.03 | 353 | 457 | 457 | 353 | cpu:8 16 |  |
| `cpu:8` | 5.00 | 2.90 | 2786 | 5418 | 5447 | 343 | cpu:8 38 | saturated after 38 calls |
| `cpu:8` | 10.00 | 2.82 | 2790 | 5221 | 5584 | 350 | cpu:8 22 | saturated after 22 calls |
| `cpu:8` | 20.00 | 2.92 | 2895 | 5453 | 5453 | 343 | cpu:8 19 | saturated after 19 calls |

