# Laya throughput experiment (2026-09-22T22:16:33.458Z)

Machine: Intel(R) Core(TM) i9-14900KF (32 threads), 64 GiB RAM, GPU: NVIDIA GeForce RTX 4070; Node v25.3.0, onnxruntime-node 1.30.0

Lanes: `webgpu:fp16`, `cpu:8`. 3 question(s) per call (~255 input tokens). All calls via `LayaRouter.decide()`; "total" = wall time seen by the caller incl. the per-lane queue, "inference" = `routing.ms`.

## A) Single lane, closed loop (24 calls, k callers in flight, lane forced)

| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |
|---|---|---|---|---|---|---|---|---|
| `webgpu:fp16` | 1 | 20.51 | 61.5 | 48 | 54 | 48 | 0 | 0 |
| `webgpu:fp16` | 2 | 21.01 | 63.0 | 95 | 96 | 47 | 48 | 49 |
| `webgpu:fp16` | 4 | 20.76 | 62.3 | 192 | 198 | 48 | 144 | 151 |
| `webgpu:fp16` | 8 | 20.99 | 63.0 | 380 | 384 | 47 | 332 | 336 |
| `cpu:8` | 1 | 3.83 | 11.5 | 262 | 289 | 262 | 0 | 0 |
| `cpu:8` | 2 | 3.76 | 11.3 | 529 | 591 | 262 | 262 | 314 |
| `cpu:8` | 4 | 3.90 | 11.7 | 1020 | 1045 | 256 | 764 | 791 |
| `cpu:8` | 8 | 3.81 | 11.4 | 2076 | 2117 | 257 | 1820 | 1867 |

## B) All lanes, closed loop (24 calls): policy auto vs prefer-gpu

| policy | k | calls/s | q/s | total p50 (ms) | total p95 | total max | lanes used | median total/predicted |
|---|---|---|---|---|---|---|---|---|
| auto | 1 | 20.50 | 61.5 | 48 | 54 | 54 | webgpu:fp16 24 | 0.95 |
| auto | 2 | 19.01 | 57.0 | 100 | 122 | 124 | webgpu:fp16 24 | 0.94 |
| auto | 4 | 10.15 | 30.5 | 213 | 879 | 885 | webgpu:fp16 18, cpu:8 6 | 1.00 |
| auto | 8 | 6.58 | 19.8 | 1125 | 1515 | 1742 | webgpu:fp16 12, cpu:8 12 | 1.06 |
| prefer-gpu | 1 | 19.17 | 57.5 | 49 | 71 | 72 | webgpu:fp16 24 | 0.90 |
| prefer-gpu | 2 | 20.05 | 60.2 | 99 | 103 | 106 | webgpu:fp16 24 | 0.96 |
| prefer-gpu | 4 | 19.68 | 59.1 | 196 | 224 | 226 | webgpu:fp16 24 | 0.95 |
| prefer-gpu | 8 | 20.02 | 60.1 | 391 | 412 | 414 | webgpu:fp16 24 | 0.96 |

## C) Cross-lane interference (sequential calls on one lane while the other lane runs back-to-back in the same process)

| lane | isolated p50 (ms) | isolated p95 | other lane busy p50 | busy p95 | slowdown | other lane |
|---|---|---|---|---|---|---|
| `webgpu:fp16` | 49 | 51 | 321 | 464 | 6.55x | `cpu:8` |
| `cpu:8` | 358 | 383 | 480 | 630 | 1.34x | `webgpu:fp16` |

## D) Open loop: fixed arrival rate (~8 s per rate, min 8 / max 60 calls, cold GPU at the start of each rate)

| set | offered calls/s | achieved calls/s | total p50 (ms) | total p95 | total max | lanes used | note |
|---|---|---|---|---|---|---|---|
| auto | 0.33 | 0.37 | 269 | 379 | 379 | webgpu:fp16 8 |  |
| auto | 1.00 | 1.11 | 139 | 329 | 329 | webgpu:fp16 8 |  |
| auto | 2.00 | 2.09 | 128 | 526 | 526 | webgpu:fp16 16 |  |
| auto | 5.00 | 5.05 | 111 | 138 | 377 | webgpu:fp16 40 |  |
| auto | 10.00 | 4.48 | 1792 | 3387 | 3729 | webgpu:fp16 16, cpu:8 12 | saturated after 28 calls |
| auto | 20.00 | 5.06 | 1641 | 3082 | 3189 | webgpu:fp16 13, cpu:8 8 | saturated after 21 calls |
| `webgpu:fp16` | 0.33 | 0.37 | 235 | 357 | 357 | webgpu:fp16 8 |  |
| `webgpu:fp16` | 1.00 | 1.11 | 183 | 433 | 433 | webgpu:fp16 8 |  |
| `webgpu:fp16` | 2.00 | 2.10 | 120 | 250 | 250 | webgpu:fp16 16 |  |
| `webgpu:fp16` | 5.00 | 5.05 | 112 | 135 | 281 | webgpu:fp16 40 |  |
| `webgpu:fp16` | 10.00 | 10.02 | 82 | 245 | 384 | webgpu:fp16 60 |  |
| `webgpu:fp16` | 20.00 | 10.95 | 834 | 1280 | 1290 | webgpu:fp16 31 | saturated after 31 calls |
| `cpu:8` | 0.33 | 0.37 | 353 | 425 | 425 | cpu:8 8 |  |
| `cpu:8` | 1.00 | 1.08 | 385 | 427 | 427 | cpu:8 8 |  |
| `cpu:8` | 2.00 | 2.03 | 375 | 477 | 477 | cpu:8 16 |  |
| `cpu:8` | 5.00 | 2.73 | 3218 | 5780 | 5815 | cpu:8 36 | saturated after 36 calls |
| `cpu:8` | 10.00 | 3.76 | 2135 | 4250 | 4253 | cpu:8 26 | saturated after 26 calls |
| `cpu:8` | 20.00 | 3.78 | 2084 | 3973 | 4241 | cpu:8 20 | saturated after 20 calls |

