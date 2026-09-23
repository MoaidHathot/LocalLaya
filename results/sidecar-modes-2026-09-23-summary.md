# Sidecar call paths, sporadic traffic and the GPU keep-alive (2026-09-23)

Machine: i9-14900KF / RTX 4070, Node 25.3, sidecar with the default lanes `cuda:fp16,webgpu:fp16,cpu:8`.
Question answered here: what is the fastest way to *use* the sidecar, and when does the lane matter?

## 1. Cost of one decision by call path (3-5 questions, sidecar warm)

| how the caller reaches the model | per call | of which inference | note |
|---|---|---|---|
| persistent process, `POST /decide` over HTTP (Node client, keep-alive connection) | **~13-15 ms** | 12 ms | what an agent runtime or service should do for many calls |
| PowerShell `Invoke-RestMethod` | 31 ms | 12 ms | PowerShell's own HTTP overhead |
| `scripts/laya.mjs` (skill wrapper), fast path: this Node process -> HTTP | **80-90 ms** | 12-15 ms | Node start-up ~37 ms + imports + 2 HTTP round trips |
| `node ask.mjs --sidecar --json` | 87-127 ms | 12-15 ms | + preset loading, formatting modules |
| old wrapper (spawned `ask.mjs`: two Node processes) | 130-165 ms | 12 ms | replaced by the fast path |
| `node ask.mjs --local` (load the model in this process) | 2.2-2.5 s | 12 ms | no background process; every call pays the load |

A shell call per decision cannot go below ~70 ms of process start-up + module loading; the inference is 15 % of
it. Anything that makes more than a few calls per minute should hold an HTTP connection to the sidecar.

## 2. Sporadic traffic: the one regime where the lane matters

Per lane, 3 questions, p50 of 8 calls per gap (`experiments/sporadic.mjs`; cuda/webgpu on the optimised fp16 bundle):

| gap between calls | `cuda:fp16` | `webgpu:fp16` | `cpu:8` |
|---|---|---|---|
| 0 (back-to-back) | 12-13 ms | 32 | ~270 |
| 250 ms | 13-18 | 32 | ~270 |
| 1 s | 39-45 (bimodal, max 250) | 81 | ~270 |
| 3 s | 68-217 (**bimodal: ~50 or ~200-300**) | 156 | ~270 |
| 1 question after 3 s | 153-185 (min 30, max 240) | ~101 | ~105 |

The CUDA process lane pays the GPU's idle-clock ramp like WebGPU, and its slow mode is worse (200-300 ms).
For a *single* question after a 3 s pause the CPU lane (~105 ms, gap-insensitive) or WebGPU (~101 ms) beat the
CUDA lane's median; the router's cold-state EMA learns this per lane and bucket (`DEFAULT_PRIORS.cuda` are the
measured medians without keep-alive).

### GPU keep-alive (a tiny GPU call every 500 ms for `--gpu-keepalive` after each real call; default 30 s)

Through the sidecar over HTTP, 3 questions every 3 s, **n = 20 per arm, arms alternated** (the bimodal
distribution fooled two earlier n = 6-8 comparisons in opposite directions - this is the number to trust):

| `--gpu-keepalive` | p50 | mean | p90 | max | slow calls (> 120 ms) |
|---|---|---|---|---|---|
| `0` | 174 ms | 157 | 240 | 333 | **12 / 20** |
| `30s` (default) | **46 ms** | 77 | 202 | 276 | **4 / 20** |

Cost while the keep-alive runs (nvidia-smi, 6 s windows): idle 26 W / 345 MHz -> keep-alive at 500 ms 30 W /
690 MHz (at 250 ms 27 W / 555 MHz); busy inference 106 W. Interval: 250 ms -> 3 q after 3 s p50 44 ms max 135;
500 ms -> p50 52 max 60; 1000 ms -> p50 152 (no longer helps). The keep-alive stops `--gpu-keepalive` after
the last real call, so an idle sidecar costs nothing extra. It targets the GPU lane that served the last call and
counts as GPU work for the thermal state. On the WebGPU lane the same mechanism did not help (2026-09-21).

## 3. What concurrent small calls could gain: coalescing (not built)

`cuda:fp16`, same state, measured 10x: 8 sequential calls of 1 question **75 ms** vs one forward pass with the
same 8 rows **21 ms** -> **3.6x** for 8 concurrent single-question callers. The CUDA EP is launch-bound, so rows
are almost free. Requires a `systemMany` on the lane handle (collate rows from several requests, one
`session.run`, split) and router support (gather queued calls bound to the same lane); the vendored library's
`systemOne` takes one state per call, so the collate / split would be reimplemented against its sequence
builder. Worth it only if the traffic has several concurrent callers with small calls.

## 4. Decision guide

| situation | do this | expected |
|---|---|---|
| a script / agent step needs one decision now and then | `node scripts/laya.mjs ...` (or `ask.mjs --sidecar`) | 80-90 ms; first call after idle 2.5-4.5 s |
| a service / agent runtime makes many decisions | `node ask.mjs --start`, then `POST /decide` from a kept connection | 13-15 ms |
| several questions about the same state | one call with all questions | 3 q 12 ms, 10 q 24 ms - not 3 / 10 calls |
| calls seconds apart (interactive assistant) | keep the default `--gpu-keepalive 30s`; longer if the pauses are longer | p50 ~46 ms instead of ~174 |
| calls minutes apart | let it idle-exit (`--idle 5m`) or keep it (`--idle 0 --max-age 24h`, ~3.3 GiB RAM + 2 GiB VRAM held) | reload 2.5 s vs held memory |
| one isolated run, no background process wanted | `ask.mjs --local` | 2.5 s |
| RAM matters more than the fallback | `LAYA_LANES=cuda:fp16,webgpu:fp16` (drops the 1.6 GiB CPU lane) | same speed on a GPU machine |
| a hard latency budget | `deadlineMs` in the request; `routing.reason` tells what happened | - |

Which lane serves is the router's decision per call (`routing.lane`, `routing.reason`); with the CUDA lane
present it is CUDA for everything except a cold single question, where the CPU / WebGPU lanes tie with it.
