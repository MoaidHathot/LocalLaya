# CUDA process lane (2026-09-23)

`onnxruntime-node` has no CUDA EP on Windows. The Python wheel has: `onnxruntime-gpu` 1.30.0 (CUDA 13 build) from
Microsoft's release feed + CUDA 13.2 / cuDNN 9 runtime wheels from NVIDIA's index - both reachable from this
network, unlike `files.pythonhosted.org`. `tools/cuda_lane.py` holds one CUDA session and answers over stdio;
`src/lane.mjs` `ProcessLane` presents it behind the same handle as the worker lanes, with a `RemoteSession`
implementing the two methods `@receptron/laya` calls, so tokenising, temperatures and answer formatting stay
in the vendored library. Setup: `npm run cuda:setup` (pinned versions, `--no-deps`), check: `npm run cuda:check`.

## Speed (experiments/ab.mjs: interleaved, paired ratio to the fp32 WebGPU reference, bootstrap 95 % CI)

| workload | `webgpu:fp16` (optimised bundle) | `cuda:fp16` (same bundle, process lane) | cuda vs webgpu |
|---|---|---|---|
| PoC 1/3/10 q + smart-home eval (24 items x 5 q) | 0.429 [0.425, 0.433] | **0.131 [0.130, 0.132]** | **3.3x** |
| dev-request eval (40 items x 4 q) | 0.423 [0.419, 0.425] | **0.140 [0.138, 0.141]** | **3.0x** |

Per-call, single session, P-core pinned (`openLane` end to end, IPC + tokenising included):

| questions | webgpu:fp16 | cuda:fp16 | of which stdio round trip + tokenising |
|---|---|---|---|
| 1 (83 tokens) | 21.1 ms | **8.9 ms** | 0.7 ms |
| 3 (236 tokens) | 32.0 ms | **12.1 ms** | 0.9 ms |
| 10 (775 tokens) | 83.1 ms | **24.1 ms** | 1.8 ms |

The CUDA EP is launch-bound at ~9-12 ms for this 1753-node graph and almost flat in batch size, so the gain
grows with questions per call. With 2 CPU-side threads pinned to the P-cores; the unpinned 24-thread default
measured 20 ms for the same call (the E-core effect of finding 7 again).

Throughput (`experiments/throughput.mjs --lanes cuda:fp16`, 3 q): **66 calls/s** with one caller, **83-86 calls/s**
(≈ 260 questions/s) with 4-8 callers in the queue (12 ms per inference under load); 40 calls/s offered are served
at 13 ms p50; 60/s offered -> 59/s with a growing queue. 10 q: 41 calls/s (409 questions/s). 1 q: 96 calls/s.
Sidecar: 8 parallel 5-question calls in **202 ms** (676 ms on WebGPU); first call 2.5 s (the CUDA lane is ready
first, at ~2.4 s after spawn).

## Fidelity

| comparison | arg-max agreement | max abs dp | accuracy |
|---|---|---|---|
| cuda vs fp32 reference, smart-home (134 answers) | 134/134 | 0.0375 | 0.708 (= fp32) |
| cuda vs fp32 reference, dev-request (120 answers) | 119/120 (webgpu: 118/120) | 0.0416 | 0.800 (= fp32) |
| cuda vs webgpu:fp16, full smart-home eval (65 x 3 = 195 answers, `test/cuda-lane.test.mjs`) | 195/195 | 0.0348 | - |

## Behaviour under failure (`test/cuda-lane.test.mjs`)

- Python process killed mid-queue: in-flight calls fail over to the next lane, the lane is marked gone and
  never picked again; no leaked queue entries. (Killing a separate process is safe - unlike terminating a
  worker thread under a running inference.)
- `close()` ends the process; a wrong Python path (or no venv) makes the lane "unavailable" in the router
  log and the other lanes serve - the default lane list `cuda:fp16,webgpu:fp16,cpu:8` works on machines
  without the venv, minus the CUDA lane.

## Cost

RSS: Python process ~1.0 GiB; VRAM: +~830 MiB for the fp16 weights + ~400 MiB CUDA context. The three default
lanes together hold ~2 GiB VRAM (freed on idle exit). Disk: the venv grows by ~1.6 GiB of CUDA / cuDNN wheels.

## Not done / next

- CUDA Graph capture (`--cuda-graph` in `tools/cuda_lane.py`, `enable_cuda_graph`) would remove most of the
  ~9 ms launch floor but needs fixed input shapes: pad `[n, L]` to shape buckets and keep one capture per
  bucket. Not started.
- The same lane mechanism would carry a TensorRT EP session (`onnxruntime-gpu` ships it) - engine build time
  per shape makes it unattractive for variable-length inputs without the same bucketing.
