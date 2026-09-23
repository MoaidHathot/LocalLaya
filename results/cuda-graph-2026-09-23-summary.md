# CUDA Graph replay for the `cuda:fp16` lane (2026-09-23)

Machine: i9-14900KF (P-cores = logical 0-15), RTX 4070 12 GB, driver 596.49 (CUDA 13.2), Windows; Node 25.3;
`onnxruntime-gpu` 1.30.0 (CUDA 13 build) in `.venv` (Python 3.12). Bundle: `models/laya-onnx-fp16` (optimised
fp16 graph, `tools/optimize_graph.py`). All latencies are round trips through `openLane("cuda:fp16")` / the
router, i.e. they include the stdio hop and Node's tokenising (0.7-1.8 ms), hot GPU unless stated.

## Starting point

- Every CUDA call costs **6.0 ms + 21.6 us per token** (`experiments/cuda_graph_probe.py`): ORT launches ~1400
  kernels per forward pass, one by one, and the GPU idles between them. For a 1-question call that floor is ~80 %.
- `enable_cuda_graph` on the exported graph: **CUDA failure 700 (illegal memory access)** during the capture run,
  CUDA context dead afterwards (`experiments/cuda_capture_attempt.py`; one attempt per process, a failed capture
  kills the context for good).

## Cause and fix

1. **Bisect** (`experiments/cuda_capture_bisect.py`, prefixes of the graph, opt level disabled): the capture
   breaks at node #3, `GatherND node_GatherND_69` - the attention-mask broadcast with a constant index tensor.
   ORT 1.30's CUDA GatherND kernel copies a host-side vector to the device with `cudaMemcpyAsync` inside `Compute`
   (`onnxruntime/core/providers/cuda/tensor/gather_nd.cc:70-75`); that copy is recorded by the capture and replays
   from a stack buffer that no longer exists.
2. **Static graphs** (`tools/static_graph.py`): for a bucket (rows `n`, length `L`, options `K`) the input dims
   are fixed, every `Shape` output is observed in one run and baked in, constant subgraphs are folded through
   per-node ORT evaluation (`symbolic_shape_infer` fails on this export), and a `GatherND` with constant indices
   becomes a `Reshape` when it is numerically a pure re-indexing. 1753 -> 1517 nodes, 2.9 MB per bucket file.
   Capture then works at every optimisation level; `--check` vs the dynamic graph: max |dlogit| 0.0022.
3. **One copy of the weights**: `SessionOptions.add_initializer` refuses device buffers, so the external weights
   become graph inputs (`laya-static-*-w.onnx`, `laya-dynamic-w.onnx`) and one set of 202 CUDA `OrtValue`s
   (804 MiB) is bound to every session. A bucket session costs its activations only (+28-240 MiB).
4. **Replay inputs** are device `OrtValue`s updated in place (a CPU-bound input is never re-read by a replay),
   followed by `cudaDeviceSynchronize` (ctypes on the bundled `cudart64_13.dll`): `update_inplace` is a
   synchronous `cudaMemcpy` from pageable memory whose DMA may still be in flight when it returns.
5. **Threads** - two ORT 1.30 facts that shaped `tools/cuda_lane.py`:
   - the captured graph lives in the *calling thread's* `PerThreadContext` ("Cuda graph with multi threads will
     be supported in the future"), and a session captures on its **third** run on that thread
     (`min_num_runs_before_cuda_graph_capture_ = 2`). A graph captured on a helper thread is invisible to the
     thread that serves requests.
   - the capture uses `cudaStreamCaptureModeGlobal`: any "unsafe" CUDA call from another thread meanwhile
     (`cudaMalloc`, synchronous `cudaMemcpy`, `cudaStreamSynchronize`) fails with **CUDA error 900** and
     invalidates the capture. Measured with a naive background builder: 7 of 400 serving calls failed, 3 of 4
     builds failed.

   So the builder thread only *prepares* a bucket (static file, `InferenceSession`, IO binding, device buffers)
   under a lock, and the serving thread *captures* it under the same lock: two regular runs, the capturing run,
   one replay, then the replay is checked against the dynamic session on the last real inputs seen for the shape
   (`FINALISE_MAX_DELTA` 0.5 = "broken capture"; real text lands at 0.002-0.015). That happens in idle gaps of the
   request loop (100 ms) or is forced onto the first matching request 0.5 s after the bucket was prepared.
6. **The GIL**: ORT holds the interpreter lock for the whole `InferenceSession` construction (317 of 340 ms,
   `experiments/cuda_gil_probe.py`), which freezes the serving thread for that long. Mitigations, in the order
   they were measured (see "Start-up, memory and stalls" below for the second round): the builder waits for a
   100 ms gap in the requests before such a step (1 s for the first one, at most 1 s once requests flow), lazy
   buckets are only scheduled for shapes seen twice, the router's start-up probe / warm-up / keep-alive run with
   `exec: { graph: false }` so they neither build buckets nor measure a disturbed lane, sessions are created
   from a pre-optimised file (~100 ms instead of ~250), the capture runs in one-run steps, and the wait a
   request suffered is reported as `stallMs` and excluded from the router's estimates.

Grid: `L` in 64..512 (10 steps), `n` in 1..16 (10 steps), `K` in 8 / 16 / 32; shapes with `n x L > 1536`
(`--graph-max-work`) stay dynamic - there the launch overhead is a few percent of the call. Inputs are padded to
the bucket (`attention_mask` / `marker_mask` 0 for the padding, missing rows are copies of row 0), outputs sliced
back. LRU eviction at `--graph-max-sessions 16` / `--graph-max-vram-mib 1536`.

## Results

### Paired A/B, dynamic graph as baseline (`experiments/ab.mjs`, 8 rounds x 43 items)

`node experiments/ab.mjs --variant cudadyn="cuda:fp16?graph=false" --variant cuda=cuda:fp16 --rounds 8 --workload both --preset dev-request --max-items 40`

| variant | poc 1 q p50/p90 | poc 3 q | poc 10 q | dev-request (3 q, ~200 tokens) | all p50 | paired ratio [95 % CI] | arg-max agree | max abs dp | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| `cuda:fp16` dynamic (`exec: { graph: false }`) | 10.8 / 20.6 | 12.4 / 22.0 | 24.5 / 26.6 | 18.8 / 20.2 | 18.8 | baseline | - | - | 0.800 |
| `cuda:fp16` CUDA Graph replay (default) | **5.0 / 5.8** | **8.6 / 10.2** | 22.9 / 24.3 | **16.0 / 16.9** | 16.0 | **0.858 [0.848, 0.862]** (-14.2 %) | 134/134 | 0.0063 | 0.800 |

All 344 replay-variant calls ran on a graph (7 buckets: the 4 eager ones + `3x192x8`, `10x96x8`, `3x256x8` built
from the warm-up, 728 MiB). The paired ratio is dominated by the 40 preset items (~200 tokens x 3 rows, -15 %);
the small shapes gain 2x (1 q) and 1.4x (3 q). The p90 of the dynamic path (20-22 ms on 1-3 q) is the
launch-bound path's sensitivity to whatever else the CPU does; the replay's p90 sits 1-2 ms above its p50.

### Against the fp32 reference (6 rounds, PoC shapes)

| variant | 1 q | 3 q | 10 q | ratio to fp32 [95 % CI] | arg-max agree | max abs dp |
|---|---|---|---|---|---|---|
| `webgpu` fp32 | 36.8 | 59.4 | 156.6 | baseline | - | - |
| `webgpu:fp16` | 22.7 | 31.2 | 84.0 | 0.542 [0.519, 0.592] | 14/14 | 0.0322 |
| `cuda:fp16` dynamic | 9.8 | 12.4 | 24.6 | 0.213 [0.161, 0.266] | 14/14 | 0.0031 |
| `cuda:fp16` graph | **5.2** | **9.5** | **23.1** | **0.150 [0.142, 0.155]** | 14/14 | 0.0014 |

### Fidelity on real text (`test/cuda-lane.test.mjs`, 65 smart-home states x 3 questions)

Replay vs the dynamic graph on the same inputs: **195/195 arg-max agreement, max |dp| 0.0055**; replay vs the
WebGPU lane: 195/195, max |dp| 0.036 (unchanged from the dynamic CUDA lane). The per-bucket check at capture
time on real inputs: 0.002-0.015. On random-token soup the static graph differs from the dynamic one by up to
0.12 logits (fp16 accumulation order differs between the shape-specialised and the generic kernels), which is
why the self-test threshold is 0.15 and the runtime "broken capture" threshold 0.5.

### Throughput (`experiments/throughput.mjs --lanes cuda:fp16 --scenarios A --quick`, steady state)

| questions per call | before (dynamic) | now (graph) |
|---|---|---|
| 1 | 96 calls/s (9 ms) | **194 calls/s (5 ms)** |
| 3 | 66 calls/s one caller, 83-86 queued (12 ms) | **114-115 calls/s (8-9 ms)**, same with 2-8 callers |
| 10 | 41 calls/s = 409 q/s (24 ms) | **44.7 calls/s = 447 q/s (22 ms)** |

Parallel callers no longer add throughput on CUDA: the ~30 % the process gained from pipelining the next
request behind a 12 ms inference is gone now that the inference is 8 ms and the CPU side (tokenising, stdio,
padding) is the larger share.

### Sporadic traffic (`experiments/sporadic.mjs --ep cuda --fp16`, 3 q)

Gap 0 ms: 8.7 ms (was 12); 250 ms: 10.5; 1000 ms: 38.8 (11-78); 3000 ms: 124 (51-166). The cold side is the
GPU's clock management, as before; the keep-alive in `serve.mjs` covers it.

### Sidecar 8-call burst right after start (`test/sidecar.test.mjs`)

199 ms (202 before): a burst is the *first* sighting of its shape, so it runs dynamic; the second burst of the
same shape replays. That is the lazy policy's price: a shape must be seen twice and the lane needs a 100 ms gap
(or 1 s of continuous traffic) plus ~0.35 s to prepare and capture.

### Self-test (`tools/cuda_lane.py --self-test`, random tokens, padding included)

| call shape | bucket | prepare + capture | VRAM | max abs delta vs dynamic | graph p50 | dynamic p50 |
|---|---|---|---|---|---|---|
| 1 x 40 x 3 | 1x64x8 | 225 + 207 ms | +54 MiB | 0.0027 | **3.8 ms** | 7.8 |
| 1 x 85 x 4 | 1x96x8 | 211 + 57 | +12 | 0.0029 | **4.1** | 7.8 |
| 3 x 85 x 4 | 3x96x8 | 215 + 74 | +40 | 0.0159 | **7.6** | 10.9 |
| 3 x 96 x 8 | 3x96x8 | 220 + 71 | +34 | 0.0195 | **8.0** | 11.8 |
| 5 x 90 x 6 | 5x96x8 | 287 + 101 | +68 | 0.0515 | **11.9** | 15.1 |
| 5 x 100 x 5 | 5x128x8 | 287 + 114 | +94 | 0.1162 | 14.4 | 16.2 |
| 10 x 95 x 6 | 10x96x8 | 290 + 159 | +134 | 0.0291 | 21.0 | 25.4 |
| 2 x 250 x 4 | 2x256x8 | 305 + 109 | +82 | 0.1221 | 13.2 | 16.3 |
| 4 x 300 x 8 | 4x320x8 | 293 + 236 | +240 | 0.0059 | 32.6 | 33.0 |
| 6 x 130 x 12 | 6x160x16 | 285 + 169 | +146 | 0.0537 | 22.5 | 22.5 |

(in-process timings, no stdio hop). Above ~1300 units of `n x L` the gain is within noise: `--graph-max-work`.

## Start-up, memory and stalls (second round, same evening)

The first integration built the four default buckets *before* the ready line (+1.4 s on the lane's join time;
the sidecar was ready on WebGPU at ~2.6 s and CUDA joined at 3.4 s), and the defaults were the PoC shapes - the
default `smart-home` preset through the sidecar lands on `5x160x8`, which nothing pre-built. Changes, each
measured against the previous state:

- **Eager builds after ready, in idle gaps.** The lane reports ready after the dynamic session plus one warm-up
  run (1.1-1.3 s alone, 1.7-2.1 s while the two other lanes load) and schedules its eager set. Sidecar: ready
  with the CUDA lane at **2.1-2.4 s** after spawn (was ~2.6 s on WebGPU), first call answered at 2.2-2.5 s on
  CUDA (was 2.8-3.1 s on WebGPU incl. its first-shape shader compile). `test/sidecar.test.mjs`: first call
  3092 -> 2212-2490 ms.
- **Memory of real traffic** (`--graph-buckets-file`, `serve.mjs` default `.laya/cuda-buckets-<port>.json`):
  buckets that were hit or built from traffic are written at capture / first hit / close with their hit counts;
  the next start builds the top 6 by hits instead of the defaults. Second sidecar start on the default preset:
  the `5x160x8` bucket is captured in the first idle second and every later call replays (23 ms vs 27-32 dynamic
  under the start-up CPU contention). Preset-agnostic: whatever the callers use gets remembered.
- **The first build waits for a full idle second** (`PREPARE_FIRST_IDLE_S`), later ones 100 ms, forced after 1 s
  of continuous requests: the request that made a sidecar start arrives ~40 ms after ready and must not hit a
  250 ms GIL freeze. Measured: first four calls of a fresh sidecar 26-39 ms round trip (5 q, dynamic, other
  lanes still loading), no stall.
- **Pre-optimised session files** (`experiments/cuda_session_create_probe.py`): default optimisation 250-257 ms
  per session creation; loading the graph ORT saved via `optimized_model_filepath` with `ORT_DISABLE_ALL` 114-120
  ms (0.2 MB per file, per ORT version). The lane writes the file on a bucket's first creation; `npm run
  cuda:setup` (`--graphs`, `cuda_lane.py --prepare-buckets`) pre-generates the defaults'. Prepare step 225-300 ->
  97-140 ms, same fidelity (self-test max |delta| unchanged).
- **Capture in one-run steps**: the serving thread does one `run_with_iobinding` (10-60 ms; the first run of a
  new session ~90 ms) per idle tick or per incoming request instead of a 55-230 ms block; the check against the
  dynamic session is the fifth step.
- **Static-graph generation in a subprocess** (`static_graph.py`, low priority, on the CPUs the lane is not
  pinned to = the E-cores here, probe session 8 threads): in-process the 2-3 s of Python/ONNX work plus the
  probe made every concurrent dynamic call 2-3x slower (p50 18.7, p90 39.6, 37 calls > 40 ms of 594 in 12 s);
  as a subprocess with inherited affinity 40-50 ms calls remained; with the affinity/priority change **p50
  17.5, p90 18.6, max 25.7 ms** (stalls excluded) during a 5.6 s generation. One-time per shape per machine.
- **`stallMs`**: the process reports how long a request waited for something that was not its inference -
  queue time behind a capture step, capture steps run on its behalf, and any builder GIL hold overlapping it
  (the hold is recorded *before* the call starts: the serving thread can run before the `finally`). Measured
  under load: a 307 ms call reported 300.6 (the session creation of a new bucket without a pre-optimised file),
  a 95 ms call 90.5 (the first capture step). The router observes `ms - stallMs` for its EMA; before that, one
  stalled call taught the EMA 300+ ms and the late binding moved the rest of an 8-call burst to WebGPU (1527 ms
  for the burst, two lanes; now 200-300 ms on one lane).
- **Exit without interpreter teardown** (`os._exit` after the memory file is written; every file write is
  atomic): `--stop` left the process alive while the builder was mid-generation; destroying sessions under a
  thread inside ORT hung the exit.
- **Router bug found on the way**: with `sampleLoad: false` the one-shot `nvidia-smi` sample at `create()`
  (which caught the previous test's GPU work) inflated the CUDA lane's predictions 2.2x for the router's whole
  life - every burst went to `cpu:8`, intermittently (1 in ~4 runs). Load samples now count only while fresh
  (< 10 s, `LOAD_STALE_MS`), and with sampling off the start-up utilisation is not ingested at all. Unit test added.

Kept-connection round trips after this round (sidecar, all lanes loaded, buckets ready): **1 q 6.6 ms, 3 q
10.8, 10 q 25.1; default preset (5 q, ~150 tokens) 23.3** (p50 of 30; was 13-15 ms for 3 q on the dynamic graph).

## Costs

- Start-up: the lane is ready in 1.1-1.3 s (dynamic session + one warm-up run); buckets follow in idle gaps,
  ~0.25 s each with the files cached, plus a 5-8 s subprocess for a shape that has no static file yet
  (`npm run cuda:setup` generates the defaults').
- VRAM: 804 MiB weights (shared) + 28-240 MiB per bucket; 5 buckets 464 MiB, 7 buckets 728 MiB
  (`--graph-max-vram-mib 1536`).
- A bucket built under continuous traffic: one call stalls ~100 ms (session creation holds the GIL; ~250 without
  the pre-optimised file), 4-5 calls each carry one capture step of 10-60 ms; all reported as `stallMs`. Under
  sporadic traffic everything lands in gaps.
- Per-call: padding + in-place device update + sync ~0.1-0.3 ms.

## Files

`tools/cuda_lane.py` (lane: dynamic session + buckets, `--graph auto|off`, `--graph-buckets`,
`--graph-buckets-file`, `--graph-max-*`, `--self-test`, `--prepare-buckets`, ops `stats` / `bucket`, per-call
`exec: { graph: false }`, `stallMs`), `tools/static_graph.py` (static bucket graphs, GatherND -> Reshape, weights
as inputs, `--check`, atomic writes), `tools/setup-cuda-lane.mjs --graphs`,
`src/lane.mjs` (`ProcessLane.systemOne(state, questions, temps, exec)`, `stats()`, `bucketFor()`, `graph`),
`src/ep-router.mjs` (`routing.exec`, `detailedStats()`, `cudaGraph` / `graphBuckets` / `graphMax*` options,
warm-up with `exec: { graph: false }`), `serve.mjs --cuda-graph off` (`LAYA_CUDA_GRAPH`), `ask.mjs --no-graph`,
`demo/cli.mjs --no-graph`, `experiments/ab.mjs` (`cuda:fp16?graph=false`, settle step), `experiments/throughput.mjs`
(settle step), probes `experiments/cuda_static_probe.py`, `cuda_capture_attempt.py`, `cuda_capture_bisect.py`,
`cuda_shared_weights_probe.py`, `cuda_input_race_probe.py`, `cuda_gil_probe.py`, `cuda_session_create_probe.py`;
tests `test/cuda-lane.test.mjs` (6 scenarios), `test/unit.test.mjs` (stale load samples). Summaries: `results/cuda-graph-vs-dynamic-dev-request-summary.md`, `results/cuda-graph-vs-fp32-poc-summary.md`,
`results/throughput-cuda-graph-3q-summary.md`, `results/throughput-cuda-graph-10q-summary.md` (raw JSON next to them, git-ignored).
