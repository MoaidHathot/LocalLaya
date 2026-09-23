# Session log - 2026-09-23: throughput, the one-thread truth, the queue fix, worker lanes

Chronological record. The living status is `docs/STATUS.md`; this file is the "what happened and why" trail.

## Starting point

User question: "what about performance? in the different modes, how many prompts/responses can we do in a
second or in a minute?" Answered from the existing p50 tables (throughput = 1 / latency per lane), flagged
one suspicious log (`.laya/sidecar-8797.log`: GPU 5-q calls at 155 ms became ~950 ms while `cpu:8` ran
alongside during an 8-caller burst), and offered a throughput experiment plus a router tweak if the
suspicion held. User: "sure".

## Phase 1 - the throughput matrix (`experiments/throughput.mjs`)

Scenarios through `LayaRouter.decide()` with the default lanes `webgpu:fp16,cpu:8`, 3 questions:
A) one lane forced, closed loop k = 1/2/4/8 callers; B) policy `auto` vs `prefer-gpu`, same k; C) one lane
sequential while the other runs back-to-back; D) open loop at 0.33-20 calls/s, `auto` vs each lane forced.
Raw JSON in `results/throughput-*.json` (ignored), summaries in `results/throughput-*-summary.md`.

"Before" run (`results/throughput-before-3q-summary.md`):

- `webgpu:fp16` 20.5-21 calls/s at every k (queueing only); `cpu:8` 3.8 calls/s.
- `auto` collapsed under concurrency: 20.5 (k=1) -> 19.0 -> **10.2 (k=4, p95 879 ms) -> 6.6 (k=8, p95 1.5 s)**
  because it spilled queued calls to `cpu:8`; `prefer-gpu` stayed at 19.7-20.0.
- Open loop: `auto` saturated at 4.5-5 calls/s when offered 10-20/s; GPU-only sustained 10/s.
- C: GPU 49 ms alone -> **321 ms p50 / 464 p95 while cpu:8 ran** (6.5x); the reverse 1.3x.

## Phase 2 - why (`experiments/interference.mjs`, `experiments/worker-lanes.mjs`)

- Same-thread CPU load: GPU 78 -> 470 ms. CPU load in a **child process**: 82 ms (no effect). The GPU
  session's own CPU-side thread count (default / 1 / 2) changes nothing. So it is in-process, not the cores.
- Cause: `onnxruntime-node` 1.30 `dist/backend.js` wraps a **synchronous** `session.run()` in
  `setImmediate` + Promise. Every inference blocks the JS thread for its full duration; lanes never overlap;
  a CPU call queued into a GPU burst stalls everything behind it; `routing.ms` included that stall, which
  polluted the GPU lane's EMA (735 ms after a mixed burst) and made the router avoid the GPU afterwards.
- Worker threads: both EPs load fine in `node:worker_threads`, main-thread stall 300+ ms -> 20 ms, clean exit.
  But mixing still loses: GPU 49 -> 76 ms while the 8 pinned CPU threads spin; burst 20.6 calls/s GPU-only,
  18.5 with every 7th call on CPU, 16 with every 4th, 4.0 CPU-only. Conclusion independent of threading:
  **under load use the fastest lane for everything; the CPU lane is for sporadic small calls on a cold GPU
  and for GPU-less machines.**

## Phase 3 - router fix (`src/ep-router.mjs`)

- One FIFO (`this.queue`) for all lanes instead of a chain per lane; `this.inflight` tracks queued/running
  calls with their own predicted ms. `predictions()` returns `wait + own` per lane; the wait is shared, the
  GPU state is the one expected at start (`hot` if GPU work is queued ahead; never `cold` while anything is
  queued - a queue means traffic and the first GPU call warms the clocks). Exploration only while idle.
  `chooseLane` compares own latencies for the 2x exploration window. `routing` gains `ownMs`, `waitMs`;
  `stats()` gains `queue`. `ms` is now the inference alone (nothing else runs between start and end).
- Unit test rewritten for the model (shared wait, no spill at any depth, remaining time of the running call,
  state at start, cold + idle -> CPU for 1 q, inflation on own only). Sidecar test "both lanes used" ->
  "one queue, no spill; GPU calls not stalled".
- "After" run: `auto` = `prefer-gpu` at every k (19-20 calls/s, p95 428 ms at k=8 vs 1515 before); open loop
  `auto` sustains 10/s and 18-19/s where it saturated at 4.5-5. Sidecar 8-caller burst 2576 -> 1381 ms.

## Phase 4 - the slow-mode mystery and process affinity

- One full "after" run had the GPU at 78 ms and `cpu:8` at ~350 ms for the whole 5 minutes (12.5 calls/s);
  the next run 51-54 ms. Direct tests: idle CPU session present or not - no effect; ORT `allow_spinning` -
  no effect; background sampling - no effect; cold start - no effect. Forcing the process affinity: **E-cores
  only -> 71-80 ms, P-cores -> 49 ms**, i.e. the slow mode is the JS thread (tokeniser, WebGPU dispatch, one of
  ORT's intra-op workers) parked on an E-core by Windows.
- Fix: `pinProcessToPCores()` in `src/laya-client.mjs` (PowerShell sets the process mask, ~0.4 s, in the
  background during the model load), on by default in `LayaRouter.create({ pinProcess })`. Also
  `buildSessionConfig` now accepts `threads` / `affinity` for the WebGPU session's CPU-side pool.
- Also measured: GPU latency vs gap between calls: 48 ms back-to-back, 58-66 with 50 ms gaps, 54-81 with
  100 ms, 75-96 with 200 ms - the clock governor reacts within tenths of a second.

## Phase 5 - commit, then the next STATUS items: worker lanes, early serving, late binding, max-age, CI

User: "commit, and go ahead with the next items in docs/status.md; add tests, verify everything, fix bugs."
Items 1-3 (real labels, fine-tuning, multilingual) need data or hours of GPU; 7 (DirectML retest) was
skipped with a wrong reason ("npm cannot fetch a newer onnxruntime-node" - see Phase 6: there is nothing
newer, and it is fetchable). Done: 4 (first-lane serving, `--max-age`), 5 (worker lanes), 6 (`cpu:8`
default), 9 (GitHub Actions, `bench-all` defaults).

- **Worker lanes** (`src/lane.mjs`, `src/lane-worker.mjs`): one handle for in-thread and worker sessions
  (`systemOne(state, questions, temps)`, `close()`, `onDeath`); the router talks to `L.session`, never to
  `Laya` directly. Workers unref'd while idle. `test/router.test.mjs`: identical answers, main-thread stall
  1.1 s -> 18 ms, round trip 306 vs 294 ms (noise), two lanes ready in 2.2 s instead of 3.3-3.6.
- **Bug found by the first death test**: `worker.terminate()` while ORT runs on that thread kills the whole
  process (`0xC0000409`). Consequences: `close()` asks the worker to release and exit itself and only
  terminates an idle worker; the worker handles messages in order so `close` never releases a session under a
  run; `router.close()` drains the FIFO first. The tests inject failures instead (a rejecting `systemOne`;
  terminating an *idle* worker) and cover retry, quarantine, dead-lane exclusion and forced-lane errors.
- **Early serving** (`waitFor: "first"`, `warmup`, `onLaneReady`, `router.ready`, `pendingLanes`;
  `/health.lanesLoading`, 503 + `retryAfterMs` for a forced lane still loading, `waitAllLanes()` in the
  client, `--start` waits for every lane). First call 5.0-6.3 s -> 3.1-4.5 s; ready at 2.6-2.9 s, both lanes at 3.5-3.8 s.
- **Bug found by the sidecar suite**: an 8-call burst right after start ran on `cpu:8` for 7.6 s while the
  GPU lane joined after ~1 s and sat idle - lanes were bound at enqueue time. Fix 1: **late binding** - the
  lane is chosen when the call reaches the front of the queue (`_enqueue(..., rebind)`, provisional lane only
  for the wait estimates; exploration moved there too). Still 7.6 s: the GPU lane's probe + warm-up were
  *in the FIFO behind the burst*. Fix 2: probes and warm-ups run outside the FIFO (`_direct`). Result: 6 x 10 q
  during start-up 1.8 s instead of 5.7 s, 5 of 6 calls re-bound cpu:8 -> webgpu:fp16 (`routing.provisionalLane`).
  CPU lanes now warm with one call per size (nothing to compile).
- `--max-age` (`LAYA_MAX_AGE`) recycling; `health.lanes` in configured order; `--in-process` escape hatch for
  `serve.mjs`, `router-demo.mjs`, `experiments/throughput.mjs`.
- Test hygiene bugs: a stall ticker cannot observe a stall that ends before its next tick (measure the gap
  after the await too); a noul question with P ~ 0 cannot be "sharpened" (use `should_execute` ~0.9);
  early-serving assertions must tolerate the second lane joining during the call; every sidecar test now
  stops the sidecar in `finally` so one failure does not cascade.
- **CI**: `.github/workflows/unit-tests.yml` (Ubuntu/Windows x Node 20/22, `ONNXRUNTIME_NODE_INSTALL=skip`).
  Found on the way: `package-lock.json` resolved URLs pointed at the Microsoft feed proxy, which GitHub cannot
  reach - rewritten to `registry.npmjs.org` (npm's `replace-registry-host` keeps it working locally). The
  workflow itself is unverified: no runner here and no `act`.
- Verification: `npm test` 9/9, `npm run test:router` 7/7 (~35 s), `npm run test:sidecar` 13/13 (~65 s),
  throughput quick run in worker mode 19.8 calls/s (same as in-process), `npm run router` scenarios unchanged.
- Not done from item 4: named pipes, Windows service (no need shown); `skills-ref validate` (tool absent).

## Phase 6 - "what exactly can't be fetched, and would having it make it faster?"

User challenged the "npm cannot fetch" claim. Checked instead of repeating it:

- Blocked: `registry.npmjs.org` (TLS handshake fails). The proxy `packagefeedproxy.microsoft.io/npm/` returns
  404 for **`@receptron/laya` only** - that is the whole reason for `vendor/`. It serves `onnxruntime-node`
  completely: 181 versions, `latest` 1.30.0 (= installed, published 2026-09-14), nightly
  `1.31.0-dev.20260918`. GitHub (incl. release zips), jsDelivr, unpkg, Hugging Face, PyPI all reachable.
- So nothing fetchable would make it faster: `@receptron/laya` from the registry is the same bytes, no newer
  `onnxruntime-node` exists, and the CUDA EP is missing from the Windows build of `onnxruntime-node`
  (README matrix, `install-metadata.js`, no CUDA symbols in the binary) - not from any download.
- Where the time really goes (P-core pinned, `webgpu:fp16`): 99 % inside `session.run` (JS side 0.3-1.9 ms);
  27.8 / 50.9 / 139.8 ms for 83 / 236 / 775 tokens = **~14 ms fixed + 0.16 ms/token**, i.e. 5-10 % of the
  RTX 4070's fp16 tensor throughput. The fp16 graph has **2101 nodes** (286 Slice, 262 Mul, 182 MatMul,
  158 Transpose, 32 Softmax; only LayerNormalization is fused) -> dispatch-bound, which also explains why
  fp16 gained only 5-13 %.
- Levers checked against the installed 1.30.0 binary + binding source (`js/node/src/session_options_helper.cc`):
  WebGPU graph capture exists in the DLL but the Node binding rejects the option (lines 104-108); the binding
  forwards `preferredLayout`, `validationMode`, `*BufferCacheMode`, `enableRobustness`, `forceCpuNodeNames`.
  DML failure matches ORT #27118 (closed stale). CUDA is reachable via Python `onnxruntime-gpu` wheels (PyPI,
  ~1.5 GiB with the CUDA/cuDNN wheels) as a process lane.
- Plan agreed: (1) this correction; (2) cheap sweep of forwardable WebGPU options + CPU-fallback audit +
  nightly DML test; (3) offline transformer fusion -> fused fp16 bundle (stop and report if ORT's patterns do
  not match ModernBERT); (4) CUDA process lane behind the lane handle, default if it measures faster.
  Constraints from the user: keep everything preset-agnostic (more presets coming), and make sure gains are
  real (interleaved A/B in one session, not single runs).

## Phase 7 - the sweep that found nothing, the profile that found everything

User: "go ahead; keep it generic (more presets coming); make sure the gains are real."

- **Measurement first**: `experiments/ab.mjs` - N variants in one process, interleaved rounds, paired ratio
  with bootstrap CI, fidelity vs a reference on any preset's eval set (`--preset`, `--inputs`). Noise floor
  measured: 1.000 [0.997, 1.001], answers bit-identical. Everything below is from this tool.
- **WebGPU EP options** (8 variants incl. all combined): every ratio within +-0.5 %, every CI covers 1.0.
  Nothing adopted. Real result: the binding-forwardable knobs are not a lever for this graph.
- **CPU-fallback audit**: the 90 CPU nodes are Shape->Slice->Concat plumbing; ORT's runtime fusions all report
  `modified: 0`. **DML nightly**: same `node_view` failure; the node is `Reshape(allowzero=1)` of the QKV view.
- **ORT transformer optimizer** on the fp32 export, every model type: only Gelu (29) fuses. Attention, RoPE,
  SkipLayerNorm: no match (torch.export structure). Stopped there, as agreed.
- **ORT profiler** (the step that mattered): 28 `IsNaN` nodes (sdpa NaN guard per layer) run on the CPU because
  the WebGPU EP has no IsNaN kernel -> 30 `MemcpyToHost` + 36 `MemcpyFromHost` per call; 196 `Cast` nodes from
  the fp32 islands (LayerNorm/Softmax) of the first converter. Neither is "fusion"; both are exact, mechanical.
- `tools/optimize_graph.py` (replaces `convert_fp16.py`): `IsNaN(x)` -> `Not(Equal(x,x))`, Reshape `allowzero`
  cleared, Gelu fusion, fp16 without islands -> 1753 nodes, 2 `MemcpyToHost` per call. Two bugs on the way:
  `optimize_by_fusion` takes a `ModelProto`, and the fused `com.microsoft` ops need the opset import added by
  hand before shape inference.
- **Results** (paired, vs fp32 on WebGPU): optB 0.453 [0.448, 0.457] vs previous fp16 0.846 on PoC+smart-home
  (**1.87x**); 0.430 vs 0.916 on dev-request (**2.13x**). Fidelity identical to the previous bundle (134/134;
  118/120 with the same two near-tie flips), accuracy = fp32 on both presets. Adopted as `models/laya-onnx-fp16`
  (previous kept as `-v1`). Standard bench 21.1 / 32.0 / 83.1 ms; throughput 30-31 calls/s (3 q); sidecar
  8-call burst 676 ms; cold 1 q 101 ms (was 180) - the CPU lane's cold-GPU niche is gone.
- **DML works** with `allowzero=0`: 18-19 ms for 1 q (fastest EP), but 217 / 264 ms for 3 / 10 q even on a fixed
  shape - not a shape-cache effect (tested). Priors updated, not a default lane.
- **CPU lane**: the same clean-up in fp32 gives 1.03 [0.95, 1.06] -> keep the pinned HF bundle.
- Unit tests that hard-coded prior values were rewritten against `DEFAULT_PRIORS`; router + sidecar suites pass.

Numbers worth remembering (updated): `webgpu:fp16` 3 q 32 ms / 30 calls/s / 92 questions/s; 1 q 21 ms; 10 q
83 ms; the two fp16 arg-max flips on dev-request are 0.426 vs 0.419 and 0.194 vs 0.192.

## Phase 8 - the CUDA process lane

- **The real fetch limit, stated precisely this time**: `pypi.org` answers but `files.pythonhosted.org` (every
  PyPI wheel) and `api.nuget.org` are TLS-blocked. Reachable: Microsoft's ORT release feeds on
  `aiinfra.pkgs.visualstudio.com` (`onnxruntime-cuda-13` has `onnxruntime-gpu` 1.30.0 cp312/win_amd64 built for
  CUDA 13; `onnxruntime-cuda-12` only dev builds; `ORT-Nightly` 1.31 dev), NVIDIA's `pypi.nvidia.com` (CUDA 13.x
  runtime wheels un-suffixed: `nvidia-cublas` 13.x etc., plus `nvidia-cudnn-cu13`). Installed with `--no-deps`
  (Python deps already present): CUDA 13.2 libs matching the driver's "CUDA 13.2", cuDNN 9.14.
- First CUDA session (Python, random tokens, default 24 threads): 20 ms flat for 1-10 questions. In the lane
  (2 threads, P-core affinity): **8.9 / 12.1 / 24.1 ms** for 1 / 3 / 10 q - the E-core effect once more.
- `tools/cuda_lane.py` (stdio NDJSON, base64 tensors, ordered) + `ProcessLane` / `RemoteSession` in
  `src/lane.mjs`: a stock `Laya` is built on the remote session (`new Laya(session, tok, config, ids, dir)` - the
  constructor is public), so nothing of the sequence logic is ported. `openLane` dispatches `PROCESS_EPS`.
- Pitfalls: `preload_dlls(verbose=...)` does not exist in 1.30 and the failed call left the CUDA DLLs off the
  path ("cublasLt64_13.dll missing"); Python's stderr came out UTF-16 (`PYTHONUTF8=1`, `PYTHONIOENCODING`);
  ORT 1.30 prints a "No registered plugin EP device" notice (`set_default_logger_severity`); `--input-type`
  piped scripts break worker threads (test artefact, not a bug).
- **A/B** (vs fp32 WebGPU): cuda **0.131 [0.130, 0.132]** vs webgpu 0.429 (PoC + smart-home, 3.3x); 0.140 vs
  0.423 (dev-request, 3.0x). Fidelity 134/134 and 119/120 vs fp32, 195/195 vs WebGPU over the full eval set,
  accuracy = fp32. Throughput 66 calls/s (3 q), 83-86 with a queue, 409 questions/s at 10 q.
- Router: `cuda` in `isGpuLane`, priors, `python` / `deviceId` options; defaults `cuda:fp16,webgpu:fp16,cpu:8`
  everywhere (serve, ask, router-demo, throughput); `ask.mjs` one-shot uses `waitFor: "first"` and no load
  sampling. `tools/setup-cuda-lane.mjs` (`npm run cuda:setup` / `cuda:check`) pins the versions and verifies a
  CUDA session. Sidecar tests derive the expected lane list from a CUDA availability probe.
- `test/cuda-lane.test.mjs` (3 scenarios, skipped without the venv): fidelity + temps override + speed sanity;
  hot burst on cuda, `process.kill(pid)` mid-queue -> queued calls fail over to `cpu:8`, lane gone, no leaks;
  `close()` ends the process, bad python path -> "unavailable" in the router log, other lanes serve.
- Sidecar: first call 2.5 s, 8 parallel 5-q calls **202 ms** (day started at 2576 ms), idle exit frees ~2 GB VRAM.

Day total for the GPU lane (3 questions, 8 parallel callers): 2576 ms -> 1381 (queue fix) -> 676 (graph) ->
202 ms (CUDA); single call 47 ms -> 12 ms; throughput 19-21 -> 66-86 calls/s. Every step measured
interleaved against the previous one and against the fp32 reference for fidelity.

## Phase 9 - "update the skills; what is the most optimised sidecar setup, and when to call what?"

- Measured the call paths first (results/sidecar-modes-2026-09-23-summary.md): persistent HTTP client 13-15 ms;
  PowerShell `Invoke-RestMethod` 31; `ask.mjs --sidecar` 87-127; the skill wrapper spawned a *second* Node
  process: 130-165 ms for a 12 ms inference. Bare Node start-up is 37 ms, so a shell call cannot go below ~70.
- Wrapper fast path: `scripts/laya.mjs` now imports the project's `sidecar-client.mjs` and answers one-shot calls
  itself (80-90 ms); lifecycle / `--local` / `--pretty` / REPL / sidecar failures are delegated to `ask.mjs`.
  Found on the way: SKILL.md's inline `--questions '{...}'` example never worked (`ask.mjs` read it as a file
  path) - fixed in `ask.mjs` (inline JSON or file for `--state` and `--questions`) and covered by a test.
- Sporadic traffic per lane (`sporadic.mjs` gained `--ep cuda` via `openLane` and a `--router` mode): the CUDA
  lane is **bimodal** after a pause (3 q after 3 s: ~50 or ~200-300 ms, p50 68-217), worse in its slow mode than
  WebGPU (156) and the CPU (~270 flat); for a cold single question CPU (~105) / WebGPU (~101) beat CUDA's median.
  `DEFAULT_PRIORS.cuda` cold rows were assumed too optimistic (105/115) -> set to the measured medians.
- GPU keep-alive: a first router-level comparison (n = 8) said 217 -> 61 ms; a first sidecar comparison (n = 6)
  said no effect; `/stats` proved the keep-alive was running. Settled with n = 20 per arm, arms alternated,
  through the sidecar: **p50 174 -> 46 ms, slow calls 12/20 -> 4/20**. Power 26 -> 27-30 W while active
  (SM clock 345 -> 555-690 MHz; the driver stays awake, the clocks stay low). Interval 500 ms (250: max 135,
  500: max 60, 1000: no effect). Default in `serve.mjs`: `--gpu-keepalive 30s` (`LAYA_GPU_KEEPALIVE`), REPL
  the same; the keep-alive targets the GPU lane that served last (was: first GPU lane in map order).
  Lesson written down: with a bimodal distribution, n = 6-8 comparisons point wherever the coin lands.
- Coalescing bound measured (not built): 8 x 1 q sequential 75 ms vs 8 rows in one pass 21 ms on CUDA (3.6x).
  Recorded as open item 6 with the design cost; depends on the traffic having concurrent small callers.
- Skill updated (SKILL.md 1.1, api.md, presets.md): call-path costs, batching, lanes incl. cuda, keep-alive,
  the corrected inline-JSON example (and a note that the ad-hoc question in it is not usable as written -
  0.03 for a text that needs a reply), an HTTP example for many calls. README: "Choosing how to call it"
  decision table; STATUS: decisions, "How to call it", caveat on the bimodal cold CUDA lane.
- Tests: router keep-alive (interval, target lane, window, not started by CPU calls); sidecar suite + wrapper
  fast path (JSON shape, inline args, exit codes, delegation, not slower than `ask.mjs`): 14 scenarios.

## Phase 10 - "is the ~9 ms per request or only the first time?"

Measured instead of asserted (`experiments/cuda_graph_probe.py`): batch-1 latency = **6.0 ms + 21.6 us per
token** (r2 0.992; 1 x 32 tokens 7.0 ms, 1 x 500 17.1 ms), ~1400 CUDA kernel launches per forward pass at
~4 us each - so the floor is per request, every request, and ~80 % of a 1-question call. First-time costs are
separate (session 1.0-1.2 s, first inference ~200 ms) and absorbed by the lane warm-up. CUDA Graph capture,
which would replay those launches as one, **fails on this graph** in onnxruntime-gpu 1.30 (CUDA error 700
during the capture run, context dead afterwards) - the dynamic-shape plumbing inside the captured stream. So
"CUDA graph capture" moved from "needs shape buckets" to "needs a static-shape graph per bucket first"
(STATUS item 7), and coalescing (STATUS item 6, now with a full explanation) attacks the same 6 ms from the
other side without any ORT feature - but only for concurrent traffic.

## Phase 11 - CUDA Graph replay: from "fails on this graph" to the default path

Question: is the 6 ms launch floor fixable, and is a fix worth its complexity? Answered by building it, with
the rule that every step is measured before the next one is designed.

- **Bisect, not guess.** `experiments/cuda_capture_bisect.py` captures growing prefixes of the graph (one
  attempt per process - a failed capture kills the CUDA context): the capture breaks at node #3, `GatherND`
  with constant indices. ORT 1.30's kernel does a `cudaMemcpyAsync` from a host stack vector inside
  `Compute` (`gather_nd.cc:70-75`); the recorded copy replays from freed memory. Everything blamed before
  (shape plumbing, host<->device copies) was innocent.
- **Static bucket graphs** (`tools/static_graph.py`): fixed `[n, L, K]`, Shape outputs observed in one run and
  baked, 202 constant nodes folded through per-node ORT evaluation (`symbolic_shape_infer` fails on this
  export with an `_infer_Range` assertion), the `GatherND` rewritten to `Reshape` where it is a pure
  re-indexing. 1753 -> 1517 nodes; capture works at every optimisation level. Replay 3x96x8: 8.0 ms vs 12.0
  dynamic, 1x96x8: 4.1 vs 8.8, 4x512x8: 64 vs 66 (`experiments/cuda_static_probe.py`) - so big shapes stay
  dynamic (`--graph-max-work`).
- **One copy of the weights**: `add_initializer` refuses device buffers ("must be owned by the user"), so the
  weights became graph inputs bound as CUDA `OrtValue`s (`experiments/cuda_shared_weights_probe.py`): N buckets
  cost 804 MiB once + 28-240 MiB each. ORT also rejects external-data paths outside the model's directory, so
  the bucket files live next to `laya.onnx`.
- **Inputs must be device buffers** updated in place: a CPU-bound input is never re-read by a replay (200/200
  wrong answers, `experiments/cuda_input_race_probe.py`); `cudaDeviceSynchronize` via ctypes after the update.
- **First integration run (naive background builder): 7 of 400 calls failed, 3 of 4 builds failed** with CUDA
  error 900 "operation not permitted when stream is capturing". Two ORT facts from
  `cuda_execution_provider.h`: the captured graph lives in the *calling thread's* `PerThreadContext` and is
  captured on the session's third run on that thread; the capture is global-mode, so any other thread's
  `cudaMalloc` / sync copy / `cudaStreamSynchronize` invalidates it. Redesign: the builder thread *prepares*
  (file, session, binding, buffers) under a lock, the serving thread *captures* under the same lock in idle
  gaps of the request loop (forced after 0.5 s), and checks the replay against the dynamic session on the
  last real inputs of that shape before it serves anything. Second run: 0 errors, 5/5 builds.
- **Then the GIL.** Steady-state replays were right, but every bucket build stalled one serving call ~230 ms:
  `experiments/cuda_gil_probe.py` shows ORT holds the interpreter lock for 317 of the 340 ms of
  `InferenceSession(...)`. Mitigations, each measured: the builder waits for a 100 ms request gap (at most
  1 s) before a GIL-holding step; the eager buckets are built before the ready line (+1.4 s start-up, files
  pre-generated by `npm run cuda:setup`); lazy buckets only for shapes seen twice; the router's probe and
  warm-up run with `exec: { graph: false }` so they neither build buckets nor measure a disturbed lane. The
  first burst of a never-seen shape still runs dynamic - accepted, documented.
- **Node plumbing**: `ProcessLane.systemOne(state, questions, temps, exec)` -> `RemoteSession` sends `exec`,
  the reply's `mode` / `bucket` become `routing.exec`; `decide-core` already validated `exec`, so the HTTP
  body, `demo/`, `ask.mjs --no-graph`, `serve.mjs --cuda-graph off` and `experiments/ab.mjs`
  (`cuda:fp16?graph=false`, plus a "wait until the buckets settled" step) followed.
- **Verdict** (`ab.mjs`, 8 rounds, PoC + 40 dev-request items, dynamic as baseline): **0.858 [0.848, 0.862]**;
  1 / 3 / 10 q 10.8 -> 5.0, 12.4 -> 8.6, 24.5 -> 22.9 ms; dev-request 18.8 -> 16.0; 134/134 arg-max, max
  |dp| 0.006; 0.150 of fp32 time (was 0.213). Throughput 3 q 66-86 -> 114 calls/s, 1 q 96 -> 194, 10 q 41 ->
  45. Fidelity on the eval set 195/195 vs both the dynamic graph (0.0055) and WebGPU (0.036). Default on.
- Tests: `test/cuda-lane.test.mjs` 3 -> 5 scenarios (replay vs dynamic on the eval set, `exec.graph`,
  bucket-grid edges, 1 q speed, `cudaGraph: false`, lazy build after two sightings, `routing.exec`,
  `detailedStats()`); `test:all` green (10 + 8 + 5 + 14 + 5).
- Numbers: 202 ms for the sidecar's 8-call start-up burst is unchanged (the burst is the first sighting of its
  shape and runs dynamic; the second burst replays). Sporadic 3 q: back-to-back 8.7 ms (was 12), 250 ms gaps
  10.5, 1 s 39 (11-78), 3 s 124 - the cold side is still the GPU's clocks.
- Things that looked like problems and were not: the static graph differs from the dynamic one by up to 0.12
  logits on *random tokens* (kernel selection changes with the shape; real text 0.002-0.02) - the self-test
  threshold is 0.15, the runtime "broken capture" threshold 0.5; `nvidia-smi` for VRAM deltas was replaced by
  `cudaMemGetInfo` (the 5 ms subprocess call sat inside the capture window).

## Phase 12 - "anything to do about the CLI shape?" -> start-up, memory, stalls

The question was about the default preset's shape (`5x160x8`) not being among the eager buckets, so a fresh
sidecar answered its first calls on the dynamic graph. The answer became a second pass over how buckets come
to exist (`results/cuda-graph-2026-09-23-summary.md`, "Start-up, memory and stalls"):

- Eager buckets are built *after* the ready line, in idle gaps: the CUDA lane is now the first lane ready
  (sidecar ready 2.1-2.4 s, first call 2.2-2.5 s; was WebGPU at ~2.6 s / 2.8-3.1 s). The lane warms its
  dynamic session once before ready so the first real call does not pay the ~180 ms cuDNN/kernel-load cost.
- The shapes real traffic used are remembered per port (`.laya/cuda-buckets-<port>.json`, hits accumulate) and
  built first on the next start - preset-agnostic; the built-in defaults only matter for a first start ever.
- Every stall the build path could inflict on a caller was measured and either removed or accounted for: first
  build waits a full idle second (the spawn-on-demand call arrives 40 ms after ready); session creation from a
  pre-optimised file (250 -> 100 ms, `experiments/cuda_session_create_probe.py`); capture in one-run steps
  (10-60 ms) instead of a 55-230 ms block; static-graph generation in a low-priority subprocess on the E-cores
  (in-process it made every concurrent call 2-3x slower for 7 s); `stallMs` in the response so the router's
  EMA never learns a stall (one stalled call had sent half a burst to WebGPU: 1527 ms for 8 calls, now 200-300).
- Two bugs found by the tests along the way: (1) the router's one-shot GPU load sample at `create()` was never
  refreshed with `sampleLoad: false` and, having caught the previous test's GPU work, inflated the CUDA lane 2.2x
  for the whole router life - every burst on `cpu:8`, 1 in ~4 runs; fixed with a 10 s staleness rule
  (`LOAD_STALE_MS`) and no utilisation ingest when sampling is off, unit-tested; (2) `--stop` left the Python
  process alive while the builder was mid-generation - the exit now skips the interpreter teardown
  (`os._exit`; all file writes atomic).
- Kept connection after this round: 1 / 3 / 10 q = 6.6 / 10.8 / 25.1 ms, default preset 23.3 (was 13-15 for 3 q).
- Tests: 6 CUDA-lane scenarios (bucket memory round trip added), `test:all` green; docs, skill 1.2, api.md,
  presets.md, demo README updated to the new numbers.

## Numbers worth remembering (2026-09-21/23, before the graph optimisation where GPU numbers are given)

- CUDA lane end state (2026-09-23 night): 1 / 3 / 10 q = 5.0 / 8.6 / 22.9 ms round trip with graph replay,
  10.8 / 12.4 / 24.5 dynamic; 114 calls/s for 3 q; kept HTTP connection 6.6 / 10.8 / 25.1 ms; sidecar first call
  2.2-2.5 s. ORT holds the GIL for a session build (~250 ms, ~100 from a pre-optimised file); a capture is owned
  by the thread that ran it; other threads' CUDA calls during a global-mode capture = error 900.
- `webgpu:fp16` 3 q: 19-21 calls/s back-to-back (57-63 questions/s), 10/s at 53-60 ms, 5/s at 61-85 ms,
  1/s at 112-143 ms, 1 per 3 s at 172-257 ms. `cpu:8`: 2.5-3.8 calls/s, saturates above ~3/s.
- Concurrency never adds throughput; 8 callers each wait ~410 ms at 19 calls/s.
- A CPU call in a GPU burst costs every queued GPU call its full duration; `auto` lost 3x at 8 callers.
- The JS thread on an E-core: 1.6x slower everything, for minutes at a time.
- Machine noise: ~18 % CPU busy from other apps at "idle"; `cpu:8` ranged 257-375 ms p50 across runs today.
- Sidecar first call 3.1-4.5 s (was 5.0-6.3); worker lanes: 18-44 ms main-thread stall during a 1 s CPU inference.
- `worker.terminate()` under a running ORT inference = process crash 0xC0000409.
