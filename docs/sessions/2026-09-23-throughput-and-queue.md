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

## Numbers worth remembering

- `webgpu:fp16` 3 q: 19-21 calls/s back-to-back (57-63 questions/s), 10/s at 53-60 ms, 5/s at 61-85 ms,
  1/s at 112-143 ms, 1 per 3 s at 172-257 ms. `cpu:8`: 2.5-3.8 calls/s, saturates above ~3/s.
- Concurrency never adds throughput; 8 callers each wait ~410 ms at 19 calls/s.
- A CPU call in a GPU burst costs every queued GPU call its full duration; `auto` lost 3x at 8 callers.
- The JS thread on an E-core: 1.6x slower everything, for minutes at a time.
- Machine noise: ~18 % CPU busy from other apps at "idle"; `cpu:8` ranged 257-375 ms p50 across runs today.
- Sidecar first call 3.1-4.5 s (was 5.0-6.3); worker lanes: 18-44 ms main-thread stall during a 1 s CPU inference.
- `worker.terminate()` under a running ORT inference = process crash 0xC0000409.
