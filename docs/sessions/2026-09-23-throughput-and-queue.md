# Session log - 2026-09-23: throughput, the one-thread truth, and the queue fix

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

## Numbers worth remembering

- `webgpu:fp16` 3 q: 19-21 calls/s back-to-back (57-63 questions/s), 10/s at 53-60 ms, 5/s at 61-85 ms,
  1/s at 112-143 ms, 1 per 3 s at 172-257 ms. `cpu:8`: 2.5-3.8 calls/s, saturates above ~3/s.
- Concurrency never adds throughput; 8 callers each wait ~410 ms at 19 calls/s.
- A CPU call in a GPU burst costs every queued GPU call its full duration; `auto` lost 3x at 8 callers.
- The JS thread on an E-core: 1.6x slower everything, for minutes at a time.
- Machine noise: ~18 % CPU busy from other apps at "idle"; `cpu:8` ranged 257-375 ms p50 across runs today.
