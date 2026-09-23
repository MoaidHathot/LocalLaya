# LocalLaya - project status

Living document: goals, decisions, what exists, what is next. Update it when the picture changes.
The chronological record of how we got here is in `docs/sessions/`.

## Goal

Run [Laya](https://huggingface.co/convaiinnovations/laya) (Convai Innovations' non-autoregressive
"System 1" decision model: typed `choice` / `score` / `noul` answers with probabilities in one forward
pass) **locally on Windows**, through the ONNX port [`@receptron/laya`](https://github.com/receptron/laya),
and make it usable as a fast, cheap decision step for other workflows and agents: classify, route, gate,
triage, score - in ~0.3 s, without an LLM call.

Hardware: i9-14900KF (8 P + 16 E cores) / RTX 4070 12 GB / 64 GB / SSD (`W:` is ReFS).
Toolchain: Node 25.3 (>= 20 required), npm 11.4, Python 3.12 + uv (only for the fp16 conversion).

## Decisions (and why)

| decision | rationale |
|---|---|
| ONNX port (`@receptron/laya` 0.1.2 + `onnxruntime-node` 1.30) rather than the original PyTorch package | Node integration, no Python/PyTorch/CUDA at runtime, output matches the reference to 4 decimals. The original is needed only for multilingual / typed-decisions checkpoints, fine-tuning, `predict_shortlist`. |
| Vendored npm tarball (`vendor/receptron-laya-0.1.2.tgz`) | npm on this machine routes through `packagefeedproxy.microsoft.io`, which returns 404 for `@receptron/laya` (the only package we need that it lacks); `registry.npmjs.org` fails at TLS. Everything else - `onnxruntime-node` (all 181 versions incl. nightlies), `@huggingface/tokenizers` - is served by the proxy; GitHub, jsDelivr, unpkg, Hugging Face and PyPI are reachable. Fetched from jsDelivr, SHA256-verified against an independent unpkg copy (23/23 files), `dist/` audited against GitHub source. |
| Model pinned to HF commit `68f27dfe5a27a54fb2b1fefc432f43f972e90868`, SHA256-verified, cached in `models/` | The library follows `main` and only compares byte sizes. After the first download loading is offline (`modelDir`). |
| GPU path = WebGPU EP; DirectML abandoned | `onnxruntime-node` does not build the CUDA EP for Windows at all (README matrix, `install-metadata.js` `requirements['win32/x64'] = []`, no CUDA symbols in the win32 binding) - a build gap, not a download gap. DML loads the graph but every inference fails in a `Reshape` node at all optimisation levels (same class as ORT issue #27118: DML + int64 indices in transformer graphs, closed stale). WebGPU works and matches CPU answers. |
| fp16 bundle (`models/laya-onnx-fp16`) as the GPU default lane, built by `tools/optimize_graph.py` (2026-09-23) | The ORT profiler showed the export's 28 `IsNaN` NaN guards running on the CPU (WebGPU EP has no IsNaN): 28 GPU->CPU round trips per call, plus 196 Cast dispatches from the first converter's fp32 islands. Exact rewrites (`IsNaN` -> `Not(Equal(x,x))`, Reshape `allowzero` cleared, ORT Gelu fusion) + fp16 without islands: 2101 -> 1753 nodes, **1.87-2.13x faster** than the previous fp16 bundle (paired, interleaved, CI [0.448, 0.457] of fp32 time), arg-max agreement with fp32 134/134 + 118/120 = identical to before, same accuracy. Attention / RoPE / SkipLayerNorm patterns do not match this export - not hand-written (agreed stop point). Previous bundle kept as `models/laya-onnx-fp16-v1` for A/B. |
| DirectML: fixed but not a lane | The failure was `Reshape(allowzero=1)` + `-1` (torch.export emits it for every `view`); clearing the attribute makes DML run on 1.30.0 and the nightly. 18-19 ms for 1 question (fastest EP) but 6-10x slower than WebGPU for batch > 1 even on a fixed shape (3 q 217 ms). Priors updated; not in the default lanes. |
| CUDA lane as a Python process, default first lane (`cuda:fp16,webgpu:fp16,cpu:8`) (2026-09-23) | `onnxruntime-node` has no CUDA EP on Windows; the Python wheel has, and Microsoft's `onnxruntime-cuda-13` feed + NVIDIA's wheel index are reachable here (PyPI's file host is not). `tools/cuda_lane.py` (stdio, NDJSON + base64 tensors) behind `ProcessLane` / `RemoteSession` in `src/lane.mjs`, so `@receptron/laya` runs unchanged against a remote session. Paired ratio to fp32: **0.131 [0.130, 0.132]** vs 0.429 WebGPU (3.3x), 8.9 / 12.1 / 24.1 ms for 1 / 3 / 10 q; 195/195 arg-max agreement with WebGPU, accuracy = fp32. Dropped with a log line where the venv is missing, so the defaults work everywhere. User decision: default on if faster. |
| CUDA Graph replay on static bucket graphs, default on in the CUDA lane (`--cuda-graph off`, per call `exec: { graph: false }`) (2026-09-23) | The lane was launch-bound (6 ms + 22 us/token, ~1400 kernels per call); capture failed on the exported graph because ORT 1.30's CUDA `GatherND` copies a host buffer during `Compute` (bisected to one node). `tools/static_graph.py` builds a static-shape graph per (rows, tokens, options) bucket with that node rewritten; `tools/cuda_lane.py` captures it and replays with padded inputs, sharing one device copy of the weights across buckets. Paired vs the dynamic path: **0.858 [0.848, 0.862]** overall, 1 q 10.8 -> 5.0 ms, 3 q 12.4 -> 8.6, 10 q 24.5 -> 22.9; 195/195 arg-max agreement, max abs dp 0.006; throughput 3 q 66-86 -> 114 calls/s. Threading follows two ORT facts (graph owned by the capturing thread; other threads' CUDA calls during a capture fail with error 900): a builder thread prepares sessions, the serving thread captures in idle gaps, every capture is checked against the dynamic graph before it serves. Buckets follow the traffic (a shape seen twice) plus four eager ones before ready. |
| GPU keep-alive on by default in the sidecar (`--gpu-keepalive 30s`, tiny call every 500 ms on the lane that served last) (2026-09-23) | The CUDA lane is bimodal after a pause (3 q after 3 s: ~50 or ~200-300 ms). Through the sidecar, n = 20 per arm, arms alternated: median 174 -> 46 ms, slow calls 12/20 -> 4/20, for 1-4 W of GPU power while active and nothing when idle. Two earlier n = 6-8 comparisons pointed in opposite directions - the bimodality needs n >= 20. A 1 s interval does not help; on WebGPU the trick never did. |
| Skill wrapper answers one-shot calls itself over HTTP (no second Node process) | 130-165 ms -> 80-90 ms per agent call; the remaining cost is Node start-up + imports around a 12 ms inference, so the guidance for many calls is a kept HTTP connection (13-15 ms). Lifecycle / `--local` / `--pretty` / REPL / sidecar failures are still delegated to `ask.mjs`. |
| The CPU lane keeps the pinned, hash-verified HF fp32 bundle | The same graph clean-up gives nothing on the CPU EP (1.030 [0.953, 1.058]; IsNaN is native there), fp16 on the CPU EP is not faster either. |
| Gains are only claimed from `experiments/ab.mjs` | Variants in one process, interleaved rounds, paired ratio with bootstrap 95 % CI, fidelity vs a reference on any preset's eval set. Noise floor 1.000 [0.997, 1.001]. Sequential single runs on this machine swing 1.1-1.3x and cannot resolve a 5 % effect. |
| CPU lanes pin 16 (or 8) intra-op threads to the P-cores | ORT's default 24-thread pool spans E-cores; Windows scheduling makes 3 questions take 215 ms or 1200 ms (p95 > 1.1 s in half the sessions). Pinning: p95 284-350 ms for ~10 % cost. `cpu:8` pinned equals `cpu:16` pinned at half the CPU share. |
| Per-call execution-provider router (`src/ep-router.mjs`) | Back-to-back traffic: WebGPU 3-5x faster. Sporadic traffic: the GPU drops to 225 MHz between calls and a single question takes ~180 ms vs ~105 ms on CPU. The router predicts per (lane, GPU thermal state at start, question bucket, work) plus the shared queue wait, and picks. |
| One FIFO queue for all lanes; a burst never spills to the CPU lane (2026-09-23) | `onnxruntime-node` runs `session.run()` synchronously on the JS thread, so inferences never overlap in one process; a CPU call inside a GPU burst stalled every GPU call behind it (49 -> 321-479 ms) and policy `auto` fell to 6.6 calls/s at 8 callers vs 20 GPU-only. Even with lanes in worker threads mixing loses (GPU 1.5x slower while the pinned CPU pool spins; 16 vs 20.6 calls/s). The CPU lane is for sporadic single questions on a cold GPU and for GPU-less machines. |
| Process restricted to the P-cores on Windows (`pinProcessToPCores`, default in `LayaRouter.create`) | The JS thread tokenises, drives WebGPU and is one of ORT's workers; Windows parks it on an E-core for minutes: whole sessions at 78 ms instead of 49 ms per 3-q GPU call (1.6x), reproducible by forcing the E-cores. Node has no thread-affinity API; PowerShell sets the process mask in ~0.4 s during the model load. |
| Lanes in worker threads by default (`src/lane.mjs`, `workers: false` for in-thread) (2026-09-23) | `session.run()` blocks the calling thread; in the main thread a 10-q CPU call froze the server for 1.1 s (no /health, no new requests, no idle timer). In a worker: 18 ms stall, identical answers, no measurable round-trip cost (306 vs 294 ms), and two lanes load in 2.2 s instead of 3.3-3.6 s (session creation no longer serialised). The FIFO across lanes stays (mixing loses). Rule: never `terminate()` a worker under a running inference - the process dies with 0xC0000409; workers release their session and exit on request. |
| Sidecar serves as soon as the first lane is warm; lanes are bound at the front of the queue; probes/warm-ups bypass the FIFO | First call 5.0-6.3 s -> 3.1-4.5 s; ready with one lane at 2.6-2.9 s, both at 3.5-3.8 s. Binding at enqueue time sent a start-up burst to the CPU lane for 7.6 s while the GPU lane sat idle after joining; binding when a call reaches the front lets queued calls move over (6 x 10 q: 1.8 s instead of 5.7 s). Probes/warm-ups in the FIFO delayed the joining lane behind the burst, hence outside. |
| `--max-age` recycling for the sidecar (default off, `LAYA_MAX_AGE`) | Cheap guard against slow leaks in an always-on instance; exits once nothing is in flight, the next call spawns a fresh one. Named-pipe transport and a Windows service were not done (no need shown yet). |
| `package-lock.json` resolved URLs point at `registry.npmjs.org` | The lock had the Microsoft feed proxy's URLs, which GitHub Actions cannot reach; npm's `replace-registry-host` maps npmjs URLs to whatever registry is configured locally, so both work. |
| Calibration = per-bucket temperature refit on labelled data; wording is the accuracy lever | Temperature never changes the arg-max; it fixes over-confidence where errors are spread, not systematic confusions. Wording moved smart-home intent 0.63 -> 0.72 but hurt another question - measure every question. |
| CLI default stays in-process; `--sidecar` (or `LAYA_SIDECAR=1`) opts into the shared background instance | User choice (2026-09-22). No background process unless asked for. |
| Sidecar idle exit 5 min; open REPL keeps it alive | User choice. Reload costs ~5 s; holding costs ~2 GB RAM + 0.8 GB VRAM. |
| Sidecar client uses `node:http`, not `fetch` | Node 25.3 on Windows: undici crashes the process at exit (`0xC0000409`) after two requests - wrong exit codes for every orchestration. |
| Project license Unlicense (GitHub repo choice); third-party notices kept | Vendored library MIT, weights Apache-2.0, ORT MIT. |

## What exists (done)

- **PoC + client** (`poc.mjs`, `src/laya-client.mjs`): pinned download, SHA256 verification, offline load,
  EP selection, P-core pinning, fp16 bundle, calibration tables, `createDecider()` facade.
- **Benchmarks + metrics** (`bench.mjs`, `bench-all.mjs`, `src/metrics.mjs`, `experiments/`): EP x 1/3/10
  questions with process CPU / RSS / GPU util / VRAM / power; sporadic-vs-burst, length sweep, shape and
  concurrency, fp16 fidelity; throughput matrix (`experiments/throughput.mjs`: lanes x closed-loop concurrency
  x open-loop arrival rate, policy auto vs prefer-gpu, cross-lane interference), interference mechanism
  (`experiments/interference.mjs`: same thread vs child process), worker-thread lanes
  (`experiments/worker-lanes.mjs`), CUDA per-call floor + graph-capture probes (`experiments/cuda_graph_probe.py`,
  `cuda_capture_attempt.py`, `cuda_capture_bisect.py`, `cuda_static_probe.py`, `cuda_shared_weights_probe.py`,
  `cuda_input_race_probe.py`, `cuda_gil_probe.py`, Python). Results in `results/*-summary.md` and README.
- **Router** (`src/ep-router.mjs`, `router-demo.mjs`): lanes `webgpu[:fp16]`, `cpu[:N][:nopin]`, `cpu:auto`,
  `dml` (default `webgpu,cpu:8`); probe with real inference; EMA latency model normalised by estimated work;
  contention inflation; one FIFO queue for all lanes with `wait + own` predictions, the GPU state expected at
  start and the lane bound when the call reaches the front; policies `auto` / `prefer-gpu` / `prefer-cpu` /
  `min-cpu`, `deadlineMs`, forced lane; quarantine + fallback, dead-lane handling; per-call calibration;
  `pauseSampling`/`resumeSampling`; P-core process affinity on Windows; `waitFor: "first"` + `warmup` +
  `onLaneReady` + `router.ready` for early serving; `close()` drains the queue.
- **Lanes** (`src/lane.mjs`, `src/lane-worker.mjs`, `tools/cuda_lane.py`): one Laya session per lane in a worker
  thread (default), in-thread, or in a Python process (CUDA EP), same handle (`systemOne(state, questions, temps)`,
  `close()`, `onDeath`); per-call temperature override applied inside the lane; workers unref'd while idle;
  ordered message handling so `close` never releases a session under a running inference; the process lane
  builds a stock `Laya` on a `RemoteSession` (run / release over stdio) and passes a per-call `exec`
  (`{ graph: false }`) through; the result's `exec` (`mode`, `bucket`, `remoteMs`) becomes `routing.exec`.
  `tools/cuda_lane.py`: dynamic session + CUDA Graph bucket sessions on one shared device copy of the weights,
  buckets built from the traffic in idle gaps (`--graph auto|off`, `--graph-buckets`, `--graph-max-sessions`,
  `--graph-max-vram-mib`, `--graph-max-work`, `--self-test`, ops `stats` / `bucket`); `tools/static_graph.py`
  makes the static bucket graphs. `tools/setup-cuda-lane.mjs` (`npm run cuda:setup` / `cuda:check`, `--graphs`)
  installs the pinned wheels, verifies a CUDA session and pre-generates the eager buckets' graphs.
- **Graph tool** (`tools/optimize_graph.py`, `npm run fp16:convert`): IsNaN -> Not(Equal), Reshape allowzero,
  ORT Gelu fusion, fp16 without islands; `optimize-report.json` next to the bundle.
- **A/B tool** (`experiments/ab.mjs`, `npm run fp16:check`): interleaved variants, paired ratio + bootstrap CI,
  fidelity vs a reference on any preset's eval set (`--preset`, `--inputs`).
- **Calibration** (`src/calibration.mjs`, `calibrate.mjs`, `data/`, `calibration/`): raw-logit capture,
  accuracy / NLL / Brier / ECE, reliability tables, per-bucket temperature refit with leave-one-out,
  majority/chance baselines with a verdict per question; generic `--preset --eval` for any domain.
- **Presets** (`data/presets.mjs`, `presets/`): built-ins `smart-home` (measured, calibrated), `triage`,
  `guard`, `moderation`, `route`, `sentiment` (unmeasured); file presets `presets/<name>.json` with a
  `$TEXT` state template; `dev-request` worked example with 40 labelled items.
- **Ask surface** (`ask.mjs`): one-shot CLI, interactive REPL (ad-hoc `/choice` `/noul` `/score`, `/save`),
  `--json`, own `--state`/`--questions`.
- **Server / sidecar** (`serve.mjs`, `src/sidecar-client.mjs`): `POST /decide`, `/presets`, `/health`
  (`lanes`, `lanesLoading`, `maxAgeS`, `workers`), `/stats`, `/touch`, `/shutdown`, browser page;
  listen-before-load (port = mutex), 503 while loading (and for a forced lane still loading, with
  `retryAfterMs`), ready as soon as the first lane is warm, idle exit with nothing in flight, `--max-age`
  recycling, sampling pause, presets/calibration re-read by mtime, per-request calibration override,
  `--in-process` escape hatch. CLI `--sidecar` / `--local` / `--start` (waits for every lane) / `--status` /
  `--stop` / `--idle` / `--max-age` / `--port`; client `waitAllLanes()`.
- **Agent skill** (`skills/laya-decisions/`): `SKILL.md`, `references/api.md`, `references/presets.md`,
  `scripts/laya.mjs` (resolves the project via `LAYA_DIR`, runs `ask.mjs --sidecar --json`).
- **Tests**: `npm test` (10 unit tests, no model; also in GitHub Actions on Ubuntu/Windows x Node 20/22 -
  `.github/workflows/unit-tests.yml`, not yet seen running on GitHub), `npm run test:router` (8 worker-lane /
  failover / early-serving / keep-alive scenarios, ~45 s), `npm run test:cuda` (5 process-lane scenarios: fidelity
  vs WebGPU on the full eval set, CUDA Graph replay vs the dynamic graph on the eval set + `exec.graph` + bucket
  grid + 1 q speed, `cudaGraph: false` + lazy bucket build, router `routing.exec` + kill -> failover, close / bad
  python; ~40 s, skipped without the venv), `npm run test:sidecar` (14 lifecycle scenarios, ~65 s, port 8797),
  `npm run test:demo` (5), `npm run test:all`.

## Key measurements (this machine; treat as +-20 %)

| lane | 1 q | 3 q | 10 q | machine CPU | RSS | VRAM |
|---|---|---|---|---|---|---|
| `cpu` (16 pinned) | 115 ms | 276 | 911 | 49 % | 1.6-1.8 GiB | 0 |
| `cpu:8` (8 pinned) | 113 | 272 | 921 | 24 % | 1.6-1.8 GiB | 0 |
| `webgpu` fp32 (HF export as is) | 30.5 | 53 | 143 | 2 % | 1.1 GiB | +1636 MiB |
| `webgpu:fp16` first converter (2026-09-21) | 29.1 | 47 | 122 | 2 % | 0.6 GiB | +832 MiB |
| `webgpu:fp16` optimised graph (2026-09-23) | 21.1 | 32.0 | 83.1 | 2 % | 0.6 GiB | +832 MiB |
| `cuda:fp16` process lane, dynamic graph (2026-09-23) | 8.9 | 12.1 | 24.1 | 3 % | 1.0 GiB | +~1250 MiB |
| **`cuda:fp16` + CUDA Graph replay (2026-09-23, default first lane)** | **5.0** | **8.6** | **22.9** | 3 % | 1.0 GiB | +~1250 MiB + 28-240 MiB per bucket |
| `dml` optimised graph (not a lane) | 18-19 | 217 | 264 | | | |

Sporadic 1-question calls (3 s gaps): WebGPU ~101 ms (was ~180), CPU ~105 ms. Sidecar: first call 3.1-4.5 s
(was 5.0-6.3 s with lanes loading one after the other in the main thread), later calls ~0.15-0.4 s; ready with
the first lane 2.6-2.9 s after spawn, both lanes 3.5-3.8 s; 8 parallel 5-q calls in 0.7 s on one lane (1.1-1.4 s
before the graph optimisation, 2.6 s when they were spread over GPU + CPU). Worker lanes: main-thread stall
during a 10-q CPU call 1.1 s -> 18 ms; two lanes load in 2.2-2.3 s instead of 3.3-3.6 s. Accuracy zero-shot:
smart-home intent 0.72, should_execute 0.66, target_device 0.77 (65 items); dev-request task 0.75, language
0.85 (40 items) - unchanged by fp16 or the graph optimisation.

Throughput (3 q per call, `results/throughput-*-summary.md`): `cuda:fp16` with CUDA Graph replay 114-115
calls/s at 1-8 callers (8-9 ms per inference; 1 q 194 calls/s, 10 q 45 calls/s = 447 questions/s); on the
dynamic graph 66 calls/s with one caller, 83-86 with 4-8 queued (12 ms), 40/s offered served at 13 ms.
`webgpu:fp16` (optimised bundle) 30-31 calls/s
back-to-back (91-93 questions/s; was 19-21), 20/s offered served at 33 ms p50 (was 110-146 with a queue), 10/s
at 35 ms, 5/s at 70 ms (GPU clocks sag between calls), 1 per 3 s at 156 ms; `cpu:8` 2.5-3.8 calls/s, saturates
above ~3/s. Concurrency adds nothing (one FIFO): 8 callers = same calls/s, each waits ~130 ms. 1 q per call
~1.5x the calls/s of 3 q, 10 q ~0.4x - batch questions per call. Policy `auto` before the queue fix: 10.2
calls/s at 4 callers and 6.6 at 8 (p95 1.5 s); after: equal to `prefer-gpu` at every concurrency.

## Open items / next steps

Ordered roughly by value. Done on 2026-09-23: sidecar serves from the first lane + `--max-age`, lanes in
worker threads, `cpu:8` as the default CPU lane, `bench-all` defaults, GitHub Actions for the unit tests,
optimised fp16 graph (1.9-2.1x), CUDA process lane (another 3x), generic A/B tooling, GPU keep-alive default,
skill wrapper fast path, call-path decision guide (README "Choosing how to call it"), `demo/`, CUDA Graph replay
on static bucket graphs (1 q 2x, 3 q 1.4x; item 7 below is now the record of how).

1. **Real traffic, real labels.** Everything above is measured on hand-written examples. Collect 50+ real
   inputs per preset that matters, label them, run `calibrate.mjs`, act on the verdicts.
2. **Fine-tuning path** when wording stops helping: the original repo's RLCD notebook (Kaggle 2xT4, 4-5 h),
   then `export/export_onnx.py` -> load via `modelDir`. Not started.
3. **Multilingual checkpoint**: only the English root is published as ONNX; export
   `convaiinnovations/laya-multilingual` ourselves if non-English input is needed.
4. **Sidecar, if ever needed**: named-pipe transport (no port); a Windows service / Task Scheduler entry for
   always-on use (`--idle 0 --max-age 24h`).
5. **Router**: the CPU lane is only chosen for sporadic small calls on a cold GPU and as a fallback - decide
   whether loading it is worth 1.6 GiB RAM on GPU machines; token-length feature is an estimate
   (`estimateWork`); the start-up EMA of a GPU lane warmed while CPU calls run is ~1.5x pessimistic for its
   first few calls (self-corrects). The `cuda` cold priors are the measured medians of a bimodal distribution
   (`experiments/sporadic.mjs --ep cuda --fp16`); the keep-alive makes them pessimistic in practice.
6. **Coalescing concurrent requests into one forward pass** (measured upper bound, not built).
   *What:* when a call reaches the front of the FIFO, take every other call already queued for the same lane,
   run all their question rows as **one** `session.run`, split the logits back per request.
   *Why it works:* a CUDA-lane call costs **6.0 ms + 21.6 us per token** whatever the number of rows
   (`experiments/cuda_graph_probe.py`): the 6 ms is ~1400 kernel launches, paid per call, not per row. Eight
   1-question requests served one by one = 8 x 6 ms of launches; as one batch = 1 x 6 ms. Measured: 8 x (1 q)
   sequential **75 ms**, the same 8 rows in one pass **21 ms** (3.6x). The gain shrinks with bigger calls
   (a 10-question call is already 75 % arithmetic) and is zero for a single caller.
   *Design:* `systemMany(items)` on the lane handle (`src/lane.mjs`) that collates the rows of several
   `(state, questions, temps)` items into one `[n, L]` / `[n, K]` batch - padded to the longest sequence and
   widest option set, as `systemOne` already does within one call - runs it, and splits the answers; per-item
   temperature tables are applied per row (softmax is per row, so per-request calibration overrides stay
   correct); `usage.input_tokens` per item; `routing.coalesced = n`. The vendored library's `systemOne` is one
   state per call and its sequence builder (`dist/sequence.js`) is not exported, so the collate / split is
   re-implemented against a copy (or deep import) of it - ~150 lines, plus a fidelity test against
   `systemOne` per item. Router: in `_enqueue`, when a call starts, gather the `inflight` entries that are queued
   (not started) and re-bind to the same lane, up to a cap (rows x tokens <= e.g. 64 x 512 so a long state
   does not pad everyone). No caller is delayed: only calls that were already waiting join, and each of them
   would otherwise run after the current one.
   *Limits:* different lanes cannot coalesce; a request with a `lane` override or `deadlineMs` runs alone;
   mixed long / short states waste padding (cap); the coalesced call's `ms` is shared, so `routing.ms`
   becomes "batch ms" and the EMA needs the per-row share (batch ms / n, work-normalised).
   *When it pays:* several concurrent callers sending small calls (agents in parallel, a service fan-out).
   The sidecar's 8-caller burst today: 8 x 9 ms = ~70 ms end to end for the last caller; coalesced ~25 ms.
   With graph replay the per-call floor is ~2 ms rather than 6, so the bound has shrunk to ~2x for 1-question
   callers.
   *Decide from real traffic:* `routing.queueMs > 0` on a meaningful share of calls is the signal.
7. **CUDA lane: the per-call floor - DONE via CUDA Graph replay (2026-09-23), what remains.** Measured
   (`experiments/cuda_graph_probe.py`): every `session.run` cost **6.0 ms + 21.6 us/token** on this graph
   because ORT issues ~1400 CUDA kernels per forward pass at ~4 us each. `enable_cuda_graph` on the exported
   graph failed ("CUDA failure 700: illegal memory access" during the capture run, context dead) - not the
   shape plumbing in general but one kernel: ORT 1.30's CUDA `GatherND` copies a host vector during `Compute`
   (`cuda_capture_bisect.py`). Built: `tools/static_graph.py` (static bucket graphs, that node -> `Reshape`,
   weights as inputs) + bucket sessions in `tools/cuda_lane.py` with the threading the ORT design forces
   (details and all numbers: `results/cuda-graph-2026-09-23-summary.md`). Result: 1 q 10.8 -> 5.0 ms, 3 q
   12.4 -> 8.6, dev-request 18.8 -> 16.0; paired 0.858 [0.848, 0.862]; same answers.
   *Left on the table:* (a) the remaining ~4 ms of a 1-question replay is the stdio hop + tokenising in Node
   (~1.5 ms) and the padded forward pass itself; a TensorRT EP session per bucket (fused kernels) is the next
   lever, same static-graph prerequisite, untested; (b) buckets are per exact (rows, tokens, options) grid
   cell, so a burst of a never-seen shape runs dynamic once and pays ~250 ms once for the session creation
   (ORT holds the GIL) - `--graph-buckets` can pre-build known shapes; (c) the grid (`GRAPH_L/N/K`) is
   hand-chosen from this machine's traffic - a preset with long states (300+ tokens x 5+ rows) sits above
   `--graph-max-work` and gains nothing, by design (the gain there is < 5 %); (d) the fidelity check at
   capture uses the last real inputs of that shape (threshold 0.5 logits = broken capture); on random-token
   soup the static and dynamic fp16 graphs differ by up to 0.12 logits (kernel selection differs with the
   shape), on real text by 0.002-0.02. For machines with a CUDA 12 driver: `node tools/setup-cuda-lane.mjs
   --cuda 12` (PyPI wheels; pinned but untested here, PyPI's file host is blocked on this network). The
   remaining sporadic slow mode (4/20 calls ~200 ms with keep-alive) is the GPU's power management; user-side
   levers are the NVIDIA "prefer maximum performance" setting or a locked clock (`nvidia-smi -lgc`, admin) -
   not something this project should set.
8. **DirectML**: works on the optimised graph (`allowzero` fix) and is the fastest EP for a single question
   (18-19 ms) but 6-10x slower than WebGPU for batch > 1 on a fixed shape - cause unknown (not shape
   recompilation). Not a lane. Worth a look only if single-question traffic dominates somewhere.
9. **Unmeasured presets** (`triage`, `guard`, `moderation`, `route`, `sentiment`): label and measure before
   relying on them; the `guard` preset answering 100 % on obvious injections says nothing about subtle ones.
   `experiments/ab.mjs --preset <name>` validates fp16 / CUDA fidelity on any preset that has an eval file.
10. **Housekeeping**: confirm the GitHub Actions run is green after the first push (written blind: no runner
   here; GitHub's runners reach `registry.npmjs.org`, this machine does not), `skills-ref validate` on the
    skill (tool not installed here), decide whether to publish the skill separately.

## How to call it (the short version; measured in results/sidecar-modes-2026-09-23-summary.md)

- Many decisions from a process: `node ask.mjs --start`, then `POST /decide` on a kept connection - ~10 ms.
- A decision from a shell / agent tool call: `node skills/laya-decisions/scripts/laya.mjs ...` - 80-90 ms
  (Node start-up; the inference is 5-9 ms of it). First call after idle 2.5-4.5 s.
- All questions about one text in one call (10 q = 23 ms on CUDA, not 10 x 5 ms).
- Calls seconds apart: the default GPU keep-alive handles it (median 46 ms instead of 174); calls minutes
  apart: idle-exit and reload (2.5 s) or `--idle 0`.
- The lane is never the caller's decision; `routing.lane` / `routing.reason` show what the router did.

## Known issues / caveats

- CPU latency on this hybrid CPU fluctuates 1.2-1.6x between sessions even when pinned; the main cause found so
  far is the JS thread landing on an E-core (whole sessions 1.6x slower on both lanes; fixed by the P-core
  process affinity in `LayaRouter.create`). Residual 1.1-1.3x swings remain from other processes' load (~18 %
  of the machine busy at "idle" here) and thermals. The 5x E-core straggler mode is gone with pinning.
- `session.run()` blocks the thread it runs on. With lanes in worker threads (default) the main thread is free;
  with `--in-process` / `workers: false` the HTTP server is unresponsive while a call runs (up to ~1 s for 10
  CPU questions). Never `worker.terminate()` a lane with an inference in flight: the process dies with
  `0xC0000409` (terminating an idle worker is fine).
- Early serving trades the first second: a call that arrives while only the CPU lane is up runs there
  (~300 ms for 3 q) rather than waiting ~0.5-1 s for the GPU lane; bursts move over as soon as it joins.
- The CUDA lane replays CUDA Graphs only for shapes it has seen twice (plus four eager buckets): the first
  burst of a new shape runs on the dynamic graph (~2x slower for 1 question), and under continuous traffic the
  bucket build stalls one call ~250 ms (ORT holds the GIL while it creates a session). `routing.exec.mode`
  tells which path served a call. Turn off with `--cuda-graph off` / `exec: { graph: false }` if that ever
  matters more than the steady-state gain.
- The CUDA lane after a pause of >= 1 s is bimodal (~50 or ~200-300 ms); the keep-alive removes most but not
  all slow calls (4/20 remain). A cold single question is a tie between CUDA, WebGPU (~101 ms) and CPU
  (~105 ms) that the router settles from its EMA; expect either lane in `routing.lane` for such calls.
- GPU latency depends on the gap between calls, not only on long pauses (optimised bundle, 3 q): 32 ms
  back-to-back and with 250 ms gaps, 81 ms after 1 s, 156 ms after 3 s. Rates quoted from back-to-back runs
  are upper bounds.
- WebGPU EP is marked experimental by ORT; first call after load ~130-180 ms (shader compile).
- Several ORT sessions in one process slow each other's small calls (5 WebGPU sessions: 1 q 35-37 ms instead
  of 21-28; 3 pinned CPU sessions: 1 q 500+ ms) - only relevant to `experiments/ab.mjs`, whose comparisons are
  interleaved and therefore unaffected; production has one session per lane per process.
- `confidence` in answers is 1 - normalised entropy, not P(correct); gate on `probabilities[choice]` / `noul`.
- Base checkpoints are "a fast base to specialise, not a zero-shot engine" (model card): expect mediocre
  zero-shot accuracy, systematic confusions (device *questions* vs device *commands*), English only.
- Temperatures are per (type x option-count) bucket: one chance-level `noul` question flattens all `noul`
  questions of a preset - remove it.
- Node 25 `fetch` exit crash (see decisions); anything that adds HTTP calls to the CLI must use `node:http`.

## How to resume work

```powershell
cd W:\Github\LocalLaya
npm test                      # 10 unit tests, no model needed
npm run test:router           # worker lanes, failover, early serving (~45 s, needs model + GPU)
npm run test:cuda             # CUDA process lane incl. CUDA Graph replay vs dynamic (~40 s, needs the venv)
node ask.mjs --status         # is a sidecar running?
node ask.mjs --sidecar "..."  # start using it
npm run test:sidecar          # full lifecycle check (~1 min)
npm run fp16:check            # fp16 + cuda lanes vs fp32 reference: speed (paired CI) + fidelity; add --preset <yours>
npm run cuda:check            # is the Python side of the CUDA lane installed and working? (npm run cuda:setup to install)
```

Models are in `models/` (ignored by git, 3.2 GB incl. the previous fp16 bundle `laya-onnx-fp16-v1`). If
missing, `node poc.mjs` re-downloads and verifies the pinned fp32 bundle; `npm run fp16:convert` rebuilds the
optimised fp16 bundle with `tools/optimize_graph.py` (needs `.venv`: `uv venv .venv` +
`uv pip install --python .venv/Scripts/python.exe onnx onnxruntime`; `npm run cuda:setup` then swaps
`onnxruntime` for `onnxruntime-gpu` + the CUDA / cuDNN wheels, ~1.6 GiB, from Microsoft's and NVIDIA's feeds,
and pre-generates the static bucket graphs `laya-static-*-w.onnx` / `laya-dynamic-w.onnx` next to `laya.onnx`;
`tools/cuda_lane.py` regenerates any that are missing on its first start).
