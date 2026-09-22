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
| Vendored npm tarball (`vendor/receptron-laya-0.1.2.tgz`) | npm on this machine routes through `packagefeedproxy.microsoft.io`, which does not carry the package; `registry.npmjs.org` is blocked at TLS. Fetched from jsDelivr, SHA256-verified against an independent unpkg copy (23/23 files), `dist/` audited against GitHub source. |
| Model pinned to HF commit `68f27dfe5a27a54fb2b1fefc432f43f972e90868`, SHA256-verified, cached in `models/` | The library follows `main` and only compares byte sizes. After the first download loading is offline (`modelDir`). |
| GPU path = WebGPU EP; DirectML abandoned | `onnxruntime-node` ships no CUDA EP for Windows. DML loads the graph but every inference fails in a `Reshape` node at all optimisation levels. WebGPU works and matches CPU answers. |
| fp16 bundle (`models/laya-onnx-fp16`) as the GPU default lane | Built with ORT's transformer float16 pass (onnxconverter-common left a broken Cast). 195/195 arg-max agreement, max delta p 0.034; half the VRAM (830 MiB) and RSS, 5-13 % faster. |
| CPU lanes pin 16 (or 8) intra-op threads to the P-cores | ORT's default 24-thread pool spans E-cores; Windows scheduling makes 3 questions take 215 ms or 1200 ms (p95 > 1.1 s in half the sessions). Pinning: p95 284-350 ms for ~10 % cost. `cpu:8` pinned equals `cpu:16` pinned at half the CPU share. |
| Per-call execution-provider router (`src/ep-router.mjs`) | Back-to-back traffic: WebGPU 3-5x faster. Sporadic traffic: the GPU drops to 225 MHz between calls and a single question takes ~180 ms vs ~105 ms on CPU. The router predicts per (lane, GPU thermal state at start, question bucket, work) plus the shared queue wait, and picks. |
| One FIFO queue for all lanes; a burst never spills to the CPU lane (2026-09-23) | `onnxruntime-node` runs `session.run()` synchronously on the JS thread, so inferences never overlap in one process; a CPU call inside a GPU burst stalled every GPU call behind it (49 -> 321-479 ms) and policy `auto` fell to 6.6 calls/s at 8 callers vs 20 GPU-only. Even with lanes in worker threads mixing loses (GPU 1.5x slower while the pinned CPU pool spins; 16 vs 20.6 calls/s). The CPU lane is for sporadic single questions on a cold GPU and for GPU-less machines. |
| Process restricted to the P-cores on Windows (`pinProcessToPCores`, default in `LayaRouter.create`) | The JS thread tokenises, drives WebGPU and is one of ORT's workers; Windows parks it on an E-core for minutes: whole sessions at 78 ms instead of 49 ms per 3-q GPU call (1.6x), reproducible by forcing the E-cores. Node has no thread-affinity API; PowerShell sets the process mask in ~0.4 s during the model load. |
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
  (`experiments/worker-lanes.mjs`). Results in `results/*-summary.md` and README.
- **Router** (`src/ep-router.mjs`, `router-demo.mjs`): lanes `webgpu[:fp16]`, `cpu[:N][:nopin]`, `cpu:auto`,
  `dml`; probe with real inference; EMA latency model normalised by estimated work; contention inflation;
  one FIFO queue for all lanes with `wait + own` predictions and the GPU state expected at start; policies
  `auto` / `prefer-gpu` / `prefer-cpu` / `min-cpu`, `deadlineMs`, forced lane; quarantine + fallback; per-call
  calibration; `pauseSampling`/`resumeSampling`; P-core process affinity on Windows.
- **Calibration** (`src/calibration.mjs`, `calibrate.mjs`, `data/`, `calibration/`): raw-logit capture,
  accuracy / NLL / Brier / ECE, reliability tables, per-bucket temperature refit with leave-one-out,
  majority/chance baselines with a verdict per question; generic `--preset --eval` for any domain.
- **Presets** (`data/presets.mjs`, `presets/`): built-ins `smart-home` (measured, calibrated), `triage`,
  `guard`, `moderation`, `route`, `sentiment` (unmeasured); file presets `presets/<name>.json` with a
  `$TEXT` state template; `dev-request` worked example with 40 labelled items.
- **Ask surface** (`ask.mjs`): one-shot CLI, interactive REPL (ad-hoc `/choice` `/noul` `/score`, `/save`),
  `--json`, own `--state`/`--questions`.
- **Server / sidecar** (`serve.mjs`, `src/sidecar-client.mjs`): `POST /decide`, `/presets`, `/health`,
  `/stats`, `/touch`, `/shutdown`, browser page; listen-before-load (port = mutex), 503 while loading,
  idle exit with nothing in flight, sampling pause, presets/calibration re-read by mtime, per-request
  calibration override. CLI `--sidecar` / `--local` / `--start` / `--status` / `--stop` / `--idle` / `--port`.
- **Agent skill** (`skills/laya-decisions/`): `SKILL.md`, `references/api.md`, `references/presets.md`,
  `scripts/laya.mjs` (resolves the project via `LAYA_DIR`, runs `ask.mjs --sidecar --json`).
- **Tests**: `npm test` (9 unit tests), `npm run test:sidecar` (11 lifecycle scenarios, ~1 min, port 8797).

## Key measurements (this machine; treat as +-20 %)

| lane | 1 q | 3 q | 10 q | machine CPU | RSS | VRAM |
|---|---|---|---|---|---|---|
| `cpu` (16 pinned) | 115 ms | 276 | 911 | 49 % | 1.6-1.8 GiB | 0 |
| `cpu:8` (8 pinned) | 113 | 272 | 921 | 24 % | 1.6-1.8 GiB | 0 |
| `webgpu` fp32 | 30.5 | 53 | 143 | 2 % | 1.1 GiB | +1636 MiB |
| `webgpu:fp16` | 29.1 | 47 | 122 | 2 % | 0.6 GiB | +832 MiB |

Sporadic 1-question calls (3 s gaps): WebGPU ~180 ms, CPU ~105 ms. Sidecar: first call 5.4 s, later
calls ~0.2-0.4 s; 8 parallel 5-q calls in 1.4 s on one lane (2.6 s when they were spread over GPU + CPU).
Accuracy zero-shot: smart-home intent 0.72, should_execute 0.66, target_device 0.77 (65 items); dev-request
task 0.75, language 0.85 (40 items).

Throughput (3 q per call, `results/throughput-*-summary.md`): `webgpu:fp16` 19-21 calls/s back-to-back
(57-63 questions/s), 10/s sustained at 53-60 ms, 5/s at 61-85 ms, 1/s at 112-143 ms (GPU clocks sag between
calls); `cpu:8` 2.5-3.8 calls/s, saturates above ~3/s. Concurrency adds nothing (one FIFO): 8 callers = same
calls/s, each waits ~410 ms. 1 q per call ~1.6x the calls/s of 3 q, 10 q ~0.4x - batch questions per call.
Policy `auto` before the queue fix: 10.2 calls/s at 4 callers and 6.6 at 8 (p95 1.5 s); after: 19-20 at every
concurrency, identical to `prefer-gpu`.

## Open items / next steps

Ordered roughly by value.

1. **Real traffic, real labels.** Everything above is measured on hand-written examples. Collect 50+ real
   inputs per preset that matters, label them, run `calibrate.mjs`, act on the verdicts.
2. **Fine-tuning path** when wording stops helping: the original repo's RLCD notebook (Kaggle 2xT4, 4-5 h),
   then `export/export_onnx.py` -> load via `modelDir`. Not started.
3. **Multilingual checkpoint**: only the English root is published as ONNX; export
   `convaiinnovations/laya-multilingual` ourselves if non-English input is needed.
4. **Sidecar**: serve as soon as the first lane is ready (~1.6 s instead of ~5 s first call); optional
   `--max-age` recycling; named-pipe transport (no port) if ever needed. Consider a Windows service / Task
   Scheduler entry for always-on use.
5. **Lanes in worker threads** (`experiments/worker-lanes.mjs` proves it works: both EPs load in
   `node:worker_threads`, clean exit). Not for throughput - mixing lanes loses even in parallel - but for the
   server: today every inference blocks the event loop, so `serve.mjs` cannot accept or parse requests, answer
   `/health`, or run its idle timer while a 300 ms CPU call runs (main-thread stall 300+ ms vs 20 ms with
   workers). Worth it if the sidecar is ever shared by several agents at once.
6. **Router**: `cpu:8` measured equal to `cpu:16` at half the CPU share - consider making it the default
   CPU lane after another measurement session; token-length feature is an estimate (`estimateWork`). With one
   FIFO the CPU lane is only chosen for sporadic small calls on a cold GPU; consider whether loading it at all
   is worth 1.6 GiB RAM on GPU machines (it still covers WebGPU failures).
7. **DirectML**: re-test with newer `onnxruntime-node` releases (Reshape `node_view` failure, ORT 1.30).
8. **Unmeasured presets** (`triage`, `guard`, `moderation`, `route`, `sentiment`): label and measure before
   relying on them; the `guard` preset answering 100 % on obvious injections says nothing about subtle ones.
9. **Housekeeping**: `bench-all` default configs, GitHub Actions (unit tests only - no model in CI),
   `skills-ref validate` on the skill, decide whether to publish the skill separately.

## Known issues / caveats

- CPU latency on this hybrid CPU fluctuates 1.2-1.6x between sessions even when pinned; the main cause found so
  far is the JS thread landing on an E-core (whole sessions 1.6x slower on both lanes; fixed by the P-core
  process affinity in `LayaRouter.create`). Residual 1.1-1.3x swings remain from other processes' load (~18 %
  of the machine busy at "idle" here) and thermals. The 5x E-core straggler mode is gone with pinning.
- Every inference blocks the Node event loop for its duration (`onnxruntime-node` runs synchronously); the HTTP
  server is unresponsive while a call runs (up to ~1 s for 10 CPU questions). See open item 5.
- GPU latency depends on the gap between calls, not only on long pauses: 48 ms back-to-back, 58-66 ms with
  50 ms gaps, 75-96 ms with 200 ms gaps (3 q). Rates quoted from back-to-back runs are upper bounds.
- WebGPU EP is marked experimental by ORT; first call after load ~180 ms (shader compile).
- `confidence` in answers is 1 - normalised entropy, not P(correct); gate on `probabilities[choice]` / `noul`.
- Base checkpoints are "a fast base to specialise, not a zero-shot engine" (model card): expect mediocre
  zero-shot accuracy, systematic confusions (device *questions* vs device *commands*), English only.
- Temperatures are per (type x option-count) bucket: one chance-level `noul` question flattens all `noul`
  questions of a preset - remove it.
- Node 25 `fetch` exit crash (see decisions); anything that adds HTTP calls to the CLI must use `node:http`.

## How to resume work

```powershell
cd W:\Github\LocalLaya
npm test                      # 9 unit tests, no model needed
node ask.mjs --status         # is a sidecar running?
node ask.mjs --sidecar "..."  # start using it
npm run test:sidecar          # full lifecycle check (~1 min)
```

Models are in `models/` (ignored by git, 2.4 GB). If missing, `node poc.mjs` re-downloads and verifies the
pinned fp32 bundle; `npm run fp16:convert` rebuilds the fp16 bundle (needs `.venv`: `uv venv .venv` +
`uv pip install --python .venv/Scripts/python.exe onnx onnxruntime`).
