# LocalLaya - Laya on Windows via ONNX Runtime: CLI, sidecar, router, calibration, benchmarks

Runs [Laya](https://huggingface.co/convaiinnovations/laya) (Convai Innovations' non-autoregressive "System 1"
decision model: typed `choice` / `score` / `noul` answers with probabilities, one forward pass) from Node.js on
Windows through [`@receptron/laya`](https://github.com/receptron/laya) + `onnxruntime-node`, on an
i9-14900KF / RTX 4070 / 64 GB box.

Everything downloaded lives under this directory (`models/`, `node_modules/`, `.venv/`, `.uv-cache/`).

## Layout

| path | what |
|---|---|
| `ask.mjs` | ask it things: one-shot CLI or interactive REPL with presets and ad-hoc questions |
| `serve.mjs` | local HTTP API (`POST /decide`) + browser page; model stays loaded; also the on-demand sidecar (`--idle`, `--max-age`, `/health`, `/shutdown`); ready as soon as the first lane serves |
| `src/sidecar-client.mjs` | discover / spawn / wait / call / stop the sidecar; no ONNX import (used by `ask.mjs --sidecar`) |
| `data/presets.mjs`, `presets/` | built-in question presets (smart-home calibrated; triage, guard, moderation, route, sentiment unmeasured) + your own `presets/<name>.json`; `dev-request` worked example with labelled eval |
| `poc.mjs` | load once, answer 3 questions in one pass, 10 timed runs, sanity check on contrasting states |
| `src/laya-client.mjs` | reusable loader: pinned HF revision, project-local cache, SHA256 verification, no network after first download, EP selection, P-core pinning (threads and, on Windows, the process), calibration, `createDecider()` facade |
| `src/ep-router.mjs` | per-call execution-provider router (`LayaRouter`): probes lanes, predicts latency per (lane, GPU thermal state, question bucket, work), one FIFO queue for all lanes with the lane bound at the front of the queue, contention-aware, quarantines failing lanes, serves from the first lane while the others load |
| `src/lane.mjs`, `src/lane-worker.mjs`, `tools/cuda_lane.py` | a lane = one Laya session in a worker thread (default), in this thread, or in a Python process (CUDA EP over stdio, CUDA Graph replay on static bucket graphs, finding 12), behind one handle: `systemOne(state, questions, temps, exec)`, `close()`, death notification |
| `tools/setup-cuda-lane.mjs` | `npm run cuda:setup` / `cuda:check`: onnxruntime-gpu (Microsoft's CUDA 13 feed) + CUDA / cuDNN wheels (NVIDIA's index) into `.venv`, pinned, verified with a CUDA session; `--graphs` pre-generates the static bucket graphs |
| `tools/static_graph.py` | static-shape variant of the graph per (rows, length, options) bucket for CUDA Graph capture: Shape outputs baked, constants folded, `GatherND` -> `Reshape` (the kernel that broke the capture), weights as inputs so every bucket shares one device copy; `--check` vs the dynamic graph |
| `router-demo.mjs` | the router under burst / sporadic / batch traffic, deadline and forced-lane calls |
| `src/calibration.mjs`, `calibrate.mjs` | raw-logit capture, accuracy + NLL / Brier / ECE + reliability tables, per-bucket temperature refit with leave-one-out |
| `data/smart-home-eval.mjs`, `data/question-variants.mjs` | 65 hand-labelled utterances; three question wordings (v1 original, v2 explicit, v3 best per question) |
| `calibration/*.json` | fitted temperature tables (`loadLaya({ calibration })`) |
| `bench.mjs`, `bench-all.mjs`, `src/metrics.mjs` | latency matrix (EP x 1/3/10 questions) with process CPU / RSS / GPU util / VRAM / power sampling |
| `experiments/*.mjs` | `ab.mjs`: interleaved A/B of lane variants (speed with paired CI + fidelity vs a reference, any preset's eval set); shape sensitivity + concurrency, sporadic-vs-burst latency, length sweep, throughput matrix (lanes x concurrency x arrival rate), cross-lane interference, worker-thread lanes |
| `tools/optimize_graph.py` | fp32 export -> optimised fp16 bundle: IsNaN -> Not(Equal), Reshape allowzero, ORT Gelu fusion, fp16 without fp32 islands (no PyTorch); writes `optimize-report.json` |
| `verify-model.mjs` | re-hash the cached bundle against the pinned SHA256 values |
| `vendor/receptron-laya-0.1.2.tgz` | the exact published npm tarball (see Supply chain) |
| `demo/` | use it from your own code today: `laya.mjs` library (modes auto / sidecar / local / http, keep-alive connection), `cli.mjs`, five worked examples with sample data, `npm run demo`; see `demo/README.md` |
| `skills/laya-decisions/` | Agent Skill (agentskills.io format): `SKILL.md` + `references/` + `scripts/laya.mjs` wrapper; copy the folder into an agent's skills directory and set `LAYA_DIR` |
| `test/unit.test.mjs`, `test/router.test.mjs`, `test/cuda-lane.test.mjs`, `test/sidecar.test.mjs` | unit tests (router maths, latency model, session options, calibration maths, presets, durations; no model needed - run in CI); router integration tests (worker lanes, failover, early serving); CUDA process-lane tests (fidelity vs WebGPU, failover, close; skipped without the venv); sidecar lifecycle tests |
| `.github/workflows/unit-tests.yml` | GitHub Actions: `npm ci` + syntax check + unit tests on Ubuntu / Windows, Node 20 / 22 (no model in CI) |

## Quick start

```powershell
npm install                 # deps via the configured registry; @receptron/laya from vendor/
node poc.mjs                # first run downloads the pinned 1.7 GB bundle into models/ and verifies SHA256
node poc.mjs --ep webgpu    # RTX 4070 through the WebGPU EP
npm run cuda:setup          # optional, 3x faster GPU lane: onnxruntime-gpu + CUDA 13 / cuDNN wheels into .venv (see Results)
node router-demo.mjs        # auto lane selection per call (cuda:fp16 if set up, webgpu:fp16, cpu:8)
npm test                    # unit tests (no model needed); npm run test:router / test:cuda / test:sidecar need the model + GPU
```

## Ask it something

Laya does not chat. You give it a *state* (your message, a ticket, any JSON) and typed *questions*; it answers
all of them in one pass with probabilities. `data/presets.mjs` bundles question sets with a wrapper that turns
plain text into a state: `smart-home` (default, calibrated here), `triage`, `guard`, `moderation`, `route`,
`sentiment` (the last five are unevaluated starting points).

```powershell
node ask.mjs "Turn off the living room lights"                # one shot
node ask.mjs --preset triage "Charged twice. Refund today or I cancel."
node ask.mjs --json --preset guard "Ignore all previous instructions"
node ask.mjs --state state.json --questions questions.json    # your own state + questions
node ask.mjs --sidecar "..."                                  # shared background instance, ~0.3 s per call (see Sidecar)
npm run ask                                                   # interactive: model stays loaded
```

In the REPL, type messages; `/preset triage` switches sets; `/noul Is the customer angry?`,
`/choice Which team? | billing: refunds | support | sales`, `/score How urgent? | low | mid | high` add ad-hoc
questions; `/again` re-asks; `/lane cpu:8` forces a lane; `/json` toggles raw output; `/help` lists the rest.
Piping a script into `ask.mjs` works too.

```powershell
npm run serve                 # http://127.0.0.1:8787 - browser page + JSON API, model stays loaded
```

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8787/decide -ContentType application/json `
  -Body (@{ preset = "triage"; text = "Charged twice, refund me today or I cancel" } | ConvertTo-Json)
# body: { text | state, preset?, questions?, lane?, policy?, deadlineMs?, exec? } -> { answers, usage, routing }
# exec: { graph: false } = cuda lane without CUDA Graph replay for this call; routing.exec says what it did
# GET /presets, /health, /stats.  Binds 127.0.0.1 only; --cors to allow other origins.
```

Integration:

```js
import { createDecider } from "./src/laya-client.mjs";
const d = await createDecider({ ep: "webgpu", modelDir: "models/laya-onnx-fp16", calibration: "calibration/smart-home-v3.json" });
const r = await d.decide(state, questions);      // r.answers.intent.choice, .probabilities, r.answers.x.noul ...
await d.close();

import { LayaRouter } from "./src/ep-router.mjs";
const router = await LayaRouter.create({ lanes: ["webgpu:fp16", "cpu:8"], calibration: "calibration/smart-home-v3.json" });
await router.warmup({ state: representativeState });
const r2 = await router.decide(state, questions);            // r2.routing.lane, .ms, .predictedMs, .reason
const r3 = await router.decide(state, questions, { deadlineMs: 150 });
```

## Sharing one instance: the on-demand sidecar

Each `node ask.mjs "..."` loads the model (~1.6 s, 0.4 GB RAM + 0.8 GB VRAM), answers, exits. Ten programs
calling it at once = ten copies. `serve.mjs` keeps one copy forever, even when nobody calls. The middle ground:

```powershell
node ask.mjs --sidecar "Lock the front door"   # first call: starts serve.mjs in the background (~2.5-4.5 s), answers
node ask.mjs --sidecar "..."                    # every later call: ~0.14-0.4 s, no model load
$env:LAYA_SIDECAR = "1"                         # make --sidecar the default for a shell / orchestration
node ask.mjs --status | --stop | --start [--idle 10m] [--max-age 12h] [--lanes ...]
```

- The sidecar is `serve.mjs --sidecar --idle 5m`, spawned detached (no window, log in `.laya/sidecar-<port>.log`).
  It exits by itself after 5 minutes without requests (`--idle`, `LAYA_IDLE`; `0` = never), freeing RAM and
  VRAM; the next call starts a fresh one. An open REPL (`node ask.mjs --sidecar`) pings it so it stays warm.
  `--max-age 12h` (`LAYA_MAX_AGE`) additionally recycles it once it is that old (in-flight calls finish first).
- GPU keep-alive (`--gpu-keepalive`, `LAYA_GPU_KEEPALIVE`, default 30 s): for that long after each call a tiny
  GPU call every 500 ms keeps the GPU awake, because a call that arrives a few seconds after the previous one
  otherwise pays the GPU's idle-clock ramp (CUDA lane, 3 questions every 3 s, n = 20 per arm: median 174 -> 46 ms,
  slow calls > 120 ms 12/20 -> 4/20) for 1-4 W meanwhile; off when the sidecar is idle. `0` disables it.
- The port is the mutex: the sidecar binds `127.0.0.1:8787` (`--port`, `LAYA_PORT`) *before* loading, so
  launchers racing at the same instant produce exactly one instance (the loser exits with code 3 without
  loading); callers arriving during the load wait for `status: ready`. Anything else on the port is
  detected via `/health` (`service: "laya"`) and never touched; the CLI then falls back to in-process.
- Default lanes `cuda:fp16,webgpu:fp16,cpu:8` (`--lanes`, `LAYA_LANES`): the CUDA lane is a Python process
  (`npm run cuda:setup`; without it the lane is dropped with a log line and WebGPU serves), the others live in
  worker threads; all load side by side and the sidecar is ready as soon as the first lane has been probed and
  warmed (~2.4 s), the others join while it serves (`/health.lanesLoading`; a call that forces a lane still
  loading gets a 503 the client retries). The main thread never runs an inference, so `/health`, the idle
  timer and new requests are served while a 1 s CPU inference runs.
- One queue: inferences of different lanes never run at the same time - mixing them is a net loss (see
  Results) - and a call is bound to a lane when it reaches the front of the queue, not when it is queued: 8
  parallel callers are served FIFO on the fastest lane (`routing.queueMs` tells you how long a call waited),
  and calls queued during start-up move over to a faster lane the moment it joins. Presets and calibration
  tables are re-read when their files change. Background CPU/GPU sampling pauses after 10 s idle.
- Programs that would rather call HTTP directly: `node ask.mjs --start` prints `{ url, pid, lanes }` once every
  lane is loaded; then `POST /decide` (see Ask it something). `/health` shows `idleRemainingS`; `POST /touch` extends it.
- The CLI client uses `node:http`, not `fetch`: on this Node (25.3, Windows) undici crashes the process at exit
  (`0xC0000409`) after a couple of requests - visible only as a wrong exit code.

Measured here: first call 2.5-4.5 s (spawn + load + warm-up + answer; 5.0-6.3 s when the lanes loaded one after
the other in the main thread), second call 137-366 ms; ready with the first lane 2.4-2.9 s after spawn, all
lanes 3.5-4.0 s; 8 parallel 5-question calls in **202 ms** on the CUDA lane, zero errors (676 ms on the
optimised WebGPU lane, 1.1-1.4 s on the first fp16 bundle, 2.6 s when the router still spread them over GPU +
CPU; 7.6 s when a start-up burst was bound to the CPU lane before the GPU lane had joined - fixed by binding
at the front of the queue); VRAM 7077 -> 5090 MiB after the idle exit (three lanes). `npm run test:sidecar`
runs the 13 lifecycle scenarios (~1 min; uses port 8797), `npm run test:router` the 7 worker-lane scenarios,
`npm run test:cuda` the 3 process-lane scenarios (skipped without the venv).
## Using it from agents: the skill

`skills/laya-decisions/` is an [Agent Skill](https://agentskills.io): `SKILL.md` tells an agent what Laya
decides, when (and when not) to use it, how to call it and how to read probabilities;
`references/api.md` and `references/presets.md` hold the details; `scripts/laya.mjs` is the entry point: it
finds this project (via `LAYA_DIR`, or its own location inside the repo) and answers one-shot calls itself over
HTTP to the sidecar (one Node process, 80-90 ms per call of which ~12 ms is the inference); lifecycle flags,
`--local`, `--pretty`, the REPL and any sidecar failure are delegated to `ask.mjs`.

```powershell
node skills/laya-decisions/scripts/laya.mjs --preset dev-request "is this valid json {bla: 1}"
# from anywhere: $env:LAYA_DIR = "W:\Github\LocalLaya"; node <skills-dir>\laya-decisions\scripts\laya.mjs ...
```

### Choosing how to call it

Which *lane* runs a call is the router's job (`routing.lane` / `routing.reason`); with the CUDA lane present it
is CUDA for everything except a cold single question, where CPU and WebGPU tie with it. What the caller does
decide is the call path and the call shape (`results/sidecar-modes-2026-09-23-summary.md`):

| situation | do this | per decision |
|---|---|---|
| a service or agent runtime making many decisions | `node ask.mjs --start` once, then `POST /decide` from a kept HTTP connection | **13-15 ms** |
| a shell step / agent tool call now and then | `node scripts/laya.mjs ...` (or `ask.mjs --sidecar`) | 80-90 ms (Node start-up dominates) |
| several questions about one text | one call with all the questions | 3 q 12 ms, 10 q 24 ms - not 3 / 10 calls |
| calls seconds apart (interactive) | keep the default `--gpu-keepalive 30s`; lengthen it if the pauses are longer | ~46 ms median instead of ~174 |
| calls minutes apart | let it idle-exit (`--idle 5m`, reload 2.5 s) or hold it (`--idle 0 --max-age 24h`, ~3.3 GiB RAM + 2 GiB VRAM) | - |
| one isolated run, no background process | `node ask.mjs --local` | 2.2-2.5 s |
| RAM over fallback | `LAYA_LANES=cuda:fp16,webgpu:fp16` drops the 1.6 GiB CPU lane | same speed on a GPU machine |
| a hard latency budget | `deadlineMs` in the request | `routing.reason` says what was chosen and why |

Not built, measured as an upper bound: coalescing several concurrent small requests into one forward pass
would serve 8 concurrent single-question callers in 21 ms instead of 75 (3.6x) on the CUDA lane - relevant
only for that traffic pattern (see `docs/STATUS.md`, open items).
## Your own domain (custom presets)

The output you get is always *the preset's questions answered about your text*. Asking the smart-home preset
"is this valid JSON {bla: 1}" yields "intent: ask_question 34 %, no device, not urgent" - correct, and useless,
because those are the wrong questions. Laya never answers your question itself; it makes typed decisions that
route or gate the step that does (a parser, a tool, an LLM, a human).

1. **Write the questions** - `presets/<name>.json` (or build them in the REPL with `/choice`, `/noul`, `/score`
   and `/save <name>`):

   ```json
   {
     "description": "Developer-assistant request triage",
     "state": { "request": "$TEXT" },
     "questions": {
       "task": { "type": "choice", "instructions": "What is the user asking the assistant to do?",
                 "criteria": { "validate": "check whether a given input is valid", "write_code": "...", "explain": "..." } },
       "effort": { "type": "score", "instructions": "How much work is this?", "criteria": ["trivial", "small", "medium", "large"] }
     }
   }
   ```

   `state` is a template (`"$TEXT"` = the message; other fields are literal context such as time, app, user
   role); `"state": "request"` is shorthand for a single key. Option descriptions are model input: short,
   concrete, mutually exclusive; < 20 options; the whole option block shares 192 tokens.

2. **Try it**: `node ask.mjs --preset <name> "..."`, `/preset <name>` in the REPL, or `{"preset": "<name>"}`
   against `serve.mjs` (files are re-read per request, so edit and retry without restarting).

3. **Measure** with 30-60 labelled examples - `presets/<name>.eval.json`:

   ```json
   { "items": [ { "text": "is this valid json {bla: 1}", "gold": { "task": "validate", "language": "json" } } ] }
   ```

   `node calibrate.mjs --preset <name> --eval presets/<name>.eval.json` prints, per question, accuracy against
   the majority/chance baselines with a verdict, calibration before/after, the confident mistakes, and writes
   `calibration/<name>.json`, which `ask.mjs`/`serve.mjs` apply automatically for that preset.

4. **Iterate**: reword or drop questions that fail; re-measure. Fine-tune (PyTorch, original repo) when wording
   stops helping.

Worked example, `presets/dev-request.json` (40 labelled items):

| question | acc | baseline | verdict |
|---|---|---|---|
| task (7 options) | 0.75 | 0.15 | usable; answers given at >= 80 % were right 90 % of the time |
| language (8) | 0.85 | 0.23 | usable; >= 80 % bin: 100 % right |
| needs_tool (noul) | 0.55 | 0.63 | **not usable** - coin flip; calibration fitted T = 13.5, i.e. probabilities collapse to ~50 % ("don't know") |
| has_input (noul) | 0.55 | 0.57 | **not usable** - and a regex answers it exactly |

The last two were removed from the preset (v2). "is this a balid json {bla: 1}" now yields
`task: validate 91 %, language: json 97 %, effort: trivial-small` - i.e. *route to the JSON parser*, which is
the decision Laya is for.

Limits: temperatures are per (type, option-count) bucket, so a broken `noul` question flattens the good `noul`
questions in the same preset - remove it rather than keep it. Presets other than `smart-home` and
`dev-request` are unmeasured starting points.
## Results (Node 25.3, onnxruntime-node 1.30.0, Laya English fp32 421M, `results/*-summary.md`)

Latency of one `systemOne` call, p50 ms, 20 runs back-to-back after warm-up:

| lane | 1 q | 3 q | 10 q | machine CPU | RSS | VRAM | GPU power |
|---|---|---|---|---|---|---|---|
| `cpu` = 16 threads pinned to P-cores | 115 | 276 | 911 | 49 % | 1.6-1.8 GiB | 0 | - |
| `cpu:8` pinned (1 thread / physical P-core) | 113 | 272 | 921 | 24 % | 1.6-1.8 GiB | 0 | - |
| `cpu:auto` ORT default (24 threads, unpinned) | 101 | 248 | 809 | 74 % | 1.6-1.8 GiB | 0 | - |
| `webgpu` fp32 (pinned HF export as is) | 30.5 | 53 | 143 | 2 % | 1.1 GiB | +1636 MiB | 42-161 W |
| `webgpu:fp16`, first converter (2026-09-21) | 29.1 | 47 | 122 | 2 % | 0.6 GiB | +832 MiB | 53-144 W |
| `webgpu:fp16`, optimised graph (2026-09-23) | 21.1 | 32.0 | 83.1 | 2 % | 0.6 GiB | +832 MiB | 38-166 W |
| `cuda:fp16`, same bundle in a Python process (2026-09-23), dynamic graph | 8.9 | 12.1 | 24.1 | 3 % | 1.0 GiB (process) | +~1250 MiB incl. CUDA context | - |
| **`cuda:fp16` + CUDA Graph replay (2026-09-23, default first lane)** | **5.0** | **8.6** | **22.9** | 3 % | 1.0 GiB (process) | +~1250 MiB + 28-240 MiB per shape bucket | - |
| `dml` (DirectML), optimised graph | 18-19 | 217 | 264 | | | | batch > 1 pathological, not a lane |

Model-card reference on a Tesla T4 (PyTorch): 39.5 ms (1 q), 158.6 ms (10 q). All lanes return identical
answers to 4 decimals (fp16 vs fp32: 134/134 + 118/120 arg-max agreement over the smart-home and dev-request
eval sets, max |delta p| 0.05, same accuracy; the two flips are near-ties, see finding 10).

### Throughput (`experiments/throughput.mjs`, `results/throughput-*-summary.md`)

Laya is not generative, so the unit is one call (state + N questions, one forward pass), not tokens.
Throughput = 1 / latency: inferences never overlap in one process (see finding 4), so more callers only
lengthen the queue. Sustained rates, 3 questions per call (~255 tokens), through `LayaRouter.decide()`, with
the optimised fp16 bundle (the 2026-09-21 numbers in brackets):

| lane | calls/s | questions/s | per minute | latency at that rate |
|---|---|---|---|---|
| **`cuda:fp16` (graph replay), back-to-back, 1-8 callers** | **114-115** | 343-346 | ~6900 | 8-9 ms per inference, 17-52 ms end to end with a queue |
| **`cuda:fp16`, 1 question per call** | **194** | 194 | ~11600 | 5 ms |
| `cuda:fp16`, 10 questions per call | 45 | **447** | 2680 | 22 ms |
| `cuda:fp16` dynamic graph (`exec: { graph: false }`), 3 q | 66-86 | 198-260 | ~4000-5000 | 12 ms |
| `cuda:fp16`, 40 calls/s offered | 40 | 120 | 2400 | 13 ms |
| `webgpu:fp16`, back-to-back | **30-31** (19-21) | 91-93 | ~1800 | 32-33 ms |
| `webgpu:fp16`, 20 calls/s offered | 20 (queued) | 60 | 1200 | 33 ms (110-146 with a queue) |
| `webgpu:fp16`, 10 calls/s offered | 10 | 30 | 600 | 35 ms (53-60) |
| `webgpu:fp16`, 5 calls/s offered | 5 | 15 | 300 | 70 ms (GPU clocks sag between calls) |
| `webgpu:fp16`, 1 call / 3 s | 0.33 | 1 | 20 | 156 ms cold (172-257) |
| `cpu:8`, back-to-back | 2.5-3.8 | 7-11 | 150-230 | 260-375 ms |
| `cpu:8`, anything above ~3 calls/s | saturates | | | queue grows without bound |

Batching questions into one call is the lever, most of all on CUDA where the per-call cost is almost flat:
447 questions/s at 10 per call vs 194 at 1 per call (WebGPU: ~120 vs ~47). Parallel callers do not add
throughput (one FIFO; each caller just sees `queueMs` grow) - the ~30 % the CUDA process used to gain from
pipelining the next request behind a 12 ms inference is gone now that the inference is 8 ms and the CPU side
(tokenising, stdio, padding) is the larger share.

Deployment overhead on top: HTTP `serve.mjs` +1-3 ms; `ask.mjs --sidecar` (new CLI process per call) ~140 ms
per call, so ~7 calls/s from a shell loop; `ask.mjs --local` one-shot pays the 1.5-2.5 s load every time.

### Findings that change how you should run it

1. **Hybrid CPU straggler mode (5x).** ORT's default pool (24 threads = all physical cores) splits every op over
   P- and E-cores and waits for the slowest thread. Depending on how Windows schedules the threads, 3 questions
   take either ~215 ms or ~1200 ms (p95 1174-1251 ms measured in 3 of 6 sessions; E-cores only: 1327 ms).
   Pinning the pool to the P-cores (`session.intra_op_thread_affinities`, exposed via `sessionOptions.extra`)
   costs ~10 % in the good case and removes the slow mode: p95 284-350 ms across all sessions.
   `pinToPCores: true` is the default for the router's CPU lanes; override the P-core count with `LAYA_PCORE_LOGICAL`.
2. **GPU idle clocks.** Between sporadic calls the RTX 4070 drops to ~225 MHz. With the optimised bundle a
   3-question call is 32 ms back-to-back and with 250 ms gaps, 81 ms after a 1 s pause, ~156 ms after 3 s; a
   single question after 3 s ~101 ms (before the graph optimisation: 48 / 91-140 / 170-235 / ~180). The CPU is
   insensitive to gaps (1 q ~105 ms), so a cold single question is now a tie the router settles from its EMA.
   A keep-alive (tiny GPU calls every 250 ms) was tried and does not help; left off.
3. **DirectML** failed on the exported graph in a `Reshape` node (`node_view`, HRESULT 80070057) at all
   graph-optimisation levels, on 1.30.0 and the 1.31 nightly. Cause: torch.export emits `Reshape(allowzero=1)`
   for every `view` and DML rejects that together with a `-1`; `tools/optimize_graph.py` clears the attribute
   (exact: no shape here contains a 0) and DML runs. It is the fastest EP for a single question (18-19 ms) but
   6-10x slower than WebGPU for batch > 1 even on a fixed shape (3 q 217 ms), so it is not a default lane.
   WebGPU is the NVIDIA path in `onnxruntime-node` on Windows (no CUDA EP is built for Windows).
4. **Inferences never overlap in one thread, and mixing lanes under load is a net loss.** `onnxruntime-node`
   1.30 runs `session.run()` synchronously on the JS thread (`dist/backend.js`: `setImmediate` + blocking run),
   so a CPU inference blocks the event loop - and every other lane in that thread - for its full duration:
   webgpu:fp16 3 q went from 49 ms alone to 321-479 ms p50 while `cpu:8` ran in the same thread, and policy
   `auto` (which spilled queued calls to the CPU) fell from 20 to 6.6 calls/s at 8 parallel callers
   (`experiments/interference.mjs`, `experiments/throughput.mjs`). Putting each lane in a worker thread makes
   them overlap (`experiments/worker-lanes.mjs`), but the 8 spinning P-core threads of the CPU lane still slow
   the GPU lane 1.5x (49 -> 76 ms), so a burst with every 4th call on the CPU runs at 16 calls/s against 20.6
   GPU-only. The router therefore keeps one FIFO for all lanes and never spills a burst to the CPU; the CPU lane
   is for sporadic single questions on a cold GPU and for machines without a usable GPU. No per-shape
   recompilation on WebGPU beyond the first call (~180 ms); unseen sequence lengths run at normal speed.
5. **Latency scales with tokens**: CPU 3 q goes 236 -> 400 ms for 77 -> 136 tokens per question. Longer option
   descriptions (see v2 wording) cost latency on every call.
6. **fp16 bundle**: half the VRAM and RSS, 0.8 s load; built by `tools/optimize_graph.py` (finding 10) with
   `npm run fp16:convert`, validated against the fp32 reference with `npm run fp16:check` (`experiments/ab.mjs`;
   add `--preset <yours>` to validate on your own eval set).
7. **The JS thread's core matters (1.6x).** It tokenises, drives the WebGPU dispatch and is one of ORT's
   intra-op workers (the pool is threads-1 pinned workers plus the caller). Windows parks it on an E-core for
   minutes at a time: whole sessions ran webgpu:fp16 3 q at 78 ms instead of 49 and `cpu:8` ~1.3x slower, and
   forcing the process onto the E-cores reproduces exactly that (71-80 ms). Node has no thread-affinity API, so
   `LayaRouter.create()` restricts the whole process to the P-cores via PowerShell (`pinProcessToPCores`, ~0.4 s
   in the background during the model load; `pinProcess: false` to opt out, `LAYA_PCORE_LOGICAL` for the
   layout). This is the likely source of the 1.2-1.6x session-to-session swings noted earlier.
8. **Lanes in worker threads (default since 0.5): same answers, free main thread, parallel loading, one hard
   rule.** Each lane's session lives in a `node:worker_threads` worker (`src/lane.mjs`); the round trip costs
   nothing measurable (cpu:8 3 q 306 vs 294 ms, answers identical), the main thread stalls 18 ms instead of
   1.1 s during a 10-question CPU call, and two lanes load in 2.2-2.3 s instead of 3.3-3.6 s because the
   synchronous session creation no longer serialises them. The hard rule: never `worker.terminate()` a worker
   with an inference in flight - ORT is running native code on that thread and the whole process dies with
   `0xC0000409`. `close()` asks the worker to release the session and exit by itself; an idle worker can be
   terminated safely (the router marks the lane gone, fails in-flight calls over to another lane and never
   picks it again - `test/router.test.mjs`).
9. **Where the time went: GPU->CPU round trips and Cast dispatches, not arithmetic.** 99 % of a `webgpu:fp16`
   call is inside `session.run` (tokenising, padding and softmax take 0.3-1.9 ms), and the ORT profiler showed
   what that was: the exported graph has 28 `IsNaN` nodes (the sdpa NaN guard after every attention softmax)
   that the WebGPU EP does not implement, so each call did **28 GPU->CPU copies + 28 copies back** - 28 pipeline
   drains - plus **196 `Cast` dispatches** from the first fp16 converter's fp32 islands (LayerNorm, Softmax).
   The arithmetic itself was 5-10 % of the RTX 4070's fp16 throughput. ORT's runtime fusions change nothing on
   this torch.export graph, and nothing downloadable helps: `onnxruntime-node` 1.30.0 is the newest release
   and its Windows build has no CUDA EP (see `docs/STATUS.md` decisions); WebGPU graph capture exists in the
   DLL but the Node binding rejects the option; all eight forwardable WebGPU EP options are within +-0.5 %
   (`results/webgpu-options-2026-09-23-summary.md`).
10. **Optimised graph (`tools/optimize_graph.py`, current `webgpu:fp16` bundle): 1.9-2.1x, same answers.**
    Exact rewrites - `IsNaN(x)` -> `Not(Equal(x, x))`, `Reshape allowzero=1` -> default, ORT's Gelu fusion
    (the only transformer pattern that matches this export; attention / RoPE / SkipLayerNorm do not) - and fp16
    without fp32 islands: 2101 -> 1753 nodes, 2 `MemcpyToHost` per call instead of 30. Measured with
    `experiments/ab.mjs` (interleaved, paired, bootstrap CI, against the fp32 reference): 0.453 [0.448, 0.457] of
    fp32 time on the PoC + smart-home workload vs 0.846 for the previous bundle (**1.87x**), 0.430 vs 0.916 on
    dev-request (**2.13x**); arg-max agreement with fp32 134/134 and 118/120 - identical to the previous fp16
    bundle, the two flips being near-ties (0.426 vs 0.419) - and identical accuracy. The same clean-up does
    nothing for the CPU lane (1.03 [0.95, 1.06]), which keeps the pinned, hash-verified HF bundle. Details:
    `results/graph-opt-2026-09-23-summary.md`.
11. **CUDA lane in a Python process: another 3x, same answers.** `onnxruntime-node` has no CUDA EP on Windows,
    but the Python wheel has, and Microsoft's release feed for the CUDA 13 build plus NVIDIA's wheel index are
    reachable where PyPI's file host is not (`npm run cuda:setup`, pinned versions, ~1.6 GiB). `tools/cuda_lane.py`
    holds one CUDA session and answers over stdio; `src/lane.mjs` presents it behind the same handle as the
    worker lanes, with a `RemoteSession` implementing the two methods `@receptron/laya` calls, so nothing of
    the sequence logic is duplicated. The CUDA EP is launch-bound: every call costs **6.0 ms + 21.6 us per
    token** (`experiments/cuda_graph_probe.py`) because ORT issues ~1400 CUDA kernels per forward pass at
    ~4 us each - paid on every call, not only the first - so it is almost flat in batch size: 8.9 / 12.1 /
    24.1 ms for 1 / 3 / 10 questions vs 21 / 32 / 83 on WebGPU (stdio round trip + tokenising 0.7-1.8 ms of
    that; 2 CPU-side threads on the P-cores - the unpinned default measured 20 ms). CUDA Graph capture, the
    standard fix for the floor, fails on this graph as exported (illegal memory access during capture) - solved
    with static-shape bucket graphs in finding 12.
    Paired ratio to fp32 (`experiments/ab.mjs`): **0.131 [0.130, 0.132]** vs 0.429 for WebGPU on PoC +
    smart-home (3.3x), 0.140 vs 0.423 on dev-request (3.0x); arg-max agreement with fp32 134/134 and 119/120,
    195/195 with the WebGPU lane over the full eval set (max |dp| 0.035), same accuracy. Throughput 66 calls/s
    (3 q, one caller), 83-86 with a queue, 409 questions/s at 10 per call. Killing the Python process fails
    in-flight calls over and marks the lane gone (`test/cuda-lane.test.mjs`); without the venv the lane is
    dropped at start-up and WebGPU serves. Details: `results/cuda-lane-2026-09-23-summary.md`.
12. **CUDA Graph replay on static bucket graphs: 2x on 1 question, 1.4x on 3, same answers.** The capture failure
    was one kernel: `experiments/cuda_capture_bisect.py` narrowed it to `GatherND` (the attention-mask broadcast
    with constant indices), whose ORT 1.30 CUDA implementation copies a host vector during `Compute` and so
    replays from a dead stack buffer. `tools/static_graph.py` makes a static-shape graph per (rows, tokens,
    options) bucket - Shape outputs baked, 202 constant nodes folded, that `GatherND` rewritten to a `Reshape` -
    and the capture works. `tools/cuda_lane.py` keeps the dynamic session for anything else and replays
    buckets: inputs padded up (mask 0), outputs sliced, one shared device copy of the weights (weights are graph
    inputs), buckets built from the shapes that arrive (a shape seen twice gets one, in an idle gap) plus a few
    eager ones before the lane reports ready. Two ORT facts dictated the threading: the captured graph belongs
    to the *calling thread* and any other thread's CUDA call during the (global-mode) capture fails with error
    900 - so a builder thread prepares sessions under a lock and the serving thread captures them, in idle gaps,
    each replay checked against the dynamic graph on real inputs before it serves (`results/cuda-graph-2026-09-23-summary.md`).
    Paired against the dynamic CUDA path (`experiments/ab.mjs`, 8 rounds, PoC + 40 dev-request items):
    **0.858 [0.848, 0.862]**; 1 / 3 / 10 questions **5.0 / 8.6 / 22.9 ms** vs 10.8 / 12.4 / 24.5, dev-request
    18.8 -> 16.0; 0.150 of fp32 time vs 0.213. Fidelity: 134/134 arg-max with the dynamic graph (max |dp| 0.006),
    195/195 on the eval set (0.0055), same accuracy. Throughput 194 / 114 / 45 calls/s for 1 / 3 / 10 questions
    (was 96 / 66-86 / 41). Costs: +1.4 s start-up (eager buckets), 28-240 MiB VRAM per bucket, and the first
    burst of a never-seen shape still runs dynamic. Off per call with `exec: { graph: false }`
    (`--no-graph`), per lane with `serve.mjs --cuda-graph off`; `routing.exec` reports `mode` and `bucket`.

## The execution-provider router (`src/ep-router.mjs`)

Lanes: `cuda`, `cuda:fp16` (Python process, finding 11), `webgpu`, `webgpu:fp16`, `dml`, `cpu` (16 threads pinned),
`cpu:8` (8 pinned), `cpu:24:nopin`, `cpu:auto`. Default `cuda:fp16` + `webgpu:fp16` + `cpu:8` (a lane that cannot
load is dropped; `cpu:8` equals `cpu` in latency at half the CPU share). Each lane is a separate `Laya` session in
its own worker thread (`workers: false` for in-thread sessions) or process; RAM ~1.6 GiB per CPU lane, ~1.1 / 0.6 GiB
+ VRAM per WebGPU lane, ~1.0 GiB + ~1.25 GiB VRAM for the CUDA process.

- **Probe**: every lane is loaded and must pass a real inference; a lane that cannot load (no venv for `cuda`,
  no GPU, DML on an unfixed graph) is dropped with a log line. A lane that throws later is quarantined for 60 s
  and the call is retried on the next-best lane; a lane whose worker or process exits is marked gone, its
  in-flight calls fail over, and it is never picked again.
- **Start-up**: lanes load in parallel; with `waitFor: "first"` (what `serve.mjs` uses) `create()` returns as soon
  as one lane has been probed and warmed (`warmup: { state, sizes }`), the others join while calls are served
  (`router.pendingLanes`, `router.ready`, `onLaneReady`). Probes and warm-ups run outside the FIFO so a lane
  joining late is never stuck behind a burst on the lane that is already serving.
- **Prediction**: EMA of ms-per-work-unit keyed by (lane, GPU state `hot` < 0.4 s / `warm` < 2 s / `cold`,
  question bucket 1 / 2-3 / 4-6 / 7-10 / 11+), seeded with priors from this machine; work = questions x
  estimated padded sequence length, so short and long states share estimates.
- **Queue**: one FIFO for all lanes (mixing lanes loses, finding 4). A call's prediction is `wait + own`: the
  wait is the predicted remaining time of everything queued (the same for every lane), `own` the lane's latency
  in the GPU state expected *when it starts* (`hot` if GPU work is queued ahead, never `cold` while anything is
  queued). The lane is bound when the call reaches the front of the queue - lanes that joined, died or got
  quarantined while it waited count (`routing.provisionalLane` says when that changed the choice; a start-up
  burst queued on the CPU lane moves to the GPU the moment it joins: 1.8 s instead of 5.7 s for 6 x 10 q).
  `routing.ms` is the inference alone; `routing.queueMs` the wait.
- **Contention**: background sampling of other processes' CPU (os.cpus deltas minus own usage) and GPU load
  (nvidia-smi utilisation weighted by SM clock, sampled while we are idle) inflates the affected lane's `own`.
- **Decision**: `auto` = fastest predicted, with 5 % exploration among lanes within 2x when nothing is queued
  behind the call; `prefer-gpu`, `prefer-cpu`, `min-cpu`; per call `{ lane }` or `{ deadlineMs }` (meet the
  deadline, queue included, with the least CPU share); `{ exec: { graph: false } }` runs a CUDA call on the
  dynamic graph instead of a CUDA Graph replay (finding 12). `routing.exec = { mode, bucket, remoteMs }` says what
  the process lane did; `router.detailedStats()` adds the lane's buckets, hits and VRAM.
- **Process affinity**: on Windows hybrid CPUs the process is restricted to the P-cores (finding 7; `pinProcess: false` to opt out).
- **Shutdown**: `close()` rejects new calls, lets the queue drain, then releases every session (a worker
  releases its own session and exits; see finding 8 for why it is never terminated under a running inference).

Measured behaviour (`router-demo.mjs`): bursts -> WebGPU (32-50 ms / 3 q); one question every 3 s -> whichever of
CPU (~105 ms) and cold WebGPU (~101 ms) the EMA currently rates lower; 10 questions after a pause -> WebGPU
(~300-400 ms vs CPU ~1100 predicted).
Warm up with a representative state: estimates depend on state length.

## Over-confidence, calibration, accuracy (`calibrate.mjs`)

**What the numbers mean.** Laya scores every option at its own `[MASK]` token and returns
`softmax(logit / T)`; `T` is a temperature from `laya_config.json`, one per (question type, option-count bucket):
`choice:3-5` 1.76, `noul:2` 1.98, `score:3-5` 1.25, ... `confidence` in the response is
`1 - normalised entropy` of that distribution, not a probability of being right.

**Calibrated** means: of all answers given with probability 0.8, about 80 % are correct. **Over-confident**
means the stated probability is higher than the hit rate. Temperature scaling only stretches or flattens the
distribution - the arg-max (accuracy) never changes - so it can fix over-confidence but not wrong answers.
The model card reports mean ECE 0.466 as shipped -> 0.081 after refitting `T` per bucket on in-domain data.

**Measured here** (65 labelled smart-home utterances, `data/smart-home-eval.mjs`, WebGPU; v1 = PoC wording):

| question | acc | mean conf raw (T=1) | mean conf shipped T | ECE raw -> shipped -> refit (LOO) | fitted T |
|---|---|---|---|---|---|
| intent (choice, 4) | 0.63 | 0.82 | 0.69 | 0.204 -> 0.147 -> 0.157 | 1.50 (pooled bucket) |
| should_execute (noul) | 0.66 | 0.89 | 0.80 | 0.264 -> 0.178 -> **0.066** | 4.18 |
| target_device (choice, 5) | 0.77 | 0.83 | 0.70 | 0.109 -> 0.124 -> 0.127 | 1.50 (pooled bucket) |

Reading it: raw the model says 82-89 % while being right 63-77 %; the shipped temperatures already remove
part of that. Refitting helps a lot where errors are spread across confidence levels (`should_execute`:
T 1.98 -> 4.18, ECE 0.178 -> 0.066) and not at all where they are systematic: for `intent`, answers given
with ~48 % confidence were right 13 % of the time (16 of 65) - these are *questions about a device*
("Which lights are on right now?") classified as `control_device` at p = 0.86. No scalar fixes that.

**Accuracy lever 1 - wording** (`data/question-variants.mjs`). Option texts are model input. v2 (explicit
descriptions) lifted `intent` 0.63 -> 0.72 and cut `should_execute` raw ECE 0.26 -> 0.14, but dropped
`target_device` 0.77 -> 0.66 ("Lights, lamps, brightness, light colour" attracted unrelated messages).
Effects are non-monotonic: measure every question. v3 = best per question is the shipped default:
intent 0.72, should_execute 0.66, target_device 0.77; `calibration/smart-home-v3.json` = `choice:3-5` 1.54, `noul:2` 1.93.

**Accuracy lever 2 - fine-tuning.** The model card is explicit: base checkpoints are "near chance" on its
typed-decisions benchmark zero-shot (0.36 vs 0.32 random); 0.77 comes from fine-tuning on that domain
(Kaggle notebook, 2xT4, 4-5 h). Laya is a fast base to specialise, not a zero-shot engine. Fine-tuning is the
PyTorch path (original repo), after which `export/export_onnx.py` produces a bundle this project loads via `modelDir`.

Caveats: 65 examples give noisy estimates (LOO fits ranged 1.28-1.66 for `choice:3-5`); the labels are mine;
`score` questions have no gold labels here. Replace the eval set with your traffic before trusting the tables.

## License

This project: [Unlicense](LICENSE) (public domain). Third-party components keep their own licenses: the
vendored `@receptron/laya` 0.1.2 is MIT (Receptron), the Laya model weights are Apache-2.0
(Convai Innovations), `onnxruntime-node` is MIT (Microsoft).

## Supply chain / safety

- npm on this machine routes through `packagefeedproxy.microsoft.io`, which does not carry `@receptron/laya`,
  and `registry.npmjs.org` is blocked at TLS. The exact published tarball 0.1.2 was fetched from jsDelivr,
  every file's SHA256 checked against jsDelivr's manifest **and** an independent unpkg copy (23/23 match),
  `dist/` audited against the GitHub source (1:1; only network call is `huggingface.co`), and installed from
  `vendor/receptron-laya-0.1.2.tgz` (`vendor/SHA256SUMS`). Runtime deps (`onnxruntime-node` 1.30.0,
  `@huggingface/tokenizers` 0.2.0) resolve through the proxy.
- Model pinned to HF commit `68f27dfe5a27a54fb2b1fefc432f43f972e90868` (immutable URLs), files verified by
  size + SHA256 against the repo's LFS oids (`src/laya-client.mjs`, `verify-model.mjs`). The library itself
  only compares byte sizes. After the first download loading uses `modelDir`: zero network at start-up.
- The fp16 bundle is derived locally and is not pinned to anything upstream.
- No global npm/git/python configuration was changed. Python deps live in `.venv/` (uv, cache in `.uv-cache/`).

## Original (PyTorch) vs this ONNX port

`mizorewww/laya-coreml` is an independent Apple Core ML / Neural Engine port - not usable on Windows. The
original is `convaiinnovations/laya` (`pip install laya`, Python 3.10+, torch 2.14, transformers 5).

| need | use |
|---|---|
| Node.js integration, small runtime (no Python / PyTorch / CUDA), deterministic pinned artefacts | this port |
| Multilingual or `typed-decisions` checkpoints (only the English root is published as ONNX) | original, or export them yourself with `export/export_onnx.py` |
| Fine-tuning (RLCD notebook), `predict_shortlist` for 50+ options, built-in question presets | original |
| Lowest possible latency on the RTX 4070 | probably original with CUDA (T4 CUDA already matches WebGPU here); not measured |
| Python-native pipeline | original |

## Known limits / next steps

- WebGPU EP is marked experimental by ORT; first call ~180 ms; sporadic calls pay GPU clock ramp-up.
- CUDA Graph buckets follow the traffic: the first burst of a never-seen shape runs on the dynamic graph (1 q
  10 ms instead of 5), the bucket is ready ~0.5 s later in sporadic traffic, ~1.5 s under continuous load (one
  call stalls ~250 ms while ORT creates the session). `--graph-buckets` names shapes to prepare before ready.
- `head_max_len` 192 tokens shared by all options of a question; < 20 options recommended; state truncated at 512.
- English checkpoint only; non-Latin scripts fail confidently (model card).
- Next: collect real traffic -> labels -> `calibrate.mjs`; iterate wording per question; if accuracy is not
  enough, fine-tune and re-export; consider exporting the multilingual checkpoint.
