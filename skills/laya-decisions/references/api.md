# Laya decisions - API and CLI reference

All commands run in the LocalLaya project directory (or through `scripts/laya.mjs`, which resolves it via
`LAYA_DIR` or its own location).

## Which path to call (measured on the reference machine, sidecar warm, CUDA lane)

| caller | path | per decision |
|---|---|---|
| a long-running process making many decisions | `node ask.mjs --start` once, then `POST /decide` over a kept HTTP connection | **13-15 ms** (12 ms inference) |
| a shell step / agent tool call | `node scripts/laya.mjs ...` (fast path: this Node process -> HTTP) | 80-90 ms |
| the same through `node ask.mjs --sidecar --json` | + preset loading and formatting modules | 87-127 ms |
| no background process wanted | `node ask.mjs --local` | 2.2-2.5 s (loads the model) |

Batch questions about one state into one call: 1 / 3 / 10 questions cost 9 / 12 / 24 ms on the CUDA lane.
Parallel callers do not add throughput (one queue; each waits `routing.queueMs`).

## Wrapper: `scripts/laya.mjs`

Same flags as `ask.mjs`. One-shot calls (text or `--state`, not `--local` / `--pretty`) are answered by the
wrapper itself over HTTP - one Node process, no `ask.mjs`; output is `ask.mjs --json`'s
`{ state, answers, usage, routing, backend: "remote" }`. `--status` / `--stop` / `--start`, `--local`,
`--pretty` and the REPL are delegated to `ask.mjs`; so is any failure to reach or start the sidecar (`ask.mjs`
then falls back to an in-process model with a warning on stderr). Exit codes as `ask.mjs`.

## CLI: `ask.mjs`

```
node ask.mjs [flags] "<text>"            one shot
node ask.mjs [flags]                      interactive REPL (/help)
```

| flag | meaning |
|---|---|
| `--preset <name>` | question set + state wrapper; default `smart-home`; built-ins + `presets/*.json` |
| `--questions <file\|json>` | JSON question set replacing the preset's questions (file path or inline JSON) |
| `--state <file\|json>` | JSON state used verbatim (text argument ignored; file path or inline JSON) |
| `--json` | one JSON object on stdout: `{ state, answers, usage, routing, backend }` |
| `--sidecar` | use / spawn the shared background instance (also `LAYA_SIDECAR=1`) |
| `--local` | load in this process (default without the flag/env; overrides `LAYA_SIDECAR`) |
| `--idle <dur>` | sidecar idle exit when this call spawns it: `30s`, `5m`, `1h`, `0` = never (default `LAYA_IDLE` or `5m`) |
| `--max-age <dur>` | sidecar recycles itself after this long (default `LAYA_MAX_AGE` or never); also only when spawning |
| `--port <n>` | sidecar port (default `LAYA_PORT` or 8787) |
| `--lanes a,b` | lanes to load: `cuda:fp16` (Python process, `npm run cuda:setup`), `webgpu:fp16`, `webgpu`, `cpu` (16 threads pinned), `cpu:8`, `cpu:auto`, `dml`; default `cuda:fp16,webgpu:fp16,cpu:8`, a lane that cannot load is dropped |
| `--lane <lane>` | force a lane for this call |
| `--calibration <file>` | temperature table; default `calibration/<preset>.json` if present |
| `--start` | ensure the sidecar is running; prints `{ url, pid, lanes, idleS, spawned }` |
| `--status` / `--stop` | show / stop the sidecar |
| `--no-color` | plain text output |

Exit codes: 0 ok; 1 error (message on stderr); 2 unknown or invalid preset; other = crash.
Stderr carries progress (`starting one ...`, `sidecar ready in 4.9 s`) and warnings; stdout only the result.

## HTTP: `serve.mjs` / sidecar

Base URL `http://127.0.0.1:8787` (or `--port` / `LAYA_PORT`). JSON everywhere.

| method + path | purpose |
|---|---|
| `GET /health` | `{ service: "laya", version, status: loading\|ready\|failed\|stopping, pid, port, sidecar, lanes, lanesLoading, workers, uptimeS, idleS, idleRemainingS, maxAgeS, gpuKeepAliveS, inFlight, sampling }` |
| `GET /presets` | `{ <name>: { description, source: built-in\|file, state (template with "$TEXT"), questions } }` |
| `GET /stats` | router statistics: per-lane calls / pending / EMA latencies, queue depth + predicted wait, external CPU/GPU load, GPU state |
| `POST /decide` | body below -> `{ answers, usage, routing, state, questions, preset, calibration }`; 503 `{ error: "loading" }` while the model loads (retry after `retryAfterMs`) |
| `POST /touch` | reset the idle timer -> `{ ok, idleRemainingS }` |
| `POST /shutdown` | graceful exit -> `{ ok, pid }` |

`POST /decide` body:

```json
{
  "text": "Charged twice, refund me today",      // wrapped by the preset's state template, OR
  "state": { "subject": "...", "body": "..." },   // used verbatim
  "preset": "triage",                              // default smart-home; also selects calibration/<preset>.json
  "questions": { "...": {} },                      // optional: replaces the preset's questions
  "lane": "cpu:8",                                 // optional: force a lane
  "deadlineMs": 150,                               // optional: meet the deadline with the least CPU share
  "calibration": { "temperature_by_options": { "noul:2": 1.9 } }   // optional: override the table for this call
}
```

Errors: 400 with `{ error }` for bad input (unknown preset, invalid questions, missing text/state, unknown lane);
413 body > 1 MB; 503 while loading; 500 otherwise.

## Result shape

```json
{
  "answers": {
    "department": { "type": "choice", "choice": "billing",
                    "probabilities": { "billing": 0.97, "technical": 0.01, "account": 0.01, "sales": 0.01, "other": 0.01 },
                    "confidence": 0.93, "rl_agent": { "act_probability": 1 } },
    "churn_risk": { "type": "noul", "noul": 0.76, "rl_agent": { "act_probability": 1 } },
    "urgency":    { "type": "score", "score": 1.85, "legend": { "0": "not urgent", "1": "soon", "2": "urgent", "3": "critical" },
                    "probabilities": { "0": 0.02, "1": 0.14, "2": 0.80, "3": 0.04 }, "confidence": 0.6, "rl_agent": { "act_probability": 1 } }
  },
  "usage": { "input_tokens": 330, "output_tokens": 0 },
  "routing": { "lane": "webgpu:fp16", "ms": 154, "queueMs": 0, "n": 5, "gpuState": "hot", "predictedMs": 160,
               "ownMs": 160, "waitMs": 0, "reason": "fastest predicted (...)", "alternatives": [ { "lane": "cpu:8", "predictedMs": 330 } ] }
}
```

- `choice.probabilities` sum to 1 over the options; `choice` is the arg-max.
- `noul` is P(true). `score` is the expectation over levels (0-based); `legend` maps index -> level text.
- `confidence` = 1 - normalised entropy of the distribution (0 = uniform, 1 = certain). Not P(correct).
- `routing.ms` is inference time on the chosen lane; `queueMs` time spent waiting behind other callers. Calls
  are served one at a time (one queue for all lanes): ~66 calls/s for 3 questions on the CUDA lane (83-86 with
  a queue), ~30 on WebGPU, ~3.5 on the CPU; parallel callers only wait longer. Batch questions into one call
  rather than calling in parallel. The lane is chosen when the call reaches the front of the queue;
  `routing.provisionalLane` appears when that differed from the lane expected at enqueue time (e.g. a faster
  lane joined during start-up). `routing.gpuState` (`hot` / `warm` / `cold`) explains slower sporadic calls.

## Lifecycle of the sidecar

- Started by `ask.mjs --sidecar` / `--start` as `node serve.mjs --sidecar --idle <dur> [--max-age <dur>]`,
  detached, hidden window, log appended to `.laya/sidecar-<port>.log` in the project.
- Binds the port before loading; racing launchers -> one instance (loser exits 3). Lanes (default `cuda:fp16`
  in a Python process, `webgpu:fp16` and `cpu:8` in worker threads) load in parallel; `status: ready` comes as
  soon as the first lane is probed and warmed (~2.4 s after spawn), the others join ~1-1.5 s later
  (`/health.lanesLoading`). Callers during the load wait for `ready`; a
  `/decide` that forces a lane still loading gets `503 { retryAfterMs }` (the CLI client retries).
- Exits on its own after `idle` without `/decide` or `/touch` calls, with nothing in flight, and at `maxAge`
  if set. RAM and VRAM are released; the next call spawns a new one.
- GPU keep-alive: for `--gpu-keepalive` (default 30 s) after each call a tiny GPU call every 500 ms keeps the
  GPU awake, so a call arriving seconds later takes ~45 ms instead of ~175 ms (median; 1-4 W meanwhile). Set
  `--gpu-keepalive 0` to disable, longer for longer pauses between calls.
- Inferences run in worker threads: `/health`, `/stats` and new requests are answered while a call runs.
- Background load sampling (`nvidia-smi`, CPU) pauses after 10 s idle and resumes on the next request.
- Presets and calibration files are re-read when they change on disk.
- `/health` from a foreign service (no `service: "laya"`) is never stopped; the CLI falls back to in-process.

## Environment variables

| var | effect |
|---|---|
| `LAYA_DIR` | project directory for `scripts/laya.mjs` |
| `LAYA_SIDECAR=1` | `ask.mjs` uses the sidecar by default (`--local` overrides) |
| `LAYA_PORT` | sidecar port (default 8787) |
| `LAYA_IDLE` | default idle exit (default `5m`) |
| `LAYA_MAX_AGE` | default max age before the sidecar recycles itself (default never) |
| `LAYA_GPU_KEEPALIVE` | default GPU keep-alive window of `serve.mjs` (default `30s`; `0` = off) |
| `LAYA_LANES` | default lanes for `serve.mjs` (default `cuda:fp16,webgpu:fp16,cpu:8`; `cuda:fp16` needs the Python venv from `npm run cuda:setup` and is dropped otherwise) |
| `LAYA_PYTHON` | Python with onnxruntime-gpu for the CUDA lane (default `<project>/.venv/Scripts/python.exe`) |
| `LAYA_CACHE` | model cache directory (default `<project>/models`) |
| `LAYA_PCORE_LOGICAL` | number of logical P-core processors for CPU pinning (default 16) |
