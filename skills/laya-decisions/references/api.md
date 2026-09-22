# Laya decisions - API and CLI reference

All commands run in the LocalLaya project directory (or through `scripts/laya.mjs`, which resolves it via
`LAYA_DIR` or its own location).

## CLI: `ask.mjs`

```
node ask.mjs [flags] "<text>"            one shot
node ask.mjs [flags]                      interactive REPL (/help)
```

| flag | meaning |
|---|---|
| `--preset <name>` | question set + state wrapper; default `smart-home`; built-ins + `presets/*.json` |
| `--questions <file>` | JSON question set replacing the preset's questions |
| `--state <file\|json>` | JSON state used verbatim (text argument ignored) |
| `--json` | one JSON object on stdout: `{ state, answers, usage, routing, backend }` |
| `--sidecar` | use / spawn the shared background instance (also `LAYA_SIDECAR=1`) |
| `--local` | load in this process (default without the flag/env; overrides `LAYA_SIDECAR`) |
| `--idle <dur>` | sidecar idle exit when this call spawns it: `30s`, `5m`, `1h`, `0` = never (default `LAYA_IDLE` or `5m`) |
| `--port <n>` | sidecar port (default `LAYA_PORT` or 8787) |
| `--lanes a,b` | lanes to load: `webgpu:fp16`, `webgpu`, `cpu` (16 threads pinned), `cpu:8`, `cpu:auto`, `dml` |
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
| `GET /health` | `{ service: "laya", version, status: loading\|ready\|failed\|stopping, pid, port, sidecar, lanes, uptimeS, idleS, idleRemainingS, inFlight, sampling }` |
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
  are served one at a time (one queue for all lanes): throughput is ~20 calls/s for 3 questions on the GPU
  lane whatever the number of parallel callers; parallel callers only wait longer. Batch questions into one
  call rather than calling in parallel.

## Lifecycle of the sidecar

- Started by `ask.mjs --sidecar` / `--start` as `node serve.mjs --sidecar --idle <dur>`, detached, hidden
  window, log appended to `.laya/sidecar-<port>.log` in the project.
- Binds the port before loading; racing launchers -> one instance (loser exits 3). Callers during the load
  wait for `status: ready` (about 4-6 s with the GPU + CPU lanes).
- Exits on its own after `idle` without `/decide` or `/touch` calls, with nothing in flight. RAM and VRAM are
  released; the next call spawns a new one.
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
| `LAYA_LANES` | default lanes for `serve.mjs` (default `webgpu:fp16,cpu:8`) |
| `LAYA_CACHE` | model cache directory (default `<project>/models`) |
| `LAYA_PCORE_LOGICAL` | number of logical P-core processors for CPU pinning (default 16) |
