# demo/ - use Laya decisions from your own code, today

Everything here runs against the project as it is (`npm run cuda:setup` optional, see the README) and uses
only the public surfaces - the sidecar's HTTP API and the router - so it keeps working while the internals
change. Copy `laya.mjs` into your own project (set `LAYA_DIR` to this checkout) and start from an example.

```powershell
node demo/cli.mjs --preset triage "Charged twice. Refund today or I cancel."      # one decision
node demo/cli.mjs --preset triage --file demo/data/tickets.json                    # a file of inputs, one line each
node demo/examples/triage-inbox.mjs                                                 # route tickets to teams
node demo/examples/gate-prompts.mjs                                                 # allow / review / block prompts
node demo/examples/route-requests.mjs                                               # which tool handles a dev request
node demo/examples/own-questions.mjs                                                # your own questions on JSON states
node demo/examples/many-calls.mjs                                                   # what each call costs, by call pattern
npm run demo                                                                        # all of the above
```

## `laya.mjs` - the library

Choose just `model` and `mode` for automatic configuration:
`createLaya({ model: "multilingual", mode: "sidecar" })` or
`createLaya({ model: "multilingual", mode: "local" })`. Library instances are reusable by default;
`lifetime: "one-shot"` chooses the startup profile for a private isolated call. The CLI infers that
profile automatically for single local inputs, and uses the reuse profile for batches.
One-shot Multilingual auto uses eager execution to skip TileLang startup; reusable sessions enable its
available fused kernels. `acceleration: "on"` opts into kernel initialization for a one-shot.
`graphs: "auto"|"on"|"off"` overrides graph policy; other explicit model/device/runtime options still win.

```js
import { createLaya, top, gate } from "./demo/laya.mjs";   // or from your copy, with LAYA_DIR set

const laya = await createLaya();                          // mode "auto"
const r = await laya.decide("Charged twice, refund me", { preset: "triage" });
r.answers.department.choice                               // "billing"
r.answers.department.probabilities                        // { billing: 0.97, technical: 0.01, ... }
top(r.answers.department)                                 // { label: "billing", p: 0.97 }
gate(r.answers.department)                                // "act" (>= 0.8) | "ask" (>= 0.55) | "unsure"
r.routing                                                 // { lane: "cuda:fp16", ms: 18, queueMs: 0, ... }

const many = await laya.decideMany(["...", "...", { state: { subject: "...", body: "..." } }], { preset: "triage" });
await laya.close();
```

| mode | what happens | first call | per call |
|---|---|---|---|
| `auto` (default) | shared sidecar, spawned if not running; falls back to `local` if the port is foreign or the spawn fails | 2.2-2.5 s once per idle period, then reused by every process | inference + ~1.5 ms (one keep-alive connection) |
| `sidecar` | the sidecar or an error | same | same |
| `local` | the model in this process (no background process; N processes = N copies) | 2-3 s every process | inference |
| `http` | a server you started yourself (`url`); never spawns | - | inference + ~1.5 ms |

Options: `{ mode, model, device, maxLen, readout, acceleration, graphs, lifetime, timeoutMs, port, url, idle, maxAge, lanes, calibration, log }`. `decide(input, opts)`: `input` is a string
(wrapped by the preset's state template) or `{ state }`; `opts` = `{ preset, questions, lane, policy, deadlineMs,
calibration, exec, model, device, maxLen, readout, acceleration, graphs }`, all optional. `decideMany(inputs, opts, { concurrency })` keeps the input order.

Questions accept `type: "predicate"` as an alias for `noul`, `choices` for a choice question's `criteria`,
and `levels` for a score question's `criteria`. Answers and echoed question definitions use the canonical
schema. See [the request guide](../docs/REQUESTS.md) and [complete ticket JSON](data/ticket-request.json)
for reading a file and passing its state/questions to the library or sidecar.
For a complete single-request JSON file, use `node ask.mjs --file demo/data/ticket-request.json --sidecar --json`.
`demo/cli.mjs --file` accepts batch arrays or `{ items: [...] }`.

## Vega-U and multilingual

After `npm run python:setup` and `npm run models:download -- --model all`:

```js
const d = await createLaya({ mode: "sidecar", model: "vega", device: "cuda" });
try {
  const a = await d.decide({ state: longDocument }, { questions });
  const b = await d.decide("تم خصم المبلغ مرتين", { model: "multilingual", preset: "triage" });
  const c = await d.decide("Turn off the lights", { model: "laya" });
} finally { await d.close(); }
```

`model` is `laya` (default, original ONNX), `vega` (73,728-token limit), or `multilingual` (8,192).
`device` is `auto`, `cpu`, or `cuda`. `maxLen` sets a Python model's context budget; Vega also accepts
`readout: "auto"|"engine"`. CLI uses `--model`, `--device`, `--max-len`, `--readout`, and
`--text-file` for a whole UTF-8 document. The mode table's timing numbers describe original ONNX Laya.

Both Python models accept `acceleration: "auto"|"on"|"off"` (`--acceleration` in the CLI).
`npm run vega:setup` installs Vega's optional kernels; `npm run multilingual:setup` installs the optional
TileLang runtime for Multilingual. On tries the available kernels regardless of lifetime; off restores the
original eager/reference path. `routing.exec.acceleration` reports the actual kernels/backend and graph use.
Fresh Python processes still pay runtime import/model-loading costs. The first successful result includes
`routing.initialization` with phase timings; a `log` callback receives progress on the local path.

The sidecar loads one model family at a time; model/device/context switches release and reload its backend.
Calibration is model-scoped, and Python results report actual truncation and peak allocated VRAM.
`models()` / `/health.models` report selection and available profiles. See [the full guide](../docs/MODELS.md).

## Which lane / mode handles a call

For `model: "laya"`, the sidecar loads three lanes - `cuda:fp16` (Python process, if `npm run cuda:setup`
was run), `webgpu:fp16`, `cpu:8` - and its router picks, per call, the lane with the lowest predicted latency
including the queue and the GPU's thermal state; `routing.lane` / `routing.reason` say what it did. Measured
on the reference machine (RTX 4070), 3 questions: CUDA ~9 ms (~12 on its dynamic graph), WebGPU ~32 ms, CPU
~270 ms; a cold single question is a three-way tie around 100 ms, which the router settles from what it has seen.

The CUDA lane replays a captured CUDA Graph for shapes it has seen before (`routing.exec.mode: "graph"`,
`bucket: [rows, tokens, options]`) and falls back to its generic graph for the first two calls of a new shape
or for large calls (`mode: "dynamic"`); it remembers the shapes across restarts, so after the first session your
presets replay from the sidecar's first idle second. Answers agree to |dp| < 0.01; the difference is speed
(1 question 5 vs 10 ms). `routing.exec.stallMs`, when present, is time the call waited for a bucket being built.

Overrides, per call: `lane` (`"webgpu:fp16"`, `"cpu:8"`, ...), `policy` (`"prefer-gpu"`, `"prefer-cpu"`, `"min-cpu"`),
`deadlineMs` (meet it with the least CPU share), `exec: { graph: false }` (cuda lane: skip CUDA-graph replay).
CLI: `--lane`, `--policy`, `--deadline`, `--no-graph`.

## What a call costs, and how to call cheaply (`many-calls.mjs` measures it on your machine)

| pattern | per decision (5 questions) |
|---|---|
| one process, one connection, sequential | inference + ~1.5 ms (24 ms here) |
| 4-8 callers in parallel | the same throughput, each waits its turn (`routing.queueMs`) |
| a new `node demo/cli.mjs` per call | ~85 ms (Node start-up + imports) |
| `--mode local` | inference, after a 2-3 s load |

Rules: hold a connection when you make many calls; put all questions about one text into one call (10
questions cost twice a single one, not ten times); do not spawn a process per decision.

## Files

| file | purpose |
|---|---|
| `laya.mjs` | the library: `createLaya`, `top`, `gate`, `summarize`, `readInputs` |
| `cli.mjs` | one input, a file, or `--health` / `--stats` / `--presets`; `--json` for machine output |
| `examples/*.mjs` | one workflow each; every example accepts `--mode local` and a JSON file of inputs |
| `data/*.json` | sample tickets, prompts, dev requests, emails + an ad-hoc question set |

The presets used here: `triage`, `guard` and the email questions are **unmeasured** starting points; `dev-request`
and `smart-home` were measured (task 0.75 / language 0.85; intent 0.72). Before you gate on a question, label
30-60 real inputs and run `node calibrate.mjs --preset <name> --eval <file>` - see `skills/laya-decisions/references/presets.md`.
