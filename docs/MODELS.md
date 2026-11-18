# Models and execution modes

LocalLaya exposes three explicit model choices through `ask.mjs`, `serve.mjs`, the browser page,
`demo/laya.mjs`, `demo/cli.mjs`, and the skill wrapper.
All three accept the same [request/question schema](REQUESTS.md), including the `predicate`, `choices`
and `levels` aliases.

| model | checkpoint | context limit | runtime |
|---|---|---|---|
| `laya` (default) | English ModernBERT-large, 421M | 512 | existing `@receptron/laya` 0.1.2 + ONNX lanes |
| `vega` (`vega-u`) | Vega-U engine + Qwen3.5-0.8B reader | 73,728 | Python Laya, optional FLA/Triton + head CUDA Graphs |
| `multilingual` (`laya-multilingual`) | mmBERT-base, 322M | 8,192 | Python Laya, optional TileLang CUDA kernels/graphs |

The limit includes each question and its options. This project's multilingual profile explicitly enables
8,192 tokens; upstream's checkpoint configuration defaults to 1,024. Vega's configured limit is a capacity,
not a latency promise or a guarantee of document accuracy. Both new models use their own tokenizers.

## Setup

Requires Node 20+, Python 3.12 and `uv`. Run from the project directory:

```powershell
npm run python:setup                         # create .venv-laya, install the pinned runtime; CUDA selected by uv
npm run python:check                         # versions, GPU, and a real CUDA tensor operation
npm run vega:setup                           # optional fused reader kernels (Triton Windows + FLA)
npm run vega:check                           # compile/run both fused operations on this GPU
npm run models:download -- --model vega       # engine (~66 MB) + reader (~1.75 GB) + tokenizer
npm run models:download -- --model multilingual # weights (~644 MB) + tokenizer (~34 MB)
npm run multilingual:setup                   # optional fused Multilingual CUDA runtime (after model download)
npm run multilingual:check                   # real Arabic decision through that CUDA path
npm run models:download -- --model all --check  # offline digest verification
```

CPU-only installation: `npm run python:setup -- --torch-backend cpu`. To choose another supported PyTorch
wheel index, use `--torch-backend <uv-backend>`. `--device cuda` requires a CUDA-enabled installation.
On the development machine, the default PyPI file host fails TLS negotiation; this mirror worked:

```powershell
npm run python:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
npm run vega:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
npm run multilingual:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

The Laya and vegaml wheel hashes on that mirror matched PyPI's published hashes. TLS verification stays
enabled. Package versions and model revisions are recorded in `data/python-models.json`; PyTorch's CUDA
build suffix depends on `--torch-backend` (tested here: 2.15.0+cu132).

Python packages live in `.venv-laya`, model files in `models/python`, and the uv cache in `.uv-cache`.
The existing ONNX `.venv`, bundles, graph buckets and calibration files have their original roles.
`LAYA_MODEL_PYTHON` selects another prepared Python executable; `LAYA_PYTHON` is still the ONNX CUDA worker.
All inference loads local paths with Hugging Face / Transformers offline mode enabled. Downloads happen
only through the setup command. The preparation tool verifies SHA256 for LFS data and Git blob SHA1 for
small files against the pinned Hub revision.

Laya normalizes mmBERT's tokenizer config on load. The adapter supplies a derived `runtime-laya-<version>`
view with a normalized config and hard-linked weights/tokenizer data, so the pinned files remain verifiable.
On filesystems without hard links, those data files are copied instead.

## Choose independently

### Automatic configuration: model + local/sidecar

The shortest commands are `node ask.mjs --model vega --local "..."` and
`node ask.mjs --model vega --sidecar "..."`. Substitute `laya` or `multilingual` as needed.
No device, acceleration, context or graph flags are required.

| model | local one-shot | sidecar / foreground server / REPL / persistent library |
|---|---|---|
| `laya` | probe CUDA FP16, then WebGPU FP16, then CPU:8; stop after the first usable lane; no CUDA Graph setup | load the normal three lanes, serve from the first ready lane; use routing, graph reuse and GPU keep-alive |
| `vega` | CUDA when available, otherwise CPU; available fused kernels and bounded long prefills; eager decision head | same device/kernel selection, plus validated head graph replay |
| `multilingual` | CUDA when available, otherwise CPU; eager execution, skipping TileLang initialization and graph capture | available TileLang kernels, plus repeat-aware graph caching |

These are benchmark-informed defaults: they favor total time to the first answer for an isolated call and
amortized latency for a reused model. Multilingual's conservative one-shot policy skips TileLang for any
input length; `--acceleration on` opts into its setup cost for a workload that benefits from it. Optional
kernel compilation may still cost time on first use. Full context defaults
are 512 / 73,728 / 8,192, including question overhead. Calibration remains model-scoped.

`routing.profile` reports the lifetime (`one-shot` or `persistent`), acceleration/graph policy and reason; `routing.exec`
reports what the backend actually did. Automatic Python device selection may retry a CUDA allocation failure
on CPU. Explicit `--device cuda` remains strict, and model/checkpoint errors are surfaced.

Explicit flags override the defaults. `--acceleration on` tries the available fused kernels even for a
one-shot, keeping the normal capability checks and eager fallback. It does not force graph capture.
`--graphs on` enables graph reuse (and its required backend) for a one-shot, while `--graphs off`
disables it for a reusable session. Backends still apply their normal capture policy (for example,
Multilingual captures a newly repeated shape). Python `--acceleration off` uses eager execution;
combining it with `--graphs on` is an error. Original Laya also retains `--no-graph` for a per-call bypass.
Explicit `--lanes` loads the specified set, and a startup `--lane` alone loads that lane. A routing policy
or deadline keeps its candidate lanes available instead of choosing a single one at startup.

The demo CLI infers one-shot versus batch use. The Node library defaults to a reusable model; use
`createLaya({ model: "vega", mode: "local", lifetime: "one-shot" })` for an isolated call. A remote server
always owns a reusable session regardless of how short-lived its client is.

### Cold startup versus warm inference

The millisecond performance tables measure an already-loaded model. Each local one-shot starts Python,
imports PyTorch/Laya, loads the checkpoint, then answers and exits. A fresh process still takes seconds
with eager execution. A sidecar's first request pays startup too; later requests and a local REPL reuse it.

Python startup now reports phases on stderr: runtime imports, CUDA initialization, checkpoint loading,
backend selection, and the first inference. A five-second heartbeat also reports that loading is in progress.
`--json` keeps stdout as one result object. Its first successful decision includes `routing.initialization`:

- `processMs`: Node's time from starting the Python worker until ready, including process startup;
- `totalMs`: worker-side startup time;
- `phases`: `{ stage, ms, status }` records for imports/loading/accelerator setup.

`routing.ms` measures the decision after readiness, including first-use kernel work. Startup metadata is
reported once per loaded backend and remains available through `/stats`. The reproducible
`experiments/cold-start.mjs` measures time to the stdout answer separately from shutdown time.
On the reference machine, the paired short-input median fell from 9.60 s with TileLang to 5.96 s with
one-shot auto/eager. See the [cold-start report](../results/multilingual-cold-start-2026-11-17-summary.md).

### Explicit controls

1. **Model:** `--model laya|vega|multilingual`.
2. **Call path:** `--local` / `--mode local`, or `--sidecar` / `--mode sidecar`.
3. **Device:** `--device auto|cpu|cuda`. Python `auto` selects CUDA when available, otherwise CPU.
   For original Laya, `auto` retains the EP router; `cpu` or `cuda` selects a matching loaded lane.
4. **Context:** `--max-len <tokens>` for the Python models, up to their profile's limit.
5. **Vega readout:** `--readout auto|engine`. Both use the shipped engine here: this adapter currently loads
   no fitted TTT heads. These modes are independent of CUDA versus CPU and local versus sidecar.
6. **Python model acceleration:** `--acceleration auto|on|off`. Auto follows the model/lifetime table above.
   On tries the available fused kernels even for a one-shot; off selects the original eager/reference forward.
   CPU and unavailable optional kernels use eager/reference execution. Graph policy is a separate control.
7. **Graph reuse:** `--graphs auto|on|off`, for all models. Auto follows the lifetime profile above; graph
   capture is available on supported CUDA backends. `/graphs` changes it in the REPL.

```powershell
node ask.mjs --model laya --local --lanes webgpu:fp16 --no-color "Turn on the TV"
node ask.mjs --model laya --local --device cpu "Turn on the TV"
node ask.mjs --model vega --local --device cuda --readout engine --preset triage "Charged twice, refund me"
node ask.mjs --model vega --sidecar --device cpu --preset triage "Charged twice, refund me"
node ask.mjs --model multilingual --sidecar --device cuda --max-len 8192 --preset triage "تم خصم المبلغ مرتين"
node ask.mjs --model vega --sidecar --text-file document.txt --questions questions.json --json
```

`--text-file` is one UTF-8 document, passed through the preset's state template. `--state` accepts JSON as
before. An explicit `--mode sidecar` fails if the sidecar cannot start; the older `--sidecar` flag retains
its local fallback for connection/startup failures. An inference failure never changes the selected model.

`--file ticket.json` on `ask.mjs` reads a complete JSON request (`state`/`text`, `questions`/`preset`, optional
model/runtime controls). Explicit CLI flags override its settings; file settings override CLI environment
defaults. For example: `node ask.mjs --model multilingual --sidecar --file demo/data/ticket-request.json --json`.
See [request-file input](REQUESTS.md#run-a-complete-json-file-from-the-cli).

Start a REPL with `node ask.mjs --sidecar --model vega`. Use `/model laya`, `/model multilingual`,
`/device cpu`, `/device cuda`, `/context 4096`, `/readout engine`, `/acceleration off`, `/graphs auto`, `/models`, and `/again`.
The new selection takes effect on the next question. `/model` resets the context to that model's default
and clears a forced ONNX lane. `/device` also clears a forced lane.

`--lanes`, `--lane`, `--policy`, `--no-graph` / `exec.graph`, and `deadlineMs` control original ONNX Laya.
Sending them with a Python model is an error. If `--lanes` is explicitly supplied, a requested legacy
`--device` must have a matching lane in that set.

## Sidecar and HTTP

```powershell
node serve.mjs --model vega --device cuda --port 8787
# Or detached, with idle exit:
node ask.mjs --start --model vega --device cuda --idle 10m
```

`GET /models` lists model choices, defaults, active selection and an in-progress switch. `/health.models`
contains the same information. The browser page has model, device, context and readout controls.

```json
{
  "model": "vega",
  "device": "cuda",
  "maxLen": 73728,
  "readout": "engine",
  "acceleration": "auto",
  "graphs": "auto",
  "preset": "triage",
  "text": "Charged twice. Please refund the duplicate."
}
```

POST that body to `/decide`. Change `model` to `multilingual` and omit `maxLen` and `readout` to use its
8,192-token default; `acceleration` applies to both Python models. For `laya`, omit all three to use the
original bundle and its automatic EP selection.
Omitted controls inherit the server defaults, except that changing models resets model-specific context
and readout defaults. `LAYA_MODEL` / `LAYA_DEVICE` set defaults for the CLI, demo library and new server.

One model family is resident per server. Requests execute FIFO. Switching model, device, acceleration, effective graph policy or Python context
budget waits for earlier requests, closes that backend, and loads the requested one. Changing Vega readout
reuses the loaded weights. Original Laya retains its multiple EP lanes while it is the active model.
The process release also frees PyTorch's CUDA allocator. For simultaneous resident models, use separate
sidecar ports and account for their combined memory.

A client targeting a pre-0.7 sidecar refuses Python-model requests instead of letting that server silently
ignore the selection. Restart that server or select another port. `--start` configures a new sidecar; when
attaching to an existing one, its returned `models` field describes that server, and decision requests
still carry their own explicit model selection.

## Node library

```js
import { createLaya } from "../demo/laya.mjs";

const d = await createLaya({ mode: "sidecar", model: "vega", device: "cuda" });
try {
  const a = await d.decide({ state: longDocument }, { questions });
  const b = await d.decide("تم خصم المبلغ مرتين", { model: "multilingual", preset: "triage" });
  const c = await d.decide("Turn off the lights", { model: "laya" });
  console.log(a.routing.model, b.routing.device, c.routing.lane);
} finally {
  await d.close();
}
```

The library supports `mode: "auto"|"sidecar"|"local"|"http"`; HTTP mode takes an `http://host:port` `url`.
The low-level `loadLaya`, `createDecider` and `LayaRouter` APIs are ONNX-specific; model selection uses
`createLaya` or `ModelRouter`.
Each call can override `model`, `device`, `maxLen`, `readout`, `acceleration` and `graphs`. Explicit graph on/off
selection and `acceleration: "on"` need a 0.10 server. The client and Python worker default to a
ten-minute decision timeout for long documents; library callers can set `timeoutMs`. `close()` in local
mode drains accepted work and releases the model; a remote client's `close()` releases its HTTP connection.
On Windows a timed-out Python worker is stopped as a process tree, including the interpreter behind the
virtual environment's launcher. HTTP shutdown waits for accepted long-document calls to finish.

## Results and calibration

- `routing.model`, `backend`, `device`, and `contextTokens` identify what ran. `routing.ms` measures the
  decision; `queueMs` measures waiting, and `loadMs` appears when a decision triggers a backend switch.
- `routing.profile` explains the automatic runtime policy. `graphReplayEnabled` in the Python execution
  metadata reports whether that backend can use graph replay for the call.
- Python `routing.exec.acceleration.requested` preserves `auto|on|off`; `selected` is the resolved on/off
  policy. The backend/reader fields report actual execution, including eager fallback on CPU or unavailable
  kernels. First-use startup timings are in `routing.initialization` (see above).
- Python `routing.exec` includes `mode: "torch-eager"|"torch-accelerated"`, `device`, `requestedDevice`, `maxLen`, `remoteMs`,
  `peakVramMiB`, and `cpuFallback`. Multilingual Laya may retry a GPU OOM on CPU; that execution is reported
  as CPU. PyTorch peak allocated bytes exclude the CUDA context and other applications.
- Python `usage` includes `state_tokens`, `state_tokens_dropped`, `truncated`, `truncated_questions` and
  `context_tokens`. `input_tokens` sums the question rows; it can exceed the context limit even when each
  row fits. The Vega adapter measures actual prompt spans to correct the upstream sum-based truncation flag.
- `score` always means expected ordinal level, as in original LocalLaya. Vega's original argmax is retained
  as `selected_level`; its `expected_score`, prediction sets, abstention and routing fields are preserved.
- `laya` uses existing `calibration/<preset>.json`. `multilingual` uses
  `calibration/multilingual/<preset>.json` if present, otherwise its shipped temperatures. A server
  `--calibration` file belongs only to its startup model. Vega uses its checkpoint calibration and rejects
  LocalLaya temperature-table overrides. `calibration: null` explicitly selects shipped calibration.
- A larger context does not imply better decisions on every preset. Existing calibration/evaluation results
  belong to the English ONNX checkpoint; new models need evaluation on the intended domain.

## Verification and measurement

```powershell
npm test                     # model-selection, lifecycle/protocol, calibration-isolation and existing unit tests
npm run test:models          # real CPU/CUDA, score/truncation, Unicode, switching, HTTP, CLIs, skill wrapper
npm run test:profiles        # minimal model + mode commands, device fallback and explicit overrides
npm run test:questions       # question aliases, JSON-file input, HTTP/CLI/library equivalence
npm run test:all             # original router/CUDA/sidecar/demo regression suite plus unit tests
node experiments/model-context.mjs --models vega,multilingual --device cuda --words 0,1024,4096,7800
node experiments/model-context.mjs --models vega --device cuda --words 16384,32768,70000
node experiments/cold-start.mjs --rounds 3  # fresh-process Multilingual auto vs explicit acceleration=on
```

The context script records actual token counts, times, peak PyTorch VRAM and predictions. Its repeated
neutral filler plus a final refund request checks capacity and end-to-end execution, not document accuracy.

Original eager measurements on the RTX 4070: **[2026-11-17 context results](../results/model-context-2026-11-17-summary.md)**.
Vega completed 70,014 state tokens without truncation, but that call took about 143 seconds and peaked at
10,073 MiB of PyTorch allocations. Its 16K/32K checks took about 2.3/6.8 seconds. Multilingual completed
7,814 state tokens in about 293 ms. These measurements favor testing the required budget rather than
assuming the largest context is the best default for every workload.

## Vega CUDA acceleration (0.8)

`npm run vega:setup` installs the optional packages pinned in `data/python-models.json`: FLA 0.6.0,
einops 0.8.2, and Triton Windows 3.9.0.post30 (Triton 3.9.0 on Linux). The Windows wheel includes its
compiler/toolchain. `vega:check` proves the gated-delta and convolution kernels execute on the GPU.
Kernels compile locally and cache under `.laya/triton`; a new kernel shape can make its first call slower.

With `--acceleration auto` (default) or `on`, Vega uses these fused reader kernels when available. Reusable
sessions (or explicit `--graphs on`) also enable a bounded eight-entry CUDA Graph cache for the small
decision head. Each head graph is checked against the eager
head before serving. Failed shapes use eager execution and are listed in the result metadata. Graphs are
per process and are rebuilt after a model reload; the Triton disk cache survives it. If optional packages
are missing or differ from the pins, the reader uses reference kernels and reports why. CPU uses reference
kernels. `--acceleration off` disables both changes for comparisons.

`routing.exec.acceleration` reports `reader` (`fla-triton` or `torch-reference`), optional package versions,
fallback reason, and head graph mode/hits/builds/failures. The worker follows the project's Windows P-core
affinity convention (`LAYA_PCORE_LOGICAL`, default 16; 0 disables it), independently of other processes.

For prefixes longer than 32,768 tokens, auto bounds the working set with 2,048-token causal prefills,
retaining the complete KV/recurrent cache and all tapped hidden states. Supported GPUs use symbolic
lower-right masks with Flash/GQA attention. `acceleration.prefill` reports whether this ran for the call.
This changes execution order, not the context limit or the document's contents. Fixed-capacity KV storage
avoids fragmentation while views expose only the actual history. The final Node API check read 70,014
state tokens in about 4.4 s, peaking at 3,268 MiB allocated / 3,660 MiB reserved. `reservedVramMiB` reports
the caching allocator reservation, which can exceed the peak allocated-tensor count.

The fused BF16 operations are numerically close, not bit-identical, to the reference: on 165 checked
questions the argmax answers and auto/review routes matched, with a maximum probability delta of 0.016613.
Four auxiliary abstain/unbound flags differed. Head graph replay alone changed probabilities by at most
0.000001 and changed no checked flags. See [measurements and fidelity](../results/vega-speed-2026-11-17-summary.md).

## Multilingual CUDA acceleration (0.9)

After preparing the multilingual model, `npm run multilingual:setup` installs the optional packages pinned
under `multilingualAcceleration` in `data/python-models.json`, including TileLang 0.1.15 and its CUDA
compiler/runtime wheels. They live in `.venv-laya`; compilation products live in `.laya/tilelang`.
`multilingual:check` loads the model and executes a real CUDA prediction. On this Windows machine the
installer's `--index-url` mirror option is needed, as for the other Python packages.

Auto in a reusable session, or explicit `--acceleration on`, selects `laya`'s upstream TileLang backend:
resident 16-bit weights, fused operations, local/global
attention kernels, and CUDA Graph replay. The model's existing attention pattern and 8,192-token input
budget are preserved. The adapter bounds graph residency to eight shapes. Batches use power-of-two row
buckets, and sequence lengths use 16-token buckets through 256, then 64-token buckets.
New shapes first use fused kernels. An immediate repeat captures a graph; resident shapes then replay.
This avoids graph-capture overhead for one-off shapes in a reusable session. `graphs.mode` distinguishes
kernels/capture/replay. A local one-shot with auto skips TileLang initialization entirely and uses eager.

```powershell
node ask.mjs --model multilingual --device cuda --local --acceleration auto --preset triage
node ask.mjs --model multilingual --device cuda --local --acceleration off --preset triage
```

These commands start an interactive local session. The same selection works through HTTP, the browser,
demo library and skill wrapper. A pre-0.9 sidecar must be restarted to accept Multilingual acceleration
selection. CPU and missing/incompatible optional dependencies use eager execution and report the reason.

`routing.exec.acceleration` reports `backend: "tilelang"|"eager"`, package versions, graph hits/builds,
resident shape count and current bucket. An OOM retry on CPU is reported as eager/CPU, and a reinstalled
GPU backend gets a fresh bounded cache. First installation/shape use pays compilation; later processes
reuse the disk cache, while CUDA Graph instances are rebuilt per process.

Paired checks on this machine: one short question 15.8 -> 4.1 ms; one question near 8K tokens
245 -> 99 ms. Three short questions 19.3 -> 5.0 ms; three near 8K 847 -> 337 ms. All 197 checked
decisions matched the eager reference, with a maximum probability delta of 0.0213. These are warmed
inference measurements and fidelity checks. See [the report](../results/multilingual-speed-2026-11-17-summary.md).
