# Session log - 2026-09-21/22: from "vet this plan" to CLI + sidecar + skill

Chronological record of the first working session. The living status (goals, decisions, todo) is
`docs/STATUS.md`; this file is the "what happened and why" trail. Times are local.

## Starting point

The user brought a plan written by someone else: install `@receptron/laya` from npm, download the model
automatically, run a 3-question smart-home example 10 times on CPU, report latencies, then try the GPU.
Requests on top of the plan: vet it (safe, performant, working), collect CPU/RAM/GPU metrics, be able to
call it as an API, cache all downloads under the project directory. Hardware: i9-14900KF / RTX 4070 / 64 GB.

## Phase 1 - vetting and the first PoC (commits `6a8399e` .. `d2618a0`)

- Verified the library API against the GitHub source (`Laya.load` options, `systemOne`, question types,
  cache layout). The plan's API usage was correct.
- `npm install @receptron/laya` failed: npm routes through a Microsoft feed proxy without the package;
  `registry.npmjs.org` and `npmmirror` blocked at TLS. Vendored the exact 0.1.2 tarball from jsDelivr,
  SHA256 per file checked against jsDelivr's manifest and an independent unpkg copy, `dist/` audited
  against source. Installed via `file:vendor/receptron-laya-0.1.2.tgz`.
- Pinned the model to HF commit `68f27dfe...` and added SHA256 verification against LFS oids (the library
  only compares byte sizes and follows `main`). Cache under `models/`. Offline load after first download.
- Corrected the plan's GPU assumption: no CUDA EP in `onnxruntime-node` on Windows; options are DML and
  WebGPU. DML loads but fails at inference (`Reshape node_view`, HRESULT 80070057) at every optimisation
  level. WebGPU works and matches CPU to 4 decimals.
- First matrix (20 runs): CPU 102 / 229 / 751 ms for 1 / 3 / 10 questions; WebGPU 30 / 52 / 136 ms.
  Answers consistent across lanes.

## Phase 2 - original vs port, accuracy, router, fp16 (commits `edbb144` .. `8b0afdb`)

User asked: should we use the original? commit and go ahead with performance + accuracy work; explain
over-confidence; can the CPU/GPU choice be automatic?

- **Original vs port**: `mizorewww/laya-coreml` is an Apple-only port. The true original is the Python
  package (PyTorch 2.14). Recommendation: ONNX port for Node inference; original for multilingual /
  typed-decisions checkpoints, fine-tuning, `predict_shortlist`.
- **Sporadic vs burst**: WebGPU 3 q = 56 ms back-to-back, 91 ms after 1 s pauses, ~200 ms after 3 s pauses;
  1 q after 3 s: WebGPU ~180 ms vs CPU ~105 ms. GPU idles at 225 MHz between calls. CPU is gap-insensitive.
  Keep-alive dummy calls tried; they do not help.
- **Hybrid-CPU straggler mode**: 3 questions sometimes took ~1200 ms instead of ~230. Isolated with
  affinity experiments: ORT's default 24 threads span E-cores; pinned to E-cores only = 1327 ms; 16 threads
  pinned to P-cores = 235 ms stable. ORT's `session.intra_op_thread_affinities` reachable via
  `sessionOptions.extra`. Pinning became the CPU default. Interleaved rounds confirmed: unpinned p95
  1174-1251 ms vs pinned p95 <= 350 ms.
- **Calibration tooling**: raw-logit capture by wrapping the ONNX session's `run`, metrics, per-bucket
  temperature refit with leave-one-out, 65 hand-labelled smart-home utterances. Findings: raw mean
  confidence 0.82-0.89 vs accuracy 0.63-0.77; refit fixed `should_execute` (ECE 0.178 -> 0.066) and did
  nothing for `intent`, whose errors are systematic (device questions -> `control_device`).
- **Wording**: v2 lifted intent 0.63 -> 0.72, dropped target_device 0.77 -> 0.66; v3 = best per question.
- **Router** (`LayaRouter`): probe, EMA per (lane, GPU thermal state, question bucket) normalised by
  estimated work, contention sampling, policies, quarantine + fallback. Demo confirmed: bursts -> WebGPU,
  sporadic 1 q -> CPU, 10 q -> WebGPU.
- **fp16**: onnxconverter-common produced a broken Cast; ORT's `onnxruntime.transformers.float16` worked.
  195/195 arg-max agreement, max delta p 0.034; VRAM 1645 -> 830 MiB, RSS 968 -> 368 MiB, 5-13 % faster.
- Final matrix (20 runs): cpu-pin 115 / 276 / 911; cpu:8-pin 113 / 272 / 921; cpu (unpinned, fast mode)
  101 / 248 / 809; webgpu 30.5 / 53 / 143; webgpu-fp16 29.1 / 47 / 122.
- Git repository created, 6 commits, README with all results.

## Phase 3 - "how do I test it" and "I want to ask her things" (commit `6be0664`)

- `ask.mjs`: one-shot CLI and interactive REPL (ad-hoc `/noul` `/choice` `/score`, `/again`, `/lane`,
  `/json`, `/stats`); `serve.mjs`: localhost HTTP API + browser page; question presets (`smart-home`
  calibrated; `triage`, `guard`, `moderation`, `route`, `sentiment` unmeasured).
- Fixed a readline pitfall (`rl.question()` drops piped lines while an ask is in flight -> async iterator).

## Phase 4 - custom domains (commit `06dd497`)

User tried `"is this a balid json {bla: 1}"` against the smart-home preset and asked what the output means.
Answer: Laya answers the preset's questions about the text, never the text's question; the right use is
routing (task = validate, language = json -> call the parser).

- File presets `presets/<name>.json` with a `$TEXT` state template; `/save` from the REPL; per-preset
  calibration auto-applied; `calibrate.mjs --preset --eval` for any domain with majority/chance baselines
  and a verdict per question.
- Worked example `dev-request`: task 0.75 and language 0.85 usable; `needs_tool` and `has_input` at chance
  (0.55) -> removed. Calibration fitted T = 13.5 for the chance-level noul bucket (probabilities collapse to
  ~0.5), which is the honest behaviour but also shows a broken question poisons its bucket.

## Phase 5 - the sidecar (commit `173a48d`), moved to LocalLaya

User concern: N orchestrations calling the CLI = N model loads; a permanent server wastes memory when idle.
Asked for a middle ground; chose (via questions): opt-in `--sidecar` flag, 5 min idle exit, REPL keeps it alive.

- `serve.mjs` reworked: binds the port before loading (port = mutex; a racing second instance exits 3
  without loading), 503 while loading, idle exit only with nothing in flight, `/touch`, `/shutdown`,
  richer `/health` (`service: "laya"` marker), sampling paused after 10 s idle, presets and calibration
  re-read by mtime, project-root paths (fixed a cwd bug that silently skipped calibration when the CLI ran
  from another directory).
- `src/sidecar-client.mjs`: discover / spawn (detached, hidden, log in `.laya/`) / wait / decide / stop.
  Found and worked around a Node 25 + Windows crash: `fetch` + `process.exit` after two requests ->
  `Assertion failed: !(handle->flags & UV_HANDLE_CLOSING)` (exit code 0xC0000409) - switched to `node:http`.
- Router: per-lane queue depth in predictions so parallel callers spread over GPU and CPU; inference timed
  inside the queue (`queueMs`).
- `ask.mjs`: backend abstraction, `--sidecar` / `--local` / `--start` / `--status` / `--stop` / `--idle` /
  `--port`, env `LAYA_SIDECAR` / `LAYA_PORT` / `LAYA_IDLE`, lazy `onnxruntime` import, REPL keep-alive.
- Tests: `test/sidecar.test.mjs` - 11 scenarios all passing: spawn + fast second call (5.4 s -> 184 ms),
  racing launchers -> one process, 8 parallel calls both lanes zero errors, live preset/calibration edits,
  REPL keep-alive (same pid after 30 s quiet, gone 12 s after close), hard-kill recovery, idle exit frees
  ~950 MiB VRAM, `--local` / env override, foreign port detection + fallback, `--start`.
- Agent skill `skills/laya-decisions/` (agentskills.io format) with a `scripts/laya.mjs` wrapper resolving
  the project via `LAYA_DIR`; tested from the repo, from a copy elsewhere, and without `LAYA_DIR`.

## Move to `W:\Github\LocalLaya` (2026-09-22)

- Merged the 9-commit history into the GitHub-created repo (Unlicense + Visual Studio `.gitignore`, with a
  LocalLaya section appended); moved `models/`, `node_modules/`, `.venv/`, `.uv-cache/`, `.laya/`, raw
  results (same volume: renames). Updated name/path/license references. Added `docs/`.
- Old location left with a `MOVED.md` pointer.

## Numbers worth remembering

- Model: 421M params, fp32 bundle 1.69 GB, fp16 0.84 GB. Load 1.4-1.6 s (fp32) / 0.8 s (fp16).
- ~85 padded tokens per question for the PoC state; 3 questions ~ 230 input tokens; 512 max.
- Latency scales with tokens: CPU 3 q 236 -> 400 ms for 77 -> 136 tokens per question.
- WebGPU concurrency gives no throughput (15.1 -> 15.8 calls/s at 1 -> 8 concurrent); the router therefore
  serialises per lane and spreads across lanes instead.
