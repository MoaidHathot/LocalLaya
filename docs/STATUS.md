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
| Per-call execution-provider router (`src/ep-router.mjs`) | Back-to-back traffic: WebGPU 3-5x faster. Sporadic traffic: the GPU drops to 225 MHz between calls and a single question takes ~180 ms vs ~105 ms on CPU. The router predicts per (lane, GPU thermal state, question bucket, work, queue depth) and picks. |
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
  concurrency, fp16 fidelity. Results in `results/*-summary.md` and README.
- **Router** (`src/ep-router.mjs`, `router-demo.mjs`): lanes `webgpu[:fp16]`, `cpu[:N][:nopin]`, `cpu:auto`,
  `dml`; probe with real inference; EMA latency model normalised by estimated work; contention inflation;
  queue-depth aware; policies `auto` / `prefer-gpu` / `prefer-cpu` / `min-cpu`, `deadlineMs`, forced lane;
  quarantine + fallback; per-call calibration; `pauseSampling`/`resumeSampling`.
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
calls ~0.2-0.4 s; 8 parallel calls in 3.4 s across both lanes. Accuracy zero-shot: smart-home intent 0.72,
should_execute 0.66, target_device 0.77 (65 items); dev-request task 0.75, language 0.85 (40 items).

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
5. **Router**: `cpu:8` measured equal to `cpu:16` at half the CPU share - consider making it the default
   CPU lane after another measurement session; token-length feature is an estimate (`estimateWork`).
6. **DirectML**: re-test with newer `onnxruntime-node` releases (Reshape `node_view` failure, ORT 1.30).
7. **Unmeasured presets** (`triage`, `guard`, `moderation`, `route`, `sentiment`): label and measure before
   relying on them; the `guard` preset answering 100 % on obvious injections says nothing about subtle ones.
8. **Housekeeping**: `bench-all` default configs, GitHub Actions (unit tests only - no model in CI),
   `skills-ref validate` on the skill, decide whether to publish the skill separately.

## Known issues / caveats

- CPU latency on this hybrid CPU fluctuates 1.2-1.6x between sessions even when pinned (thermal / power /
  other threads); the 5x E-core straggler mode is gone with pinning.
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
