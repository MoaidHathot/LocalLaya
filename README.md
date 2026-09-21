# Laya on Windows via ONNX Runtime - PoC, benchmarks, calibration, EP router

Runs [Laya](https://huggingface.co/convaiinnovations/laya) (Convai Innovations' non-autoregressive "System 1"
decision model: typed `choice` / `score` / `noul` answers with probabilities, one forward pass) from Node.js on
Windows through [`@receptron/laya`](https://github.com/receptron/laya) + `onnxruntime-node`, on an
i9-14900KF / RTX 4070 / 64 GB box.

Everything downloaded lives under this directory (`models/`, `node_modules/`, `.venv/`, `.uv-cache/`).

## Layout

| path | what |
|---|---|
| `poc.mjs` | load once, answer 3 questions in one pass, 10 timed runs, sanity check on contrasting states |
| `src/laya-client.mjs` | reusable loader: pinned HF revision, project-local cache, SHA256 verification, no network after first download, EP selection, P-core pinning, calibration, `createDecider()` facade |
| `src/ep-router.mjs` | per-call execution-provider router (`LayaRouter`): probes lanes, predicts latency per (lane, GPU thermal state, question bucket, work), contention-aware, quarantines failing lanes |
| `router-demo.mjs` | the router under burst / sporadic / batch traffic, deadline and forced-lane calls |
| `src/calibration.mjs`, `calibrate.mjs` | raw-logit capture, accuracy + NLL / Brier / ECE + reliability tables, per-bucket temperature refit with leave-one-out |
| `data/smart-home-eval.mjs`, `data/question-variants.mjs` | 65 hand-labelled utterances; three question wordings (v1 original, v2 explicit, v3 best per question) |
| `calibration/*.json` | fitted temperature tables (`loadLaya({ calibration })`) |
| `bench.mjs`, `bench-all.mjs`, `src/metrics.mjs` | latency matrix (EP x 1/3/10 questions) with process CPU / RSS / GPU util / VRAM / power sampling |
| `experiments/*.mjs` | shape sensitivity + concurrency, sporadic-vs-burst latency, length sweep, fp16 fidelity |
| `tools/convert_fp16.py` | fp32 -> fp16 bundle conversion (ORT's transformer float16 pass; no PyTorch) |
| `verify-model.mjs` | re-hash the cached bundle against the pinned SHA256 values |
| `vendor/receptron-laya-0.1.2.tgz` | the exact published npm tarball (see Supply chain) |
| `test/unit.test.mjs` | unit tests for router decisions, latency model, calibration maths |

## Quick start

```powershell
npm install                 # deps via the configured registry; @receptron/laya from vendor/
node poc.mjs                # first run downloads the pinned 1.7 GB bundle into models/ and verifies SHA256
node poc.mjs --ep webgpu    # RTX 4070 through the WebGPU EP
node router-demo.mjs        # auto CPU / GPU selection per call
npm test
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

## Results (Node 25.3, onnxruntime-node 1.30.0, Laya English fp32 421M, `results/*-summary.md`)

Latency of one `systemOne` call, p50 ms, 20 runs back-to-back after warm-up:

| lane | 1 q | 3 q | 10 q | machine CPU | RSS | VRAM | GPU power |
|---|---|---|---|---|---|---|---|
| `cpu` = 16 threads pinned to P-cores | 115 | 276 | 911 | 49 % | 1.6-1.8 GiB | 0 | - |
| `cpu:8` pinned (1 thread / physical P-core) | 113 | 272 | 921 | 24 % | 1.6-1.8 GiB | 0 | - |
| `cpu:auto` ORT default (24 threads, unpinned) | 101 | 248 | 809 | 74 % | 1.6-1.8 GiB | 0 | - |
| `webgpu` fp32 | 30.5 | 53 | 143 | 2 % | 1.1 GiB | +1636 MiB | 42-161 W |
| `webgpu:fp16` | 29.1 | 47 | 122 | 2 % | 0.6 GiB | +832 MiB | 53-144 W |
| `dml` (DirectML) | fails | fails | fails | | | | |

Model-card reference on a Tesla T4 (PyTorch): 39.5 ms (1 q), 158.6 ms (10 q). All lanes return identical
answers to 4 decimals (fp16: max |delta p| 0.034, 195/195 arg-max agreement on the eval set).

### Findings that change how you should run it

1. **Hybrid CPU straggler mode (5x).** ORT's default pool (24 threads = all physical cores) splits every op over
   P- and E-cores and waits for the slowest thread. Depending on how Windows schedules the threads, 3 questions
   take either ~215 ms or ~1200 ms (p95 1174-1251 ms measured in 3 of 6 sessions; E-cores only: 1327 ms).
   Pinning the pool to the P-cores (`session.intra_op_thread_affinities`, exposed via `sessionOptions.extra`)
   costs ~10 % in the good case and removes the slow mode: p95 284-350 ms across all sessions.
   `pinToPCores: true` is the default for the router's CPU lanes; override the P-core count with `LAYA_PCORE_LOGICAL`.
2. **GPU idle clocks.** Between sporadic calls the RTX 4070 drops to ~225 MHz. Back-to-back a 3-question call
   is 53 ms on WebGPU; after a 1 s pause 91 ms; after a 3 s pause ~200 ms. A single question after a 3 s pause:
   WebGPU ~180 ms vs CPU ~105 ms. The CPU is insensitive to gaps. This is the case for the router.
   A keep-alive (tiny GPU calls every 250 ms) was tried and does not help; left off.
3. **DirectML** loads the graph but every inference fails in a `Reshape` node (`node_view`, HRESULT 80070057) at
   all graph-optimisation levels; not fixable client-side. WebGPU is the NVIDIA path in `onnxruntime-node` on
   Windows (no CUDA EP is shipped for Windows).
4. **No per-shape recompilation on WebGPU** beyond the first call (~180 ms); unseen sequence lengths run at
   normal speed. Concurrency does not raise throughput (GPU work is serialised: 15.1 -> 15.8 calls/s at 1 -> 8).
5. **Latency scales with tokens**: CPU 3 q goes 236 -> 400 ms for 77 -> 136 tokens per question. Longer option
   descriptions (see v2 wording) cost latency on every call.
6. **fp16 bundle**: half the VRAM and RSS, 0.8 s load, 5-13 % faster; fidelity fine on this domain
   (validate on yours: `node experiments/fp16-fidelity.mjs`). Build: `.venv/Scripts/python tools/convert_fp16.py <fp32 dir> models/laya-onnx-fp16`.

## The execution-provider router (`src/ep-router.mjs`)

Lanes: `webgpu`, `webgpu:fp16`, `dml`, `cpu` (16 threads pinned), `cpu:8` (8 pinned), `cpu:24:nopin`, `cpu:auto`.
Each lane is a separate `Laya` session (RAM: ~1.6 GiB per CPU lane, ~1.1 / 0.6 GiB + VRAM per GPU lane).

- **Probe**: every lane is loaded and must pass a real inference; DML is dropped here. A lane that throws later is
  quarantined for 60 s and the call is retried on the next-best lane.
- **Prediction**: EMA of ms-per-work-unit keyed by (lane, GPU state `hot` < 0.4 s / `warm` < 2 s / `cold`,
  question bucket 1 / 2-3 / 4-6 / 7-10 / 11+), seeded with priors from this machine; work = questions x
  estimated padded sequence length, so short and long states share estimates.
- **Contention**: background sampling of other processes' CPU (os.cpus deltas minus own usage) and GPU load
  (nvidia-smi utilisation weighted by SM clock, sampled while we are idle) inflates the affected lane.
- **Decision**: `auto` = fastest predicted with 5 % exploration among lanes within 2x; `prefer-gpu`,
  `prefer-cpu`, `min-cpu`; per call `{ lane }` or `{ deadlineMs }` (meet the deadline with the least CPU share).

Measured behaviour (`router-demo.mjs`): bursts -> WebGPU (47-90 ms / 3 q); one question every 3 s -> CPU
(100-130 ms, WebGPU predicted ~200 cold); 10 questions after a pause -> WebGPU (505 ms vs CPU 1180 predicted).
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
- `head_max_len` 192 tokens shared by all options of a question; < 20 options recommended; state truncated at 512.
- English checkpoint only; non-Latin scripts fail confidently (model card).
- Next: collect real traffic -> labels -> `calibrate.mjs`; iterate wording per question; if accuracy is not
  enough, fine-tune and re-export; consider exporting the multilingual checkpoint.
