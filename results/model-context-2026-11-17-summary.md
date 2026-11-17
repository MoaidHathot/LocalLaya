# Selectable Python models: context and latency (2026-11-17)

Hardware: i9-14900KF / RTX 4070 12 GB / 64 GB, Windows. PyTorch 2.15.0+cu132, driver 617.42,
Transformers 5.23.0, Laya 0.4.35, vegaml 0.16.3, huggingface_hub 2.4.0. Revisions and source artifacts:
`data/python-models.json`. Each model runs alone in a Python child, using local files with network disabled.
Other desktop applications were running; this is not an isolated-hardware performance benchmark.

## Method

`experiments/model-context.mjs` asks one three-option choice question about repeated neutral filler followed
by a refund request at the end. The short case is the median of three calls after a warm-up; each longer
case is one call. The state-token counts below come from the tokenizer, not character estimates. Times
include the Python protocol round trip and tokenization, exclude loading, and are not paired comparisons
with the old ONNX benchmarks. This checks capacity/execution, not real-document accuracy or calibration.

```powershell
node experiments/model-context.mjs --models vega,multilingual --device cuda --words 0,1024,4096,7800 --out results/model-context-cuda-2026-11-17.json
node experiments/model-context.mjs --models vega --device cuda --words 16384,32768,70000 --out results/model-context-vega-long-2026-11-17.json
node experiments/model-context.mjs --models vega,multilingual --device cpu --words 0 --out results/model-context-cpu-2026-11-17.json
```

## CUDA

| model | state tokens | time | peak PyTorch GPU allocation | truncated |
|---|---:|---:|---:|---|
| Vega-U 0.8B | 14 | 118 ms | 1,547 MiB | no |
| Vega-U 0.8B | 1,038 | 235 ms | 1,658 MiB | no |
| Vega-U 0.8B | 4,110 | 609 ms | 2,043 MiB | no |
| Vega-U 0.8B | 7,814 | 1,013 ms | 2,499 MiB | no |
| Vega-U 0.8B | 16,398 | 2,273 ms | 3,542 MiB | no |
| Vega-U 0.8B | 32,782 | 6,775 ms | 5,536 MiB | no |
| Vega-U 0.8B | **70,014** | **142,985 ms** | **10,073 MiB** | **no** |
| Multilingual | 14 | 41 ms | 1,497 MiB | no |
| Multilingual | 1,038 | 83 ms | 1,507 MiB | no |
| Multilingual | 4,110 | 107 ms | 1,609 MiB | no |
| Multilingual | **7,814** | **293 ms** | **2,211 MiB** | **no** |

All these synthetic cases selected `refund`. Vega's reported probability stayed near 1; multilingual's
fell from 1 to 0.5078 at 7,814 tokens. Neither observation establishes performance on real documents.
Load times in the first CUDA sweep were 6.7 s for Vega and 6.3 s for multilingual; the first real forward
pass additionally pays initialization costs. Holding the model in a sidecar amortizes that cost.

The 70K call completed on CUDA without CPU fallback. Peak allocation is PyTorch's allocated tensor memory,
not total device use, and excludes the CUDA context and other applications. About 4,041 MiB of GPU memory
was in use after the benchmark process exited. The large 32K-to-70K latency increase should be treated as
measured behavior under this desktop load; no single cause was isolated. 16K/32K is a more practical starting
budget on this configuration when the task allows it.

The Windows Vega runtime reports reference PyTorch implementations for `causal_conv1d` and
`chunk_gated_delta_rule`: optional `causal_conv1d` / `flash-linear-attention` kernels are not installed.
The numbers above are for this working eager configuration, not the upstream optimized-kernel figures.

## CPU

The same 14-state-token, one-question case, median of three warm calls:

| model | CPU time | configured PyTorch threads |
|---|---:|---:|
| Vega-U 0.8B | 2,135 ms | 1 |
| Multilingual | 68 ms | 8 |

These are separate runs, not a paired device comparison. CUDA is the practical starting choice for Vega
on this machine. Multilingual CPU is also usable for short inputs. CPU long-context performance was not
measured in this sweep.

## Integration findings

- Vega is accessed through Laya's typed API; the same question schemas work in each selectable profile.
- Vega's upstream `score` is an argmax level while its `expected_score` is the expectation. The adapter exposes
  the latter as LocalLaya's `score`, retaining the original as `selected_level`.
- The pinned Laya facade infers Vega truncation from summed question-row tokens. The adapter uses the pinned
  prompt builder's actual state spans, so multi-question calls are not falsely flagged as truncated.
- Laya rewrites mmBERT's tokenizer config on load. A derived runtime view now absorbs that normalization;
  an offline digest check after repeated multilingual loads confirms the pinned source files are intact.
- Model switches are serialized with accepted requests and release the old backend. Original Laya bursts
  reach its own EP queue intact, preserving its queue predictions, deadlines and lane selection.
- Calibration tables are scoped to their model. English ONNX temperature tables are never automatically
  applied to multilingual or Vega.

## Verification

55 tests passed across the suites: 18 unit, 8 original router, 6 original CUDA, 14 sidecar, 5 demo,
and 4 Python-model integration tests. The new integration suite covers real CPU/CUDA predictions,
Unicode, context beyond 512 tokens, model/device switching, normalized scores, truncation, HTTP, both CLIs,
the skill wrapper, and cancellation of the actual interpreter behind Windows' venv launcher.

The original CUDA lane retained 195/195 argmax agreement with WebGPU on the 65-state evaluation;
CUDA Graph replay also retained 195/195 agreement with dynamic CUDA, with a 5.0 ms median single-question
replay versus 11.4 ms dynamic. The HTTP demo measured 1.3 ms median client/transport overhead.
All pinned Python model artifacts passed offline digest verification after the integration tests.

One existing wrapper timing assertion initially compared total wall time from calls with very different
GPU inference times. It now subtracts reported inference/queue/load time to check the client overhead it
was intended to test. The rerun measured 64 ms wrapper overhead versus 65 ms for `ask.mjs`, even though
their inference times were 204 ms and 33 ms respectively.
