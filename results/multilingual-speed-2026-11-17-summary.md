# Laya Multilingual CUDA acceleration (2026-11-17)

Hardware: RTX 4070 12 GB / i9-14900KF / Windows. No LocalLaya sidecar was running at the initial check.
The user reported the PC was idle; the desktop still occupied GPU memory. Core runtime: Laya 0.4.35,
PyTorch 2.15.0+cu132, Transformers 5.23.0. Optional runtime: TileLang 0.1.15 and the CUDA compiler/runtime
wheels pinned under `multilingualAcceleration` in `data/python-models.json`.

## Implementation

Use the upstream `laya` TileLang backend: resident 16-bit weights, fused GEMM/normalization/activation
operations, native local/global attention kernels, and CUDA Graph replay. The checkpoint, tokenizer,
calibration and 8,192-token budget are the same. The adapter adds:

- `acceleration: "auto"|"off"` through the same CLI/REPL/HTTP/library/skill controls as Vega;
- CPU/off/missing-dependency fallback to the original eager implementation;
- an eight-shape LRU CUDA Graph cache, including after an upstream OOM/CPU retry reinstalls the backend;
- repeat-aware capture: a new shape first runs fused kernels; an immediate repeat captures a graph;
  cached shapes replay. This avoids paying several extra forwards for one-off shapes;
- per-call metadata for backend, package versions, graph mode/hits/builds and bucket.

Batch rows are padded to the next power of two. Sequence buckets are multiples of 16 through 256 tokens,
then multiples of 64. Padding is masked; the existing global/sliding attention pattern is preserved.
Compilation products are local to `.laya/tilelang`; optional packages live in `.venv-laya`.

## Profile

The short eager forward issued 808 CUDA kernel launches. Of 3.64 ms measured GPU work, 1.78 ms was copying
and converting tensors; host dispatch also dominated this short call. The near-8K case spent about 138 ms
in attention, including 88 ms in the masked efficient-attention path. These instrumented timings identify
costs; the uninstrumented paired measurements below are the speed comparison.

The freshly measured eager baseline was faster than the earlier busy-desktop measurements, so gains are
computed against this session's paired baseline rather than attributed to the machine becoming idle.

## Interleaved measurements

Six alternating-order pairs after warm-up, one shared loaded model, byte-identical inputs. Synthetic
neutral filler followed by a refund request. Times include Python API processing, exclude model load,
first-kernel compilation and graph capture. These are execution/fidelity checks, not domain-accuracy scores.

| questions | filler words (approximately tokens) | eager median | graph-replay median | paired time ratio [95% bootstrap interval] |
|---:|---:|---:|---:|---|
| 1 | 0 | 15.78 ms | **4.11 ms** | 0.268 [0.236, 0.585] |
| 1 | 1,024 | 20.77 ms | **12.19 ms** | 0.573 [0.541, 0.633] |
| 1 | 4,096 | 85.75 ms | **42.71 ms** | 0.495 [0.468, 0.502] |
| 1 | 7,800 | 244.87 ms | **98.62 ms** | 0.403 [0.389, 0.404] |
| 3 | 0 | 19.35 ms | **5.02 ms** | 0.264 [0.216, 0.285] |
| 3 | 1,024 | 51.90 ms | **32.61 ms** | 0.627 [0.613, 0.632] |
| 3 | 4,096 | 321.51 ms | **150.43 ms** | 0.469 [0.460, 0.470] |
| 3 | 7,800 | 847.42 ms | **336.97 ms** | 0.397 [0.391, 0.399] |

Ratios use paired observations; intervals use 4,000 bootstrap resamples, seed 0. Raw records:
`results/multilingual-tilelang-ab-2026-11-17.json` and `multilingual-tilelang-3q-ab-2026-11-17.json`.

Both A/B arms retained the fast backend's copied weights; their allocation measurements are not standalone
memory comparisons. The production API reported about 1,885 MiB for a short request and 2,092 MiB for the
7,814-state-token request, with no truncation. An eager-only short call was about 1,497 MiB: fast residency
costs memory on small inputs, while its long-input activations are smaller.

## Cold costs and production policy

Initial core-kernel compilation took tens of seconds. The first dynamic attention/graph use took about
17 seconds, and uncached long attention buckets about 10-12 seconds. Disk-cache reuse eliminates most of
that compilation work on later processes; graph capture is still per process.

With cached kernels, the original capture-on-first-use policy gave about 129 / 222 / 364 ms for the first
1K / 4K / 8K shapes through Node. The shipped repeat-aware policy measured about 108 / 140 / 198 ms for
those one-off requests, then reaches replay latency once the bucket is repeated. These separate-process
observations include loading cached kernel modules and are not paired confidence intervals.

One Windows atomic-cache-save warning occurred on the initial build. The next process recompiled that
entry; subsequent fresh-process model checks reused the cache successfully. No global toolchain or
configuration changes were made.

## Fidelity

83 inputs / **197 typed decisions**, including English, Arabic, Spanish, French, Chinese and Hindi;
the dev-request evaluation set; single and mixed question types; and lengths around 64/128/256/512 tokens
through the full 8K budget.

- **197/197 selected choices matched** the eager reference.
- Maximum probability difference: **0.0213**. Maximum expected-score difference: **0.0114** levels.
- An experimental FP32 RoPE/type-embedding adjustment also matched all choices but had a larger maximum
  probability difference (0.0321), so the shipped path uses the upstream implementation.
- Numerical results are close, not bit-identical. `--acceleration off` selects the eager reference.

Raw fidelity output: `results/multilingual-fidelity-2026-11-17.json`.

## Reproduce / use

```powershell
npm run multilingual:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
npm run multilingual:check
node ask.mjs --model multilingual --device cuda --local --acceleration auto --preset triage
node ask.mjs --model multilingual --device cuda --local --acceleration off --preset triage
.venv-laya\Scripts\python.exe experiments\multilingual_speed.py --variants eager,tilelang --questions 3
.venv-laya\Scripts\python.exe experiments\multilingual_verify.py
```

18 Node unit tests, five CPU cache/fallback policy checks, and the targeted Multilingual plus HTTP/CLI/skill
integration scenarios passed. Checks cover CPU usability with optional CUDA packages installed, auto/off,
repeat-aware graph capture/replay, Unicode, longer inputs, switching back to original Laya, and per-model
capability checks against older sidecars. Vega performance was not re-benchmarked; shared API regression
checks still exercise its selection path.
