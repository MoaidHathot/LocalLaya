# Vega CUDA acceleration (2026-11-17)

RTX 4070 12 GB / i9-14900KF / Windows; PyTorch 2.15.0+cu132, Transformers 5.23.0,
Laya 0.4.35, vegaml 0.16.3. The pre-existing sidecar was stopped before profiling; another 8787
instance appeared during the work and was stopped again at completion. Other applications continued
to use GPU memory. These are local inference measurements,
not an isolated-hardware throughput or domain-accuracy benchmark.

## What changed

1. **Fused reader kernels:** FLA 0.6.0 on Triton Windows 3.9.0.post30 for Qwen's gated-delta attention and
   causal convolution. No system CUDA compiler installation is needed; the Triton wheel includes its toolchain.
2. **CUDA Graph replay for the decision head:** trace the existing engine with both optional branches present
   (ordinal rail and document memory). Tensor masks still select the actual behavior. Validate every new shape
   against eager results before serving; bounded eight-shape cache. Weights and calibration are unchanged.
3. **Bounded long prefills:** beyond 32,768 prefix tokens, read 2,048-token blocks through the same full KV and
   recurrent cache. Stitch all tapped hidden states back together for the original pooling. Use symbolic
   lower-right causal masks with grouped-query Flash SDPA on supported GPUs. This is a causal continuation
   through the whole document, not independently classified chunks or a reduced context window.
   Full-attention KV storage is preallocated; dynamic-length views expose only initialized tokens. This
   avoids allocator fragmentation from repeatedly concatenating growing caches. Per-question cache copies
   retain independent storage, including their aliased views.
4. **Process-local P-core affinity** follows the existing project convention on Windows (16 logical P-core
   processors here, configurable with `LAYA_PCORE_LOGICAL`).

`--acceleration auto` selects these paths; `off` restores the original reference kernels and eager head.
CPU always uses reference kernels. `routing.exec.acceleration` reports the paths and cache counters.

## Profile

The original three-question short call issued **3,274 CUDA kernel launches**. Under profiler instrumentation,
the reader accounted for 288 ms host time but only 19 ms of GPU work; the decision head accounted for 90 ms
host time and 1.5 ms GPU work. Those instrumented wall times are not performance benchmarks; they identify
the dominant dispatch overhead. Fused kernels reduce it, and graph replay launches the head as one graph.

## Interleaved short/medium A/B

One loaded model, byte-identical inputs, alternating arm order over six rounds after warm-up, three typed
questions per call. Repeated neutral filler followed by a duplicate-charge request. Times include tokenization
and Python API processing, exclude model load and first-shape compilation/capture.

| filler words (approximately tokens) | reference | fused reader + head graph | paired time ratio, 95% bootstrap interval |
|---:|---:|---:|---|
| 0 | 50.1 ms | 34.0 ms | 0.704 [0.475, 0.843] |
| 1,024 | 233.0 ms | 113.1 ms | 0.486 [0.483, 0.487] |
| 4,096 | 429.2 ms | 242.9 ms | 0.566 [0.560, 0.571] |

Ratios use the six paired observations, not the ratio of the two medians. Intervals use 4,000 bootstrap
resamples, seed 0. Raw output: `results/vega-optimized-ab-2026-11-17.json`.

## Long inputs and working set

One question, one timed call per arm after a warm-up. These small samples establish working capacity and
representative timings, not confidence intervals. The first comparison used fused kernels/head graphs without
bounded prefill; P-core affinity was enabled in both arms.

| filler words | reference | fused + head graph | reference / fused peak allocation |
|---:|---:|---:|---:|
| 8,192 | 888 ms | 366 ms | 2,543 / 2,170 MiB |
| 16,384 | 1,740 ms | 757 ms | 3,542 / 2,801 MiB |
| 32,768 | 3,851 ms | 1,720 ms | 5,536 / 4,058 MiB |
| 70,000 | 198.5 s | 55.4 s | 10,072 / 6,914 MiB |

Raw output: `results/vega-long-ab-2026-11-17.json`. Large-context time varies substantially with device memory
pressure and allocator history. PyTorch allocation excludes the CUDA context and other applications.

A subsequent comparison isolated bounded prefill with Flash-compatible masks, keeping the fused reader/head
in both arms:

| filler words | whole prefill | bounded + Flash masks | whole / bounded peak allocation |
|---:|---:|---:|---:|
| 16,384 | 773 ms | 727 ms | 2,800 / 1,938 MiB |
| **70,000** | **19.8 s** | **5.32 s** | **6,920 / 3,197 MiB** |

Raw output: `results/vega-stream-flash-ab-2026-11-17.json`. Do not combine different runs' timings into a
claimed paired speedup. Production uses bounded prefill only above 32K; the smaller case checked its numerical
behavior. Ordinary dense masks with bounded prefill were also tried: 11.7 s at 70K, 3,508 MiB. Symbolic causal
bias avoids the dense mask and preserves Flash/GQA dispatch. No input tokens are discarded by this optimization.

### Production API check and allocator fix

The actual Node/Python API exposed an additional issue that live-tensor peaks alone missed: growing KV
caches caused the CUDA allocator to reserve **14,054 MiB** while only **3,196 MiB** was live at the peak.
Fixed-capacity KV storage removes those repeated allocations while retaining the dynamic visible length.

Same 70,014-state-token input through `experiments/model-context.mjs`, one timed long call in each fresh
process after a short warm-up:

| cache storage | inference | peak allocated | reserved after call | truncated |
|---|---:|---:|---:|---|
| growing | 59.84 s | 3,196 MiB | 14,054 MiB | no |
| **fixed capacity (shipped)** | **4.42 s** | **3,268 MiB** | **3,660 MiB** | **no** |

The answers were identical between these two calls. These are separate-process observations, including
potential first-shape compilation effects, not a paired confidence interval. Raw files:
`vega-production-70k-2026-11-17.json` and `vega-production-fixed-kv-2026-11-17.json` under `results/`.

## Fidelity

`experiments/vega_verify.py` checked 65 inputs / **165 typed questions**: the dev-request evaluation set,
mixed choice/noul/score questions, single-type calls, Arabic, and lengths around memory/prefix thresholds.

- **165/165 argmax answers matched** between reference and fused/head-graph paths.
- Auto/review routes and prediction sets matched on these cases.
- Maximum probability delta from fused BF16 operations: **0.016613**. Four auxiliary fields differed:
  two `abstain` and two `unbound` flags. These paths are numerically close, not bit-identical.
- Head graph replay alone: maximum probability delta **0.000001**, no changed argmax answers or flags;
  four shapes built, no failed validation.
- In the 16K/70K prefill comparison, bounded/Flash versus whole-prefill maximum probability differences were
  0.001712 / 0.006058, with matching answers and flags. These are synthetic inputs, not accuracy estimates.
- After the fixed-KV change, varied 33K/50K-token documents made from numbered dev-request records were
  checked with all three question types: **6/6 choices matched**, no flag changes. Maximum end-to-end
  probability delta versus reference was 0.002249; bounded versus whole fused prefill delta was 0.006433.
  The combined checks cover 171 matching argmax choices. `results/vega-long-fidelity-2026-11-17.json` records
  these cases separately from the 165-question set.

## Operation

```powershell
npm run vega:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
npm run vega:check
node ask.mjs --model vega --device cuda --acceleration auto --local --preset triage "Charged twice, refund me"
node ask.mjs --model vega --device cuda --acceleration off --local --preset triage "Charged twice, refund me"
```

The first new convolution shape in the initial experiment took 5.4 s to compile/warm, and the first head
graph took 3.4 s to trace/warm/capture. These are cold costs, not steady-state latency. Triton binaries cache
under `.laya/triton`; graph instances live with the Python process. Reuse a local library instance, REPL,
or sidecar for warm latency. Switching model/device/context/acceleration reloads the worker.

Reproduce the decomposition with `experiments/vega_speed.py` (`reference`, `fla`, `fla-conv`, `optimized`,
`streamed`, `streamed-flash`); `experiments/vega_verify.py --long-only` exercises varied long documents.

## Verification

- 18 unit tests passed, including acceleration selection, worker reloads, and old-server rejection.
- 25 integration tests passed across the demo, sidecar and Python-model suites. These exercise CPU and CUDA,
  auto/off through HTTP/CLI/skill, original Laya and multilingual regression paths, a >32K public API call,
  cache history/branch isolation, and shutdown of the Windows venv launcher plus interpreter.
- The optional kernel check exercises gated-delta, convolution and lower-right grouped-query attention.
- Models are loaded from the same pinned artifacts; the optimization installs only the optional runtime
  dependencies and caches compilation products locally.
