# Graph optimisation for the GPU lanes (2026-09-23): IsNaN rewrite, Reshape allowzero, Gelu fusion, full fp16

Tool: `tools/optimize_graph.py` (replaces `tools/convert_fp16.py`). Method for every claim below:
`experiments/ab.mjs` - all variants in one process, interleaved rounds, paired ratio to the baseline with a
bootstrap 95 % CI; fidelity = arg-max agreement and |delta probability| against the **fp32 reference on the
WebGPU EP**, accuracy against the eval set's gold labels. Noise floor of the method: 1.000 [0.997, 1.001].

## Why (ORT profiler, WebGPU EP, 3 questions, before)

| finding | per call |
|---|---|
| `IsNaN` is not implemented by the WebGPU EP -> runs on the CPU | 28 nodes: 30 `MemcpyToHost` + 36 `MemcpyFromHost` = 28 GPU pipeline drains |
| fp32 islands for LayerNormalization / Softmax (previous converter) | 196 `Cast` dispatches |
| ORT runtime fusions on this torch.export graph | none fire (only `TransposeOptimizer`) |
| DirectML rejects `Reshape(allowzero=1)` with a `-1` in the shape | every inference failed (`node_view`, 80070057) |

## What the tool does (all exact except fp16 numerics)

1. `IsNaN(x)` -> `Not(Equal(x, x))` (28 nodes) - same predicate, runs on the GPU.
2. `Reshape allowzero=1` -> default (77 nodes; no shape in this graph contains a 0).
3. ORT transformer fusions (`model_type=bert`): Gelu 29, BiasGelu 1. Attention / RoPE / SkipLayerNorm patterns do
   **not** match this ModernBERT export for any model type (bert, clip, gpt_neox, qwen3, ...) - reported, not
   hand-written (as agreed).
4. fp16 with no fp32 islands (8 Cast nodes left, all at the graph's int/bool boundary).

Result: 1842 nodes -> 1753 (fp16 bundle before: 2101). After: `MemcpyToHost` 2 per call, CPU-EP time 11.2 -> 3.8 ms (profiled).

## Speed (paired ratio to fp32 on WebGPU; `cur` = previous fp16 bundle, `opt` = adopted)

| workload | cur | optA (fp32 LN/Softmax kept) | optB (all fp16, adopted) | optB vs cur |
|---|---|---|---|---|
| PoC 1/3/10 q + smart-home eval (24 items x 5 q) | 0.846 [0.840, 0.850] | 0.454 [0.450, 0.457] | **0.453 [0.448, 0.457]** | **1.87x** |
| dev-request eval (40 items x 4 q, code snippets) | 0.916 [0.908, 0.922] | 0.434 [0.428, 0.439] | **0.430 [0.423, 0.433]** | **2.13x** |

Standard bench (`bench.mjs --ep webgpu --fp16`, 20 runs, P-core pinned): 1 q **21.1** ms (was 29.1), 3 q **32.0** (47.2),
10 q **83.1** (122.1); GPU utilisation 26-93 % (was 21-59 %). Throughput 3 q back-to-back **30.3-30.9 calls/s**
(was 19-21); 20 calls/s offered now served at 33 ms p50 (was 110-146 with a queue); 8 parallel 5-q sidecar calls
676 ms (was 1.1-1.4 s). Sporadic (`sporadic.mjs --fp16`): 3 q after 1 s pause 81 ms (was 91-140), after 3 s 156
(was 172-235); **1 q after 3 s 101 ms** (was ~180) - equal to the CPU lane now.

## Fidelity (vs fp32 reference)

| workload | cur | optA | optB |
|---|---|---|---|
| smart-home: arg-max agreement / max abs dp / accuracy | 134/134 / 0.0415 / 0.708 | 134/134 / 0.0415 / 0.708 | **134/134 / 0.0415 / 0.708** |
| dev-request: arg-max agreement / max abs dp / accuracy | 118/120 / 0.0520 / 0.800 | 117/120 / 0.0520 / 0.787 | **118/120 / 0.0525 / 0.800** |

The two dev-request disagreements are near-ties that the previous fp16 bundle flips identically
(`effort` 0.426 vs 0.419; `language` python 0.194 vs none 0.192). Accuracy equals fp32 on both presets.

## Side results

- **DirectML works** on the optimised graph (1.30.0 and nightly): 1 q **18-19 ms** hot (fastest of all EPs) but
  **batch > 1 is 6-10x slower than WebGPU even on a fixed shape** (3 q 217 ms, 10 q 264 ms). Not added as a lane;
  `DEFAULT_PRIORS.dml` updated so a configured `dml` lane would only ever be picked for single questions.
- **CPU lane**: the same clean-up in fp32 gives nothing (1.030 [0.953, 1.058]; IsNaN is native on the CPU EP), so
  the CPU lane keeps the pinned, hash-verified HF bundle. fp16 on the CPU EP is not faster either (1.010 [0.961, 1.214]).
- Raw data: `graph-opt-*.json` (git-ignored) and `graph-opt-*-summary.md` in this directory.
