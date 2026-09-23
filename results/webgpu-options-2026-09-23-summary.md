# WebGPU EP option sweep, CPU-fallback audit, DML nightly retest (2026-09-23)

Machine: i9-14900KF / RTX 4070, Node 25.3, onnxruntime-node 1.30.0, `webgpu:fp16` bundle. Method:
`experiments/ab.mjs` - every variant loaded in one process, interleaved rounds (each workload item once per
variant per round, rotating order), paired ratio to the baseline with a bootstrap 95 % CI. Workload: PoC
1 / 3 / 10 questions + 24 of the 65 smart-home eval items (5 questions each). Noise floor (baseline vs an
identical second session): ratio 1.000 [0.997, 1.001]; answers bit-identical (134/134, max |dp| 0.0000).

## 1. WebGPU EP provider options the 1.30 Node binding forwards

| option | paired ratio vs default [95 % CI] | verdict |
|---|---|---|
| `validationMode=disabled` | 0.997 [0.983, 1.003] | noise |
| `enableRobustness=false` | 1.003 [0.989, 1.009] | noise |
| `uniformBufferCacheMode=bucket` | 1.002 [0.993, 1.009] | noise |
| `defaultBufferCacheMode=bucket` | 1.001 [0.990, 1.010] | noise |
| `storageBufferCacheMode=simple` | 1.002 [0.987, 1.005] | noise |
| `storageBufferCacheMode=lazyRelease` | 1.001 [0.994, 1.004] | noise |
| `preferredLayout=NHWC` | 0.998 [0.986, 1.003] | noise |
| all of validation off + robustness off + uniform/default bucket | 1.005 [0.995, 1.010] | noise |

Nothing adopted. Every variant returned bit-identical answers. Raw data: `webgpu-options-batch{1,2}.json`
(git-ignored) / `-summary.md`. `enableGraphCapture` (the option that would remove the ~14 ms fixed cost) is
rejected by the binding ("unrecognized option"), see `src/laya-client.mjs` `WEBGPU_OPTION_KEYS`.

Side observation: five WebGPU sessions in one process run each 1-question call at 35-37 ms instead of
28-31 ms in a single-session process; the comparison is unaffected (interleaved), production has one GPU
session per process.

## 2. Nodes ORT keeps on the CPU (logSeverityLevel 1, `webgpu:fp16`, 3 questions)

90 fallback candidates, all `Slice` on `Shape` outputs: `Shape -> Slice -> Concat -> Reshape` shape plumbing,
3 per transformer layer x 30 layers, int64, consumed by `Concat` only. No activation is read back to the CPU
for them. ORT's runtime fusions all report `modified: 0` on this graph (`GeluFusionL1`, `LayerNormFusionL1`,
`MatMulAddFusion`, `ReshapeFusion`, `SliceConcatToSpaceToDepthFusion`, attention / SkipLayerNorm / BiasGelu at
level 2); only `TransposeOptimizer` and `MemcpyTransformer` changed anything. The graph therefore runs with
its exported ~2100 nodes - the motivation for offline fusion (`tools/optimize_graph.py`).

## 3. DirectML on the newest onnxruntime-node (nightly `1.31.0-dev.20260918-bc8e7ed75`)

Installed in a temp project via `overrides` (the vendored `@receptron/laya` pins `^1.22.0`, which excludes
prereleases). Same failure as 1.30.0 at graph-optimisation `all` and `basic`:

    Non-zero status code returned while running Reshape node. Name:'node_view' ... 80070057 The parameter is incorrect.

`node_view` = the first QKV `view`: `Reshape(linear [batch,seq,3072] -> [batch, seq, 3, -1, 64])` with
**`allowzero=1`** (torch.export emits it on every `view`; 138 Reshape nodes, none with a 0 in a constant
shape). Hypothesis: DML rejects `allowzero=1` together with `-1`. `allowzero=0` is semantically identical
for this graph and is applied by `tools/optimize_graph.py`; result in the fusion report.
