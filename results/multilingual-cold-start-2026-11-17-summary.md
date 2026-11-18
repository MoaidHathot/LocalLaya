# Multilingual cold-start policy - 2026-11-17

## Finding

For `node ask.mjs --model multilingual "Turn on the lights"`, importing/initializing TileLang cost
about **3.6 seconds per fresh process**, even with the compiler disk cache populated. Its warm-inference
advantage did not repay that cost for this short, five-question smart-home request.

The one-shot auto profile now selects eager execution. CUDA remains the automatic device when available,
and the 8,192-token context budget applies. Sidecars, REPLs, batches and reusable library instances still
use available TileLang kernels and repeat-aware graph caching. `--acceleration on` explicitly opts into
accelerator initialization for a one-shot; `--graphs on` also enables the backend required for graph use.

## Measurement

Windows, i9-14900KF / RTX 4070, the pinned Laya/PyTorch/TileLang environment in `data/python-models.json`.
Each call started a new Node/Python process. OS file caches and compiled-kernel disk caches were retained.
Three interleaved rounds per variant, using the default smart-home preset and the text above. Timings
include the first inference; these are process-cold measurements, rather than a cold-boot/disk-cache test.

The instrumented comparison after the correction:

| median | one-shot auto (eager CUDA) | explicit acceleration on (TileLang CUDA, graphs off) |
|---|---:|---:|
| Time to stdout answer | **5.96 s** | 9.60 s |
| Through process exit, including cleanup | **6.61 s** | 10.29 s |
| Python process start to ready | 5.54 s | 9.07 s |
| First decision including kernel initialization and IPC | 365 ms | 478 ms |
| Accelerator initialization phase | <1 ms | 3,596 ms |

Auto reduced median time to the first answer by **37.9% (3.64 s)**. Its three answer times were 6.05,
5.96 and 5.44 seconds. The remaining startup cost was dominated by PyTorch import (~1.47 s) and
checkpoint/model construction (~3.85 s, including lazily loaded model code).

A preceding three-round diagnostic sweep measured whole-process medians of 10.83 s for the previous
TileLang one-shot auto policy, 6.98 s for explicit eager CUDA, and 6.76 s for eager CPU. The final paired
comparison above additionally records answer arrival separately from process exit. Device auto keeps the
GPU preference for the supported range of input lengths and question counts; explicit CPU remains available.

The repeated short-input comparison produced **15/15 matching decisions** (five questions x three
rounds). Maximum probability difference was 0.0033 and expected-score difference 0.0063. These repeated
checks exercise the startup policy; broader multilingual fidelity is covered in the
[warm-inference report](multilingual-speed-2026-11-17-summary.md).

## Diagnostics and controls

- Startup logs report runtime imports, CUDA initialization, checkpoint loading and backend setup, with
  phase durations and a five-second loading heartbeat. The first inference has start/completion messages.
- Progress stays on stderr, including with `--json`. The first successful result includes
  `routing.initialization.processMs`, worker-side `totalMs`, and `phases`; `/stats` retains startup metadata.
- `routing.profile` explains the automatic policy. `routing.exec.acceleration.requested` preserves the
  user's auto/on/off choice, `selected` records the resolved eligibility, and `backend` records actual use.
- `--acceleration on` tries the available optional backend even for a one-shot, with capability checks and
  eager fallback. `off` explicitly selects reference/eager execution. CPU uses eager execution.

```powershell
# Automatic startup policy; --local is also the default when LAYA_SIDECAR is unset.
node ask.mjs --model multilingual --local "Turn on the lights"

# Reuse the loaded model for subsequent commands, or keep it in a local REPL.
node ask.mjs --model multilingual --sidecar "Turn on the lights"
node ask.mjs --model multilingual --local

# Explicit one-shot accelerator setup; graphs still follow their own policy.
node ask.mjs --model multilingual --local --acceleration on "Turn on the lights"

# Reproduce the paired cold-process measurement.
node experiments/cold-start.mjs --rounds 3 --out results/multilingual-cold-start.json
```

Raw local evidence: `results/multilingual-cold-before-2026-11-17.json` and
`results/multilingual-cold-after-2026-11-17.json` (ignored benchmark artifacts).
