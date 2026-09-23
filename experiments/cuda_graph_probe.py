"""Where do the ~9 ms of a CUDA-lane call go, and would CUDA Graph capture remove them?

    .venv/Scripts/python.exe -u experiments/cuda_graph_probe.py [--model-dir models/laya-onnx-fp16] [--runs 30]

Three measurements on the same session configuration as tools/cuda_lane.py (CUDA EP, 2 CPU threads):
  1. latency vs tokens and vs batch, normal session.run: separates the per-call floor from the per-token slope
  2. ORT profiler for one shape: kernel launches per run and the CPU-side time they take
  3. the same shapes with enable_cuda_graph + IO binding (fixed device buffers): the first run captures the
     graph, later runs replay it with one launch. If replay is much faster than session.run for the same
     shape, the floor is per-request launch overhead (paid on every call), not arithmetic.

Result 2026-09-23 (onnxruntime-gpu 1.30.0, RTX 4070, results/cuda-lane-2026-09-23-summary.md):
  1. batch 1: 6.0 ms + 21.6 us per token (r2 0.992); 1 x 32 tokens 7.0 ms, 1 x 500 tokens 17.1 ms, 10 x 95 23.5 ms
     -> the floor is paid on every call and is ~80 % of a 1-question call
  2. ~1550 kernel entries per run (1391 CUDA, 163 CPU shape plumbing / memcpy) -> ~4 us per launch = the floor
  3. capture FAILS on this graph: "CUDA failure 700: an illegal memory access" in cuda_graph.cc during the capture
     run, and the CUDA context is dead afterwards (this part therefore runs last). The cause turned out to be one
     kernel, not the shape plumbing: ORT 1.30's CUDA GatherND copies a host vector during Compute
     (experiments/cuda_capture_bisect.py). Static bucket graphs with that node rewritten capture fine and are what
     tools/cuda_lane.py replays now (tools/static_graph.py, results/cuda-graph-2026-09-23-summary.md).
"""
import argparse
import collections
import json
import os
import statistics
import tempfile
import time

import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="models/laya-onnx-fp16")
ap.add_argument("--runs", type=int, default=30)
args = ap.parse_args()
MODEL = os.path.join(args.model_dir, "laya.onnx")

ort.set_default_logger_severity(3)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
try:
    import psutil

    psutil.Process().cpu_affinity(list(range(16)))  # P-cores, like the lane
except Exception:
    pass


def feeds_for(b, L, K=4):
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(1000, 30000, size=(b, L), dtype=np.int64),
        "attention_mask": np.ones((b, L), dtype=np.int64),
        "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (b, 1)),
        "marker_mask": np.ones((b, K), dtype=bool),
        "qtype": np.zeros((b,), dtype=np.int64),
    }


def timed(fn, runs):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[int(len(ts) * 0.9)]


def session(extra_provider_opts=None, profile_prefix=None):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 2
    if profile_prefix:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    opts = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested", **(extra_provider_opts or {})}
    return ort.InferenceSession(MODEL, so, providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])


# ---- 1. floor vs slope --------------------------------------------------------------------------------------
s = session()
print(f"onnxruntime {ort.__version__}, providers {s.get_providers()}")
print("\n1. session.run latency (p50 / min / p90 ms), CUDA EP, no graph capture")
print("   batch x tokens      p50      min      p90")
rows = []
for b, L in [(1, 32), (1, 85), (1, 160), (1, 320), (1, 500), (3, 85), (3, 160), (10, 95), (10, 200)]:
    f = feeds_for(b, L)
    p50, mn, p90 = timed(lambda: s.run(None, f), args.runs)
    rows.append((b, L, p50))
    print(f"   {b:2d} x {L:3d} tokens   {p50:7.1f}  {mn:7.1f}  {p90:7.1f}")
# fit p50 = a + c * (b * L) on batch-1 rows, then check batch rows against it
one = [(L, p) for b, L, p in rows if b == 1]
xs = np.array([L for L, _ in one], dtype=float)
ys = np.array([p for _, p in one], dtype=float)
slope, floor = np.polyfit(xs, ys, 1)
print(f"   batch 1 fit: {floor:.1f} ms + {slope * 1000:.1f} us per token  (r2 {1 - np.var(ys - (floor + slope * xs)) / np.var(ys):.3f})")

# ---- 2. profiler: launches per run ----------------------------------------------------------------------------
prefix = os.path.join(tempfile.gettempdir(), "laya-cuda-prof")
sp = session(profile_prefix=prefix)
f = feeds_for(3, 85)
for _ in range(5):
    sp.run(None, f)
N = 10
for _ in range(N):
    sp.run(None, f)
path = sp.end_profiling()
ev = json.load(open(path, encoding="utf-8"))
runs = [e for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run"][-N:]
t0, t1 = runs[0]["ts"], runs[-1]["ts"] + runs[-1]["dur"]
kern = [e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time") and t0 <= e["ts"] <= t1]
by_ep = collections.Counter(e["args"].get("provider", "?") for e in kern)
cpu_side = sum(e["dur"] for e in kern) / N / 1000
print(f"\n2. profiler, 3 x 85 tokens: model_run {statistics.median(r['dur'] for r in runs) / 1000:.1f} ms; kernel entries per run "
      f"{len(kern) // N} ({', '.join(f'{k.replace('ExecutionProvider', '')} {v // N}' for k, v in by_ep.items())}); "
      f"CPU-side time in those entries {cpu_side:.1f} ms per run")
os.remove(path)

# ---- 3. CUDA graph capture on fixed shapes -------------------------------------------------------------------
print("\n3. enable_cuda_graph + IO binding (fixed device buffers): replay vs session.run, same shape")
print("   batch x tokens   session.run p50   graph replay p50 / min     replay = launch overhead removed")
for b, L in [(1, 85), (3, 85), (10, 95), (1, 320)]:
    f = feeds_for(b, L)
    plain = next(p for bb, LL, p in rows if bb == b and LL == L) if any(bb == b and LL == L for bb, LL, _ in rows) else timed(lambda: s.run(None, f), args.runs)[0]
    try:
        g = session({"enable_cuda_graph": "1"})
        io = g.io_binding()
        dev = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in f.items()}
        for k, v in dev.items():
            io.bind_ortvalue_input(k, v)
        outs = {o.name: ort.OrtValue.ortvalue_from_shape_and_type([b, 4] if o.name == "logits" else [b, 2], np.float32, "cuda", 0) for o in g.get_outputs()}
        for k, v in outs.items():
            io.bind_ortvalue_output(k, v)
        g.run_with_iobinding(io)  # capture
        # refresh inputs in place between replays so this is a real request pattern, not a cached result
        new = feeds_for(b, L)
        def replay():
            dev["input_ids"].update_inplace(new["input_ids"])
            g.run_with_iobinding(io)
            return outs["logits"].numpy()
        p50, mn, p90 = timed(replay, args.runs)
        # sanity: replay answers equal a plain run on the same inputs
        ref = s.run(None, new)[0]
        ok = np.allclose(ref, outs["logits"].numpy(), atol=1e-2)
        print(f"   {b:2d} x {L:3d} tokens      {plain:7.1f} ms        {p50:7.1f} / {mn:5.1f} ms   {'' if ok else 'MISMATCH '}-> {plain - p50:5.1f} ms of every call was launch overhead ({100 * (plain - p50) / plain:.0f} %)")
    except Exception as e:
        print(f"   {b:2d} x {L:3d} tokens      {plain:7.1f} ms        graph capture failed: {type(e).__name__}: {str(e)[:160]}")
