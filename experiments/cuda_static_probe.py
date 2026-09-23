"""CUDA Graph capture on the static-shape graphs (tools/static_graph.py): does it work now, what does replay cost,
and can many bucket sessions share one copy of the weights?

    .venv/Scripts/python.exe -u experiments/cuda_static_probe.py [--bundle models/laya-onnx-fp16]

  1. K-dependence of the dynamic graph (does the option count matter for latency?)
  2. per bucket: static graph, enable_cuda_graph + IO binding -> capture, replay p50 vs the dynamic session on the
     same inputs, max |delta logit|; the pinned-affinity, 2-thread configuration of tools/cuda_lane.py
  3. weight sharing: N static sessions created with SessionOptions.add_initializer(<CUDA OrtValue>) for every
     external initializer vs N plain sessions - VRAM (nvidia-smi) and creation time per session
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from static_graph import bucket_name, make_static  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="models/laya-onnx-fp16")
ap.add_argument("--buckets", default="1x96x8,3x96x8,3x160x8,10x96x8,1x256x8,4x512x8")
ap.add_argument("--runs", type=int, default=40)
args = ap.parse_args()
bundle = Path(args.bundle)
ort.set_default_logger_severity(3)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
try:
    import psutil

    psutil.Process().cpu_affinity(list(range(16)))
except Exception:
    pass


def log(m):
    print(m, flush=True)


def vram():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def feeds_for(n, L, K, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(1000, 30000, size=(n, L), dtype=np.int64),
        "attention_mask": np.ones((n, L), dtype=np.int64),
        "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (n, 1)),
        "marker_mask": np.ones((n, K), dtype=bool),
        "qtype": np.zeros((n,), dtype=np.int64),
    }


def p50(xs):
    return sorted(xs)[len(xs) // 2]


def timed(fn, runs=args.runs, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return p50(ts), min(ts), sorted(ts)[int(len(ts) * 0.9)]


def sess_opts(extra_config=None):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 2
    for k, v in (extra_config or {}).items():
        so.add_session_config_entry(k, v)
    return so


def cuda_session(path, so=None, graph=False):
    opts = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}
    if graph:
        opts["enable_cuda_graph"] = "1"
    return ort.InferenceSession(str(path), so or sess_opts(), providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])


dyn = cuda_session(bundle / "laya.onnx")

# ---- 1. K dependence ---------------------------------------------------------------------------------------------
log("1. dynamic graph, 3 x 96 tokens, latency vs K (options per question):")
for K in [2, 4, 8, 16, 32]:
    f = feeds_for(3, 96, K)
    m, mn, p90 = timed(lambda: dyn.run(None, f))
    log(f"   K={K:2d}: p50 {m:6.1f} ms  (min {mn:.1f}, p90 {p90:.1f})")

# ---- 2. capture + replay per bucket -----------------------------------------------------------------------------
log("\n2. static graph + CUDA Graph capture, per bucket (same inputs as the dynamic graph):")
log("   bucket        dynamic p50   static, no graph   static + graph replay p50 / min    max|dlogit| vs dynamic   speedup")
buckets = [tuple(int(x) for x in b.split("x")) for b in args.buckets.split(",")]
static_paths = {}
for n, L, K in buckets:
    path = bundle / bucket_name(n, L, K)
    if not path.exists():
        model, report = make_static(bundle / "laya.onnx", n, L, K, verbose=False)
        onnx.save_model(model, str(path))
    static_paths[(n, L, K)] = path
    f = feeds_for(n, L, K, seed=7)
    d50, _, _ = timed(lambda: dyn.run(None, f))
    ref = dyn.run(None, f)
    s_plain = cuda_session(path)
    s50, _, _ = timed(lambda: s_plain.run(None, f))
    del s_plain
    try:
        g = cuda_session(path, graph=True)
        io = g.io_binding()
        dev = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in f.items()}
        for k, v in dev.items():
            io.bind_ortvalue_input(k, v)
        outs = {o.name: ort.OrtValue.ortvalue_from_shape_and_type([n, K] if o.name == "logits" else [n, 2], np.float32, "cuda", 0) for o in g.get_outputs()}
        for k, v in outs.items():
            io.bind_ortvalue_output(k, v)
        g.run_with_iobinding(io)  # regular run
        g.run_with_iobinding(io)  # capture
        f2 = feeds_for(n, L, K, seed=11)
        ref2 = dyn.run(None, f2)

        def replay():
            for k, v in f2.items():
                dev[k].update_inplace(np.ascontiguousarray(v))
            g.run_with_iobinding(io)
            return outs["logits"].numpy()

        r50, rmin, _ = timed(replay)
        got = replay()
        diff = float(np.max(np.abs(got - ref2[0])))
        act = float(np.max(np.abs(outs["act_probs"].numpy() - ref2[1])))
        log(f"   {n:2d}x{L:3d}x{K:<2d}   {d50:9.1f} ms   {s50:12.1f} ms   {r50:16.1f} / {rmin:5.1f} ms   {max(diff, act):20.4f}   {d50 / r50:6.2f}x")
        del g
    except Exception as e:
        log(f"   {n:2d}x{L:3d}x{K:<2d}   {d50:9.1f} ms   {s50:12.1f} ms   capture FAILED: {type(e).__name__}: {str(e)[:120]}")

# ---- 3. weight sharing across sessions -----------------------------------------------------------------------------
log("\n3. weight sharing: sessions for the same bucket, with / without add_initializer(CUDA OrtValue)")
model = onnx.load(str(bundle / "laya.onnx"), load_external_data=False)
ext = [t for t in model.graph.initializer if t.data_location == TensorProto.EXTERNAL]
full = onnx.load(str(bundle / "laya.onnx"), load_external_data=True)
weights = {t.name: numpy_helper.to_array(t) for t in full.graph.initializer if t.name in {e.name for e in ext}}
del full
log(f"   {len(weights)} external initializers, {sum(w.nbytes for w in weights.values()) / 1e6:.0f} MB")
path = static_paths[buckets[1]] if len(buckets) > 1 else static_paths[buckets[0]]
n, L, K = buckets[1] if len(buckets) > 1 else buckets[0]
v0 = vram()
t = time.perf_counter()
plain = [cuda_session(path) for _ in range(2)]
t_plain = (time.perf_counter() - t) / 2
v1 = vram()
log(f"   2 plain sessions: +{v1 - v0} MiB VRAM, {t_plain * 1000:.0f} ms per session")
del plain
time.sleep(0.5)
v2 = vram()
t = time.perf_counter()
dev_weights = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in weights.items()}
t_upload = time.perf_counter() - t
v3 = vram()
log(f"   weights uploaded once as CUDA OrtValues: +{v3 - v2} MiB, {t_upload * 1000:.0f} ms")
shared = []
t = time.perf_counter()
for _ in range(3):
    so = sess_opts()
    for k, v in dev_weights.items():
        so.add_initializer(k, v)
    shared.append(cuda_session(path, so=so))
t_shared = (time.perf_counter() - t) / 3
v4 = vram()
log(f"   3 sessions sharing them: +{v4 - v3} MiB VRAM total ({(v4 - v3) / 3:.0f} MiB each), {t_shared * 1000:.0f} ms per session")
f = feeds_for(n, L, K, seed=3)
ref = dyn.run(None, f)
got = shared[0].run(None, f)
log(f"   shared-weight session output vs dynamic: max|dlogit| {float(np.max(np.abs(got[0] - ref[0]))):.4f}")
m, _, _ = timed(lambda: shared[0].run(None, f))
log(f"   shared-weight session, no graph: p50 {m:.1f} ms")
# and with graph capture on a shared-weight session
try:
    so = sess_opts()
    for k, v in dev_weights.items():
        so.add_initializer(k, v)
    g = cuda_session(path, so=so, graph=True)
    io = g.io_binding()
    dev = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in f.items()}
    for k, v in dev.items():
        io.bind_ortvalue_input(k, v)
    outs = {o.name: ort.OrtValue.ortvalue_from_shape_and_type([n, K] if o.name == "logits" else [n, 2], np.float32, "cuda", 0) for o in g.get_outputs()}
    for k, v in outs.items():
        io.bind_ortvalue_output(k, v)
    g.run_with_iobinding(io)
    g.run_with_iobinding(io)
    m, mn, _ = timed(lambda: g.run_with_iobinding(io))
    log(f"   shared-weight session + graph replay: p50 {m:.1f} ms (min {mn:.1f}); max|dlogit| {float(np.max(np.abs(outs['logits'].numpy() - ref[0]))):.4f}; VRAM now +{vram() - v0} MiB over the start")
except Exception as e:
    log(f"   shared-weight session + graph: FAILED {type(e).__name__}: {str(e)[:160]}")
