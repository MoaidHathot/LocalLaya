"""Weights as graph inputs bound to shared device buffers: does CUDA Graph capture still work, what do sessions cost?

    .venv/Scripts/python.exe -u experiments/cuda_shared_weights_probe.py

For 6 buckets: create a session from the "-w" static graph (weights are inputs), bind ONE shared set of weight
OrtValues + per-bucket input/output buffers, capture, replay; report creation time, capture time, VRAM per
session, replay p50 and the max |delta logit| against the dynamic graph. This is the configuration
tools/cuda_lane.py will use.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from static_graph import bucket_name, make_static, weights_as_inputs  # noqa: E402

bundle = Path("models/laya-onnx-fp16")
ort.set_default_logger_severity(4)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
try:
    import psutil

    psutil.Process().cpu_affinity(list(range(16)))
except Exception:
    pass
NP = {onnx.TensorProto.FLOAT16: np.float16, onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.INT64: np.int64}


def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().split("\n")[0])


def feeds_for(n, L, K, seed=0):
    rng = np.random.default_rng(seed)
    return {"input_ids": rng.integers(1000, 30000, size=(n, L), dtype=np.int64), "attention_mask": np.ones((n, L), dtype=np.int64), "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (n, 1)), "marker_mask": np.ones((n, K), dtype=bool), "qtype": np.zeros((n,), dtype=np.int64)}


def p50(xs):
    return sorted(xs)[len(xs) // 2]


so_dyn = ort.SessionOptions()
so_dyn.log_severity_level = 4
so_dyn.intra_op_num_threads = 2
dyn = ort.InferenceSession(str(bundle / "laya.onnx"), so_dyn, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])

# ---- weights: read once from laya.onnx.data, upload once ---------------------------------------------------------
base = onnx.load(str(bundle / "laya.onnx"), load_external_data=False)
ext = [t for t in base.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
t0 = time.perf_counter()
data = np.memmap(bundle / "laya.onnx.data", dtype=np.uint8, mode="r")
weights = {}
for t in ext:
    kv = {x.key: x.value for x in t.external_data}
    off, length = int(kv["offset"]), int(kv["length"])
    weights[t.name] = np.frombuffer(data[off : off + length], dtype=NP[t.data_type]).reshape(list(t.dims))
v0 = vram()
dev_w = {k: ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(v), "cuda", 0) for k, v in weights.items()}
print(f"weights: {len(dev_w)} tensors, {sum(v.nbytes for v in weights.values()) / 1e6:.0f} MB read + uploaded in {time.perf_counter() - t0:.2f} s; VRAM +{vram() - v0} MiB", flush=True)
del data, weights

buckets = [(1, 96, 8), (3, 96, 8), (5, 128, 8), (10, 96, 8), (2, 256, 8), (4, 512, 8)]
print("\nbucket        create   capture   VRAM/session   replay p50 / min    vs dynamic p50   max|dlogit|", flush=True)
sessions = []
for n, L, K in buckets:
    path = bundle / bucket_name(n, L, K).replace(".onnx", "-w.onnx")
    if not path.exists():
        model, _ = make_static(bundle / "laya.onnx", n, L, K, verbose=False)
        weights_as_inputs(model)
        onnx.save_model(model, str(path))
    v_before = vram()
    t = time.perf_counter()
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.intra_op_num_threads = 2
    s = ort.InferenceSession(str(path), so, providers=[("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested", "enable_cuda_graph": "1"}), "CPUExecutionProvider"])
    t_create = time.perf_counter() - t
    f = feeds_for(n, L, K, seed=5)
    io = s.io_binding()
    for k, v in dev_w.items():
        io.bind_ortvalue_input(k, v)
    dev_in = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in f.items()}
    for k, v in dev_in.items():
        io.bind_ortvalue_input(k, v)
    outs = {o.name: ort.OrtValue.ortvalue_from_shape_and_type([n, K] if o.name == "logits" else [n, 2], np.float32, "cuda", 0) for o in s.get_outputs()}
    for k, v in outs.items():
        io.bind_ortvalue_output(k, v)
    t = time.perf_counter()
    s.run_with_iobinding(io)
    s.run_with_iobinding(io)  # capture
    t_capture = time.perf_counter() - t
    f2 = feeds_for(n, L, K, seed=9)
    for k, v in f2.items():
        dev_in[k].update_inplace(np.ascontiguousarray(v))
    ts = []
    for _ in range(30):
        tt = time.perf_counter()
        s.run_with_iobinding(io)
        ts.append((time.perf_counter() - tt) * 1000)
    got = outs["logits"].numpy().copy()
    ref = dyn.run(None, f2)[0]
    td = []
    for _ in range(30):
        tt = time.perf_counter()
        dyn.run(None, f2)
        td.append((time.perf_counter() - tt) * 1000)
    sessions.append((s, io, dev_in, outs))
    print(f"{n:2d}x{L:3d}x{K:<2d}   {t_create * 1000:6.0f} ms  {t_capture * 1000:6.0f} ms   {vram() - v_before:6d} MiB     {p50(ts):6.1f} / {min(ts):5.1f} ms      {p50(td):6.1f} ms        {float(np.max(np.abs(got - ref))):.4f}", flush=True)
print(f"\ntotal VRAM over start: +{vram() - v0} MiB for the weights + {len(sessions)} captured sessions", flush=True)
# replay all buckets once more to be sure the shared weights are intact after many sessions
worst = 0.0
for (s, io, dev_in, outs), (n, L, K) in zip(sessions, buckets):
    f = feeds_for(n, L, K, seed=21)
    for k, v in f.items():
        dev_in[k].update_inplace(np.ascontiguousarray(v))
    s.run_with_iobinding(io)
    worst = max(worst, float(np.max(np.abs(outs["logits"].numpy() - dyn.run(None, f)[0]))))
print(f"all {len(sessions)} sessions replayed again with fresh inputs: worst max|dlogit| vs dynamic {worst:.4f}", flush=True)
