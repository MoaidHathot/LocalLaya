"""Race test for feeding a captured CUDA graph: device-bound inputs updated with update_inplace vs CPU-bound inputs
copied by ORT inside the run. 200 random inputs each, compared with the dynamic graph.

    .venv/Scripts/python.exe -u experiments/cuda_input_race_probe.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from static_graph import bucket_name  # noqa: E402

bundle = Path("models/laya-onnx-fp16")
ort.set_default_logger_severity(4)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)

# cudaDeviceSynchronize through the CUDA runtime the wheel ships (nvidia/cu13/bin/cudart64_13.dll): after OrtValue.update_inplace
# (a synchronous cudaMemcpy from pageable memory, which may return before the DMA completes) and before the graph replays
import ctypes, glob, os

cudart = None
for pat in ("nvidia/cu13/bin/x86_64/cudart64_*.dll", "nvidia/cu13/bin/cudart64_*.dll", "nvidia/cuda_runtime/bin/cudart64_*.dll", "nvidia/cu12/bin/cudart64_*.dll"):
    hits = glob.glob(os.path.join(os.path.dirname(os.path.dirname(ort.__file__)), pat))
    if hits:
        cudart = ctypes.CDLL(hits[0])
        break
print("cudart:", cudart._name if cudart else "not found", flush=True)


def device_sync():
    if cudart is not None:
        rc = cudart.cudaDeviceSynchronize()
        if rc != 0:
            raise RuntimeError(f"cudaDeviceSynchronize failed: {rc}")
NP = {onnx.TensorProto.FLOAT16: np.float16, onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.INT64: np.int64}
n, L, K = 3, 96, 8


def feeds_for(seed):
    rng = np.random.default_rng(seed)
    return {"input_ids": rng.integers(1000, 30000, size=(n, L), dtype=np.int64), "attention_mask": np.ones((n, L), dtype=np.int64), "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (n, 1)), "marker_mask": np.ones((n, K), dtype=bool), "qtype": np.zeros((n,), dtype=np.int64)}


so = ort.SessionOptions()
so.log_severity_level = 4
so.intra_op_num_threads = 2
dyn = ort.InferenceSession(str(bundle / "laya.onnx"), so, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
base = onnx.load(str(bundle / "laya.onnx"), load_external_data=False)
data = np.memmap(bundle / "laya.onnx.data", dtype=np.uint8, mode="r")
dev_w = {}
for t in base.graph.initializer:
    if t.data_location == onnx.TensorProto.EXTERNAL:
        kv = {x.key: x.value for x in t.external_data}
        off, length = int(kv["offset"]), int(kv["length"])
        dev_w[t.name] = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(np.frombuffer(data[off : off + length], dtype=NP[t.data_type]).reshape(list(t.dims))), "cuda", 0)
path = bundle / bucket_name(n, L, K).replace(".onnx", "-w.onnx")


def make(mode):
    s = ort.InferenceSession(str(path), so, providers=[("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested", "enable_cuda_graph": "1"}), "CPUExecutionProvider"])
    io = s.io_binding()
    for k, v in dev_w.items():
        io.bind_ortvalue_input(k, v)
    f0 = feeds_for(0)
    if mode == "device":
        bufs = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in f0.items()}
    else:  # cpu: ORT copies the bound host buffer inside the run; we overwrite the same numpy arrays per call
        host = {k: np.ascontiguousarray(v.copy()) for k, v in f0.items()}
        bufs = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in host.items()}
        bufs["_host"] = host
    for k, v in bufs.items():
        if k != "_host":
            io.bind_ortvalue_input(k, v)
    outs = {o.name: ort.OrtValue.ortvalue_from_shape_and_type([n, K] if o.name == "logits" else [n, 2], np.float32, "cuda", 0) for o in s.get_outputs()}
    for k, v in outs.items():
        io.bind_ortvalue_output(k, v)
    s.run_with_iobinding(io)
    s.run_with_iobinding(io)
    return s, io, bufs, outs


for mode in ["device", "device+sync", "cpu"]:
    try:
        s, io, bufs, outs = make("device" if mode.startswith("device") else "cpu")
    except Exception as e:
        print(f"{mode}: setup failed: {type(e).__name__}: {str(e)[:140]}", flush=True)
        continue
    worst = 0.0
    bad = 0
    ts = []
    for i in range(1, 201):
        f = feeds_for(i)
        t0 = time.perf_counter()
        if mode.startswith("device"):
            for k, v in f.items():
                bufs[k].update_inplace(np.ascontiguousarray(v))
            if mode == "device+sync":
                device_sync()
        else:
            for k, v in f.items():
                bufs["_host"][k][...] = v
        s.run_with_iobinding(io)
        got = outs["logits"].numpy()
        ts.append((time.perf_counter() - t0) * 1000)
        d = float(np.max(np.abs(got - dyn.run(None, f)[0])))
        worst = max(worst, d)
        if d > 0.05:
            bad += 1
    ts.sort()
    print(f"{mode:12s} inputs: 200 random calls, worst max|dlogit| {worst:.4f}, calls beyond 0.05: {bad}; per call incl. input update p50 {ts[100]:.2f} ms", flush=True)
