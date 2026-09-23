"""How long does ORT hold the GIL to create a CUDA session for a static bucket graph, and does a pre-optimised
file (optimized_model_filepath, then ORT_DISABLE_ALL) make it cheaper? That creation time is the stall a serving
thread suffers while tools/cuda_lane.py builds a bucket (experiments/cuda_gil_probe.py)."""
import sys
import time
from pathlib import Path

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
ort.set_default_logger_severity(3)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
model_dir = Path("models/laya-onnx-fp16")
path = model_dir / "laya-static-b3-l96-k8-w.onnx"
prov = [("CUDAExecutionProvider", {"device_id": 0, "enable_cuda_graph": "1", "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]


def opts(level=None, optimized_out=None):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 2
    if level is not None:
        so.graph_optimization_level = level
    if optimized_out:
        so.optimized_model_filepath = str(optimized_out)
    return so


def timed(label, so, p, n=3):
    ts = []
    keep = None
    for _ in range(n):
        t = time.perf_counter()
        keep = ort.InferenceSession(str(p), so, providers=prov)
        ts.append((time.perf_counter() - t) * 1000)
    print(f"{label:58s} create {min(ts):5.0f} ms (min of {n}), {sorted(ts)[len(ts) // 2]:5.0f} median")
    return keep


ort.InferenceSession(str(path), opts(), providers=prov)  # warm the DLLs / context
timed("default optimisation (what the lane does today)", opts(), path)
timed("ORT_DISABLE_ALL on the raw static file", opts(ort.GraphOptimizationLevel.ORT_DISABLE_ALL), path)
timed("ORT_ENABLE_BASIC", opts(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC), path)
pre = model_dir / "_probe_preopt.onnx"
ort.InferenceSession(str(path), opts(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, pre), providers=prov)
timed("pre-optimised file (ENABLE_ALL saved) + ORT_DISABLE_ALL", opts(ort.GraphOptimizationLevel.ORT_DISABLE_ALL), pre)
timed("pre-optimised file + default optimisation", opts(), pre)
so = opts(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
so.enable_mem_pattern = False
timed("pre-optimised + DISABLE_ALL + no mem pattern", so, pre)
print(f"pre-optimised file size {pre.stat().st_size / 1e6:.1f} MB vs {path.stat().st_size / 1e6:.1f} MB")
pre.unlink(missing_ok=True)
