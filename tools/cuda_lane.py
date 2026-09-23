"""CUDA process lane for LocalLaya: one ONNX Runtime session on the CUDA EP, driven over stdio.

    .venv/Scripts/python.exe tools/cuda_lane.py --model-dir models/laya-onnx-fp16 [--device 0] [--threads 2]
                                                 [--affinity 0-15] [--cuda-graph]

Why a process: onnxruntime-node has no CUDA EP on Windows; the Python wheel (onnxruntime-gpu, Microsoft's
onnxruntime-cuda-13 feed) has. The Node side (src/lane.mjs ProcessLane) sends the five input tensors of one
systemOne call and gets logits + act_probs back, so @receptron/laya's sequence building, temperatures and
answer formatting run unchanged in Node against a remote session.

Protocol (one JSON object per line):
  -> stdout on start   {"ready": true, "providers": [...], "loadMs": n, "pid": n, "device": n}
                       or {"ready": false, "error": "..."} and exit 1
  <- stdin             {"id": n, "feeds": {name: {"dtype": "int64"|"bool"|"float32", "dims": [...], "data": "<base64>"}}}
  -> stdout            {"id": n, "outputs": {"logits": {"dtype": "float32", "dims": [...], "data": "<base64>"}, "act_probs": {...}}, "ms": x}
                       or {"id": n, "error": "..."}
  <- stdin             {"op": "close"}   -> release the session, exit 0. EOF on stdin does the same.
Logs go to stderr. Requests are handled strictly in order (the router serialises anyway).
"""
import argparse
import base64
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--threads", type=int, default=2, help="intra-op threads for the few CPU-side nodes")
ap.add_argument("--affinity", default="", help="logical CPUs for this process, e.g. 0-15 (Windows hybrid CPUs: the P-cores)")
ap.add_argument("--cuda-graph", action="store_true", help="enable_cuda_graph. Currently FAILS on this graph (CUDA error 700 during capture, onnxruntime-gpu 1.30): needs a static-shape graph per bucket first - see docs/STATUS.md")
ap.add_argument("--log-level", type=int, default=3)
args = ap.parse_args()

out = sys.stdout
err = sys.stderr


def emit(obj):
    out.write(json.dumps(obj, separators=(",", ":")) + "\n")
    out.flush()


def log(msg):
    err.write(f"[cuda_lane {os.getpid()}] {msg}\n")
    err.flush()


t0 = time.perf_counter()
if args.affinity:
    try:
        import psutil

        lo, hi = (int(x) for x in args.affinity.split("-")) if "-" in args.affinity else (int(args.affinity), int(args.affinity))
        psutil.Process().cpu_affinity(list(range(lo, hi + 1)))
    except Exception as e:  # pragma: no cover - best effort
        log(f"affinity {args.affinity} not applied: {e}")

try:
    import numpy as np
    import onnxruntime as ort

    ort.set_default_logger_severity(args.log_level)  # hides the 1.30 'No registered plugin EP device' notice at session creation
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)  # pip-installed CUDA / cuDNN DLLs, no system install needed
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"this onnxruntime build has no CUDA EP (providers: {ort.get_available_providers()})")
    so = ort.SessionOptions()
    so.log_severity_level = args.log_level
    so.intra_op_num_threads = args.threads
    cuda_opts = {"device_id": args.device, "arena_extend_strategy": "kSameAsRequested"}
    if args.cuda_graph:
        cuda_opts["enable_cuda_graph"] = "1"
    session = ort.InferenceSession(os.path.join(args.model_dir, "laya.onnx"), so, providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA EP not active (providers in use: {session.get_providers()})")
    output_names = [o.name for o in session.get_outputs()]
except Exception as e:
    emit({"ready": False, "error": f"{type(e).__name__}: {e}"})
    log(f"load failed: {type(e).__name__}: {e}")
    sys.exit(1)

emit({"ready": True, "providers": session.get_providers(), "loadMs": round((time.perf_counter() - t0) * 1000), "pid": os.getpid(), "device": args.device, "onnxruntime": ort.__version__})
log(f"ready in {(time.perf_counter() - t0):.1f} s on CUDA device {args.device} ({args.model_dir}); onnxruntime {ort.__version__}")

DTYPES = {"int64": np.int64, "int32": np.int32, "bool": np.bool_, "float32": np.float32, "float16": np.float16}


def decode(t):
    return np.frombuffer(base64.b64decode(t["data"]), dtype=DTYPES[t["dtype"]]).reshape(t["dims"])


def encode(a):
    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "dims": list(a.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        log(f"bad request line: {e}")
        continue
    if req.get("op") == "close":
        break
    rid = req.get("id")
    try:
        feeds = {k: decode(v) for k, v in req["feeds"].items()}
        t = time.perf_counter()
        results = session.run(output_names, feeds)
        ms = (time.perf_counter() - t) * 1000
        emit({"id": rid, "outputs": {n: encode(r) for n, r in zip(output_names, results)}, "ms": round(ms, 3)})
    except Exception as e:
        emit({"id": rid, "error": f"{type(e).__name__}: {e}"})

del session
log("closed")
sys.exit(0)
