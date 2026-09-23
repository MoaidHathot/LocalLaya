"""Does ort.InferenceSession(...) creation on a helper thread block Python on the main thread (GIL) or only the GPU?
Main thread: tight tick loop, records the largest gap between ticks while the helper builds a CUDA session."""
import sys
import threading
import time
from pathlib import Path

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
path = Path("models/laya-onnx-fp16") / "laya-static-b3-l96-k8-w.onnx"
so = ort.SessionOptions()
so.log_severity_level = 3
prov = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
ort.InferenceSession(str(path), so, providers=prov)  # warm the CUDA context / DLLs
done = threading.Event()
timings = {}


def build():
    t = time.perf_counter()
    s = ort.InferenceSession(str(path), so, providers=prov)
    timings["create_ms"] = (time.perf_counter() - t) * 1000
    done.set()
    timings["keep"] = s


th = threading.Thread(target=build)
gaps = []
last = time.perf_counter()
th.start()
while not done.is_set():
    now = time.perf_counter()
    gaps.append((now - last) * 1000)
    last = now
    time.sleep(0.0005)
th.join()
gaps.sort()
print(f"session creation {timings['create_ms']:.0f} ms; main-thread tick gaps: max {gaps[-1]:.1f} ms, p99 {gaps[int(len(gaps) * 0.99)]:.1f} ms, ticks {len(gaps)}")
print("verdict:", "GIL held for most of the creation" if gaps[-1] > timings["create_ms"] * 0.5 else "GIL released (Python keeps running); the stall is on the CUDA side")
