"""CUDA process lane for LocalLaya: ONNX Runtime CUDA EP sessions driven over stdio, with CUDA Graph replay.

    .venv/Scripts/python.exe tools/cuda_lane.py --model-dir models/laya-onnx-fp16 [--device 0] [--threads 2]
        [--affinity 0-15] [--graph auto|off] [--graph-buckets 1x96x8,3x96x8,5x96x8] [--graph-max-sessions 16]
        [--graph-max-vram-mib 1536] [--graph-max-work 1536]
    .venv/Scripts/python.exe tools/cuda_lane.py --model-dir models/laya-onnx-fp16 --self-test

Why a process: onnxruntime-node has no CUDA EP on Windows; the Python wheel (onnxruntime-gpu) has. The Node side
(src/lane.mjs ProcessLane) sends the five input tensors of one systemOne call and gets logits + act_probs back,
so @receptron/laya's sequence building, temperatures and answer formatting run unchanged in Node.

Two ways to run a call (results/cuda-graph-2026-09-23-summary.md):
  dynamic   the exported graph as is, one session, any shape: ~6 ms of kernel-launch floor + 22 us per token,
            because ORT launches ~1400 kernels per forward pass one by one.
  graph     a static-shape variant of the graph for a (rows n, length L, options K) "bucket" (tools/static_graph.py:
            shape plumbing folded, GatherND -> Reshape because ORT's GatherND kernel is not capture-safe), captured
            as a CUDA Graph and replayed with a single launch: the ~4 ms of launch overhead is gone (1 question
            8 -> 4 ms, 3 questions 11.5 -> 8, 10 questions 24 -> 21). Inputs are padded up to the bucket
            (attention_mask 0 / marker_mask 0 for the padding, rows duplicated), outputs sliced back. Shapes above
            --graph-max-work (n x L) stay dynamic - at that size the gain is a few percent. Every session binds the
            SAME device copy of the weights (weights are graph inputs), so N buckets cost one 843 MB copy plus
            ~50-250 MiB of activations each; --graph-max-sessions / --graph-max-vram-mib bound that (LRU eviction).

How a bucket comes to life - two threads, because of two ORT facts:
  1. ORT keeps the captured graph in the *calling thread's* per-thread context and captures on the third run of a
     session on that thread (cuda_execution_provider.h: "cuda_graph_ is put under PerThreadContext",
     min_num_runs_before_cuda_graph_capture_ = 2). A graph captured on a helper thread is invisible to the thread
     that serves requests, so the runs that capture must happen on the serving thread.
  2. The capture uses cudaStreamCaptureModeGlobal: any "unsafe" CUDA call from another thread meanwhile
     (cudaMalloc, synchronous cudaMemcpy, cudaStreamSynchronize...) fails with CUDA error 900 and invalidates the
     capture. Measured: building in the background while serving lost 7 of 400 calls and 3 of 4 builds.
  So the builder thread only *prepares* a bucket (static ONNX file, InferenceSession, IO binding, device buffers)
  under `gpu_lock`, and the serving thread *finalises* it under the same lock: two regular runs, the capturing run,
  one replay, plus a check of that replay against the dynamic session on the last real inputs seen for the shape
  (a broken capture is detected before it serves anything). Finalisation (~50-150 ms) runs in idle gaps of the
  request loop; under steady traffic it is forced onto the first request 0.5 s after the bucket was prepared, so a
  new shape costs one slow call ~1-2 s after it first appeared and replays from then on. ORT holds the GIL while
  it creates a session (317 of 340 ms measured, experiments/cuda_gil_probe.py), which freezes the serving thread
  for that long, so the builder waits for a 100 ms gap in the requests before such a step (at most 1 s), and
  --graph-buckets are prepared and captured BEFORE the ready line (+~1.4 s start-up with the static files cached,
  longer the first time; `npm run cuda:setup` pre-generates them) - the router's start-up probe and warm-up then
  measure an undisturbed lane. Lazy buckets are only scheduled for shapes seen twice; a call with
  exec.graph = false neither uses nor schedules buckets.

Protocol (one JSON object per line):
  -> stdout on start   {"ready": true, "providers": [...], "loadMs": n, "pid": n, "device": n, "graph": {...}}
                       or {"ready": false, "error": "..."} and exit 1
  <- stdin             {"id": n, "feeds": {name: {"dtype": "int64"|"bool"|"float32", "dims": [...], "data": "<base64>"}},
                        "exec": {"graph": false}?}
  -> stdout            {"id": n, "outputs": {"logits": {...}, "act_probs": {...}}, "ms": x, "mode": "graph"|"dynamic",
                        "bucket": [n, L, K] | null}       or {"id": n, "error": "..."}
  <- stdin             {"op": "close"}   -> release everything, exit 0. EOF on stdin does the same.
  <- stdin             {"id": n, "op": "stats"}                    -> {"id": n, "stats": {...}} (buckets, hits, VRAM)
  <- stdin             {"id": n, "op": "bucket", "n": 3, "L": 97, "K": 8} -> {"id": n, "bucket": [3, 128, 8] | null}
Logs go to stderr. Requests are handled strictly in order (the router serialises anyway). A CUDA error inside a
call is reported to that call and then the process exits (3): CUDA errors are sticky, and a dead process is what the
Node side knows how to fail over from.
"""
import argparse
import base64
import collections
import ctypes
import glob
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--threads", type=int, default=2, help="intra-op threads for the few CPU-side nodes")
ap.add_argument("--affinity", default="", help="logical CPUs for this process, e.g. 0-15 (Windows hybrid CPUs: the P-cores)")
ap.add_argument("--graph", default="auto", choices=["auto", "off"], help="CUDA Graph replay on static bucket graphs (default auto)")
ap.add_argument("--graph-buckets", default="1x96x8,3x96x8,5x96x8,5x128x8", help="buckets to build eagerly in the background (nxLxK)")
ap.add_argument("--graph-max-sessions", type=int, default=16)
ap.add_argument("--graph-max-vram-mib", type=int, default=1536, help="budget for the bucket sessions' activations (the shared weights are separate)")
ap.add_argument("--graph-max-work", type=int, default=1536, help="n x L above which calls stay dynamic")
ap.add_argument("--log-level", type=int, default=3)
ap.add_argument("--self-test", action="store_true", help="graph vs dynamic on real-shaped inputs for several buckets; exit 1 on a mismatch")
args = ap.parse_args()

out = sys.stdout
err = sys.stderr
GRAPH_L = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512]
GRAPH_N = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
GRAPH_K = [8, 16, 32]
FINALISE_FORCE_S = 0.5  # a prepared bucket that found no idle gap for this long is finalised on the next matching request
FINALISE_IDLE_S = 0.1  # idle time in the request loop before a prepared bucket is finalised
PREPARE_IDLE_S = 0.1  # the builder waits for this much idleness of the serving thread before a GIL-holding step...
PREPARE_FORCE_S = 1.0  # ...but not longer than this (continuous traffic: one ~250 ms freeze beats hundreds of +4 ms calls)
FINALISE_MAX_DELTA = 0.5  # replay vs dynamic on real inputs above this = broken capture (fp16 noise on real text is < 0.1)
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_probs"]
PREPARED = "\x00prepared"  # request-queue sentinel from the builder thread: a bucket is waiting for finalisation


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
    import onnx
    from onnx import TensorProto
    import onnxruntime as ort

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from static_graph import bucket_name, dummy_feeds, make_static, weights_as_inputs

    ort.set_default_logger_severity(args.log_level)
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"this onnxruntime build has no CUDA EP (providers: {ort.get_available_providers()})")
    model_dir = Path(args.model_dir)
    NP = {TensorProto.FLOAT16: np.float16, TensorProto.FLOAT: np.float32, TensorProto.INT64: np.int64, TensorProto.BOOL: np.bool_, TensorProto.INT32: np.int32}

    # ---- the CUDA runtime: cudaDeviceSynchronize after input updates, cudaMemGetInfo for the VRAM budget -------------
    cudart = None
    for pat in ("nvidia/cu13/bin/x86_64/cudart64_*.dll", "nvidia/cu13/bin/cudart64_*.dll", "nvidia/cuda_runtime/bin/cudart64_*.dll", "nvidia/cu12/bin/cudart64_*.dll"):
        hits = glob.glob(os.path.join(os.path.dirname(os.path.dirname(ort.__file__)), pat))
        if hits:
            cudart = ctypes.CDLL(hits[0])
            break
    graph_enabled = args.graph == "auto"
    graph_disabled_reason = None if graph_enabled else "--graph off"
    if graph_enabled and cudart is None:
        graph_disabled_reason = "cudart DLL not found next to onnxruntime"
        log(f"{graph_disabled_reason}; CUDA Graph replay disabled (inputs could race the replay)")
        graph_enabled = False

    def device_sync():
        rc = cudart.cudaDeviceSynchronize()
        if rc != 0:
            raise RuntimeError(f"cudaDeviceSynchronize failed: CUDA error {rc}")

    def vram_used():
        """Device-wide used MiB (all processes) via cudaMemGetInfo; only differences taken by one thread are meaningful."""
        if cudart is None:
            return -1
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return (total.value - free.value) // 2**20 if cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else -1

    def sess_opts():
        so = ort.SessionOptions()
        so.log_severity_level = args.log_level
        so.intra_op_num_threads = args.threads
        return so

    def providers(graph):
        o = {"device_id": args.device, "arena_extend_strategy": "kSameAsRequested"}
        if graph:
            o["enable_cuda_graph"] = "1"
        return [("CUDAExecutionProvider", o), "CPUExecutionProvider"]

    # ---- weights: read from laya.onnx.data once, upload once, bind to every session -------------------------------
    base = onnx.load(str(model_dir / "laya.onnx"), load_external_data=False)
    ext = [t for t in base.graph.initializer if t.data_location == TensorProto.EXTERNAL]
    data = np.memmap(model_dir / "laya.onnx.data", dtype=np.uint8, mode="r")
    dev_weights = {}
    weight_bytes = 0
    for t in ext:
        kv = {x.key: x.value for x in t.external_data}
        off, length = int(kv["offset"]), int(kv["length"])
        arr = np.ascontiguousarray(np.frombuffer(data[off : off + length], dtype=NP[t.data_type]).reshape(list(t.dims)))
        dev_weights[t.name] = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", args.device)
        weight_bytes += arr.nbytes
    del data, base

    def bind_weights(io):
        for k, v in dev_weights.items():
            io.bind_ortvalue_input(k, v)

    # Serialises the serving thread's capture (finalise) against the builder thread's CUDA calls (probe run, session
    # creation, buffer allocation). Plain serving calls do not take it: nothing captures while they run.
    gpu_lock = threading.Lock()
    serve_state = {"serving": False, "busy": False, "idle_since": time.perf_counter()}  # written by the serving thread only

    def wait_for_idle():
        """Builder thread: ORT holds the GIL while it creates a session (~250-350 ms), which freezes the serving thread
        for that long. Wait for a gap in the requests so the freeze hits nobody; give up after PREPARE_FORCE_S."""
        t = time.perf_counter()
        while serve_state["serving"] and time.perf_counter() - t < PREPARE_FORCE_S:
            if not serve_state["busy"] and time.perf_counter() - serve_state["idle_since"] >= PREPARE_IDLE_S:
                return
            time.sleep(0.005)

    def probe_shapes_cuda(probe, model_path, n, L, K):
        """Shape-observation pass for make_static on the GPU with the already-uploaded weights (~0.3 s instead of the
        CPU default's 3-8 s, which loads the weights from disk). Shape outputs do not depend on weight values."""
        weights_as_inputs(probe)
        so = sess_opts()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        wait_for_idle()
        with gpu_lock:
            s = ort.InferenceSession(probe.SerializeToString(), so, providers=providers(False))
            io = s.io_binding()
            bind_weights(io)
            for k, v in dummy_feeds(n, L, K).items():
                io.bind_cpu_input(k, np.ascontiguousarray(v))
            names = [o.name for o in s.get_outputs()]
            for name in names:
                io.bind_output(name, "cpu")
            s.run_with_iobinding(io)
            values = dict(zip(names, io.copy_outputs_to_cpu()))
            del io, s
        return values

    # ---- the dynamic session (any shape) on the shared weights ----------------------------------------------------
    dyn_path = model_dir / "laya-dynamic-w.onnx"
    if not dyn_path.exists():
        m = onnx.load(str(model_dir / "laya.onnx"), load_external_data=False)
        weights_as_inputs(m)
        onnx.save_model(m, str(dyn_path))
    dyn = ort.InferenceSession(str(dyn_path), sess_opts(), providers=providers(False))
    if dyn.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA EP not active (providers in use: {dyn.get_providers()})")
    dyn_io = dyn.io_binding()
    bind_weights(dyn_io)

    def run_dynamic(feeds):
        for k in INPUT_NAMES:
            dyn_io.bind_cpu_input(k, np.ascontiguousarray(feeds[k]))
        for k in OUTPUT_NAMES:
            dyn_io.bind_output(k, "cuda", args.device)
        dyn.run_with_iobinding(dyn_io)
        res = dyn_io.copy_outputs_to_cpu()
        dyn_io.clear_binding_outputs()
        return dict(zip(OUTPUT_NAMES, res))

    # ---- bucket sessions with CUDA Graph replay -------------------------------------------------------------------
    def bucket_for(n, L, K):
        bn = next((x for x in GRAPH_N if x >= n), None)
        bl = next((x for x in GRAPH_L if x >= L), None)
        bk = next((x for x in GRAPH_K if x >= K), None)
        if bn is None or bl is None or bk is None or bn * bl > args.graph_max_work:
            return None
        return (bn, bl, bk)

    def pad_feeds(feeds, key):
        bn, bl, bk = key
        n, L = feeds["input_ids"].shape
        K = feeds["marker_pos"].shape[1]
        ids = np.zeros((bn, bl), np.int64)
        mask = np.zeros((bn, bl), np.int64)
        mpos = np.zeros((bn, bk), np.int64)
        mmask = np.zeros((bn, bk), np.bool_)
        qt = np.zeros((bn,), np.int64)
        ids[:n, :L] = feeds["input_ids"]
        mask[:n, :L] = feeds["attention_mask"]
        mpos[:n, :K] = feeds["marker_pos"]
        mmask[:n, :K] = feeds["marker_mask"]
        qt[:n] = feeds["qtype"]
        if bn > n:  # padded rows: copies of row 0 (a row with an all-zero mask is what we never want to feed)
            ids[n:] = ids[0]
            mask[n:] = mask[0]
            mpos[n:] = mpos[0]
            mmask[n:] = mmask[0]
            qt[n:] = qt[0]
        return {"input_ids": ids, "attention_mask": mask, "marker_pos": mpos, "marker_mask": mmask, "qtype": qt}

    def max_delta(a, b):
        return max(float(np.max(np.abs(a["logits"] - b["logits"]))), float(np.max(np.abs(a["act_probs"] - b["act_probs"]))))

    class BucketSession:
        """One static-shape session. state: building -> prepared -> ready, or failed."""

        def __init__(self, n, L, K):
            self.key = (n, L, K)
            self.n, self.L, self.K = n, L, K
            self.state = "building"
            self.failed = None
            self.last_used = time.perf_counter()
            self.prepared_at = None
            self.hits = 0
            self.vram_mib = 0
            self.prepare_ms = 0.0
            self.finalise_ms = 0.0
            self.check_delta = None

        def prepare(self):
            """Builder thread: static ONNX file (cached next to laya.onnx), session, IO binding, device buffers. No runs."""
            t = time.perf_counter()
            path = model_dir / bucket_name(self.n, self.L, self.K).replace(".onnx", "-w.onnx")
            if not path.exists():
                model, _ = make_static(model_dir / "laya.onnx", self.n, self.L, self.K, verbose=False, probe_runner=probe_shapes_cuda)
                weights_as_inputs(model)
                onnx.save_model(model, str(path))
            wait_for_idle()
            with gpu_lock:
                self.session = ort.InferenceSession(str(path), sess_opts(), providers=providers(True))
                self.io = self.session.io_binding()
                bind_weights(self.io)
                self.inputs = {
                    "input_ids": ort.OrtValue.ortvalue_from_numpy(np.zeros((self.n, self.L), np.int64), "cuda", args.device),
                    "attention_mask": ort.OrtValue.ortvalue_from_numpy(np.ones((self.n, self.L), np.int64), "cuda", args.device),
                    "marker_pos": ort.OrtValue.ortvalue_from_numpy(np.zeros((self.n, self.K), np.int64), "cuda", args.device),
                    "marker_mask": ort.OrtValue.ortvalue_from_numpy(np.ones((self.n, self.K), np.bool_), "cuda", args.device),
                    "qtype": ort.OrtValue.ortvalue_from_numpy(np.zeros((self.n,), np.int64), "cuda", args.device),
                }
                for k, v in self.inputs.items():
                    self.io.bind_ortvalue_input(k, v)
                self.outputs = {"logits": ort.OrtValue.ortvalue_from_shape_and_type([self.n, self.K], np.float32, "cuda", args.device), "act_probs": ort.OrtValue.ortvalue_from_shape_and_type([self.n, 2], np.float32, "cuda", args.device)}
                for k, v in self.outputs.items():
                    self.io.bind_ortvalue_output(k, v)
            self.prepare_ms = (time.perf_counter() - t) * 1000
            self.prepared_at = time.perf_counter()
            self.state = "prepared"

        def _upload(self, feeds):
            padded = pad_feeds(feeds, self.key)
            for k in INPUT_NAMES:
                self.inputs[k].update_inplace(np.ascontiguousarray(padded[k]))
            device_sync()  # update_inplace is a synchronous cudaMemcpy from pageable memory: the DMA may still be in flight

        def _outputs(self, n, K):
            return {"logits": self.outputs["logits"].numpy()[:n, :K].copy(), "act_probs": self.outputs["act_probs"].numpy()[:n].copy()}

        def finalise(self, feeds=None):
            """Serving thread, gpu_lock held: two regular runs, the capturing run, one replay (ORT captures on the third
            run of a session on this thread and keeps the graph in this thread's context), then the replay is checked
            against the dynamic session when real inputs for the shape are known."""
            t = time.perf_counter()
            if feeds is not None:
                self._upload(feeds)
            v0 = vram_used()
            for _ in range(4):
                self.session.run_with_iobinding(self.io)
            v1 = vram_used()
            self.vram_mib = max(0, v1 - v0) if v0 >= 0 and v1 >= 0 else 0
            if feeds is not None:
                n, K = feeds["input_ids"].shape[0], feeds["marker_pos"].shape[1]
                self.check_delta = max_delta(self._outputs(n, K), run_dynamic(feeds))
                if not np.isfinite(self.check_delta) or self.check_delta > FINALISE_MAX_DELTA:
                    raise RuntimeError(f"replay disagrees with the dynamic graph on real inputs: max|delta| {self.check_delta:.3f}")
            self.finalise_ms = (time.perf_counter() - t) * 1000
            self.last_used = time.perf_counter()
            self.state = "ready"

        def run(self, feeds):
            """Serving thread: pad to the bucket, upload, replay, slice the outputs back to the call's [n, K]."""
            n, K = feeds["input_ids"].shape[0], feeds["marker_pos"].shape[1]
            self._upload(feeds)
            self.session.run_with_iobinding(self.io)
            self.hits += 1
            self.last_used = time.perf_counter()
            return self._outputs(n, K)

        def close(self):
            self.state = "failed" if self.failed else "closed"
            for attr in ("io", "inputs", "outputs", "session"):
                if hasattr(self, attr):
                    delattr(self, attr)

    buckets = {}  # key -> BucketSession, in scheduling order
    last_feeds = {}  # key -> the most recent real feeds that mapped to it (for the finalisation check)
    seen = collections.Counter()  # key -> calls that mapped to it while no bucket existed (lazy builds start at 2)
    build_queue = collections.deque()
    build_lock = threading.Lock()
    build_event = threading.Event()
    req_q = queue.Queue()  # stdin lines (reader thread) + PREPARED sentinels (builder thread); None = EOF
    stats = {"graph_calls": 0, "dynamic_calls": 0, "builds": 0, "build_failures": 0, "evictions": 0}

    def all_buckets():
        return list(buckets.values())  # snapshot: the builder inserts while the serving thread iterates

    def vram_of_buckets():
        return sum(b.vram_mib for b in all_buckets() if b.state == "ready")

    def disable_graphs(reason):
        global graph_enabled, graph_disabled_reason
        if graph_enabled:
            graph_enabled = False
            graph_disabled_reason = reason
            log(f"CUDA Graph replay disabled for the rest of this process: {reason}")

    def schedule(key, feeds=None):
        if feeds is not None:
            last_feeds[key] = feeds
        with build_lock:
            if key in buckets or key in build_queue:
                return
            build_queue.append(key)
            build_event.set()

    def builder():
        while True:
            build_event.wait()
            with build_lock:
                if not build_queue:
                    build_event.clear()
                    continue
                key = build_queue.popleft()
                b = BucketSession(*key)
                buckets[key] = b
            try:
                b.prepare()
                log(f"graph bucket {key} prepared in {b.prepare_ms:.0f} ms, waiting for the serving thread to capture")
                req_q.put(PREPARED)
            except Exception as e:
                stats["build_failures"] += 1
                b.failed = f"{type(e).__name__}: {str(e)[:160]}"
                b.close()
                log(f"graph bucket {key} FAILED to prepare: {b.failed}")

    def evict_for_room():
        ready = [b for b in all_buckets() if b.state == "ready"]
        while ready and (len(ready) >= args.graph_max_sessions or vram_of_buckets() >= args.graph_max_vram_mib):
            victim = min(ready, key=lambda b: b.last_used)
            victim.close()
            del buckets[victim.key]
            ready.remove(victim)
            stats["evictions"] += 1
            log(f"graph bucket {victim.key} evicted (LRU)")

    def finalise_pending(force=False, prefer=None):
        """Serving thread: capture the prepared bucket `prefer` (else the oldest one) unless the builder is in a CUDA step."""
        prepared = [x for x in all_buckets() if x.state == "prepared"]
        b = next((x for x in prepared if x.key == prefer), prepared[0] if prepared else None)
        if b is None or not graph_enabled:
            return False
        if not gpu_lock.acquire(blocking=False):
            return False
        try:
            evict_for_room()
            b.finalise(last_feeds.get(b.key))
            stats["builds"] += 1
            log(f"graph bucket {b.key} ready: prepare {b.prepare_ms:.0f} ms + capture {b.finalise_ms:.0f} ms{' (forced)' if force else ''}, +{b.vram_mib} MiB VRAM, check vs dynamic {'n/a' if b.check_delta is None else f'{b.check_delta:.4f}'} ({len([x for x in all_buckets() if x.state == 'ready'])} buckets, {vram_of_buckets()} MiB)")
        except Exception as e:
            stats["build_failures"] += 1
            b.failed = f"{type(e).__name__}: {str(e)[:160]}"
            b.close()
            log(f"graph bucket {b.key} FAILED to capture: {b.failed}")
            disable_graphs("a capture failed")
        finally:
            gpu_lock.release()
        return True

    def has_prepared():
        return graph_enabled and any(b.state == "prepared" for b in all_buckets())

    def run_call(feeds, exec_opts):
        n, L = feeds["input_ids"].shape
        K = feeds["marker_pos"].shape[1]
        want_graph = graph_enabled and (exec_opts or {}).get("graph", True) is not False
        key = bucket_for(n, L, K) if want_graph else None
        if key is not None:
            b = buckets.get(key)
            if b is None:
                seen[key] += 1
                if seen[key] >= 2:
                    schedule(key, feeds)
            elif b.state == "prepared":
                last_feeds[key] = feeds
                if time.perf_counter() - b.prepared_at > FINALISE_FORCE_S:
                    finalise_pending(force=True, prefer=key)
            if b is not None and b.state == "ready":
                t = time.perf_counter()
                try:
                    res = b.run(feeds)
                except Exception as e:
                    disable_graphs(f"replay failed on bucket {key}: {type(e).__name__}: {str(e)[:120]}")
                    raise
                stats["graph_calls"] += 1
                return res, (time.perf_counter() - t) * 1000, "graph", list(key)
        t = time.perf_counter()
        res = run_dynamic(feeds)
        stats["dynamic_calls"] += 1
        return res, (time.perf_counter() - t) * 1000, "dynamic", None

    def stats_snapshot():
        return {**stats, "graphEnabled": graph_enabled, "graphDisabledReason": graph_disabled_reason, "buckets": {f"{k[0]}x{k[1]}x{k[2]}": {"state": b.state, "hits": b.hits, "vramMiB": b.vram_mib, "prepareMs": round(b.prepare_ms), "captureMs": round(b.finalise_ms), "checkDelta": b.check_delta, "failed": b.failed} for k, b in list(buckets.items())}, "queued": [list(k) for k in list(build_queue)], "bucketVramMiB": vram_of_buckets(), "weightsMiB": round(weight_bytes / 2**20)}

    # ---- self-test: prepare + capture on this thread, compare with the dynamic session, time both -----------------
    if args.self_test:
        rng = np.random.default_rng(0)
        worst = 0.0
        failures = 0
        cases = [(1, 40, 3), (1, 85, 4), (3, 85, 4), (3, 96, 8), (5, 90, 6), (5, 100, 5), (10, 95, 6), (2, 250, 4), (4, 300, 8), (6, 130, 12)]
        log(f"self-test: {len(cases)} shapes, graph vs dynamic on the same inputs (padding included)")
        for n, L, K in cases:
            key = bucket_for(n, L, K)
            if key is None:
                log(f"  {n}x{L}x{K}: no bucket (stays dynamic)")
                continue
            feeds = {"input_ids": rng.integers(1000, 30000, size=(n, L), dtype=np.int64), "attention_mask": np.ones((n, L), np.int64), "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (n, 1)), "marker_mask": np.ones((n, K), np.bool_), "qtype": rng.integers(0, 3, size=(n,), dtype=np.int64)}
            feeds["attention_mask"][0, L - 5 :] = 0  # a row with real padding, as Node produces for shorter rows
            feeds["marker_mask"][1 % n, K - 1] = False
            b = BucketSession(*key)
            b.prepare()
            with gpu_lock:
                b.finalise(feeds)
            d = max_delta(b.run(feeds), run_dynamic(feeds))
            worst = max(worst, d)
            ts = []
            for _ in range(10):
                tt = time.perf_counter()
                b.run(feeds)
                ts.append((time.perf_counter() - tt) * 1000)
            td = []
            for _ in range(10):
                tt = time.perf_counter()
                run_dynamic(feeds)
                td.append((time.perf_counter() - tt) * 1000)
            ok = d < 0.15  # fp16 noise on random token soup reaches ~0.12 logits; real text agrees far closer (lane test)
            failures += 0 if ok else 1
            log(f"  {n:2d}x{L:3d}x{K:<2d} -> bucket {key}: prepare {b.prepare_ms:.0f} ms + capture {b.finalise_ms:.0f} ms, +{b.vram_mib} MiB; max|delta| {d:.4f} {'ok' if ok else 'MISMATCH'}; graph p50 {sorted(ts)[5]:.1f} ms vs dynamic {sorted(td)[5]:.1f} ms")
            b.close()
        log(f"self-test done: worst max|delta| {worst:.4f}, failures {failures}")
        emit({"selfTest": {"worst": worst, "failures": failures, "cases": len(cases)}})
        sys.exit(1 if failures else 0)

    # ---- eager buckets: prepared and captured here, before the ready line (see the header) ------------------------
    if graph_enabled:
        for spec in [s for s in args.graph_buckets.split(",") if s.strip()]:
            key = tuple(int(x) for x in spec.lower().split("x"))
            if len(key) != 3 or bucket_for(*key) != key:
                log(f"--graph-buckets {spec}: not a bucket on the grid (n in {GRAPH_N}, L in {GRAPH_L}, K in {GRAPH_K}, n x L <= {args.graph_max_work}); skipped")
                continue
            b = BucketSession(*key)
            buckets[key] = b
            try:
                b.prepare()
                with gpu_lock:
                    b.finalise(dummy_feeds(*key))  # checked against the dynamic session on synthetic inputs of the bucket's shape
                stats["builds"] += 1
                log(f"graph bucket {key} ready: prepare {b.prepare_ms:.0f} ms + capture {b.finalise_ms:.0f} ms, +{b.vram_mib} MiB VRAM, check vs dynamic {b.check_delta:.4f} (eager)")
            except Exception as e:
                stats["build_failures"] += 1
                b.failed = f"{type(e).__name__}: {str(e)[:160]}"
                b.close()
                log(f"graph bucket {key} FAILED (eager): {b.failed}")
                disable_graphs("an eager capture failed")
                break
    if graph_enabled:
        threading.Thread(target=builder, name="bucket-builder", daemon=True).start()
except Exception as e:
    emit({"ready": False, "error": f"{type(e).__name__}: {e}"})
    log(f"load failed: {type(e).__name__}: {e}")
    sys.exit(1)

emit({"ready": True, "providers": dyn.get_providers(), "loadMs": round((time.perf_counter() - t0) * 1000), "pid": os.getpid(), "device": args.device, "onnxruntime": ort.__version__, "graph": {"enabled": graph_enabled, "eager": args.graph_buckets if graph_enabled else "", "maxSessions": args.graph_max_sessions, "maxVramMiB": args.graph_max_vram_mib, "maxWork": args.graph_max_work}, "weightsMiB": round(weight_bytes / 2**20)})
log(f"ready in {(time.perf_counter() - t0):.1f} s on CUDA device {args.device} ({args.model_dir}); onnxruntime {ort.__version__}; CUDA graphs {'on' if graph_enabled else 'off'}")

DTYPES = {"int64": np.int64, "int32": np.int32, "bool": np.bool_, "float32": np.float32, "float16": np.float16}


def decode(t):
    return np.frombuffer(base64.b64decode(t["data"]), dtype=DTYPES[t["dtype"]]).reshape(t["dims"])


def encode(a):
    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "dims": list(a.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}


def reader():
    try:
        for line in sys.stdin:
            req_q.put(line)
    finally:
        req_q.put(None)


threading.Thread(target=reader, name="stdin-reader", daemon=True).start()
exit_code = 0
serve_state["serving"] = True
while True:
    try:
        item = req_q.get(timeout=FINALISE_IDLE_S if has_prepared() else None)
    except queue.Empty:
        finalise_pending()  # idle: capture a prepared bucket (~50-150 ms) while nobody is waiting
        continue
    if item is None:
        break
    if item == PREPARED:
        finalise_pending()
        continue
    line = item.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        log(f"bad request line: {e}")
        continue
    op = req.get("op")
    rid = req.get("id")
    if op == "close":
        break
    if op == "stats":
        emit({"id": rid, "stats": stats_snapshot()})
        continue
    if op == "bucket":
        key = bucket_for(int(req["n"]), int(req["L"]), int(req["K"])) if graph_enabled else None
        emit({"id": rid, "bucket": list(key) if key else None})
        continue
    serve_state["busy"] = True
    try:
        feeds = {k: decode(v) for k, v in req["feeds"].items()}
        res, ms, mode, bucket = run_call(feeds, req.get("exec"))
        emit({"id": rid, "outputs": {k: encode(v) for k, v in res.items()}, "ms": round(ms, 3), "mode": mode, "bucket": bucket})
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        emit({"id": rid, "error": msg})
        if any(w in msg for w in ("CUDA", "CUBLAS", "CUDNN", "cuda")):
            log(f"fatal CUDA error in a call, exiting so the Node side fails over: {msg[:200]}")
            exit_code = 3
            break
    finally:
        serve_state["busy"] = False
        serve_state["idle_since"] = time.perf_counter()

for b in all_buckets():
    b.close()
del dyn
log("closed" if exit_code == 0 else f"exit {exit_code}")
sys.exit(exit_code)
