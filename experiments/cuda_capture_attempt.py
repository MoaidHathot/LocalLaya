"""One CUDA-graph capture attempt per process (a failed capture kills the CUDA context), driven by
experiments/cuda_capture_bisect.py. Prints one JSON line: {ok, replay_ms, diff, error}.

    python cuda_capture_attempt.py <model.onnx> <n> <L> <K> [--opt disable|basic|extended|all] [--arena next|same]
                                   [--no-graph] [--outputs cpu|cuda] [--bind-inputs cpu|cuda]
"""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("n", type=int)
ap.add_argument("L", type=int)
ap.add_argument("K", type=int)
ap.add_argument("--opt", default="all")
ap.add_argument("--arena", default="same")
ap.add_argument("--no-graph", action="store_true")
ap.add_argument("--outputs", default="cuda")
ap.add_argument("--bind-inputs", default="cuda")
ap.add_argument("--ref", default=None, help="dynamic model for the reference output")
ap.add_argument("--prefix", type=int, default=0, help="test only the first N nodes (topological order) of the model, all dangling tensors as outputs")
ap.add_argument("--drop-ops", default="", help="comma-separated op types to cut the graph before (their outputs become graph outputs, they and everything after are removed)")
args = ap.parse_args()

if args.prefix or args.drop_ops:
    # build a truncated model next to the original (external data is referenced relatively)
    import onnx
    from onnx import helper, TensorProto
    from pathlib import Path

    m = onnx.load(args.model, load_external_data=False)
    g = m.graph
    nodes = list(g.node)
    keep = nodes[: args.prefix] if args.prefix else nodes
    if args.drop_ops:
        drop = set(args.drop_ops.split(","))
        first = next((i for i, nd in enumerate(keep) if nd.op_type in drop), None)
        if first is not None:
            keep = keep[:first]
    produced = {o for nd in keep for o in nd.output if o}
    consumed = {i for nd in keep for i in nd.input if i}
    dangling = [o for o in produced if o not in consumed]
    del g.node[:]
    g.node.extend(keep)
    del g.output[:]
    for name in dangling:
        g.output.append(helper.make_empty_tensor_value_info(name))
    # keep only the initializers the prefix reads (ORT rejects unused-but-referenced external tensors? no, but smaller is faster)
    needed = {i for nd in keep for i in nd.input if i}
    inits = [t for t in g.initializer if t.name in needed]
    del g.initializer[:]
    g.initializer.extend(inits)
    trunc = Path(args.model).with_name(f"_trunc_{args.prefix}_{abs(hash(args.drop_ops)) % 10000}.onnx")
    onnx.save_model(m, str(trunc))
    args.model = str(trunc)
    args.ref = None
    TRUNC = trunc
else:
    TRUNC = None

ort.set_default_logger_severity(4)
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
levels = {"disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def feeds_for(n, L, K, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(1000, 30000, size=(n, L), dtype=np.int64),
        "attention_mask": np.ones((n, L), dtype=np.int64),
        "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (n, 1)),
        "marker_mask": np.ones((n, K), dtype=bool),
        "qtype": np.zeros((n,), dtype=np.int64),
    }


out = {"ok": False}
try:
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.intra_op_num_threads = 2
    so.graph_optimization_level = levels[args.opt]
    popts = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested" if args.arena == "same" else "kNextPowerOfTwo"}
    if not args.no_graph:
        popts["enable_cuda_graph"] = "1"
    f = feeds_for(args.n, args.L, args.K, seed=7)
    # learn output dtypes / shapes from a plain run (truncated models have arbitrary intermediates as outputs)
    plain_so = ort.SessionOptions()
    plain_so.log_severity_level = 4
    plain_so.graph_optimization_level = levels[args.opt]
    plain = ort.InferenceSession(args.model, plain_so, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    names = [o.name for o in plain.get_outputs()]
    plain_out = dict(zip(names, plain.run(names, f)))
    del plain
    s = ort.InferenceSession(args.model, so, providers=[("CUDAExecutionProvider", popts), "CPUExecutionProvider"])
    io = s.io_binding()
    dev = {k: ort.OrtValue.ortvalue_from_numpy(v, args.bind_inputs, 0) for k, v in f.items()}
    for k, v in dev.items():
        io.bind_ortvalue_input(k, v)
    outs = {}
    for name, arr in plain_out.items():
        outs[name] = ort.OrtValue.ortvalue_from_shape_and_type(list(arr.shape), arr.dtype, args.outputs, 0)
        io.bind_ortvalue_output(name, outs[name])
    s.run_with_iobinding(io)  # regular run (also the one whose arena layout is reused)
    first = {k: v.numpy().copy() for k, v in outs.items()}
    s.run_with_iobinding(io)  # capture run (with enable_cuda_graph)
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        s.run_with_iobinding(io)  # replays
        ts.append((time.perf_counter() - t) * 1000)
    got = {k: v.numpy() for k, v in outs.items()}
    out["replay_ms"] = round(sorted(ts)[10], 2)
    out["outputs"] = len(outs)
    bad = [k for k in outs if not np.allclose(first[k].astype(np.float32), got[k].astype(np.float32), atol=1e-2, equal_nan=True)]
    out["stable"] = not bad
    if bad:
        out["unstable_outputs"] = bad[:5]
    vs_plain = [k for k in outs if not np.allclose(plain_out[k].astype(np.float32), first[k].astype(np.float32), atol=1e-2, equal_nan=True)]
    out["first_run_matches_plain"] = not vs_plain
    got_logits = got.get("logits")
    if args.ref and got_logits is not None:
        so2 = ort.SessionOptions()
        so2.log_severity_level = 4
        ref = ort.InferenceSession(args.ref, so2, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]).run(None, f)[0]
        out["diff"] = float(np.max(np.abs(got_logits - ref)))
    out["ok"] = True
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
finally:
    if TRUNC is not None:
        try:
            TRUNC.unlink()
        except Exception:
            pass
print(json.dumps(out))
sys.stdout.flush()
