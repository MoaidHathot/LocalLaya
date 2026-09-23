"""Static-shape variants of the Laya graph for CUDA Graph capture.

    .venv/Scripts/python.exe tools/static_graph.py <bundle dir> --buckets 1x96x8,3x128x8,... [--out-dir <dir>]
    .venv/Scripts/python.exe tools/static_graph.py models/laya-onnx-fp16 --buckets 3x128x8 --check

A "bucket" is a fixed (batch n, sequence length L, options K). The exported graph computes its shape plumbing at
run time (Shape -> Slice -> Concat -> Reshape, Range for positions, ...): ~160 small CPU-side nodes and host<->
device copies per call, and the reason CUDA Graph capture fails on it (results/cuda-lane-2026-09-23-summary.md).
With the five inputs fixed to a bucket every one of those values is a constant. This tool makes them so, exactly:

  1. set the input dims to the bucket,
  2. run the model ONCE (CPU) with every `Shape` output exposed as an extra graph output and replace each Shape
     node by a Constant holding the observed value - no symbolic shape inference (ORT's chokes on this graph's
     Range), just what the runtime saw,
  3. fold every node whose inputs are all constant (evaluated by onnxruntime itself, one node at a time, so the
     semantics are ORT's): the shape arithmetic, Range, the RoPE cos/sin tables, ...
  4. prune unreachable nodes and initializers.

The result references the bundle's existing `laya.onnx.data` (same offsets; weights are never rewritten), so a
bucket graph is ~3 MB and all buckets share one weight file. Written next to the bundle's laya.onnx as
`laya-static-b{n}-l{L}-k{K}.onnx` (ORT 1.30 refuses external-data paths outside the model's own directory) plus
`static-manifest.json`. `--check` runs each bucket on the CUDA EP (if
available) against the dynamic graph with the same padded inputs and reports max |delta logit| and the CPU-node
count. Padding rows/positions/markers is the caller's job (tools/cuda_lane.py): attention_mask 0 for padded
positions, marker_mask 0 for padded markers, marker_pos 0 there, duplicated rows for padded batch entries.
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

INPUT_DIMS = {"input_ids": ("n", "L"), "attention_mask": ("n", "L"), "marker_pos": ("n", "K"), "marker_mask": ("n", "K"), "qtype": ("n",)}
NP_TYPES = {TensorProto.INT64: np.int64, TensorProto.BOOL: np.bool_, TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16, TensorProto.INT32: np.int32}


def log(msg):
    print(msg, flush=True)


def dummy_feeds(n, L, K):
    rng = np.random.default_rng(1)
    return {
        "input_ids": rng.integers(1000, 30000, size=(n, L), dtype=np.int64),
        "attention_mask": np.ones((n, L), dtype=np.int64),
        "marker_pos": np.tile(np.arange(5, 5 + K, dtype=np.int64), (n, 1)),
        "marker_mask": np.ones((n, K), dtype=bool),
        "qtype": np.zeros((n,), dtype=np.int64),
    }


def set_static_dims(model, n, L, K):
    vals = {"n": n, "L": L, "K": K}
    for inp in model.graph.input:
        for d, name in zip(inp.type.tensor_type.shape.dim, INPUT_DIMS[inp.name]):
            d.ClearField("dim_param")
            d.dim_value = vals[name]
    del model.graph.value_info[:]  # stale symbolic shapes; ORT re-infers what it needs


def eval_node(node, const_values, opset_imports):
    """Run one node on constant inputs through a tiny ORT CPU session; returns {output name: ndarray}.
    ORT needs typed graph outputs, so the one-node model goes through ONNX shape inference first (data_prop=True
    lets it type Shape/Range-style outputs); ops it cannot type fall back to the ONNX reference evaluator."""
    inits = [numpy_helper.from_array(np.ascontiguousarray(const_values[i]), i) for i in node.input if i]
    outs = [helper.make_empty_tensor_value_info(o) for o in node.output if o]
    g = helper.make_graph([node], "fold", [], outs, initializer=inits)
    m = helper.make_model(g, opset_imports=opset_imports)
    m.ir_version = 9
    inferred = onnx.shape_inference.infer_shapes(m, data_prop=True)
    typed = {v.name: v for v in list(inferred.graph.output) + list(inferred.graph.value_info)}
    if all(o.name in typed and typed[o.name].type.tensor_type.elem_type != TensorProto.UNDEFINED for o in outs):
        del m.graph.output[:]
        m.graph.output.extend(typed[o.name] for o in outs)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        return dict(zip([o.name for o in s.get_outputs()], s.run(None, {})))
    from onnx.reference import ReferenceEvaluator

    ev = ReferenceEvaluator(m)
    return dict(zip([o.name for o in outs], ev.run(None, {})))


def weights_as_inputs(model):
    """Turn every external (weight) initializer into a graph input of the same name, dtype and shape. The caller
    binds one set of device buffers to all bucket sessions (IO binding accepts any OrtValue), so N sessions
    share one copy of the 843 MB of weights - SessionOptions.add_initializer refuses device buffers. Returns the
    list of (name, elem_type, dims) so the loader knows what to read from laya.onnx.data."""
    g = model.graph
    moved = []
    keep = []
    for t in g.initializer:
        if t.data_location == TensorProto.EXTERNAL:
            g.input.append(helper.make_tensor_value_info(t.name, t.data_type, list(t.dims)))
            moved.append({"name": t.name, "elem_type": t.data_type, "dims": list(t.dims), **{kv.key: kv.value for kv in t.external_data}})
        else:
            keep.append(t)
    del g.initializer[:]
    g.initializer.extend(keep)
    return moved


def probe_shapes_cpu(probe, model_path, n, L, K):
    """Default Shape-observer: a CPU session on the probe model (loads the 843 MB of weights from disk: ~3-8 s)."""
    probe_path = Path(model_path).with_name("_static_probe.onnx")  # next to the data file (relative external refs)
    onnx.save_model(probe, str(probe_path))
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep every Shape node observable
        sess = ort.InferenceSession(str(probe_path), so, providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        return dict(zip(out_names, sess.run(out_names, dummy_feeds(n, L, K))))
    finally:
        probe_path.unlink(missing_ok=True)


def make_static(model_path, n, L, K, verbose=True, probe_runner=None):
    """Return (static ModelProto referencing the bundle's external data, report dict).
    `probe_runner(probe_model, model_path, n, L, K) -> {output name: ndarray}` runs the Shape-observation pass;
    the default uses a CPU session, tools/cuda_lane.py passes one that reuses its device-resident weights."""
    t0 = time.perf_counter()
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    set_static_dims(model, n, L, K)
    opset_imports = list(model.opset_import)
    report = {"bucket": [n, L, K], "nodes_before": len(graph.node)}

    # ---- 2. observe every Shape output at run time ----------------------------------------------------------
    shape_nodes = [nd for nd in graph.node if nd.op_type == "Shape"]
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing_outputs = {o.name for o in probe.graph.output}
    for nd in shape_nodes:
        if nd.output[0] not in existing_outputs:
            probe.graph.output.append(helper.make_tensor_value_info(nd.output[0], TensorProto.INT64, None))
    values = (probe_runner or probe_shapes_cpu)(probe, model_path, n, L, K)

    const_values = {}
    for nd in shape_nodes:
        const_values[nd.output[0]] = np.asarray(values[nd.output[0]], dtype=np.int64)
    report["shape_nodes_baked"] = len(shape_nodes)

    # ---- 3. fold everything that is now constant -------------------------------------------------------------
    inits = {i.name: i for i in graph.initializer}
    small_const = {}  # initializer values we may need as fold inputs (loaded lazily, small ones only)

    def const_of(name):
        if name in const_values:
            return const_values[name]
        if name in small_const:
            return small_const[name]
        t = inits.get(name)
        if t is None:
            return None
        if t.data_location == TensorProto.EXTERNAL:
            return None  # a weight: never part of shape arithmetic; do not load 800 MB
        arr = numpy_helper.to_array(t)
        small_const[name] = arr
        return arr

    graph_inputs = {i.name for i in graph.input}
    new_nodes = []
    folded = collections.Counter()
    for nd in graph.node:
        if nd.op_type == "Shape":
            continue  # replaced by constants below
        if nd.op_type == "Constant":
            new_nodes.append(nd)
            const_values[nd.output[0]] = numpy_helper.to_array(nd.attribute[0].t) if nd.attribute[0].name == "value" else None
            continue
        inputs = [i for i in nd.input if i]
        if inputs and all(i not in graph_inputs and const_of(i) is not None for i in inputs):
            try:
                res = eval_node(nd, {i: const_of(i) for i in inputs}, opset_imports)
            except Exception as e:  # pragma: no cover - keep the node if ORT cannot fold it alone
                if verbose:
                    log(f"    keep {nd.op_type} {nd.name}: {type(e).__name__}: {str(e)[:80]}")
                new_nodes.append(nd)
                continue
            for o in nd.output:
                if o:
                    const_values[o] = res[o]
            folded[nd.op_type] += 1
            continue
        new_nodes.append(nd)

    # constants that the remaining nodes still read become initializers (small: shapes, positions, RoPE tables)
    needed = {i for nd in new_nodes for i in nd.input if i} | {o.name for o in graph.output}
    new_inits = []
    for name, arr in const_values.items():
        if name in needed and name not in inits:
            new_inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)

    # ---- 3b. CUDA-graph-safe rewrites ------------------------------------------------------------------------
    # onnxruntime 1.30's CUDA GatherND copies a stack vector into the GPU with cudaMemcpyAsync during the run
    # (gather_nd.cc:70-75); under stream capture the recorded copy keeps the host stack address and every replay
    # reads garbage offsets -> "illegal memory access". With static shapes the export's GatherND (the attention
    # mask broadcast) has constant indices; when the gather is numerically a pure re-indexing of the data it is
    # replaced by the equivalent Reshape. Checked with a test array, not assumed from the index pattern.
    inits = {i.name: i for i in graph.initializer}
    rewritten = 0
    for idx, nd in enumerate(list(graph.node)):
        if nd.op_type != "GatherND" or nd.input[1] not in inits:
            continue
        batch_dims = next((a.i for a in nd.attribute if a.name == "batch_dims"), 0)
        indices = numpy_helper.to_array(inits[nd.input[1]])
        # the data shape: known only for graph inputs / Cast of graph inputs here; find it through a dry run of the
        # producer chain would be heavy, so use the observed rank from the indices: last index dim = number of data
        # dims gathered; a pure reshape needs indices to cover the whole data exactly once in row-major order
        if batch_dims != 0 or indices.shape[-1] != 2:
            continue
        flat = indices.reshape(-1, 2)
        n_rows = int(flat[:, 0].max()) + 1
        n_cols = int(flat[:, 1].max()) + 1
        if flat.shape[0] != n_rows * n_cols:
            continue
        test = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
        gathered = test[flat[:, 0], flat[:, 1]].reshape(indices.shape[:-1])
        if not np.array_equal(gathered, test.reshape(indices.shape[:-1])):
            continue
        shape_name = nd.output[0] + "_reshape_shape"
        graph.initializer.append(numpy_helper.from_array(np.asarray(indices.shape[:-1], dtype=np.int64), shape_name))
        graph.node[idx].CopyFrom(helper.make_node("Reshape", [nd.input[0], shape_name], [nd.output[0]], name=nd.name + "_as_reshape"))
        rewritten += 1
    report["gathernd_to_reshape"] = rewritten

    # ---- 4. prune: drop nodes whose outputs nobody reads, initializers nobody reads -------------------------
    changed = True
    while changed:
        changed = False
        needed = {i for nd in graph.node for i in nd.input if i} | {o.name for o in graph.output}
        keep = [nd for nd in graph.node if any(o in needed for o in nd.output)]
        if len(keep) != len(graph.node):
            del graph.node[:]
            graph.node.extend(keep)
            changed = True
    needed = {i for nd in graph.node for i in nd.input if i}
    keep_inits = [t for t in graph.initializer if t.name in needed]
    removed_inits = len(graph.initializer) - len(keep_inits)
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)
    report.update({"folded": dict(folded), "nodes_after": len(graph.node), "new_constants": len(new_inits), "removed_initializers": removed_inits, "seconds": round(time.perf_counter() - t0, 1)})
    return model, report


def bucket_name(n, L, K):
    return f"laya-static-b{n}-l{L}-k{K}.onnx"


def check_bucket(bundle_dir, static_path, n, L, K, dynamic_session, provider):
    """Run the static graph and the dynamic graph on identical (fully valid) inputs; report differences and CPU nodes."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 2
    s = ort.InferenceSession(str(static_path), so, providers=[provider, "CPUExecutionProvider"])
    feeds = dummy_feeds(n, L, K)
    a = s.run(None, feeds)
    b = dynamic_session.run(None, feeds)
    diff = max(float(np.max(np.abs(x.astype(np.float32) - y.astype(np.float32)))) for x, y in zip(a, b))
    # node placement: count nodes ORT assigned to the CPU EP (via a verbose session's log is noisy; use the
    # optimized-model dump instead: nodes that were assigned to CPU get MemcpyToHost/FromHost neighbours)
    so2 = ort.SessionOptions()
    so2.log_severity_level = 3
    tmp = Path(static_path).with_suffix(".opt.onnx")
    so2.optimized_model_filepath = str(tmp)
    so2.add_session_config_entry("session.optimized_model_external_initializers_file_name", tmp.name + ".data")
    so2.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    ort.InferenceSession(str(static_path), so2, providers=[provider, "CPUExecutionProvider"])
    opt = onnx.load(str(tmp), load_external_data=False)
    ops = collections.Counter(nd.op_type for nd in opt.graph.node)
    tmp.unlink(missing_ok=True)
    Path(str(tmp) + ".data").unlink(missing_ok=True)
    ts = []
    for _ in range(5):
        s.run(None, feeds)
    for _ in range(20):
        t = time.perf_counter()
        s.run(None, feeds)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"max_abs_diff": diff, "memcpy_nodes": ops.get("MemcpyToHost", 0) + ops.get("MemcpyFromHost", 0), "optimized_nodes": len(opt.graph.node), "p50_ms": round(ts[10], 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir")
    ap.add_argument("--buckets", default="", help="comma-separated nxLxK, e.g. 1x96x8,3x128x8")
    ap.add_argument("--dynamic", action="store_true", help="also write laya-dynamic-w.onnx: the dynamic graph with the weights as inputs (the lane's fallback session on the shared weights)")
    ap.add_argument("--out-dir", default=None, help="default: the bundle directory itself (the graphs must sit next to laya.onnx.data)")
    ap.add_argument("--check", action="store_true", help="compare with the dynamic graph on the CUDA EP (or CPU) and time it (initializer form only)")
    ap.add_argument("--weights-as-inputs", action="store_true", help="also write laya-static-*-w.onnx with the weights as graph inputs (shared device buffers across sessions, see tools/cuda_lane.py)")
    args = ap.parse_args()
    bundle = Path(args.bundle_dir)
    model_path = bundle / "laya.onnx"
    out_dir = Path(args.out_dir) if args.out_dir else bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = [tuple(int(x) for x in b.lower().split("x")) for b in args.buckets.split(",") if b.strip()]
    if args.dynamic:
        dyn_model = onnx.load(str(model_path), load_external_data=False)
        moved = weights_as_inputs(dyn_model)
        onnx.save_model(dyn_model, str(out_dir / "laya-dynamic-w.onnx"))
        log(f"laya-dynamic-w.onnx: dynamic graph, {len(moved)} weights as inputs")
    manifest_path = out_dir / "static-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": str(model_path), "buckets": {}}
    dyn = None
    provider = None
    if args.check:
        ort.set_default_logger_severity(3)
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except Exception:
            pass
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 2
        dyn = ort.InferenceSession(str(model_path), so, providers=[provider, "CPUExecutionProvider"])
        log(f"check provider: {provider}")
    for n, L, K in buckets:
        name = bucket_name(n, L, K)
        model, report = make_static(model_path, n, L, K)
        onnx.save_model(model, str(out_dir / name))
        report["file"] = name
        report["size_bytes"] = (out_dir / name).stat().st_size
        if args.weights_as_inputs:
            moved = weights_as_inputs(model)
            wname = name.replace(".onnx", "-w.onnx")
            onnx.save_model(model, str(out_dir / wname))
            report["weights_file"] = wname
            report["weight_inputs"] = len(moved)
            manifest["weights"] = moved  # identical for every bucket
        if args.check:
            report["check"] = check_bucket(bundle, out_dir / name, n, L, K, dyn, provider)
        manifest["buckets"][f"{n}x{L}x{K}"] = report
        log(f"{name}: {report['nodes_before']} -> {report['nodes_after']} nodes ({report['shape_nodes_baked']} Shape baked, folded {sum(report['folded'].values())}: {', '.join(f'{k} {v}' for k, v in sorted(report['folded'].items(), key=lambda kv: -kv[1])[:6])}), {report['size_bytes'] / 1e6:.1f} MB, {report['seconds']} s"
            + (f"; check: max|dlogit| {report['check']['max_abs_diff']:.4f}, memcpy nodes {report['check']['memcpy_nodes']}, optimized nodes {report['check']['optimized_nodes']}, p50 {report['check']['p50_ms']} ms" if args.check else ""))
    if args.dynamic:
        manifest["dynamic_weights_file"] = "laya-dynamic-w.onnx"
        manifest["weights"] = moved
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"manifest: {manifest_path} ({len(manifest['buckets'])} buckets)")
