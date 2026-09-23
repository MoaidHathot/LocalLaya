"""Optimise the Laya ONNX graph for GPU execution providers, then convert to fp16 (no PyTorch needed).

    .venv/Scripts/python.exe tools/optimize_graph.py <src fp32 bundle dir> <dst bundle dir> [options]

    --keep-fp32 Op,Op     ops to leave in fp32 (Cast nodes are inserted around them); default: none beyond the
                          converter's own block list (+ Range, CumSum, Pow). The retired tools/convert_fp16.py kept
                          LayerNormalization and Softmax in fp32, which costs ~200 Cast dispatches per call.
    --no-isnan            keep IsNaN (default: rewrite IsNaN(x) -> Not(Equal(x, x)), exact)
    --no-allowzero        keep Reshape allowzero=1 (default: allowzero=0 where no shape can contain a 0, exact)
    --no-gelu             skip ORT's Gelu fusion (onnxruntime.transformers, model_type bert)
    --fp32                write the optimised graph in fp32 (no conversion)

What it does, and why (measured with ORT profiling on the WebGPU EP, see results/graph-opt-*.md):
  1. IsNaN is not implemented by the WebGPU EP: every layer's `Where(IsNaN(softmax), 0, softmax)` NaN guard
     forced a GPU->CPU copy of the attention probabilities, IsNaN on the CPU and a copy back - 28 pipeline
     drains per call. `Not(Equal(x, x))` is the same predicate (only NaN != NaN) and runs on the GPU.
  2. torch.export emits Reshape(allowzero=1) for every `view`; DirectML rejects that together with a -1
     ("node_view", HRESULT 80070057). No Reshape shape in this graph can contain a 0, so allowzero=0 is identical.
  3. ORT's transformer fusions on this ModernBERT export: only Gelu (29 x Erf pattern) matches; attention,
     RoPE, SkipLayerNorm do not (different export structure) - reported, not hand-written.
  4. fp16 with as few fp32 islands as possible: each kept-fp32 op costs two Cast dispatches per instance.
Everything here is architecture-agnostic (works on any exported graph with these ops) and exact except the
fp16 numerics, which experiments/ab.mjs validates against the fp32 reference on any preset's eval set.
"""
import argparse
import collections
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--keep-fp32", default="")
ap.add_argument("--no-isnan", action="store_true")
ap.add_argument("--no-allowzero", action="store_true")
ap.add_argument("--no-gelu", action="store_true")
ap.add_argument("--fp32", action="store_true")
args = ap.parse_args()
src = Path(args.src)
dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def log(msg):
    print(msg, flush=True)


log(f"loading {src / 'laya.onnx'} (+ external data) ...")
model = onnx.load(str(src / "laya.onnx"), load_external_data=True)
before = op_counts(model)
report = {"src": str(src), "dst": str(dst), "nodes_before": len(model.graph.node), "steps": {}}
log(f"  {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers, opset {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")

# ---- 1. Gelu fusion via ORT's transformer optimizer ---------------------------------------------------------------
if not args.no_gelu:
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions

    fo = FusionOptions("bert")
    # only the fusions that match this export; attention / embed-layer-norm patterns do not and cost minutes to try
    fo.enable_attention = False
    fo.enable_embed_layer_norm = False
    fo.enable_rotary_embeddings = False
    opt = optimizer.optimize_by_fusion(model, model_type="bert", num_heads=16, hidden_size=1024, optimization_options=fo)
    model = opt.model
    # the fused ops live in com.microsoft; make sure the domain is imported so shape inference (fp16 pass) accepts them
    if any(n.domain == "com.microsoft" for n in model.graph.node) and not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    fused = {k: v for k, v in opt.get_fused_operator_statistics().items() if v}
    report["steps"]["gelu_fusion"] = fused
    log(f"  ORT fusions: {fused} -> {len(model.graph.node)} nodes")

graph = model.graph

# ---- 2. IsNaN(x) -> Not(Equal(x, x)) -----------------------------------------------------------------------------
if not args.no_isnan:
    n_isnan = 0
    new_nodes = []
    for node in graph.node:
        if node.op_type == "IsNaN":
            x = node.input[0]
            eq_out = node.output[0] + "_eq"
            new_nodes.append(helper.make_node("Equal", [x, x], [eq_out], name=node.name + "_eq"))
            new_nodes.append(helper.make_node("Not", [eq_out], [node.output[0]], name=node.name + "_not"))
            n_isnan += 1
        else:
            new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    report["steps"]["isnan_rewritten"] = n_isnan
    log(f"  IsNaN -> Not(Equal(x, x)): {n_isnan} nodes rewritten")

# ---- 3. Reshape allowzero=0 --------------------------------------------------------------------------------------
if not args.no_allowzero:
    inits = {i.name: i for i in graph.initializer}
    zero_shapes = [n.name for n in graph.node if n.op_type == "Reshape" and n.input[1] in inits and 0 in numpy_helper.to_array(inits[n.input[1]]).tolist()]
    if zero_shapes:
        log(f"  WARNING: {len(zero_shapes)} Reshape nodes have a 0 in a constant shape; leaving allowzero as is for them: {zero_shapes[:5]}")
    n_fixed = 0
    for node in graph.node:
        if node.op_type == "Reshape" and node.name not in zero_shapes:
            for a in list(node.attribute):
                if a.name == "allowzero" and a.i == 1:
                    node.attribute.remove(a)
                    n_fixed += 1
    report["steps"]["allowzero_cleared"] = n_fixed
    log(f"  Reshape allowzero=1 -> default: {n_fixed} nodes")

# ---- 4. fp16 --------------------------------------------------------------------------------------------------------
if not args.fp32:
    from onnxruntime.transformers import float16

    keep = {s.strip() for s in args.keep_fp32.split(",") if s.strip()}
    block = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | {"Range", "CumSum", "Pow"} | keep)
    log(f"  converting to fp16 (fp32 islands: {sorted(keep) or 'none beyond the converter defaults'}) ...")
    t1 = time.time()
    model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False, op_block_list=block)
    graph = model.graph
    report["steps"]["fp16"] = {"keep_fp32": sorted(keep), "casts": op_counts(model).get("Cast", 0), "seconds": round(time.time() - t1)}
    log(f"  fp16 done in {time.time() - t1:.0f} s; {op_counts(model).get('Cast', 0)} Cast nodes")

after = op_counts(model)
report["nodes_after"] = len(graph.node)
report["ops_before"] = dict(before.most_common())
report["ops_after"] = dict(after.most_common())

# ---- 5. save bundle --------------------------------------------------------------------------------------------------
log(f"saving to {dst} ...")
onnx.save_model(model, str(dst / "laya.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="laya.onnx.data", size_threshold=1024, convert_attribute=False)
shutil.copy(src / "laya_config.json", dst / "laya_config.json")
(dst / "tokenizer").mkdir(exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(src / "tokenizer" / f, dst / "tokenizer" / f)
report["sizes"] = {p.name: p.stat().st_size for p in dst.iterdir() if p.is_file()}
report["seconds"] = round(time.time() - t0)
report["argv"] = sys.argv[1:]
(dst / "optimize-report.json").write_text(json.dumps(report, indent=2))
changed = {k: (before.get(k, 0), after.get(k, 0)) for k in set(before) | set(after) if before.get(k, 0) != after.get(k, 0)}
log(f"nodes {report['nodes_before']} -> {report['nodes_after']}; changed op counts: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(changed.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))))
log(f"done in {report['seconds']} s; report at {dst / 'optimize-report.json'}")
