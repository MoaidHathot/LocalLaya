"""Convert the Laya fp32 ONNX bundle to fp16 (weights + compute) without PyTorch.

    .venv/Scripts/python.exe tools/convert_fp16.py <src_bundle_dir> <dst_bundle_dir>

Uses ONNX Runtime's transformer-oriented float16 pass (onnxruntime.transformers.float16, a fixed fork of
onnxconverter-common that handles pre-existing Cast nodes) with keep_io_types=True so the graph's
inputs/outputs stay int64/bool/float32 and @receptron/laya can consume the result unchanged. Numerically
fragile ops stay fp32 via the block list. Copies laya_config.json and the tokenizer next to the converted
graph. Validate with experiments/fp16-fidelity.mjs before using it.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import onnx
from onnxruntime.transformers import float16

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print(f"loading {src / 'laya.onnx'} (+ external data) ...", flush=True)
model = onnx.load(str(src / "laya.onnx"), load_external_data=True)
n_init = len(model.graph.initializer)
fp32_bytes = sum(len(t.raw_data) for t in model.graph.initializer)
print(f"  {n_init} initializers, {fp32_bytes / 1e9:.2f} GB raw; opset {[ (o.domain or 'ai.onnx', o.version) for o in model.opset_import ]}  ({time.time() - t0:.0f} s)", flush=True)

# Extra ops kept in fp32 on top of the converter's defaults (LayerNormalization / softmax reductions are
# the usual fp16 accuracy risks in BERT-style encoders; Range / position math must stay exact).
block = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | {"LayerNormalization", "SimplifiedLayerNormalization", "Softmax", "Range", "CumSum", "Pow"})
print("converting to fp16 ...", flush=True)
t1 = time.time()
model16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    disable_shape_infer=False,  # symbolic shape inference lets the pass type every Cast correctly
    op_block_list=block,
)
print(f"  done ({time.time() - t1:.0f} s)", flush=True)

ops = {}
for n in model16.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
casts = ops.get("Cast", 0)
print(f"  nodes: {len(model16.graph.node)} ({casts} Cast nodes inserted around blocked ops)")

print(f"saving to {dst} ...", flush=True)
onnx.save_model(
    model16,
    str(dst / "laya.onnx"),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="laya.onnx.data",
    size_threshold=1024,
    convert_attribute=False,
)
shutil.copy(src / "laya_config.json", dst / "laya_config.json")
(dst / "tokenizer").mkdir(exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(src / "tokenizer" / f, dst / "tokenizer" / f)

sizes = {p.name: p.stat().st_size for p in dst.iterdir() if p.is_file()}
print(json.dumps({"dst": str(dst), "sizes": sizes, "seconds": round(time.time() - t0)}, indent=2))
