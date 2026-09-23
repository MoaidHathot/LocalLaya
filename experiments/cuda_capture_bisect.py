"""Find the node that breaks CUDA Graph capture: binary search over graph prefixes, one attempt per subprocess.

    .venv/Scripts/python.exe -u experiments/cuda_capture_bisect.py models/laya-onnx-fp16/laya-static-b3-l96-k8.onnx 3 96 8 [--opt disable]

"ok" for a prefix = capture + replay succeeded AND the replayed outputs equal the first regular run's outputs
(a capture that replays different values is as broken as a crash). Then prints the first failing node and
tries the graph with that op type cut out.
"""
import json
import subprocess
import sys
from pathlib import Path

import onnx

model, n, L, K = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
opt = sys.argv[sys.argv.index("--opt") + 1] if "--opt" in sys.argv else "all"
py = sys.executable
attempt_py = str(Path(__file__).with_name("cuda_capture_attempt.py"))
nodes = list(onnx.load(model, load_external_data=False).graph.node)
print(f"{len(nodes)} nodes, opt={opt}", flush=True)


def attempt(extra):
    r = subprocess.run([py, attempt_py, model, n, L, K, "--opt", opt, *extra], capture_output=True, text=True, timeout=600)
    line = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
    try:
        return json.loads(line)
    except Exception:
        return {"ok": False, "error": (r.stderr or r.stdout)[-200:]}


def good(res):
    return res.get("ok") and res.get("stable") and res.get("first_run_matches_plain", True)


lo, hi = 1, len(nodes)  # invariant: prefix lo is good (checked below), prefix hi is bad
r = attempt(["--prefix", str(lo)])
print(f"prefix {lo}: {json.dumps(r)}", flush=True)
if not good(r):
    print("even a 1-node prefix fails; the problem is not a single op", flush=True)
    sys.exit(1)
r = attempt([])
print(f"full graph: {json.dumps(r)}", flush=True)
if good(r):
    print("full graph is fine at this optimization level", flush=True)
    sys.exit(0)
while hi - lo > 1:
    mid = (lo + hi) // 2
    r = attempt(["--prefix", str(mid)])
    verdict = "ok " if good(r) else "BAD"
    print(f"prefix {mid:5d}: {verdict} {json.dumps({k: v for k, v in r.items() if k in ('replay_ms', 'stable', 'first_run_matches_plain', 'unstable_outputs', 'error')})}", flush=True)
    if good(r):
        lo = mid
    else:
        hi = mid
bad_node = nodes[hi - 1]
print(f"\nfirst bad prefix ends with node #{hi - 1}: {bad_node.op_type} {bad_node.name} inputs={list(bad_node.input)} outputs={list(bad_node.output)}", flush=True)
r = attempt(["--drop-ops", bad_node.op_type])
print(f"graph cut before the first {bad_node.op_type}: {json.dumps(r)}", flush=True)
