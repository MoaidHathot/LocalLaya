/**
 * Set up the Python side of the CUDA lane: onnxruntime-gpu + the CUDA / cuDNN runtime wheels in .venv.
 *
 *   npm run cuda:setup                 # CUDA 13 (driver >= 580 / "CUDA Version: 13.x" in nvidia-smi)
 *   node tools/setup-cuda-lane.mjs --cuda 12       # CUDA 12 wheels (driver 525+); needs PyPI to be reachable
 *   node tools/setup-cuda-lane.mjs --check          # only verify what is installed
 *   node tools/setup-cuda-lane.mjs --graphs         # only (re)generate the static CUDA-graph bucket files
 *
 * After the install the static-shape graphs for the lane's default eager buckets are generated next to
 * models/laya-onnx-fp16/laya.onnx (tools/static_graph.py, ~1-3 s each on the CPU) together with their
 * pre-optimised variants (tools/cuda_lane.py --prepare-buckets: what ORT would otherwise optimise at every session
 * creation, halving the ~250 ms a bucket build can stall a call); the lane generates any missing file on demand.
 * Skipped when the fp16 bundle is not there yet (`npm run fp16:check`).
 *
 * Sources (chosen because they are reachable from networks that block files.pythonhosted.org):
 *   - onnxruntime-gpu: Microsoft's release feed for the CUDA 13 build
 *       https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
 *     (the CUDA 12 build is the PyPI default `onnxruntime-gpu`)
 *   - CUDA runtime libraries + cuDNN: NVIDIA's own index https://pypi.nvidia.com
 * Versions are pinned to what was measured (results/cuda-lane-2026-09-23-summary.md); the CUDA 13.2 libraries
 * match a driver reporting CUDA 13.2. Everything is installed with --no-deps: the Python dependencies of
 * onnxruntime are already in the venv from the fp16 conversion tooling (onnx, numpy, protobuf, ...).
 *
 * Requires `uv` and an existing .venv (`uv venv .venv` + `uv pip install --python .venv/Scripts/python.exe onnx onnxruntime`
 * as documented in docs/STATUS.md). onnxruntime (CPU) is replaced by onnxruntime-gpu, which contains the same
 * `onnxruntime.transformers` tooling plus the CUDA EP.
 */
import { execFile } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs, promisify } from "node:util";

const run = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PY = process.env.LAYA_PYTHON ?? path.join(ROOT, ".venv", "Scripts", "python.exe");
const { values: args } = parseArgs({ options: { cuda: { type: "string", default: "13" }, check: { type: "boolean", default: false }, graphs: { type: "boolean", default: false }, "model-dir": { type: "string", default: "models/laya-onnx-fp16" }, buckets: { type: "string", default: "1x96x8,3x96x8,5x96x8,5x128x8" } } });

const PINS = {
  13: {
    ort: { index: "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/", spec: "onnxruntime-gpu==1.30.0" },
    nvidia: { index: "https://pypi.nvidia.com", specs: ["nvidia-cuda-runtime==13.2.86", "nvidia-cublas==13.2.2.2", "nvidia-cufft==12.2.0.57", "nvidia-curand==10.4.2.66", "nvidia-cuda-nvrtc==13.2.86", "nvidia-nvjitlink==13.2.86", "nvidia-cudnn-cu13==9.14.0.64"] },
  },
  12: {
    ort: { index: "https://pypi.org/simple/", spec: "onnxruntime-gpu==1.30.0" },
    nvidia: { index: "https://pypi.nvidia.com", specs: ["nvidia-cuda-runtime-cu12==12.9.79", "nvidia-cublas-cu12==12.9.1.4", "nvidia-cufft-cu12==11.4.1.4", "nvidia-curand-cu12==10.3.10.19", "nvidia-cuda-nvrtc-cu12==12.9.86", "nvidia-nvjitlink-cu12==12.9.86", "nvidia-cudnn-cu12==9.14.0.64"] },
  },
};

const say = (m) => console.log(m);
const exists = (p) => access(p).then(() => true, () => false);

async function check() {
  const code = `
import json
try:
    import onnxruntime as ort
    info = {"onnxruntime": ort.__version__, "providers": ort.get_available_providers(), "cuda_build": "CUDAExecutionProvider" in ort.get_available_providers()}
    if info["cuda_build"]:
        ort.set_default_logger_severity(3)
        ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        import numpy as np
        so = ort.SessionOptions(); so.log_severity_level = 3
        # a 1-node graph is enough to prove the EP can be created and run on this GPU
        import onnx
        from onnx import helper, TensorProto
        g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 9
        s = ort.InferenceSession(m.SerializeToString(), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        info["session_providers"] = s.get_providers()
        info["cuda_ok"] = s.get_providers()[0] == "CUDAExecutionProvider" and s.run(None, {"x": np.ones(2, np.float32)})[0].tolist() == [1.0, 1.0]
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
`;
  const { stdout } = await run(PY, ["-c", code], { env: { ...process.env, PYTHONUTF8: "1" } });
  return JSON.parse(stdout.trim().split("\n").pop());
}

if (!(await exists(PY))) {
  say(`no Python at ${PY}. Create the venv first: uv venv .venv && uv pip install --python .venv/Scripts/python.exe onnx onnxruntime (or set LAYA_PYTHON)`);
  process.exit(2);
}
if (args.check) {
  const info = await check();
  say(JSON.stringify(info, null, 2));
  process.exit(info.cuda_ok ? 0 : 1);
}

/** Static-shape graphs for the lane's eager buckets (+ the dynamic graph with the weights as inputs). */
async function buildGraphs() {
  const dir = path.resolve(ROOT, args["model-dir"]);
  if (!(await exists(path.join(dir, "laya.onnx")))) {
    say(`no fp16 bundle at ${dir} - skipping the static graphs (tools/cuda_lane.py generates them on first start)`);
    return;
  }
  say(`> static graphs for ${args.buckets} in ${dir}`);
  const { stdout } = await run(PY, ["-u", path.join(ROOT, "tools", "static_graph.py"), dir, "--buckets", args.buckets, "--weights-as-inputs", "--dynamic"], { env: { ...process.env, PYTHONUTF8: "1" }, maxBuffer: 16 * 1024 * 1024 });
  for (const l of stdout.trim().split(/\r?\n/)) say(`  ${l}`);
  say(`> pre-optimised session files (onnxruntime ${(await check()).onnxruntime})`);
  const prep = await run(PY, ["-u", path.join(ROOT, "tools", "cuda_lane.py"), "--model-dir", dir, "--graph-buckets", args.buckets, "--prepare-buckets"], { env: { ...process.env, PYTHONUTF8: "1" }, maxBuffer: 16 * 1024 * 1024 });
  for (const l of prep.stderr.trim().split(/\r?\n/).filter((l) => /files ready|FAILED|failed/.test(l))) say(`  ${l.replace(/^\[cuda_lane \d+\] /, "")}`);
}

if (args.graphs) {
  await buildGraphs();
  process.exit(0);
}
const pins = PINS[args.cuda];
if (!pins) {
  say(`--cuda must be 12 or 13`);
  process.exit(2);
}
const uv = async (argv) => {
  say(`> uv ${argv.join(" ")}`);
  const { stdout, stderr } = await run("uv", argv, { maxBuffer: 16 * 1024 * 1024 });
  const lines = (stdout + stderr).trim().split(/\r?\n/).filter((l) => /^(\s*[+-] |Installed|Uninstalled|Audited|error)/.test(l));
  for (const l of lines) say(`  ${l.trim()}`);
};
try {
  await uv(["pip", "uninstall", "--python", PY, "onnxruntime"]).catch(() => {});
  await uv(["pip", "install", "--python", PY, "--no-deps", "--index-url", pins.ort.index, pins.ort.spec]);
  await uv(["pip", "install", "--python", PY, "--no-deps", "--index-url", pins.nvidia.index, ...pins.nvidia.specs]);
} catch (e) {
  say(`install failed: ${String(e.stderr ?? e.message).split("\n").slice(-6).join("\n")}`);
  process.exit(1);
}
const info = await check();
say(JSON.stringify(info, null, 2));
if (!info.cuda_ok) {
  say("CUDA EP did not come up. Check `nvidia-smi` (driver / CUDA version) and the versions pinned in this script.");
  process.exit(1);
}
await buildGraphs().catch((e) => say(`static graphs not generated (${String(e.message).split("\n")[0].slice(0, 120)}); tools/cuda_lane.py generates them on first start`));
say(`CUDA lane ready: onnxruntime-gpu ${info.onnxruntime}. Try: node router-demo.mjs --lanes cuda:fp16,webgpu:fp16,cpu:8`);
