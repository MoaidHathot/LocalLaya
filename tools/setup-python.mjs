#!/usr/bin/env node
/**
 * Isolated Python runtime for Vega-U and Laya Multilingual (the ONNX .venv is separate).
 *   npm run python:setup
 *   npm run python:setup -- --torch-backend cpu
 *   npm run python:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
 *   npm run python:check
 *   npm run vega:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple  # optional fused kernels
 *   npm run vega:check
 *   npm run multilingual:setup -- --index-url https://pypi.tuna.tsinghua.edu.cn/simple
 *   npm run multilingual:check
 *   npm run models:download -- --model vega
 *   npm run models:download -- --model multilingual
 *   npm run models:download -- --model all --check    # offline digest verification
 */
import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { PROJECT_ROOT, PYTHON, PYTHON_MANIFEST } from "../src/models.mjs";

const { values: a } = parseArgs({ options: {
  check: { type: "boolean" }, download: { type: "boolean" }, model: { type: "string", default: "vega" },
  accelerate: { type: "boolean" },
  "multilingual-fast": { type: "boolean" },
  "torch-backend": { type: "string", default: "auto" }, "index-url": { type: "string" },
} });
const run = (command, argv) => new Promise((resolve, reject) => {
  const child = spawn(command, argv, { cwd: PROJECT_ROOT, stdio: "inherit", windowsHide: true,
    env: { ...process.env, UV_CACHE_DIR: path.join(PROJECT_ROOT, ".uv-cache"), PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" } });
  child.on("error", reject);
  child.on("exit", (code, signal) => code === 0 ? resolve() : reject(new Error(`${path.basename(command)} exited with ${code ?? signal}`)));
});

try {
  if (a.download) {
    if (![...Object.keys(PYTHON_MANIFEST.models), "all"].includes(a.model)) throw new Error("--model must be vega, multilingual or all");
    await run(PYTHON, ["tools/prepare_models.py", "--model", a.model, ...(a.check ? ["--check"] : [])]);
  } else if (a["multilingual-fast"]) {
    if (!(await access(PYTHON).then(() => true, () => false))) throw new Error("Run npm run python:setup first");
    if (!a.check) {
      await run("uv", ["pip", "install", "--python", PYTHON, "--no-deps",
        ...(a["index-url"] ? ["--index-url", a["index-url"]] : []),
        ...Object.entries(PYTHON_MANIFEST.multilingualAcceleration).map(([name, version]) => `${name}==${version}`)]);
    }
    await run(PYTHON, ["tools/multilingual_fast.py"]);
  } else if (a.accelerate) {
    if (!(await access(PYTHON).then(() => true, () => false))) throw new Error("Run npm run python:setup first");
    if (!a.check) {
      const skip = process.platform === "win32" ? "triton" : "triton-windows";
      const pins = Object.entries(PYTHON_MANIFEST.acceleration).filter(([name]) => name !== skip);
      await run("uv", ["pip", "install", "--python", PYTHON, "--no-deps",
        ...(a["index-url"] ? ["--index-url", a["index-url"]] : []), ...pins.map(([name, version]) => `${name}==${version}`)]);
    }
    await run(PYTHON, ["tools/vega_fast.py"]);
  } else {
    if (!a.check) {
      if (!(await access(PYTHON).then(() => true, () => false))) {
        if (process.env.LAYA_MODEL_PYTHON) throw new Error(`LAYA_MODEL_PYTHON does not exist: ${PYTHON}`);
        await run("uv", ["venv", "--python", "3.12", ".venv-laya"]);
      }
      await run("uv", ["pip", "install", "--python", PYTHON, "--torch-backend", a["torch-backend"],
        ...(a["index-url"] ? ["--index-url", a["index-url"]] : []),
        ...Object.entries(PYTHON_MANIFEST.runtime).map(([name, version]) => `${name}==${version}`)]);
    }
    await run(PYTHON, ["tools/laya_worker.py", "--check"]);
    if (!a.check) console.log("Runtime ready. Next: npm run models:download -- --model vega");
  }
} catch (e) {
  console.error(e.message);
  process.exitCode = 1;
}
