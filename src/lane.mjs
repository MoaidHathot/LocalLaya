/**
 * A lane = one Laya session for one execution provider, living in this thread, in a worker thread, or in a
 * separate process (CUDA, via Python onnxruntime-gpu). All three expose the same handle so the router does not
 * care where the session runs:
 *
 *   const L = await openLane("webgpu:fp16", { ep: "webgpu", modelDir }, { worker: true });
 *   const r = await L.systemOne(state, questions, tempsOverride);   // one call at a time (the router serialises)
 *   L.config, L.loadMs, L.mode, L.dead, L.onDeath(cb)
 *   await L.close();
 *
 * Why worker threads: onnxruntime-node runs session.run() synchronously on the calling JS thread, so an
 * in-process lane blocks the event loop for the whole inference (300 ms for 3 CPU questions, ~1 s for 10).
 * A server cannot accept requests, answer /health or run its idle timer meanwhile, and two lanes cannot even
 * load in parallel. In a worker the main thread stays free (max stall ~20 ms measured) and lanes load side by
 * side. The router still serialises inferences across lanes: mixing them is a net loss (see ep-router.mjs).
 *
 * Why a process for CUDA: onnxruntime-node has no CUDA EP on Windows; the Python wheel has. tools/cuda_lane.py
 * holds the session and answers over stdio; here a RemoteSession implements the two methods @receptron/laya
 * calls (run, release), so tokenising, temperatures and answer formatting stay in the vendored library.
 *
 * The worker is unref'd while idle and ref'd while a call is in flight, so a script that forgets close()
 * still exits like it did with in-process sessions.
 *
 * Never worker.terminate() while a call is in flight: session.run() is synchronous native code on that
 * thread and tearing the isolate down under it kills the whole process (0xC0000409, measured). close() asks
 * the worker to release the session and exit by itself; terminate() is only used once it is idle.
 */
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import { Laya } from "@receptron/laya";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-node";
import { loadLaya, PCORE_LOGICAL, resolveModelDir } from "./laya-client.mjs";

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
/** Execution providers that run in a separate process (Python onnxruntime-gpu) rather than in onnxruntime-node. */
export const PROCESS_EPS = ["cuda"];
/** Python with onnxruntime-gpu for the CUDA lane; override with LAYA_PYTHON. */
export const LANE_PYTHON = process.env.LAYA_PYTHON ?? path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe");

export class LaneDeadError extends Error {
  constructor(lane, cause) {
    super(`lane ${lane} is gone (${cause})`);
    this.code = "LANE_DEAD";
    this.lane = lane;
  }
}

/** Replace the session's temperature table for the next call: base table + optional override. */
function applyTemps(laya, baseTemps, override) {
  const temps = laya.config.temperature_by_options;
  for (const k of Object.keys(temps)) delete temps[k];
  Object.assign(temps, baseTemps, override ?? {});
}

class InProcessLane {
  constructor(lane, laya, loadMs) {
    this.lane = lane;
    this.mode = "in-process";
    this.laya = laya;
    this.config = laya.config;
    this.baseTemps = { ...laya.config.temperature_by_options };
    this.loadMs = loadMs;
    this.dead = false;
    this.closed = false;
    this._onDeath = [];
  }
  onDeath(cb) {
    this._onDeath.push(cb);
  }
  async systemOne(state, questions, temps) {
    if (this.closed) throw new LaneDeadError(this.lane, "closed");
    applyTemps(this.laya, this.baseTemps, temps);
    return this.laya.systemOne(state, questions);
  }
  async close() {
    if (this.closed) return;
    this.closed = true;
    await this.laya.close();
  }
}

class WorkerLane {
  constructor(lane) {
    this.lane = lane;
    this.mode = "worker";
    this.worker = null;
    this.config = null;
    this.loadMs = null;
    this.dead = false;
    this.closed = false;
    this.pending = new Map(); // id -> { resolve, reject }
    this.seq = 0;
    this._onDeath = [];
    this._closing = null;
  }
  onDeath(cb) {
    this._onDeath.push(cb);
  }

  static open(lane, loadOpts, { log = () => {} } = {}) {
    const L = new WorkerLane(lane);
    return new Promise((resolve, reject) => {
      const w = new Worker(new URL("./lane-worker.mjs", import.meta.url), { workerData: { lane, loadOpts }, name: `laya ${lane}` });
      L.worker = w;
      let settled = false;
      const fail = (e) => {
        if (!settled) {
          settled = true;
          reject(e);
        }
      };
      w.on("message", (m) => {
        switch (m.type) {
          case "log":
            log(m.message);
            break;
          case "ready":
            L.config = m.config;
            L.loadMs = m.loadMs;
            L.source = m.source;
            L.modelDir = m.modelDir;
            settled = true;
            w.unref(); // idle: do not keep the process alive
            resolve(L);
            break;
          case "load-error":
            fail(Object.assign(new Error(m.message), { workerStack: m.stack }));
            break;
          case "result": {
            const p = L.pending.get(m.id);
            if (!p) break;
            L.pending.delete(m.id);
            if (!L.pending.size && !L.closed) w.unref();
            m.error !== undefined ? p.reject(new Error(m.error)) : p.resolve(m.result);
            break;
          }
          case "closed":
            L._closing?.resolve();
            break;
        }
      });
      w.on("error", (e) => {
        fail(e);
        L._die(`worker error: ${String(e?.message ?? e).split("\n")[0]}`);
      });
      w.on("messageerror", (e) => L._die(`message error: ${String(e?.message ?? e)}`));
      w.on("exit", (code) => {
        fail(new Error(`worker exited with code ${code} before the session was ready`));
        L._die(L.closed ? "closed" : `worker exited with code ${code}`);
      });
    });
  }

  /** The worker is gone: fail every in-flight call and tell the router. */
  _die(cause) {
    if (this.dead) return;
    this.dead = true;
    for (const p of this.pending.values()) p.reject(new LaneDeadError(this.lane, cause));
    this.pending.clear();
    this._closing?.resolve();
    if (!this.closed) for (const cb of this._onDeath) cb(cause);
  }

  systemOne(state, questions, temps) {
    if (this.dead || this.closed) return Promise.reject(new LaneDeadError(this.lane, this.closed ? "closed" : "worker gone"));
    return new Promise((resolve, reject) => {
      const id = this.seq++;
      this.pending.set(id, { resolve, reject });
      if (this.pending.size === 1) this.worker.ref(); // a call in flight keeps the process alive
      this.worker.postMessage({ type: "run", id, state, questions, temps });
    });
  }

  /**
   * Release the session and end the worker. Idempotent. Calls already sent to the worker finish first (the
   * worker handles messages in order); after `timeoutMs` without the worker's "closed" the handle is given
   * up but the thread is only terminated when it is idle (see the header).
   */
  async close({ timeoutMs = 10_000 } = {}) {
    if (this.closed) return this._closing?.promise;
    this.closed = true;
    let resolve;
    const promise = new Promise((r) => (resolve = r));
    this._closing = { promise, resolve };
    if (!this.dead) {
      this.worker.ref();
      this.worker.postMessage({ type: "close" });
      const t = setTimeout(() => resolve(), timeoutMs);
      await promise;
      clearTimeout(t);
    }
    if (!this.pending.size) await this.worker.terminate().catch(() => {});
    else this.worker.unref();
    this.dead = true;
  }
}

/**
 * The two methods @receptron/laya calls on an onnxruntime-node InferenceSession, implemented over a child
 * process that speaks the tools/cuda_lane.py protocol. Tensors travel as base64 (a 10-question call is ~80 KB
 * in, a few hundred bytes out). Requests are answered in order.
 */
class RemoteSession {
  constructor(lane) {
    this.lane = lane; // the ProcessLane (child, pending map, death handling)
  }
  async run(feeds) {
    const wire = {};
    for (const [name, t] of Object.entries(feeds)) {
      const buf = t.data instanceof BigInt64Array ? Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength) : Buffer.from(t.data.buffer ?? t.data, t.data.byteOffset ?? 0, t.data.byteLength ?? t.data.length);
      wire[name] = { dtype: t.type, dims: t.dims, data: buf.toString("base64") };
    }
    const res = await this.lane._request({ feeds: wire });
    const out = {};
    for (const [name, t] of Object.entries(res.outputs)) {
      const buf = Buffer.from(t.data, "base64");
      const data = t.dtype === "float32" ? new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4) : t.dtype === "float16" ? new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2) : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      out[name] = new ort.Tensor(t.dtype === "float16" ? "float16" : t.dtype === "float32" ? "float32" : "bool", data, t.dims);
    }
    this.lane.lastInferenceMs = res.ms;
    return out;
  }
  async release() {
    // the lane owns the child; nothing to do here (ProcessLane.close ends the process)
  }
}

/** Read the bundle's config + tokenizer exactly like Laya.load() does, then build a Laya on any session object. */
async function layaOnSession(session, modelDir) {
  const read = async (f) => JSON.parse(await readFile(path.join(modelDir, f), "utf8"));
  const config = await read("laya_config.json");
  const tok = new Tokenizer(await read("tokenizer/tokenizer.json"), await read("tokenizer/tokenizer_config.json"));
  const id = (t) => {
    const v = tok.token_to_id(t);
    if (v === undefined) throw new Error(`special token ${t} missing from tokenizer`);
    return v;
  };
  const ids = { cls: id("[CLS]"), sep: id("[SEP]"), mask: id("[MASK]"), pad: id("[PAD]"), maskTok: "[MASK]" };
  return new Laya(session, tok, config, ids, modelDir);
}

class ProcessLane {
  constructor(lane) {
    this.lane = lane;
    this.mode = "process";
    this.child = null;
    this.pid = null;
    this.laya = null;
    this.config = null;
    this.loadMs = null;
    this.dead = false;
    this.closed = false;
    this.pending = new Map(); // id -> { resolve, reject }
    this.seq = 0;
    this._onDeath = [];
    this._exit = null;
    this.lastInferenceMs = null;
  }
  onDeath(cb) {
    this._onDeath.push(cb);
  }

  /**
   * @param {object} loadOpts { ep: "cuda", modelDir, deviceId, threads, calibration, python }
   */
  static async open(lane, loadOpts, { log = () => {} } = {}) {
    const L = new ProcessLane(lane);
    const python = loadOpts.python ?? LANE_PYTHON;
    const modelDir = resolveModelDir(loadOpts.modelDir);
    const argv = [path.join(PROJECT_ROOT, "tools", "cuda_lane.py"), "--model-dir", modelDir, "--device", String(loadOpts.deviceId ?? 0), "--threads", String(loadOpts.threads ?? 2)];
    if (process.platform === "win32" && PCORE_LOGICAL > 0) argv.push("--affinity", `0-${PCORE_LOGICAL - 1}`);
    if (loadOpts.cudaGraph) argv.push("--cuda-graph");
    const t0 = performance.now();
    const child = spawn(python, argv, { cwd: PROJECT_ROOT, stdio: ["pipe", "pipe", "pipe"], windowsHide: true, env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1" } });
    L.child = child;
    L.pid = child.pid;
    L._exit = new Promise((resolve) => child.once("exit", (code, signal) => resolve({ code, signal })));
    const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
    createInterface({ input: child.stderr, crlfDelay: Infinity }).on("line", (l) => log(l));
    const ready = await new Promise((resolve, reject) => {
      const onExit = ({ code }) => reject(new Error(`cuda lane process exited with code ${code} before it was ready`));
      child.once("error", (e) => reject(new Error(`could not start ${python}: ${e.message}`)));
      L._exit.then(onExit);
      lines.once("line", (line) => {
        let m;
        try {
          m = JSON.parse(line);
        } catch {
          return reject(new Error(`unexpected first line from cuda lane: ${line.slice(0, 200)}`));
        }
        if (!m.ready) return reject(new Error(m.error ?? "cuda lane failed to load"));
        resolve(m);
      });
    }).catch(async (e) => {
      child.kill();
      throw e;
    });
    lines.on("line", (line) => L._onLine(line));
    L._exit.then(({ code, signal }) => L._die(L.closed ? "closed" : `process exited (code ${code}${signal ? `, signal ${signal}` : ""})`));
    L.laya = await layaOnSession(new RemoteSession(L), modelDir);
    if (loadOpts.calibration) {
      const { applyCalibration } = await import("./calibration.mjs");
      const table = typeof loadOpts.calibration === "string" ? JSON.parse(await readFile(path.resolve(PROJECT_ROOT, loadOpts.calibration), "utf8")) : loadOpts.calibration;
      applyCalibration(L.laya, table);
    }
    L.config = L.laya.config;
    L.baseTemps = { ...L.laya.config.temperature_by_options };
    L.loadMs = performance.now() - t0;
    L.providers = ready.providers;
    L.remoteLoadMs = ready.loadMs;
    return L;
  }

  _onLine(line) {
    let m;
    try {
      m = JSON.parse(line);
    } catch {
      return;
    }
    const p = this.pending.get(m.id);
    if (!p) return;
    this.pending.delete(m.id);
    m.error !== undefined ? p.reject(new Error(m.error)) : p.resolve(m);
  }

  _request(body) {
    if (this.dead || this.closed) return Promise.reject(new LaneDeadError(this.lane, this.closed ? "closed" : "process gone"));
    return new Promise((resolve, reject) => {
      const id = this.seq++;
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(JSON.stringify({ id, ...body }) + "\n", (e) => {
        if (e) {
          this.pending.delete(id);
          reject(new LaneDeadError(this.lane, `stdin: ${e.message}`));
        }
      });
    });
  }

  _die(cause) {
    if (this.dead) return;
    this.dead = true;
    for (const p of this.pending.values()) p.reject(new LaneDeadError(this.lane, cause));
    this.pending.clear();
    if (!this.closed) for (const cb of this._onDeath) cb(cause);
  }

  async systemOne(state, questions, temps) {
    if (this.dead || this.closed) throw new LaneDeadError(this.lane, this.closed ? "closed" : "process gone");
    applyTemps(this.laya, this.baseTemps, temps);
    return this.laya.systemOne(state, questions);
  }

  /** Ask the process to release the session and exit; kill it after `timeoutMs`. Killing a separate process is safe. */
  async close({ timeoutMs = 5000 } = {}) {
    if (this.closed) return;
    this.closed = true;
    if (!this.dead) {
      try {
        this.child.stdin.write(JSON.stringify({ op: "close" }) + "\n");
        this.child.stdin.end();
      } catch {
        /* already gone */
      }
      const t = setTimeout(() => this.child.kill(), timeoutMs);
      await this._exit;
      clearTimeout(t);
    }
    this.dead = true;
  }
}

/**
 * Open a lane. `loadOpts` are loadLaya() options ({ ep, threads, pinToPCores, modelDir, calibration }); for
 * PROCESS_EPS (cuda) also { deviceId, python, cudaGraph }.
 * @param {{ worker?: boolean, log?: (m:string)=>void }} o  worker=true (default) runs an in-node session in a worker thread
 * @returns {Promise<InProcessLane|WorkerLane|ProcessLane>}
 */
export async function openLane(lane, loadOpts, { worker = true, log = () => {} } = {}) {
  if (PROCESS_EPS.includes(loadOpts.ep)) return ProcessLane.open(lane, loadOpts, { log });
  if (worker) return WorkerLane.open(lane, loadOpts, { log });
  const { laya, loadMs } = await loadLaya({ ...loadOpts, log, logSeverityLevel: loadOpts.logSeverityLevel ?? 3 });
  return new InProcessLane(lane, laya, loadMs);
}
