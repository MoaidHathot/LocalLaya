/**
 * A lane = one Laya session for one execution provider, living either in this thread or in a worker thread.
 * Both flavours expose the same handle so the router does not care where the session runs:
 *
 *   const L = await openLane("webgpu:fp16", { ep: "webgpu", modelDir }, { worker: true });
 *   const r = await L.systemOne(state, questions, tempsOverride);   // one call at a time (the router serialises)
 *   L.config, L.loadMs, L.dead, L.onDeath(cb)
 *   await L.close();
 *
 * Why worker threads: onnxruntime-node runs session.run() synchronously on the calling JS thread, so an
 * in-process lane blocks the event loop for the whole inference (300 ms for 3 CPU questions, ~1 s for 10).
 * A server cannot accept requests, answer /health or run its idle timer meanwhile, and two lanes cannot even
 * load in parallel. In a worker the main thread stays free (max stall ~20 ms measured) and lanes load side by
 * side. The router still serialises inferences across lanes: mixing them is a net loss (see ep-router.mjs).
 *
 * The worker is unref'd while idle and ref'd while a call is in flight, so a script that forgets close()
 * still exits like it did with in-process sessions.
 *
 * Never worker.terminate() while a call is in flight: session.run() is synchronous native code on that
 * thread and tearing the isolate down under it kills the whole process (0xC0000409, measured). close() asks
 * the worker to release the session and exit by itself; terminate() is only used once it is idle.
 */
import { Worker } from "node:worker_threads";
import { loadLaya } from "./laya-client.mjs";

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
 * Open a lane. `loadOpts` are loadLaya() options ({ ep, threads, pinToPCores, modelDir, calibration }).
 * @param {{ worker?: boolean, log?: (m:string)=>void }} o  worker=true (default) runs the session in a worker thread
 * @returns {Promise<InProcessLane|WorkerLane>}
 */
export async function openLane(lane, loadOpts, { worker = true, log = () => {} } = {}) {
  if (worker) return WorkerLane.open(lane, loadOpts, { log });
  const { laya, loadMs } = await loadLaya({ ...loadOpts, log, logSeverityLevel: loadOpts.logSeverityLevel ?? 3 });
  return new InProcessLane(lane, laya, loadMs);
}
