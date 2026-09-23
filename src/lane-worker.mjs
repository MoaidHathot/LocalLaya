/**
 * Worker-thread side of a lane (see lane.mjs): loads one Laya session and answers systemOne requests.
 *
 * Protocol (main -> worker):  { type: "run", id, state, questions, temps? } | { type: "close" }
 *          (worker -> main): { type: "ready", config, loadMs, source, modelDir } | { type: "load-error", message }
 *                            | { type: "log", message } | { type: "result", id, result } | { type: "result", id, error }
 *                            | { type: "closed" }
 *
 * Messages are handled strictly in order: a "close" received behind a "run" releases the session only after
 * that run has finished (session.run() is synchronous native code; releasing under it, or terminating the
 * thread during it, crashes the whole process with 0xC0000409).
 *
 * `temps` is a per-call override of temperature_by_options applied on top of the lane's base table (the
 * table the session was created with, i.e. shipped or LayaRouter.create({ calibration })).
 */
import { parentPort, workerData } from "node:worker_threads";
import { loadLaya } from "./laya-client.mjs";

const post = (m) => parentPort.postMessage(m);
let laya = null;
let baseTemps = {};

try {
  const loaded = await loadLaya({ ...workerData.loadOpts, log: (message) => post({ type: "log", message }), logSeverityLevel: workerData.loadOpts.logSeverityLevel ?? 3 });
  laya = loaded.laya;
  baseTemps = { ...laya.config.temperature_by_options };
  post({ type: "ready", config: laya.config, loadMs: loaded.loadMs, source: loaded.source, modelDir: loaded.modelDir });
} catch (e) {
  post({ type: "load-error", message: String(e?.message ?? e), stack: String(e?.stack ?? "") });
  process.exit(1); // in a worker this ends the thread, not the process
}

async function handle(m) {
  if (m.type === "close") {
    try {
      await laya.close();
    } finally {
      post({ type: "closed" });
      process.exit(0);
    }
  }
  if (m.type !== "run") return;
  try {
    const temps = laya.config.temperature_by_options;
    for (const k of Object.keys(temps)) delete temps[k];
    Object.assign(temps, baseTemps, m.temps ?? {});
    const result = await laya.systemOne(m.state, m.questions);
    post({ type: "result", id: m.id, result });
  } catch (e) {
    post({ type: "result", id: m.id, error: String(e?.message ?? e) });
  }
}

let chain = Promise.resolve();
parentPort.on("message", (m) => {
  chain = chain.then(() => handle(m));
});
