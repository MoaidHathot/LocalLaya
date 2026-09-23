/**
 * demo/laya.mjs - Laya decisions from your own Node code, one import.
 *
 *   import { createLaya, top, gate } from "./laya.mjs";
 *   const laya = await createLaya();                       // mode "auto": shared sidecar, spawned on demand
 *   const r = await laya.decide("Charged twice, refund me", { preset: "triage" });
 *   r.answers.department.choice           // "billing"
 *   top(r.answers.department)             // { label: "billing", p: 0.97 }
 *   gate(r.answers.department)            // "act" | "ask" | "unsure"
 *   await laya.close();
 *
 * Modes (option `mode`, default "auto"):
 *   "auto"     use the shared background instance (sidecar); spawn it if it is not running; if the port is taken
 *              by something else or the sidecar cannot start, load the model in this process instead
 *   "sidecar"  the sidecar only (throws if it cannot be used)
 *   "local"    load the model in this process (2-3 s; no background process; N processes = N copies)
 *   "http"     an already running server at `url` (never spawns, never falls back)
 * Every mode returns the same result shape as the HTTP API: { answers, usage, routing, state, questions, preset,
 * calibration }; `routing.lane` / `routing.ms` say where and how fast it ran, `routing.queueMs` how long it
 * waited behind other callers. Which lane runs a call is the router's decision unless you pass `lane`.
 *
 * decide(input, opts)
 *   input   a string (wrapped by the preset's state template) or { state: <any JSON> }
 *   opts    { preset, questions, lane, policy, deadlineMs, calibration, exec }  (all optional; see references/api.md)
 * decideMany(inputs, opts, { concurrency = 4 })   same, for a list; results in input order; the sidecar serves
 *   them one at a time (one queue), concurrency only overlaps the HTTP round trips.
 *
 * Costs measured on the reference machine (RTX 4070, CUDA lane): decide() over the sidecar's keep-alive
 * connection ~13-15 ms for 3-5 questions; local mode the same minus ~1 ms once loaded. The first call of a
 * process pays either the sidecar spawn (2.5-4.5 s, once per idle period) or the local load (2-3 s, every process).
 *
 * This file only needs the LocalLaya project: it finds it through LAYA_DIR or its own location
 * (<project>/demo/laya.mjs), so it can be copied into another project with LAYA_DIR set.
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const candidates = [process.env.LAYA_DIR, path.resolve(here, "..")].filter(Boolean);
export const PROJECT_DIR = candidates.find((d) => existsSync(path.join(d, "serve.mjs")) && existsSync(path.join(d, "src", "sidecar-client.mjs")));
if (!PROJECT_DIR) throw new Error(`demo/laya.mjs: cannot find the LocalLaya project (looked at ${candidates.join(", ")}); set LAYA_DIR`);
const mod = (rel) => import(pathToFileURL(path.join(PROJECT_DIR, rel)).href);

/** Top answer of any question type: { label, p } (for score: the nearest level and its probability). */
export function top(answer) {
  if (!answer) return null;
  if (answer.type === "choice") return { label: answer.choice, p: answer.probabilities[answer.choice] };
  if (answer.type === "noul") return { label: answer.noul >= 0.5 ? "yes" : "no", p: answer.noul >= 0.5 ? answer.noul : 1 - answer.noul };
  if (answer.type === "score") {
    const levels = Object.values(answer.legend);
    const i = Math.min(levels.length - 1, Math.max(0, Math.round(answer.score)));
    return { label: levels[i], p: answer.probabilities[String(i)] ?? 0, score: answer.score, max: levels.length - 1 };
  }
  return null;
}

/**
 * Gate on the top probability (not on `confidence`, which is 1 - entropy): "act" at >= act, "ask" between, "unsure"
 * below. Defaults 0.8 / 0.55 are the starting point from the README; tune per question with labelled data.
 */
export function gate(answer, { act = 0.8, ask = 0.55 } = {}) {
  const t = top(answer);
  if (!t) return "unsure";
  return t.p >= act ? "act" : t.p >= ask ? "ask" : "unsure";
}

/** One line per answers object, e.g. "department=billing 0.97, urgency=urgent 0.61, churn_risk=yes 0.76". */
export const summarize = (answers) =>
  Object.entries(answers)
    .map(([id, a]) => {
      const t = top(a);
      return `${id}=${t.label} ${t.p.toFixed(2)}`;
    })
    .join(", ");

const toBody = (input, opts = {}) => {
  const body = {};
  if (typeof input === "string") body.text = input;
  else if (input && typeof input === "object" && "state" in input) body.state = input.state;
  else if (input && typeof input === "object") body.state = input;
  else throw new Error("decide(): input must be a string or an object");
  for (const k of ["preset", "questions", "lane", "policy", "deadlineMs", "calibration", "exec"]) if (opts[k] !== undefined) body[k] = opts[k];
  return body;
};

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
      for (;;) {
        const i = next++;
        if (i >= items.length) return;
        out[i] = await fn(items[i], i);
      }
    }),
  );
  return out;
}

/**
 * @param {object} [o]
 * @param {"auto"|"sidecar"|"local"|"http"} [o.mode="auto"]
 * @param {number} [o.port]            sidecar port (default LAYA_PORT or 8787)
 * @param {string} [o.url]             mode "http": base URL of a running server, e.g. http://127.0.0.1:8787
 * @param {string} [o.idle]            sidecar idle exit when this call spawns it (default LAYA_IDLE or "5m")
 * @param {string} [o.maxAge]          sidecar max age when this call spawns it
 * @param {string} [o.lanes]           lanes for a spawned sidecar or the local router, e.g. "cuda:fp16,webgpu:fp16,cpu:8"
 * @param {string} [o.calibration]     one calibration file for every preset (local mode; the sidecar uses calibration/<preset>.json)
 * @param {(m:string)=>void} [o.log]   progress messages (spawning, loading ...)
 */
export async function createLaya(o = {}) {
  const mode = o.mode ?? "auto";
  const log = o.log ?? (() => {});
  const sidecar = await mod("src/sidecar-client.mjs");
  const port = Number(o.port ?? sidecar.DEFAULT_PORT);
  const progress = (state, ms, health) => {
    if (state === "spawning") log(`no sidecar on :${port}; starting one ...`);
    else if (state === "attaching") log(`sidecar on :${port} is loading (pid ${health?.pid}); waiting ...`);
    else if (state === "ready") log(`sidecar ready in ${(ms / 1000).toFixed(1)} s (pid ${health?.pid}, lanes ${health?.lanes?.join(", ")})`);
  };

  const remote = (client, info) => ({
    mode: info.mode,
    info,
    decide: async (input, opts) => client.decide(toBody(input, opts)), // async: a bad input rejects instead of throwing synchronously
    decideMany: async (inputs, opts, { concurrency = 4 } = {}) => mapLimit(inputs, concurrency, (input) => client.decide(toBody(input, opts))),
    health: () => client.health(),
    stats: () => client.stats(),
    presets: () => client.presets(),
    close: async () => client.close(),
  });

  if (mode === "http") {
    if (!o.url) throw new Error('mode "http" needs url');
    const u = new URL(o.url);
    const client = sidecar.createClient({ port: Number(u.port || 80), host: u.hostname });
    const h = await client.health();
    if (h?.service !== "laya") throw new Error(`${o.url} is not a Laya server`);
    return remote(client, { mode: "http", url: o.url, pid: h.pid, lanes: h.lanes });
  }

  if (mode === "auto" || mode === "sidecar") {
    try {
      const { health } = await sidecar.ensureSidecar({ port, idle: o.idle ?? sidecar.DEFAULT_IDLE, maxAge: o.maxAge, lanes: o.lanes, onProgress: progress });
      const client = sidecar.createClient({ port });
      return remote(client, { mode: "sidecar", url: `http://127.0.0.1:${port}`, pid: health.pid, lanes: health.lanes, lanesLoading: health.lanesLoading ?? [] });
    } catch (e) {
      if (mode === "sidecar") throw e;
      log(`sidecar unavailable (${e.code ?? "error"}: ${String(e.message).split("\n")[0]}); loading the model in this process`);
    }
  }

  // local: the same request logic as the server, on an in-process router
  const [{ LayaRouter }, { loadPresets }, { CalibrationCache, decideRequest }] = await Promise.all([mod("src/ep-router.mjs"), mod("data/presets.mjs"), mod("src/decide-core.mjs")]);
  const t0 = performance.now();
  const router = await LayaRouter.create({
    lanes: (o.lanes ?? "cuda:fp16,webgpu:fp16,cpu:8").split(","),
    waitFor: "first",
    gpuKeepAliveMs: 30_000,
    warmup: { state: { text: "Please turn off the lights in the living room now" }, sizes: [3, 5] },
    log: (m) => log(`  ${m}`),
  });
  log(`model loaded in this process in ${((performance.now() - t0) / 1000).toFixed(1)} s (lanes ${[...router.lanes.keys()].join(", ")})`);
  const calibration = new CalibrationCache({ root: PROJECT_DIR, override: o.calibration ?? null });
  let presets = await loadPresets();
  const decide = async (input, opts) => {
    presets = await loadPresets();
    try {
      return await decideRequest(router, toBody(input, opts), presets, calibration);
    } catch (e) {
      if (e.status === 503) {
        await router.ready; // a forced lane still loading: wait for it, then retry once
        return decideRequest(router, toBody(input, opts), presets, calibration);
      }
      throw e;
    }
  };
  return {
    mode: "local",
    info: { mode: "local", lanes: [...router.lanes.keys()], router },
    decide,
    decideMany: (inputs, opts, { concurrency = 4 } = {}) => mapLimit(inputs, concurrency, (input) => decide(input, opts)),
    health: async () => ({ service: "laya", status: "ready", lanes: [...router.lanes.keys()], lanesLoading: router.pendingLanes, mode: "local" }),
    stats: async () => router.stats(),
    presets: async () => Object.fromEntries(Object.entries(await loadPresets()).map(([k, p]) => [k, { description: p.description, source: p.source, questions: p.questions }])),
    close: () => router.close(),
  };
}

/** Read a JSON file of inputs: an array of strings, or of { text | state, ... } objects (extra fields are kept). */
export function readInputs(file) {
  const raw = JSON.parse(readFileSync(file, "utf8"));
  const list = Array.isArray(raw) ? raw : raw.items;
  if (!Array.isArray(list)) throw new Error(`${file}: expected a JSON array (or { items: [...] })`);
  return list;
}
