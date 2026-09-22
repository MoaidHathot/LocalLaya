#!/usr/bin/env node
/**
 * Local HTTP API for Laya (model stays loaded, router picks the lane per request) + a minimal browser page.
 * Doubles as the on-demand sidecar that ask.mjs --sidecar spawns and talks to.
 *
 *   node serve.mjs                         # foreground server on http://127.0.0.1:8787, never exits on its own
 *   node serve.mjs --idle 5m               # exit after 5 minutes without requests (frees RAM + VRAM); 30s, 10m, 0 = never
 *   node serve.mjs --sidecar               # spawned by ask.mjs: --idle defaults to $LAYA_IDLE or 5m
 *   node serve.mjs --port 9000 --lanes webgpu:fp16,cpu:8
 *   node serve.mjs --calibration calibration/smart-home-v3.json   # one table for every preset (default: per preset,
 *                                                                 # calibration/<preset>.json when present)
 *   node serve.mjs --cors                  # allow browser pages from other origins to call the API (off by default)
 *
 * Lifecycle: the port is bound BEFORE the model loads, so the port doubles as the mutex between racing
 * launchers (a second instance gets EADDRINUSE and exits with code 3 without loading anything). While
 * loading, /health reports status "loading" and /decide answers 503. Idle exit only happens with no request
 * in flight. Background load sampling (nvidia-smi / CPU) pauses after 10 s idle.
 *
 * Endpoints (all JSON):
 *   GET  /                 browser UI
 *   GET  /health           { service: "laya", status: loading|ready|failed|stopping, pid, port, lanes, uptimeS,
 *                            idleS, idleRemainingS, inFlight, sidecar }
 *   GET  /presets          { name: { description, source, state, questions } }
 *   GET  /stats            router statistics (latency estimates, load, per-lane calls / pending)
 *   POST /decide           body: { text?: string, state?: any, preset?: string, questions?: {...}, lane?: string, deadlineMs?: number,
 *                                  calibration?: { temperature_by_options } }
 *                          -> { answers, usage, routing, state, questions, preset, calibration }
 *                          `text` is wrapped by the preset's state builder; `state` is used verbatim;
 *                          `questions` replaces the preset's question set; `calibration` overrides the per-preset
 *                          table for this call. Resets the idle timer.
 *   POST /touch            resets the idle timer without doing work (REPL keep-alive) -> { ok, idleRemainingS }
 *   POST /shutdown         graceful stop -> { ok: true }, then the process exits
 *
 * Presets (built-in + presets/*.json|*.mjs) and calibration tables (calibration/<preset>.json) are re-read
 * when their files change, so a long-running instance picks up edits without a restart.
 */
import { createServer } from "node:http";
import { access, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { LayaRouter } from "./src/ep-router.mjs";
import { loadPresets, DEFAULT_PRESET, describePresets, PRESETS_DIR } from "./data/presets.mjs";
import { parseDuration } from "./src/sidecar-client.mjs";

const PROJECT_ROOT = path.dirname(fileURLToPath(import.meta.url));
const VERSION = JSON.parse(await readFile(path.join(PROJECT_ROOT, "package.json"), "utf8")).version;

const { values: args } = parseArgs({
  options: {
    port: { type: "string", default: process.env.LAYA_PORT ?? "8787" },
    host: { type: "string", default: "127.0.0.1" },
    lanes: { type: "string", default: process.env.LAYA_LANES ?? "webgpu:fp16,cpu:8" },
    calibration: { type: "string" },
    cors: { type: "boolean", default: false },
    idle: { type: "string" },
    sidecar: { type: "boolean", default: false },
  },
});
const idleMs = parseDuration(args.idle ?? (args.sidecar ? process.env.LAYA_IDLE ?? "5m" : "0"));
const SAMPLING_PAUSE_MS = 10_000;
const log = (m) => console.log(`[serve${args.sidecar ? ":sidecar" : ""} ${process.pid}] ${new Date().toISOString().slice(11, 19)} ${m}`);
process.title = args.sidecar ? "laya-sidecar" : "laya-serve";

// ---- state ------------------------------------------------------------------------------------------------------
const started = Date.now();
let status = "loading"; // loading | ready | failed | stopping
let loadError = null;
let router = null;
let PRESETS = {};
let inFlight = 0;
let lastRequestAt = Date.now();
let idleTimer = null;
let samplingPauseTimer = null;
let shuttingDown = false;
const idleRemainingS = () => (idleMs > 0 ? Math.max(0, Math.round((lastRequestAt + idleMs - Date.now()) / 1000)) : null);

function touch() {
  lastRequestAt = Date.now();
  if (router && !router.sampling) router.resumeSampling();
  armTimers();
}

function armTimers() {
  if (idleTimer) clearTimeout(idleTimer);
  if (samplingPauseTimer) clearTimeout(samplingPauseTimer);
  samplingPauseTimer = setTimeout(() => {
    if (router && inFlight === 0) router.pauseSampling();
  }, SAMPLING_PAUSE_MS);
  samplingPauseTimer.unref();
  if (idleMs > 0) {
    idleTimer = setTimeout(onIdle, Math.max(250, lastRequestAt + idleMs - Date.now()));
    // deliberately NOT unref'd: the timer is what keeps the parked process alive
  }
}

function onIdle() {
  idleTimer = null;
  if (shuttingDown) return;
  if (inFlight > 0 || Date.now() < lastRequestAt + idleMs) return armTimers();
  log(`idle for ${Math.round(idleMs / 1000)} s with nothing in flight; exiting`);
  shutdown("idle", 0);
}

async function shutdown(reason, code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  status = "stopping";
  log(`shutting down (${reason}) ...`);
  if (idleTimer) clearTimeout(idleTimer);
  if (samplingPauseTimer) clearTimeout(samplingPauseTimer);
  server.close();
  // let in-flight decides finish (bounded), then drop keep-alive sockets so close() completes
  const deadline = Date.now() + 10_000;
  while (inFlight > 0 && Date.now() < deadline) await new Promise((r) => setTimeout(r, 50));
  server.closeAllConnections?.();
  try {
    await router?.close();
  } catch (e) {
    log(`router close: ${e?.message ?? e}`);
  }
  log("bye");
  process.exit(code);
}
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("uncaughtException", (e) => {
  log(`uncaught exception: ${e?.stack ?? e}`);
  shutdown("crash", 1);
});
process.on("unhandledRejection", (e) => {
  log(`unhandled rejection: ${e?.stack ?? e}`);
  shutdown("crash", 1);
});

// ---- per-preset calibration tables, re-read when the file changes ----------------------------------------------
const calibrationCache = new Map(); // preset -> { file, mtimeMs, table }
async function calibrationFor(name) {
  const file = path.resolve(PROJECT_ROOT, args.calibration ?? `calibration/${name}.json`);
  let mtimeMs;
  try {
    mtimeMs = (await stat(file)).mtimeMs;
  } catch {
    calibrationCache.delete(name);
    return null;
  }
  const cached = calibrationCache.get(name);
  if (cached && cached.file === file && cached.mtimeMs === mtimeMs) return cached.table;
  const table = { ...JSON.parse(await readFile(file, "utf8")), file: path.relative(PROJECT_ROOT, file).replace(/\\/g, "/") };
  calibrationCache.set(name, { file, mtimeMs, table });
  return table;
}

// ---- http helpers --------------------------------------------------------------------------------------------------
const json = (res, statusCode, body) => {
  const headers = { "content-type": "application/json; charset=utf-8" };
  if (args.cors) headers["access-control-allow-origin"] = "*";
  res.writeHead(statusCode, headers);
  res.end(JSON.stringify(body));
};

const readBody = (req, limit = 1_000_000) =>
  new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(Object.assign(new Error("body too large"), { status: 413 }));
        req.destroy();
      } else chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });

function validateQuestions(q) {
  if (!q || typeof q !== "object" || !Object.keys(q).length) throw Object.assign(new Error("questions must be a non-empty object"), { status: 400 });
  for (const [id, x] of Object.entries(q)) {
    if (!x || !["choice", "score", "noul"].includes(x.type)) throw Object.assign(new Error(`question ${id}: type must be choice | score | noul`), { status: 400 });
    if (x.instructions === undefined) throw Object.assign(new Error(`question ${id}: instructions required`), { status: 400 });
    if (x.type === "choice" && !(Array.isArray(x.criteria) ? x.criteria.length >= 2 : x.criteria && Object.keys(x.criteria).length >= 2)) throw Object.assign(new Error(`question ${id}: choice needs >= 2 criteria`), { status: 400 });
    if (x.type === "score" && !(Array.isArray(x.criteria) && x.criteria.length >= 2)) throw Object.assign(new Error(`question ${id}: score needs an ordered array of >= 2 levels`), { status: 400 });
  }
}

async function decide(body) {
  PRESETS = await loadPresets();
  const presetName = body.preset ?? DEFAULT_PRESET;
  const preset = PRESETS[presetName];
  if (!preset || preset.invalid) throw Object.assign(new Error(preset?.invalid ? `preset ${presetName} is invalid: ${preset.description}` : `unknown preset ${presetName}; have ${Object.keys(PRESETS).join(", ")}`), { status: 400 });
  const questions = body.questions ?? preset.questions;
  validateQuestions(questions);
  let state;
  if (body.state !== undefined) state = body.state;
  else if (typeof body.text === "string" && body.text.trim()) state = preset.state(body.text.trim());
  else throw Object.assign(new Error("provide text (string) or state (any JSON)"), { status: 400 });
  const opts = { calibration: body.calibration?.temperature_by_options ? { temperature_by_options: body.calibration.temperature_by_options, file: body.calibration.file ?? "request" } : await calibrationFor(presetName) };
  if (body.lane) {
    if (!router.lanes.has(body.lane)) throw Object.assign(new Error(`lane ${body.lane} not loaded; have ${[...router.lanes.keys()].join(", ")}`), { status: 400 });
    opts.lane = body.lane;
  }
  if (body.deadlineMs) opts.deadlineMs = Number(body.deadlineMs);
  const r = await router.decide(state, questions, opts);
  return { answers: r.answers, usage: r.usage, routing: r.routing, state, questions, preset: body.questions ? null : presetName, calibration: opts.calibration?.file ?? null };
}

const health = () => ({
  service: "laya",
  version: VERSION,
  status,
  error: loadError,
  pid: process.pid,
  port: Number(args.port),
  sidecar: args.sidecar,
  lanes: router ? [...router.lanes.keys()] : [],
  uptimeS: Math.round((Date.now() - started) / 1000),
  idleS: idleMs ? Math.round(idleMs / 1000) : 0,
  idleRemainingS: idleRemainingS(),
  inFlight,
  sampling: !!router?.sampling,
});

// ---- server ----------------------------------------------------------------------------------------------------------
const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (req.method === "OPTIONS" && args.cors) {
      res.writeHead(204, { "access-control-allow-origin": "*", "access-control-allow-headers": "content-type", "access-control-allow-methods": "GET, POST" });
      return res.end();
    }
    if (req.method === "GET" && url.pathname === "/") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      return res.end(PAGE);
    }
    if (req.method === "GET" && url.pathname === "/health") return json(res, 200, health());
    if (req.method === "GET" && url.pathname === "/presets") return json(res, 200, describePresets(await loadPresets()));
    if (req.method === "POST" && url.pathname === "/shutdown") {
      json(res, 200, { ok: true, pid: process.pid });
      setTimeout(() => shutdown("requested"), 20);
      return;
    }
    if (status !== "ready") return json(res, 503, { error: status === "loading" ? "loading" : `not available: ${status}${loadError ? ` (${loadError})` : ""}`, status, retryAfterMs: 500 });
    if (req.method === "POST" && url.pathname === "/touch") {
      touch();
      return json(res, 200, { ok: true, idleRemainingS: idleRemainingS() });
    }
    if (req.method === "GET" && url.pathname === "/stats") return json(res, 200, router.stats());
    if (req.method === "POST" && url.pathname === "/decide") {
      const raw = await readBody(req);
      let body;
      try {
        body = raw ? JSON.parse(raw) : {};
      } catch {
        return json(res, 400, { error: "invalid JSON body" });
      }
      inFlight++;
      touch();
      const t0 = performance.now();
      try {
        const out = await decide(body);
        log(`decide ${out.preset ?? "custom"} ${Object.keys(out.questions).length} q -> ${out.routing.lane} ${out.routing.ms.toFixed(0)} ms${out.routing.queueMs > 5 ? ` (+${out.routing.queueMs.toFixed(0)} queued)` : ""} (total ${(performance.now() - t0).toFixed(0)} ms)`);
        return json(res, 200, out);
      } finally {
        inFlight--;
        touch();
      }
    }
    return json(res, 404, { error: "not found" });
  } catch (e) {
    const statusCode = e.status ?? 500;
    if (statusCode === 500) log(`error: ${e.stack ?? e}`);
    return json(res, statusCode, { error: e.message ?? String(e) });
  }
});
server.keepAliveTimeout = 5_000;

server.on("error", (e) => {
  if (e.code === "EADDRINUSE") {
    log(`port ${args.port} is already in use (another instance is probably running); exiting with code 3`);
    process.exit(3);
  }
  log(`server error: ${e.stack ?? e}`);
  process.exit(1);
});

server.listen(Number(args.port), args.host, async () => {
  log(`listening on http://${args.host}:${args.port} (v${VERSION}${idleMs ? `, idle exit after ${Math.round(idleMs / 1000)} s` : ", no idle exit"}); loading lanes ${args.lanes} ...`);
  armTimers();
  try {
    router = await LayaRouter.create({ lanes: args.lanes.split(","), log: (m) => log(`  ${m}`) });
    PRESETS = await loadPresets();
    await router.warmup({ state: PRESETS[DEFAULT_PRESET].state("Please turn off the lights in the living room now"), sizes: [3, 5] });
    status = "ready";
    touch();
    log(`ready in ${((Date.now() - started) / 1000).toFixed(1)} s; lanes: ${[...router.lanes.keys()].join(", ")}; presets: ${Object.keys(PRESETS).join(", ")} (${PRESETS_DIR})`);
  } catch (e) {
    status = "failed";
    loadError = String(e?.message ?? e).split("\n")[0];
    log(`load failed: ${e?.stack ?? e}`);
    setTimeout(() => shutdown("load failed", 4), 1500); // leave /health readable for a moment
  }
});

// ---- browser page --------------------------------------------------------------------------------------------
const PAGE = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Laya - ask</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; font-family: system-ui, Segoe UI, sans-serif; }
  body { max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.4; }
  textarea { width: 100%; min-height: 4.5rem; font: inherit; padding: .5rem; box-sizing: border-box; }
  .row { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; margin: .5rem 0; }
  select, button { font: inherit; padding: .35rem .6rem; }
  button { cursor: pointer; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  td, th { text-align: left; padding: .35rem .5rem; vertical-align: top; border-bottom: 1px solid #8884; }
  .bar { display: inline-block; height: .7rem; background: #4a90e2; vertical-align: middle; border-radius: 2px; }
  .muted { opacity: .65; font-size: .9em; }
  .ans { font-weight: 600; }
  pre { background: #8881; padding: .75rem; overflow: auto; font-size: .85em; }
  details { margin-top: 1rem; }
</style></head>
<body>
<h2>Laya <span class="muted">typed decisions, one forward pass</span></h2>
<div class="row">
  <label>Preset <select id="preset"></select></label>
  <label>Lane <select id="lane"><option value="">auto</option></select></label>
  <span id="desc" class="muted"></span>
</div>
<textarea id="text" placeholder="Type a message, e.g. Turn off the living room lights">Turn off the living room lights</textarea>
<div class="row"><button id="ask">Ask (Ctrl+Enter)</button><span id="status" class="muted"></span></div>
<table id="out"></table>
<details><summary class="muted">questions sent / raw response</summary><pre id="raw"></pre></details>
<script>
const $ = (id) => document.getElementById(id);
let presets = {};
async function init() {
  presets = await (await fetch('/presets')).json();
  for (const [k, p] of Object.entries(presets)) { const o = document.createElement('option'); o.value = k; o.textContent = k; $('preset').append(o); }
  const h = await (await fetch('/health')).json();
  for (const l of h.lanes) { const o = document.createElement('option'); o.value = l; o.textContent = l; $('lane').append(o); }
  $('preset').onchange = () => { $('desc').textContent = presets[$('preset').value].description; };
  $('preset').onchange();
  if (h.status !== 'ready') { $('status').textContent = 'model ' + h.status + ' ...'; setTimeout(init, 1000); }
}
const pct = (p) => Math.round(p * 100) + '%';
const bar = (p) => '<span class="bar" style="width:' + Math.round(p * 120) + 'px"></span>';
function render(r) {
  const rows = [];
  for (const [id, a] of Object.entries(r.answers)) {
    if (a.type === 'choice') {
      const p = a.probabilities[a.choice];
      const others = Object.entries(a.probabilities).filter(([k]) => k !== a.choice).sort((x, y) => y[1] - x[1]).map(([k, v]) => k + ' ' + pct(v)).join(', ');
      rows.push('<tr><td>' + id + '</td><td class="ans">' + a.choice + '</td><td>' + pct(p) + ' ' + bar(p) + '</td><td class="muted">' + others + '</td></tr>');
    } else if (a.type === 'noul') {
      rows.push('<tr><td>' + id + '</td><td class="ans">' + (a.noul >= .5 ? 'yes' : 'no') + '</td><td>' + pct(a.noul >= .5 ? a.noul : 1 - a.noul) + ' ' + bar(a.noul) + '</td><td class="muted">P(true) = ' + a.noul.toFixed(3) + '</td></tr>');
    } else {
      const levels = Object.values(a.legend), max = levels.length - 1;
      const dist = Object.entries(a.probabilities).map(([i, v]) => levels[i] + ' ' + pct(v)).join(', ');
      rows.push('<tr><td>' + id + '</td><td class="ans">' + levels[Math.min(max, Math.max(0, Math.round(a.score)))] + '</td><td>' + a.score.toFixed(2) + ' / ' + max + ' ' + bar(a.score / max) + '</td><td class="muted">' + dist + '</td></tr>');
    }
  }
  $('out').innerHTML = '<tr><th>question</th><th>answer</th><th>probability</th><th>distribution</th></tr>' + rows.join('');
  $('status').textContent = r.routing.lane + ' ' + r.routing.ms.toFixed(0) + ' ms (predicted ' + r.routing.predictedMs.toFixed(0) + ', GPU ' + r.routing.gpuState + '), ' + r.usage.input_tokens + ' tokens';
  $('raw').textContent = JSON.stringify(r, null, 2);
}
async function ask() {
  $('status').textContent = 'thinking ...';
  const body = { text: $('text').value, preset: $('preset').value };
  if ($('lane').value) body.lane = $('lane').value;
  const res = await fetch('/decide', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const r = await res.json();
  if (!res.ok) { $('status').textContent = 'error: ' + r.error; return; }
  render(r);
}
$('ask').onclick = ask;
$('text').addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.ctrlKey) ask(); });
init();
</script>
</body></html>`;
