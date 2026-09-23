#!/usr/bin/env node
/**
 * laya.mjs - call Laya from anywhere (the entry point of the laya-decisions skill).
 *   1. finds the LocalLaya project ($LAYA_DIR, or three levels up from this script when the skill lives in
 *      <project>/skills/laya-decisions/), and
 *   2. answers through the shared sidecar with machine-readable JSON on stdout.
 *
 *   node scripts/laya.mjs --preset triage "Charged twice, refund me today"
 *   node scripts/laya.mjs --state '{"text":"..."}' --questions q.json          # own state + questions (file or inline JSON)
 *   node scripts/laya.mjs --local "..."         # in-process instead of the sidecar (no background process)
 *   node scripts/laya.mjs --status | --stop | --start
 *   node scripts/laya.mjs --pretty "..."        # human-readable instead of JSON
 *
 * Fast path (the default for a one-shot decision): this process itself talks HTTP to the sidecar through the
 * project's src/sidecar-client.mjs (no onnxruntime import, no second Node process) - ~80 ms per call on the
 * reference machine of which ~12 ms is the inference, vs ~130 ms when going through ask.mjs. Everything else
 * (lifecycle flags, --local, --pretty, the REPL, and any failure to reach or start the sidecar) is delegated
 * to ask.mjs, which has the full behaviour incl. the fall-back to an in-process model.
 *
 * Output on the fast path = ask.mjs --json: { state, answers, usage, routing, backend: "remote" }.
 * Exit codes are ask.mjs's: 0 ok, 1 error (stderr), 2 unknown / invalid preset or bad question set.
 */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const here = path.dirname(fileURLToPath(import.meta.url));
const candidates = [process.env.LAYA_DIR, path.resolve(here, "..", "..", "..")].filter(Boolean);
const projectDir = candidates.find((d) => existsSync(path.join(d, "ask.mjs")) && existsSync(path.join(d, "serve.mjs")));
if (!projectDir) {
  console.error(`laya.mjs: cannot find the LocalLaya project (looked at ${candidates.join(", ")}). Set LAYA_DIR to the directory that contains ask.mjs.`);
  process.exit(2);
}

const argv = process.argv.slice(2);
let parsed;
try {
  parsed = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false, // unknown flags are passed through to ask.mjs untouched
    options: {
      preset: { type: "string" },
      questions: { type: "string" },
      state: { type: "string" },
      lane: { type: "string" },
      lanes: { type: "string" },
      calibration: { type: "string" },
      port: { type: "string" },
      idle: { type: "string" },
      "max-age": { type: "string" },
      pretty: { type: "boolean" },
      json: { type: "boolean" },
      local: { type: "boolean" },
      sidecar: { type: "boolean" },
      start: { type: "boolean" },
      status: { type: "boolean" },
      stop: { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });
} catch (e) {
  console.error(`laya.mjs: ${e.message}`);
  process.exit(1);
}
const { values: a, positionals } = parsed;
const text = positionals.join(" ").trim();
const lifecycle = a.status || a.stop || a.start || a.help;
const oneShot = !lifecycle && !a.local && !a.pretty && (text || a.state);

/** Everything the fast path does not cover goes to ask.mjs in the project directory (same exit codes). */
function delegate(extra = []) {
  const passthrough = argv.filter((x) => x !== "--pretty");
  const finalArgs = ["ask.mjs", ...passthrough, ...extra];
  if (!lifecycle) {
    if (!a.local && !a.sidecar && !extra.includes("--local")) finalArgs.push("--sidecar");
    // one-shot answers as JSON unless --pretty; the REPL (no text / state) stays human-readable
    if (!a.pretty && !a.json && (text || a.state)) finalArgs.push("--json");
  }
  const child = spawn(process.execPath, finalArgs, { cwd: projectDir, stdio: "inherit", windowsHide: true });
  child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
  child.on("error", (e) => {
    console.error(`laya.mjs: failed to start node: ${e.message}`);
    process.exit(1);
  });
}

if (!oneShot) {
  delegate();
} else {
  const sidecar = await import(pathToFileURL(path.join(projectDir, "src", "sidecar-client.mjs")).href);
  const port = Number(a.port ?? sidecar.DEFAULT_PORT);
  const idle = a.idle ?? sidecar.DEFAULT_IDLE;
  const maxAge = a["max-age"] ?? process.env.LAYA_MAX_AGE;
  const jsonArg = (v, what) => {
    try {
      return v.trim().startsWith("{") || v.trim().startsWith("[") ? JSON.parse(v) : JSON.parse(readFileSync(path.resolve(v), "utf8"));
    } catch (e) {
      console.error(`error: ${what}: ${e.message}`);
      process.exit(2);
    }
  };
  const body = {};
  if (a.preset) body.preset = a.preset;
  if (a.state) body.state = jsonArg(a.state, "--state");
  else body.text = text;
  if (a.questions) body.questions = jsonArg(a.questions, "--questions");
  if (a.lane) body.lane = a.lane;
  if (a.calibration) {
    const t = jsonArg(a.calibration, "--calibration");
    body.calibration = { temperature_by_options: t.temperature_by_options, file: a.calibration.replace(/\\/g, "/") };
  }
  const progress = (state, ms, health) => {
    if (state === "spawning") console.error(`no sidecar on :${port}; starting one (idle exit after ${idle === "0" ? "never" : idle}) ...`);
    else if (state === "attaching") console.error(`sidecar on :${port} is loading (pid ${health?.pid}); waiting ...`);
    else if (state === "ready") console.error(`  sidecar ready in ${(ms / 1000).toFixed(1)} s (pid ${health?.pid}, lanes ${health?.lanes?.join(", ")})`);
  };
  try {
    await sidecar.ensureSidecar({ port, idle, maxAge, lanes: a.lanes, onProgress: progress });
    const r = await sidecar.decide(body, { port });
    process.stdout.write(JSON.stringify({ state: r.state, answers: r.answers, usage: r.usage, routing: r.routing, backend: "remote" }, null, 2) + "\n");
    process.exit(0);
  } catch (e) {
    if (e?.code === "BAD_REQUEST") {
      // the server validated the request: unknown preset / invalid questions / missing text -> ask.mjs's exit codes
      console.error(`error: ${e.message}`);
      process.exit(/preset/.test(e.message) || /question/.test(e.message) ? 2 : 1);
    }
    if (e?.code === "SERVER_ERROR") {
      console.error(`error: ${e.message}`);
      process.exit(1);
    }
    // cannot reach / start / use the sidecar (FOREIGN_PORT, SPAWN_FAILED, LOAD_FAILED, TIMEOUT, ECONNREFUSED ...):
    // let ask.mjs handle it - it explains the situation and falls back to an in-process model
    console.error(`warning: sidecar unavailable (${e?.code ?? e?.name ?? "error"}: ${String(e?.message ?? e).split("\n")[0]}); handing over to ask.mjs`);
    delegate();
  }
}
