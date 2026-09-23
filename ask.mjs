#!/usr/bin/env node
/**
 * Ask Laya things - one-shot from the command line or interactively.
 *
 * Laya does not generate text. You give it a STATE (your message, a ticket, JSON ...) and typed QUESTIONS;
 * it answers every question in one forward pass with probabilities. Presets bundle a question set with a
 * wrapper that turns your text into a state (built-ins in data/presets.mjs, your own in presets/<name>.json).
 *
 *   node ask.mjs "Turn off the living room lights"                     # smart-home preset, one shot (loads in-process, ~2 s)
 *   node ask.mjs --preset triage "Charged twice. Refund today or I cancel."
 *   node ask.mjs --sidecar "Lock the front door"                       # use the shared background instance (spawned on
 *                                                                      # first use, exits after 5 min idle); later calls ~0.3 s
 *   node ask.mjs --state state.json --questions questions.json         # your own state + questions
 *   node ask.mjs --json "..."                                          # full JSON (answers, usage, routing)
 *   node ask.mjs                                                       # interactive REPL (/help)
 *   node ask.mjs --sidecar                                             # REPL over the sidecar (kept alive while open)
 *
 * Where the model runs:
 *   default / --local     load the model in this process, answer, exit. Simple; N concurrent callers = N copies.
 *   --sidecar             talk to serve.mjs on 127.0.0.1:$PORT; spawn it detached if it is not running and wait
 *                         until it is ready. One shared instance for every caller; it exits by itself after
 *                         --idle without requests. Set LAYA_SIDECAR=1 to make this the default for a shell/orchestration.
 *   --start [--idle 10m] [--lanes ...]   make sure the sidecar is running, print port/pid, exit (for programs that call HTTP directly)
 *   --status              show whether a sidecar is running (pid, lanes, idle countdown)
 *   --stop                stop the sidecar gracefully
 *
 * Options:
 *   --preset <name>       smart-home (default) | triage | guard | moderation | route | sentiment | presets/*.json
 *   --questions <file|json>  JSON question set, replaces the preset's questions
 *   --state <file|json>   JSON state, replaces the preset's text wrapper (text arguments are ignored)
 *   --lanes a,b           lanes to load (default cuda:fp16,webgpu:fp16 one-shot, + cpu:8 for the REPL / sidecar; a lane that cannot load is dropped)
 *   --lane <lane>         force a lane for every call (default: router decides)
 *   --calibration <file>  temperature table for the active preset; default: calibration/<preset>.json if it exists
 *   --idle <dur>          sidecar idle exit, used only when this call spawns it (default $LAYA_IDLE or 5m; 0 = never)
 *   --max-age <dur>       sidecar recycles itself after this long (default $LAYA_MAX_AGE or never); also only when spawning
 *   --port <n>            sidecar port (default $LAYA_PORT or 8787)
 *   --json                machine-readable output
 *   --no-color
 *
 * Your own domain: put presets/<name>.json next to the built-ins (see data/presets.mjs for the format, or build
 * the questions in the REPL and /save <name>), then `node ask.mjs --preset <name> "..."`. Calibrate with
 * `node calibrate.mjs --preset <name> --eval presets/<name>.eval.json` once you have labelled examples.
 */
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin, stdout, stderr } from "node:process";
import { parseArgs } from "node:util";
import { formatAnswers, formatHeader, colors as c } from "./src/format.mjs";
import { loadPresets, presetToJson, savePreset, PRESETS_DIR, DEFAULT_PRESET } from "./data/presets.mjs";
import * as sidecar from "./src/sidecar-client.mjs";

const { values: args, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    preset: { type: "string", default: DEFAULT_PRESET },
    questions: { type: "string" },
    state: { type: "string" },
    lanes: { type: "string" },
    lane: { type: "string" },
    calibration: { type: "string" },
    sidecar: { type: "boolean", default: false },
    local: { type: "boolean", default: false },
    idle: { type: "string" },
    "max-age": { type: "string" },
    port: { type: "string" },
    start: { type: "boolean", default: false },
    status: { type: "boolean", default: false },
    stop: { type: "boolean", default: false },
    json: { type: "boolean", default: false },
    "no-color": { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
});
if (args.help) {
  stdout.write((await readFile(new URL(import.meta.url), "utf8")).split("*/")[0].replace(/^#!.*\n|^\/\*\*\n|^ \* ?/gm, "") + "\n");
  process.exit(0);
}
const color = !args["no-color"] && stdout.isTTY;
const say = (s) => stderr.write(`${s}\n`);
const dim = (s) => (color ? c.dim(s) : s);
const text = positionals.join(" ").trim();
const port = Number(args.port ?? sidecar.DEFAULT_PORT);
const idle = args.idle ?? sidecar.DEFAULT_IDLE;
const maxAge = args["max-age"] ?? process.env.LAYA_MAX_AGE;
const envSidecar = /^(1|true|yes|on)$/i.test(process.env.LAYA_SIDECAR ?? "");
const useSidecar = !args.local && (args.sidecar || envSidecar);
const interactive = !text && !args.state && !args.start && !args.status && !args.stop;

// ---- lifecycle commands ----------------------------------------------------------------------------------------
if (args.status) {
  const d = await sidecar.discover({ port });
  if (d.state === "none") stdout.write(`no sidecar on 127.0.0.1:${port}\n`);
  else if (d.state === "foreign") stdout.write(`port ${port} is used by something else (not the Laya sidecar)\n`);
  else {
    const h = d.health;
    stdout.write(`sidecar ${h.status} on 127.0.0.1:${port}  pid ${h.pid}  v${h.version}  up ${h.uptimeS} s\n  lanes: ${h.lanes.join(", ") || "-"}${h.lanesLoading?.length ? `  (loading: ${h.lanesLoading.join(", ")})` : ""}\n  idle exit: ${h.idleS ? `${h.idleS} s (${h.idleRemainingS} s remaining)` : "never"}${h.maxAgeS ? `  max age: ${h.maxAgeS} s` : ""}  in flight: ${h.inFlight}  sampling: ${h.sampling ? "on" : "paused"}\n  log: ${sidecar.logFileFor(port)}\n`);
  }
  process.exit(0);
}
if (args.stop) {
  try {
    const stopped = await sidecar.stop({ port });
    stdout.write(stopped ? `sidecar on :${port} stopped\n` : `no sidecar on :${port}\n`);
    process.exit(0);
  } catch (e) {
    say(`${c.red("error:")} ${e.message}`);
    process.exit(1);
  }
}
if (args.start) {
  try {
    const { health, spawned } = await sidecar.ensureSidecar({ port, idle, maxAge, lanes: args.lanes, calibration: args.calibration, onProgress: progress });
    // --start is for programs that will call HTTP directly: report the full lane set, not just the first one serving
    const h = health.lanesLoading?.length ? await sidecar.waitAllLanes({ port }) : health;
    if (!spawned && args.lanes && h.lanes.join(",") !== args.lanes) say(dim(`note: sidecar already running with lanes ${h.lanes.join(",")}; --lanes ignored`));
    stdout.write(`${JSON.stringify({ url: `http://127.0.0.1:${port}`, pid: h.pid, lanes: h.lanes, idleS: h.idleS, spawned })}\n`);
    process.exit(0);
  } catch (e) {
    say(`${c.red("error:")} ${e.message}`);
    process.exit(1);
  }
}

function progress(state, ms, health) {
  const t = `${(ms / 1000).toFixed(1)} s`;
  if (state === "spawning") say(dim(`no sidecar on :${port}; starting one (idle exit after ${idle === "0" ? "never" : idle}) ...`));
  else if (state === "attaching") say(dim(`sidecar on :${port} is loading (pid ${health?.pid}); waiting ...`));
  else if (state === "loading") say(dim(`  sidecar loading (pid ${health?.pid}) ...`));
  else if (state === "ready") say(dim(`  sidecar ready in ${t} (pid ${health?.pid}, lanes ${health?.lanes?.join(", ")}${health?.lanesLoading?.length ? `; still loading ${health.lanesLoading.join(", ")}` : ""})`));
}

// ---- presets + calibration -------------------------------------------------------------------------------------
let PRESETS = await loadPresets();
if (!PRESETS[args.preset] || PRESETS[args.preset].invalid) {
  say(`unknown or invalid preset "${args.preset}"; available: ${Object.keys(PRESETS).join(", ")} (files in ${PRESETS_DIR})`);
  if (PRESETS[args.preset]?.invalid) say(PRESETS[args.preset].description);
  process.exit(2);
}

// calibration table per preset: --calibration for the initial preset, else calibration/<preset>.json if present
const calibrationCache = new Map();
async function calibrationFor(name) {
  if (calibrationCache.has(name)) return calibrationCache.get(name);
  const rel = name === args.preset && args.calibration ? args.calibration : `calibration/${name}.json`;
  const file = path.resolve(sidecar.PROJECT_ROOT, rel);
  let table = null;
  try {
    await access(file);
    const t = JSON.parse(await readFile(file, "utf8"));
    table = { temperature_by_options: t.temperature_by_options, file: rel.replace(/\\/g, "/") };
  } catch {
    if (name === args.preset && args.calibration) throw new Error(`calibration file not found: ${file}`);
  }
  calibrationCache.set(name, table);
  return table;
}

// ---- session state ---------------------------------------------------------------------------------------------
let presetName = args.preset;
let preset = PRESETS[presetName];
/** `--state` / `--questions` take inline JSON or a file path. */
const jsonArg = async (v) => (v.trim().startsWith("{") || v.trim().startsWith("[") ? JSON.parse(v) : JSON.parse(await readFile(v, "utf8")));
let questions = args.questions ? await jsonArg(args.questions) : { ...preset.questions };
let extraFields = {};
let fixedState = args.state ? await jsonArg(args.state) : null;
let lastState = null;
let lastText = "";
let jsonOut = args.json;
let forcedLane = args.lane;
let calibration = await calibrationFor(presetName);

// ---- backends --------------------------------------------------------------------------------------------------
async function localBackend() {
  const lanes = (args.lanes ?? (interactive ? "cuda:fp16,webgpu:fp16,cpu:8" : "cuda:fp16,webgpu:fp16")).split(",");
  const t0 = performance.now();
  say(dim(`loading ${lanes.join(" + ")} in this process ...`));
  const { LayaRouter } = await import("./src/ep-router.mjs"); // lazy: pulls in onnxruntime
  // one-shot: answer from whichever lane is ready first, no background sampling; REPL: full router + GPU keep-alive (sporadic calls)
  const router = await LayaRouter.create({ lanes, waitFor: interactive ? "all" : "first", sampleLoad: interactive, gpuKeepAliveMs: interactive ? 30_000 : 0, log: (m) => say(dim(`  ${m}`)) });
  if (interactive) await router.warmup({ state: preset.state("Please turn off the lights in the living room now"), sizes: [Object.keys(questions).length] });
  say(dim(`ready in ${((performance.now() - t0) / 1000).toFixed(1)} s; lanes: ${[...router.lanes.keys()].join(", ")}`));
  return {
    kind: "local",
    via: "in-process",
    lanes: [...router.lanes.keys()],
    decide: (state, qs, o) => router.decide(state, qs, o),
    stats: async () => router.stats(),
    close: () => router.close(),
  };
}

async function remoteBackend() {
  const { health, spawned } = await sidecar.ensureSidecar({ port, idle, maxAge, lanes: args.lanes, calibration: args.calibration, onProgress: progress });
  if (!spawned && args.lanes && health.lanes.join(",") !== args.lanes) say(dim(`note: sidecar already running with lanes ${health.lanes.join(",")}; --lanes ignored`));
  let keepAlive = null;
  if (interactive && health.idleS > 0) {
    // an open REPL counts as "in use": ping at half the idle period so the sidecar does not exit under us
    keepAlive = setInterval(() => sidecar.touch({ port }).catch(() => {}), Math.max(5_000, (health.idleS * 1000) / 2));
    keepAlive.unref();
  }
  return {
    kind: "remote",
    via: `sidecar :${port} pid ${health.pid}`,
    lanes: health.lanes,
    // lanes still loading when we attached join later: re-read /health on demand
    refreshLanes: async function () {
      const d = await sidecar.status({ port });
      if (d.health?.lanes) this.lanes = d.health.lanes;
      return this.lanes;
    },
    decide: async (state, qs, o) => {
      const r = await sidecar.decide({ state, questions: qs, preset: presetName, lane: o.lane, calibration: o.calibration ?? undefined }, { port });
      return { answers: r.answers, usage: r.usage, routing: r.routing };
    },
    stats: () => sidecar.stats({ port }),
    close: async () => {
      if (keepAlive) clearInterval(keepAlive);
    },
  };
}

let backend;
if (useSidecar) {
  try {
    backend = await remoteBackend();
  } catch (e) {
    say(`${c.yellow("warning:")} sidecar unavailable (${e.code ?? "error"}: ${String(e.message).split("\n")[0]}); falling back to in-process`);
    if (e.code !== "FOREIGN_PORT") say(dim(`  log: ${sidecar.logFileFor(port)}`));
    backend = await localBackend();
  }
} else {
  backend = await localBackend();
}
say(dim(`preset ${presetName}${calibration ? ` (calibration ${calibration.file})` : " (shipped temperatures)"}; ${backend.via}`));

async function ask(state, shownText) {
  const result = await backend.decide(state, questions, { ...(forcedLane ? { lane: forcedLane } : {}), calibration });
  lastState = state;
  lastText = shownText;
  if (jsonOut) {
    stdout.write(JSON.stringify({ state, answers: result.answers, usage: result.usage, routing: result.routing, backend: backend.kind }, null, 2) + "\n");
  } else {
    stdout.write(`${formatHeader(shownText, result, { color, via: backend.kind === "remote" ? "sidecar" : "" })}\n${formatAnswers(result.answers, questions, { color })}\n`);
  }
  return result;
}

// ---- one-shot --------------------------------------------------------------------------------------------------
if (!interactive) {
  try {
    await ask(fixedState ?? preset.state(text), fixedState ? JSON.stringify(fixedState) : text);
  } catch (e) {
    say(`${c.red("error:")} ${e.message}`);
    await backend.close();
    process.exit(1);
  }
  await backend.close();
  process.exit(0);
}

// ---- REPL ------------------------------------------------------------------------------------------------------
const HELP = `
Type a message and press Enter: it becomes the state, the current questions are answered.
Commands:
  /preset <name>            switch question set + state wrapper (${Object.keys(PRESETS).join(", ")})
  /presets                  list presets (built-in and presets/*.json|*.mjs)
  /save <name> [description]  save the current questions + state wrapper as presets/<name>.json
  /show                     print the current questions as JSON
  /noul <question>          add a yes/no question
  /choice <question> | a | b: description | c    add a pick-one question (options after '|')
  /score <question> | low | mid | high           add an ordered-scale question (levels after '|')
  /drop <id>                remove a question        /clear   back to the preset's questions
  /set key=value            add a field to the state wrapper (e.g. /set livingRoomLights=on)
  /state {json}             ask about an explicit JSON state
  /again                    re-ask the last state with the current questions
  /lane auto|cpu:8|webgpu:fp16   force a lane (loaded: ${backend.lanes.join(", ")})
  /json                     toggle JSON output     /stats   router statistics     /exit
`.trim();

const slug = (s) => {
  const base = s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").split("_").slice(0, 3).join("_") || "q";
  let id = base;
  for (let i = 2; questions[id]; i++) id = `${base}${i}`;
  return id;
};
const parseOptions = (rest) => rest.split("|").map((s) => s.trim()).filter(Boolean);

async function command(line) {
  const [cmd, ...restArr] = line.slice(1).split(" ");
  const rest = restArr.join(" ").trim();
  switch (cmd) {
    case "help":
      stdout.write(HELP + "\n");
      break;
    case "presets":
      PRESETS = await loadPresets();
      for (const [k, p] of Object.entries(PRESETS)) {
        const cal = await calibrationFor(k);
        stdout.write(`  ${k.padEnd(14)} ${p.source === "file" ? "file    " : "built-in"}  ${cal ? "calibrated" : "shipped T "}  ${p.description}\n`);
      }
      stdout.write(`  (add your own: ${PRESETS_DIR}\\<name>.json - see data/presets.mjs, or /save)\n`);
      break;
    case "preset":
      PRESETS = await loadPresets();
      if (!PRESETS[rest] || PRESETS[rest].invalid) return stdout.write(`${PRESETS[rest]?.description ?? "unknown preset"}; try: ${Object.keys(PRESETS).join(", ")}\n`);
      presetName = rest;
      preset = PRESETS[rest];
      questions = { ...preset.questions };
      extraFields = {};
      calibration = await calibrationFor(rest);
      stdout.write(`preset ${rest}: ${preset.description}\n  questions: ${Object.keys(questions).join(", ")}\n  state: ${JSON.stringify(preset.template ?? preset.state("$TEXT"))}\n  temperatures: ${calibration ? calibration.file : "shipped (no calibration/" + rest + ".json)"}\n`);
      break;
    case "save": {
      const [name, ...descArr] = rest.split(" ");
      if (!name) return stdout.write("usage: /save <name> [description]\n");
      try {
        const file = await savePreset(name, presetToJson(preset, questions, extraFields, descArr.join(" ") || undefined));
        stdout.write(`saved ${file}\n  use it with: node ask.mjs --preset ${name} "..."   or   /preset ${name}\n  calibrate with: node calibrate.mjs --preset ${name} --eval presets/${name}.eval.json\n`);
        PRESETS = await loadPresets();
      } catch (e) {
        stdout.write(`could not save: ${e.message}\n`);
      }
      break;
    }
    case "show":
      stdout.write(JSON.stringify(questions, null, 2) + "\n");
      break;
    case "noul": {
      if (!rest) return stdout.write("usage: /noul <question>\n");
      const id = slug(rest);
      questions[id] = { type: "noul", instructions: rest };
      stdout.write(`added ${id} (noul)\n`);
      break;
    }
    case "choice":
    case "score": {
      const [ins, ...opts] = parseOptions(rest);
      if (!ins || opts.length < 2) return stdout.write(`usage: /${cmd} <question> | option | option ...  (at least 2 options)\n`);
      const id = slug(ins);
      if (cmd === "choice") {
        const criteria = {};
        for (const o of opts) {
          const [k, ...d] = o.split(":");
          criteria[k.trim().replace(/\s+/g, "_")] = d.length ? d.join(":").trim() : null;
        }
        questions[id] = { type: "choice", instructions: ins, criteria };
      } else {
        questions[id] = { type: "score", instructions: ins, criteria: opts };
      }
      stdout.write(`added ${id} (${cmd}, ${opts.length} options)\n`);
      break;
    }
    case "drop":
      if (!questions[rest]) return stdout.write(`no question "${rest}"; have ${Object.keys(questions).join(", ")}\n`);
      delete questions[rest];
      stdout.write(`dropped ${rest}\n`);
      break;
    case "clear":
      questions = { ...preset.questions };
      extraFields = {};
      stdout.write(`back to preset ${presetName}: ${Object.keys(questions).join(", ")}\n`);
      break;
    case "set": {
      const m = rest.match(/^([^=]+)=(.*)$/);
      if (!m) return stdout.write("usage: /set key=value\n");
      extraFields[m[1].trim()] = m[2].trim();
      stdout.write(`state wrapper now adds ${JSON.stringify(extraFields)}\n`);
      break;
    }
    case "state": {
      let st;
      try {
        st = JSON.parse(rest);
      } catch {
        return stdout.write('usage: /state {"key": "value", ...}\n');
      }
      await ask(st, JSON.stringify(st));
      break;
    }
    case "again":
      if (!lastState) return stdout.write("nothing asked yet\n");
      await ask(lastState, lastText);
      break;
    case "lane":
      if (rest === "auto" || !rest) forcedLane = undefined;
      else if (!backend.lanes.includes(rest) && !(await backend.refreshLanes?.())?.includes(rest)) return stdout.write(`lane not loaded; have ${backend.lanes.join(", ")}\n`);
      else forcedLane = rest;
      stdout.write(`lane: ${forcedLane ?? "auto (router decides)"}\n`);
      break;
    case "json":
      jsonOut = !jsonOut;
      stdout.write(`json output ${jsonOut ? "on" : "off"}\n`);
      break;
    case "stats": {
      const s = await backend.stats();
      for (const [lane, L] of Object.entries(s.lanes)) stdout.write(`  ${lane.padEnd(12)} ${L.mode ?? ""} calls=${L.calls} pending=${L.pending ?? 0} healthy=${L.healthy}${L.dead ? " GONE" : ""}  ${Object.entries(L.ema).map(([k, v]) => `${k}: ${v.ms.toFixed(0)} ms`).join("  ")}\n`);
      if (s.loading?.length) stdout.write(`  loading: ${s.loading.join(", ")}\n`);
      stdout.write(`  queue: ${s.queue?.pending ?? 0} pending (~${(s.queue?.waitMs ?? 0).toFixed(0)} ms)  others: CPU ${(s.load.cpuOthers * 100).toFixed(0)}%  GPU ${(s.load.gpuOthersUtil * 100).toFixed(0)}%  GPU clock ${s.load.gpuSmClockMHz ?? "-"} MHz  GPU state ${s.gpuState}  sampling ${s.sampling ? "on" : "paused"}  (${backend.via})\n`);
      break;
    }
    case "exit":
    case "quit":
    case "q":
      return "exit";
    default:
      stdout.write(`unknown command /${cmd} - /help\n`);
  }
}

stdout.write(`${c.bold("Laya")} ${dim(`- preset ${presetName} (${Object.keys(questions).join(", ")}). Type a message, or /help. Ctrl+C to quit.`)}\n`);
const rl = createInterface({ input: stdin, output: stdout, terminal: stdin.isTTY, prompt: color ? "\x1b[36m> \x1b[0m" : "> " });
rl.on("SIGINT", () => rl.close());
rl.prompt();
// The async iterator buffers lines that arrive while an ask is in flight (rl.question() would drop them),
// so piping a script into ask.mjs works as well as typing.
for await (const raw of rl) {
  const line = raw.trim();
  if (line) {
    if (line.startsWith("/")) {
      if ((await command(line)) === "exit") break;
    } else {
      try {
        await ask({ ...preset.state(line), ...extraFields }, line);
      } catch (e) {
        stdout.write(`${c.red("error:")} ${e.message}\n`);
      }
    }
  }
  if (!rl.closed) rl.prompt();
}
if (!rl.closed) rl.close();
await backend.close();
process.exit(0);
