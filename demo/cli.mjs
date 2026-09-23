#!/usr/bin/env node
/**
 * demo/cli.mjs - decide from the command line, one input or a whole file, through demo/laya.mjs.
 *
 *   node demo/cli.mjs "Turn off the living room lights"                       # smart-home preset, shared sidecar
 *   node demo/cli.mjs --preset triage "Charged twice. Refund today or I cancel."
 *   node demo/cli.mjs --preset triage --file demo/data/tickets.json          # every item, one line each + timing
 *   node demo/cli.mjs --preset guard --file demo/data/prompts.json --json    # full JSON array on stdout
 *   node demo/cli.mjs --state '{"email":"..."}' --questions demo/data/email-questions.json
 *   node demo/cli.mjs --mode local --lanes cpu:8 "..."                        # no background process, CPU only
 *   node demo/cli.mjs --lane webgpu:fp16 "..."                                # force a lane for this call
 *   node demo/cli.mjs --health | --stats | --presets
 *
 * Modes: --mode auto (default: sidecar, spawned on demand, fall back to local) | sidecar | local | http --url <base>.
 * Per-call overrides: --lane <lane>, --policy auto|prefer-gpu|prefer-cpu|min-cpu, --deadline <ms>, --no-graph
 * (cuda lane: skip CUDA-graph replay). Without them the router picks; `routing` in the output says what it did.
 * Exit codes: 0 ok, 1 error (stderr), 2 bad arguments / unknown preset.
 */
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { createLaya, readInputs, summarize, top } from "./laya.mjs";

const { values: a, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    preset: { type: "string" },
    questions: { type: "string" },
    state: { type: "string" },
    file: { type: "string" },
    mode: { type: "string", default: "auto" },
    url: { type: "string" },
    port: { type: "string" },
    lanes: { type: "string" },
    lane: { type: "string" },
    policy: { type: "string" },
    deadline: { type: "string" },
    "no-graph": { type: "boolean", default: false },
    concurrency: { type: "string", default: "4" },
    json: { type: "boolean", default: false },
    quiet: { type: "boolean", default: false },
    health: { type: "boolean", default: false },
    stats: { type: "boolean", default: false },
    presets: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
});
if (a.help) {
  process.stdout.write(readFileSync(new URL(import.meta.url), "utf8").split("*/")[0].replace(/^#!.*\n|^\/\*\*\n|^ \* ?/gm, "") + "\n");
  process.exit(0);
}
const say = (m) => !a.quiet && process.stderr.write(`${m}\n`);
const jsonArg = (v, what) => {
  try {
    return v.trim().startsWith("{") || v.trim().startsWith("[") ? JSON.parse(v) : JSON.parse(readFileSync(v, "utf8"));
  } catch (e) {
    say(`error: ${what}: ${e.message}`);
    process.exit(2);
  }
};

const text = positionals.join(" ").trim();
if (!a.health && !a.stats && !a.presets && !text && !a.state && !a.file) {
  say("nothing to decide: give a text, --state, or --file (or --health / --stats / --presets); -h for help");
  process.exit(2);
}
const opts = {};
if (a.preset) opts.preset = a.preset;
if (a.questions) opts.questions = jsonArg(a.questions, "--questions");
if (a.lane) opts.lane = a.lane;
if (a.policy) opts.policy = a.policy;
if (a.deadline) opts.deadlineMs = Number(a.deadline);
if (a["no-graph"]) opts.exec = { graph: false };

let laya;
try {
  laya = await createLaya({ mode: a.mode, url: a.url, port: a.port ? Number(a.port) : undefined, lanes: a.lanes, log: say });
} catch (e) {
  say(`error: ${e.message}`);
  process.exit(1);
}
say(`mode ${laya.mode}${laya.info.lanes ? `; lanes ${laya.info.lanes.join(", ")}` : ""}`);

try {
  if (a.health || a.stats || a.presets) {
    const out = a.health ? await laya.health() : a.stats ? await laya.stats() : await laya.presets();
    process.stdout.write(JSON.stringify(out, null, 2) + "\n");
  } else if (a.file) {
    const inputs = readInputs(a.file).map((it) => (typeof it === "string" ? it : it.state !== undefined ? { state: it.state } : it.text));
    const t0 = performance.now();
    const results = await laya.decideMany(inputs, opts, { concurrency: Number(a.concurrency) });
    const wall = performance.now() - t0;
    if (a.json) process.stdout.write(JSON.stringify(results, null, 2) + "\n");
    else {
      const w = Math.min(60, Math.max(...inputs.map((i) => (typeof i === "string" ? i : JSON.stringify(i.state)).length)));
      for (let i = 0; i < inputs.length; i++) {
        const shown = typeof inputs[i] === "string" ? inputs[i] : JSON.stringify(inputs[i].state);
        process.stdout.write(`${(shown.length > w ? shown.slice(0, w - 3) + "..." : shown).padEnd(w)}  ${summarize(results[i].answers)}  [${results[i].routing.lane} ${results[i].routing.ms.toFixed(0)} ms]\n`);
      }
    }
    const lanes = [...new Set(results.map((r) => r.routing.lane))];
    say(`${inputs.length} decisions in ${wall.toFixed(0)} ms (${(wall / inputs.length).toFixed(1)} ms each end to end, inference p50 ${results.map((r) => r.routing.ms).sort((x, y) => x - y)[Math.floor(results.length / 2)].toFixed(0)} ms; lanes ${lanes.join(", ")})`);
  } else {
    const input = a.state ? { state: jsonArg(a.state, "--state") } : text;
    const t0 = performance.now();
    const r = await laya.decide(input, opts);
    const wall = performance.now() - t0;
    if (a.json) process.stdout.write(JSON.stringify(r, null, 2) + "\n");
    else {
      const w = Math.max(...Object.keys(r.answers).map((k) => k.length));
      for (const [id, ans] of Object.entries(r.answers)) {
        const t = top(ans);
        const dist = ans.type === "noul" ? `P(true) ${ans.noul.toFixed(3)}` : Object.entries(ans.probabilities).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${ans.type === "score" ? ans.legend[k] : k} ${(100 * v).toFixed(0)}%`).join(", ");
        process.stdout.write(`  ${id.padEnd(w)}  ${t.label.padEnd(18)} ${(100 * t.p).toFixed(0).padStart(3)}%   ${dist}\n`);
      }
      say(`${r.routing.lane} ${r.routing.ms.toFixed(0)} ms${r.routing.queueMs > 5 ? ` (+${r.routing.queueMs.toFixed(0)} queued)` : ""}, ${wall.toFixed(0)} ms end to end, ${r.usage.input_tokens} tokens${r.preset ? `, preset ${r.preset}` : ""}${r.calibration ? ` (${r.calibration})` : ""}`);
    }
  }
} catch (e) {
  say(`error: ${e.message}`);
  await laya.close().catch(() => {});
  process.exit(e.code === "BAD_REQUEST" || e.status === 400 ? 2 : 1);
}
await laya.close();
process.exit(0);
