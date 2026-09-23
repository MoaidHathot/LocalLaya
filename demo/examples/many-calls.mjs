/**
 * Many decisions from one process: what a service or agent runtime pays per call when it holds one connection
 * to the sidecar, compared with spawning a CLI per decision. Also the effect of concurrent callers (the sidecar
 * serves one call at a time; concurrency only overlaps the HTTP round trips) and of the `local` mode.
 *
 *   node demo/examples/many-calls.mjs [--n 60] [--mode auto|local] [--no-cli]
 */
import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createLaya, PROJECT_DIR } from "../laya.mjs";

const arg = (name, dflt) => (process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : dflt);
const N = Number(arg("--n", "60"));
const mode = arg("--mode", "auto");
const texts = ["Turn off the living room lights", "Is the front door locked?", "Set the thermostat to 21 degrees", "Play some jazz in the kitchen", "What's the weather tomorrow?", "Dim the bedroom lights to 30 percent"];
const p50 = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const p90 = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length * 0.9)];

const laya = await createLaya({ mode, log: (m) => console.error(`  ${m}`) });
console.log(`mode ${laya.mode}; lanes ${laya.info.lanes?.join(", ")}\n`);
for (let i = 0; i < 5; i++) await laya.decide(texts[i % texts.length]); // warm

console.log(`${"pattern".padEnd(46)} ${"per call p50".padStart(13)} ${"p90".padStart(7)} ${"calls/s".padStart(8)}   inference p50   lanes`);
async function run(label, concurrency) {
  const t0 = performance.now();
  const per = [];
  const results = await laya.decideMany(Array.from({ length: N }, (_, i) => texts[i % texts.length]), {}, { concurrency });
  const wall = performance.now() - t0;
  // per-call latency as seen by one caller = wall / (N / concurrency); with concurrency 1 that is the true round trip
  const each = wall / N;
  console.log(`${label.padEnd(46)} ${(each * concurrency).toFixed(1).padStart(10)} ms ${"".padStart(7)} ${(N / wall * 1000).toFixed(1).padStart(8)}   ${p50(results.map((r) => r.routing.ms)).toFixed(1).padStart(6)} ms       ${[...new Set(results.map((r) => r.routing.lane))].join(",")}`);
  return results;
}
// sequential: one caller, one connection
{
  const t = [];
  const r = [];
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    r.push(await laya.decide(texts[i % texts.length]));
    t.push(performance.now() - t0);
  }
  console.log(`${"1 caller, sequential (round trip incl. HTTP)".padEnd(46)} ${p50(t).toFixed(1).padStart(10)} ms ${p90(t).toFixed(1).padStart(7)} ${(1000 / (t.reduce((a, b) => a + b, 0) / N)).toFixed(1).padStart(8)}   ${p50(r.map((x) => x.routing.ms)).toFixed(1).padStart(6)} ms       ${[...new Set(r.map((x) => x.routing.lane))].join(",")}`);
}
await run("4 callers in parallel (each waits its turn)", 4);
await run("8 callers in parallel", 8);

if (!process.argv.includes("--no-cli") && laya.mode !== "local") {
  // the same decision as a fresh CLI process per call (what a shell loop or a per-call tool invocation costs)
  const exec = promisify(execFile);
  const cli = path.join(PROJECT_DIR, "demo", "cli.mjs");
  const t = [];
  for (let i = 0; i < Math.min(N, 8); i++) {
    const t0 = performance.now();
    await exec(process.execPath, [cli, "--quiet", "--json", texts[i % texts.length]], { cwd: PROJECT_DIR });
    t.push(performance.now() - t0);
  }
  console.log(`${"a new CLI process per call (node demo/cli.mjs)".padEnd(46)} ${p50(t).toFixed(1).padStart(10)} ms ${p90(t).toFixed(1).padStart(7)} ${(1000 / p50(t)).toFixed(1).padStart(8)}   (Node start-up + imports around the same inference)`);
}
console.log(`\nN = ${N}, 5 smart-home questions per call. Hold a connection when you make many calls; batch questions about one text into one call.`);
await laya.close();
