#!/usr/bin/env node
/** Fresh-process, end-to-end measurements. Keeps downloaded models and on-disk compiler caches. */
import { execFile } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs, promisify } from "node:util";
import { PROJECT_ROOT } from "../src/models.mjs";

const run = promisify(execFile);
const { values: a } = parseArgs({ options: {
  rounds: { type: "string", default: "3" }, variants: { type: "string", default: "auto,forced-cuda" },
  text: { type: "string", default: "Turn on the lights" }, out: { type: "string", default: "results/cold-start.json" },
} });
const rounds = Number(a.rounds);
if (!Number.isSafeInteger(rounds) || rounds < 1) throw new Error("--rounds must be a positive integer");
const choices = {
  auto: [],
  "eager-cuda": ["--device", "cuda", "--acceleration", "off"],
  "eager-cpu": ["--device", "cpu", "--acceleration", "off"],
  "forced-cuda": ["--device", "cuda", "--acceleration", "on"],
};
const variants = a.variants.split(",");
for (const v of variants) if (!choices[v]) throw new Error(`unknown variant ${v}`);
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !["LAYA_DEVICE", "LAYA_MODEL", "LAYA_SIDECAR", "LAYA_LANES"].includes(k)));
const report = { date: new Date().toISOString(), text: a.text, rounds, records: [], note: "Fresh Node/Python process per call; OS file cache and compiled-kernel disk cache retained. answerMs ends at stdout result; wallMs includes shutdown." };
const file = path.resolve(PROJECT_ROOT, a.out);
await mkdir(path.dirname(file), { recursive: true });
const median = xs => { const x = [...xs].sort((a,b) => a-b); return (x[Math.floor((x.length-1)/2)] + x[Math.floor(x.length/2)]) / 2; };
for (let round = 0; round < rounds; round++) {
  const order = [...variants.slice(round % variants.length), ...variants.slice(0, round % variants.length)];
  for (const variant of order) {
    const argv = ["ask.mjs", "--model", "multilingual", "--local", "--json", ...choices[variant], a.text];
    const t0 = performance.now();
    const pending = run(process.execPath, argv, { cwd: PROJECT_ROOT, env, timeout: 240_000, maxBuffer: 8 * 1024 * 1024 });
    let answerMs;
    pending.child.stdout.on("data", () => { answerMs = performance.now() - t0; });
    const { stdout, stderr } = await pending;
    const wallMs = performance.now() - t0;
    const result = JSON.parse(stdout);
    const row = { variant, round, answerMs, wallMs, inferenceMs: result.routing.ms, initialization: result.routing.initialization,
      execution: result.routing.exec, answers: result.answers, log: stderr };
    report.records.push(row);
    console.log(`${variant} #${round+1}: ${(answerMs/1000).toFixed(2)}s to answer, ${(wallMs/1000).toFixed(2)}s through exit, ${(result.routing.initialization?.processMs/1000).toFixed(2)}s startup, ${result.routing.ms.toFixed(1)}ms inference`);
    await writeFile(file, JSON.stringify(report, null, 2) + "\n");
  }
}
report.summary = Object.fromEntries(variants.map(v => { const rows = report.records.filter(r => r.variant === v); return [v, {
  answerMs: median(rows.map(r => r.answerMs)), wallMs: median(rows.map(r => r.wallMs)),
  initializationMs: median(rows.map(r => r.initialization.processMs)), inferenceMs: median(rows.map(r => r.inferenceMs)),
}]; }));
await writeFile(file, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report.summary, null, 2));
