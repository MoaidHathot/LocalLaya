/**
 * Run every demo example in sequence (npm run demo). Exit code 1 if any example fails.
 *
 *   node demo/run-all.mjs [--mode local] [--n 20]
 */
import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const run = promisify(execFile);
const here = path.dirname(fileURLToPath(import.meta.url));
const extra = process.argv.slice(2);
const examples = ["triage-inbox.mjs", "gate-prompts.mjs", "route-requests.mjs", "own-questions.mjs", "many-calls.mjs"];
let failed = 0;
for (const ex of examples) {
  console.log(`\n\u2500\u2500 ${ex} ${"\u2500".repeat(Math.max(0, 70 - ex.length))}`);
  try {
    const { stdout, stderr } = await run(process.execPath, [path.join(here, "examples", ex), ...extra], { cwd: path.resolve(here, ".."), timeout: 300_000, maxBuffer: 8 * 1024 * 1024 });
    process.stdout.write(stdout);
    if (stderr.trim()) process.stderr.write(stderr.split("\n").filter((l) => !/^\s*(sidecar|mode )/.test(l)).join("\n"));
  } catch (e) {
    failed++;
    console.error(`FAILED (${e.code}): ${(e.stderr || e.message).trim().split("\n").slice(-5).join("\n")}`);
  }
}
console.log(`\n${examples.length - failed}/${examples.length} examples ok`);
process.exit(failed ? 1 : 0);
