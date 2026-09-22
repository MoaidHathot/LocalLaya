#!/usr/bin/env node
/**
 * laya.mjs - call Laya from anywhere. Thin wrapper around the project's ask.mjs that
 *   1. finds the TestLayaONNX project ($LAYA_DIR, or three levels up from this script when the skill lives in
 *      <project>/skills/laya-decisions/), and
 *   2. defaults to --sidecar --json (shared background instance, machine-readable stdout).
 *
 *   node scripts/laya.mjs --preset triage "Charged twice, refund me today"
 *   node scripts/laya.mjs --state '{"text":"..."}' --questions q.json
 *   node scripts/laya.mjs --local "..."         # in-process instead of the sidecar
 *   node scripts/laya.mjs --status | --stop | --start
 *   node scripts/laya.mjs --pretty "..."        # human-readable instead of JSON
 *
 * Exit codes are ask.mjs's: 0 ok, 1 error (stderr), 2 unknown preset.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const candidates = [process.env.LAYA_DIR, path.resolve(here, "..", "..", "..")].filter(Boolean);
const projectDir = candidates.find((d) => existsSync(path.join(d, "ask.mjs")) && existsSync(path.join(d, "serve.mjs")));
if (!projectDir) {
  console.error(`laya.mjs: cannot find the TestLayaONNX project (looked at ${candidates.join(", ")}). Set LAYA_DIR to the directory that contains ask.mjs.`);
  process.exit(2);
}

const args = process.argv.slice(2);
const has = (f) => args.includes(f);
const passthrough = args.filter((a) => a !== "--pretty");
const lifecycle = has("--status") || has("--stop") || has("--start");
const finalArgs = ["ask.mjs", ...passthrough];
if (!lifecycle) {
  if (!has("--local") && !has("--sidecar")) finalArgs.push("--sidecar");
  if (!has("--pretty") && !has("--json")) finalArgs.push("--json");
}
if (finalArgs.length === 1 || (!lifecycle && !passthrough.some((a) => !a.startsWith("--")) && !has("--state"))) {
  // interactive REPL when no text / state / lifecycle flag: keep stdout human-readable
  const i = finalArgs.indexOf("--json");
  if (i > 0 && !has("--json")) finalArgs.splice(i, 1);
}

const child = spawn(process.execPath, finalArgs, { cwd: projectDir, stdio: "inherit", windowsHide: true });
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
child.on("error", (e) => {
  console.error(`laya.mjs: failed to start node: ${e.message}`);
  process.exit(1);
});
