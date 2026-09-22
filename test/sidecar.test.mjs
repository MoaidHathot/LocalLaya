/**
 * Integration test for the on-demand sidecar (needs the model bundle + GPU; ~2 minutes).
 *
 *   npm run test:sidecar          (= node --test test/sidecar.test.mjs)
 *
 * Uses port 8797 and short idle timeouts so it never touches a sidecar you may have running on 8787.
 * Scenarios: spawn on first use / fast second call / two launchers racing -> exactly one process /
 * idle exit frees the process and VRAM / hard kill recovery / --stop / --local / REPL over the sidecar with
 * keep-alive / 8 parallel calls spread over both lanes with zero errors / preset + calibration edits picked
 * up live / foreign service on the port detected.
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm, writeFile, copyFile, utimes } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { discover, ensureSidecar, decide, stop, waitReady, spawnSidecar, PROJECT_ROOT, logFileFor } from "../src/sidecar-client.mjs";
import { queryGpu } from "../src/metrics.mjs";

const PORT = 8797;
const run = promisify(execFile);
const node = process.execPath;
const ask = (args, opts = {}) => run(node, ["ask.mjs", "--port", String(PORT), ...args], { cwd: PROJECT_ROOT, timeout: 180_000, ...opts });
/** Run ask.mjs with a scripted stdin (the REPL); execFile has no `input` option, spawnSync would block the loop. */
const askStdin = (args, script) =>
  new Promise((resolve, reject) => {
    const p = spawn(node, ["ask.mjs", "--port", String(PORT), ...args], { cwd: PROJECT_ROOT, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    p.stdout.on("data", (d) => (stdout += d));
    p.stderr.on("data", (d) => (stderr += d));
    const timer = setTimeout(() => {
      p.kill();
      reject(new Error(`REPL did not exit within 120 s\n${stdout}\n${stderr}`));
    }, 120_000);
    p.on("exit", (code) => {
      clearTimeout(timer);
      code === 0 ? resolve({ stdout, stderr }) : reject(new Error(`REPL exited with ${code}\n${stdout}\n${stderr}`));
    });
    p.stdin.end(script);
  });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};
const waitGone = async (port, ms = 30_000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    if ((await discover({ port })).state === "none") return true;
    await sleep(200);
  }
  return false;
};

before(async () => {
  await stop({ port: PORT }).catch(() => {});
});
after(async () => {
  await stop({ port: PORT }).catch(() => {});
});

test("first --sidecar call spawns the sidecar; the second call reuses it and is fast", async () => {
  assert.equal((await discover({ port: PORT })).state, "none");
  const t0 = Date.now();
  const a = await ask(["--sidecar", "--idle", "2m", "--json", "Turn off the living room lights"]);
  const firstMs = Date.now() - t0;
  const r1 = JSON.parse(a.stdout);
  assert.equal(r1.backend, "remote");
  assert.equal(r1.answers.intent.choice, "control_device");
  assert.match(a.stderr, /starting one/);
  const h = (await discover({ port: PORT })).health;
  assert.equal(h.status, "ready");
  assert.equal(h.idleS, 120);
  assert.ok(alive(h.pid), "sidecar process is alive after the CLI exited");

  const t1 = Date.now();
  const b = await ask(["--sidecar", "--json", "--preset", "triage", "Charged twice, refund me today"]);
  const secondMs = Date.now() - t1;
  const r2 = JSON.parse(b.stdout);
  assert.equal(r2.answers.department.choice, "billing");
  assert.doesNotMatch(b.stderr, /starting one/);
  assert.ok(secondMs < firstMs / 3, `second call (${secondMs} ms) should be far faster than the first (${firstMs} ms)`);
  assert.ok(secondMs < 2500, `second call took ${secondMs} ms`);
  console.log(`      first ${firstMs} ms (spawn + load), second ${secondMs} ms`);
});

test("--status / --stop; --stop frees the port and the process", async () => {
  const s = await ask(["--status"]);
  assert.match(s.stdout, /sidecar ready on 127\.0\.0\.1:8797/);
  const pid = (await discover({ port: PORT })).health.pid;
  const st = await ask(["--stop"]);
  assert.match(st.stdout, /stopped/);
  assert.equal((await discover({ port: PORT })).state, "none");
  await sleep(500);
  assert.ok(!alive(pid), "process exited");
  const s2 = await ask(["--status"]);
  assert.match(s2.stdout, /no sidecar/);
});

test("two launchers racing produce exactly one sidecar (port = mutex, loser exits with code 3)", async () => {
  assert.equal((await discover({ port: PORT })).state, "none");
  const c1 = await spawnSidecar({ port: PORT, idle: "2m" });
  const c2 = await spawnSidecar({ port: PORT, idle: "2m" });
  // one of them must exit with 3 (EADDRINUSE) quickly; the other becomes ready
  const exitOf = (c) => new Promise((r) => (c.exitCode !== null ? r(c.exitCode) : c.once("exit", r)));
  const loserCode = await Promise.race([exitOf(c1), exitOf(c2), sleep(20_000).then(() => "timeout")]);
  assert.equal(loserCode, 3, "the racing launcher exits with code 3 without loading");
  const h = await waitReady({ port: PORT, timeoutMs: 120_000 });
  assert.equal(h.status, "ready");
  const winner = [c1, c2].find((c) => c.pid === h.pid);
  assert.ok(winner, "the ready sidecar is one of the two spawned processes");
  const loser = [c1, c2].find((c) => c !== winner);
  assert.equal(loser.exitCode, 3);
});

test("8 parallel callers: zero errors, both lanes used, queueing visible", async () => {
  const bodies = Array.from({ length: 8 }, (_, i) => ({ preset: "smart-home", text: `Set the ${["kitchen", "bedroom", "office", "porch", "garage", "hall", "living room", "bathroom"][i]} lights to ${20 + i * 10} percent` }));
  const t0 = Date.now();
  const results = await Promise.all(bodies.map((b) => decide(b, { port: PORT })));
  const wall = Date.now() - t0;
  const lanes = new Set(results.map((r) => r.routing.lane));
  const queued = results.filter((r) => r.routing.queueMs > 5).length;
  console.log(`      8 parallel calls in ${wall} ms; lanes used: ${[...lanes].join(", ")}; ${queued} waited in a queue`);
  assert.equal(results.length, 8);
  assert.ok(results.every((r) => r.answers.intent), "every call answered");
  assert.ok(lanes.size >= 2, `expected both lanes to be used under load, got ${[...lanes].join(", ")}`);
});

test("preset and calibration edits are picked up without a restart", async () => {
  const presetFile = path.join(PROJECT_ROOT, "presets", "zz-test-live.json");
  const calFile = path.join(PROJECT_ROOT, "calibration", "zz-test-live.json");
  try {
    await writeFile(presetFile, JSON.stringify({ description: "live test", state: { msg: "$TEXT" }, questions: { yes: { type: "noul", instructions: "Is the message positive?" } } }));
    const r1 = await decide({ preset: "zz-test-live", text: "what a wonderful day" }, { port: PORT });
    assert.equal(r1.preset, "zz-test-live");
    assert.equal(r1.calibration, null);
    assert.deepEqual(Object.keys(r1.answers), ["yes"]);
    // add a question to the preset file -> next call sees it
    await writeFile(presetFile, JSON.stringify({ description: "live test", state: { msg: "$TEXT" }, questions: { yes: { type: "noul", instructions: "Is the message positive?" }, short: { type: "noul", instructions: "Is the message short?" } } }));
    const r2 = await decide({ preset: "zz-test-live", text: "what a wonderful day" }, { port: PORT });
    assert.deepEqual(Object.keys(r2.answers), ["yes", "short"]);
    // add a calibration table with a huge temperature -> the noul probability moves towards 0.5
    const p1 = r1.answers.yes.noul;
    await writeFile(calFile, JSON.stringify({ temperature_by_options: { "noul:2": 50 } }));
    const r3 = await decide({ preset: "zz-test-live", text: "what a wonderful day" }, { port: PORT });
    assert.equal(r3.calibration, "calibration/zz-test-live.json");
    const p3 = r3.answers.yes.noul;
    assert.ok(Math.abs(p3 - 0.5) < Math.abs(p1 - 0.5) - 0.05, `T=50 should flatten P(true) (shipped ${p1} -> ${p3})`);
    // change the table (new mtime) -> re-read: back to a sharp answer
    await writeFile(calFile, JSON.stringify({ temperature_by_options: { "noul:2": 0.5 } }));
    await utimes(calFile, new Date(), new Date(Date.now() + 2000));
    const r4 = await decide({ preset: "zz-test-live", text: "what a wonderful day" }, { port: PORT });
    const p4 = r4.answers.yes.noul;
    assert.ok(Math.abs(p4 - 0.5) > Math.abs(p3 - 0.5) + 0.05, `T=0.5 should sharpen again (${p3} -> ${p4})`);
  } finally {
    await rm(presetFile, { force: true });
    await rm(calFile, { force: true });
  }
});

test("REPL over the sidecar: commands work, keep-alive keeps it warm", async () => {
  const script = ["Is the front door locked?", "/preset triage", "/noul Is the customer angry?", "The app crashed again and nobody answers my emails", "/stats", "/lane cpu:8", "/again", "/exit", ""].join("\n");
  const out = await askStdin(["--sidecar", "--no-color"], script);
  assert.match(out.stdout, /target_device\s+locks/);
  assert.match(out.stdout, /department\s+technical/);
  assert.match(out.stdout, /is_the_customer/);
  assert.match(out.stdout, /via sidecar/);
  assert.match(out.stdout, /cpu:8 .* via sidecar/);
  assert.match(out.stderr, /sidecar :8797 pid/);
});

test("hard kill: the next call notices and respawns", async () => {
  const pid = (await discover({ port: PORT })).health.pid;
  process.kill(pid);
  assert.ok(await waitGone(PORT, 10_000), "port freed after kill");
  const a = await ask(["--sidecar", "--idle", "15s", "--json", "Turn off the living room lights"]);
  assert.match(a.stderr, /starting one/);
  const r = JSON.parse(a.stdout);
  assert.equal(r.answers.target_device.choice, "lights");
  const h = (await discover({ port: PORT })).health;
  assert.notEqual(h.pid, pid);
  assert.equal(h.idleS, 15);
});

test("idle exit: the process goes away by itself and VRAM is released", async () => {
  const h = (await discover({ port: PORT })).health;
  assert.equal(h.idleS, 15);
  const gpuBusy = await queryGpu();
  const gone = await waitGone(PORT, 40_000);
  assert.ok(gone, "sidecar exited after its idle period");
  await sleep(1500);
  assert.ok(!alive(h.pid), "process is gone");
  const gpuIdle = await queryGpu();
  if (gpuBusy && gpuIdle) {
    console.log(`      VRAM used: ${gpuBusy.memUsedMiB} MiB with sidecar -> ${gpuIdle.memUsedMiB} MiB after exit`);
    assert.ok(gpuBusy.memUsedMiB - gpuIdle.memUsedMiB > 500, "at least ~800 MiB of VRAM released");
  }
  const log = await readFile(logFileFor(PORT), "utf8");
  assert.match(log, /idle for 15 s with nothing in flight; exiting/);
});

test("--local ignores the sidecar; LAYA_SIDECAR=1 enables it; --local overrides the env", async () => {
  const a = await ask(["--local", "--json", "--lanes", "cpu:8", "Turn on the TV"]);
  assert.equal(JSON.parse(a.stdout).backend, "local");
  assert.equal((await discover({ port: PORT })).state, "none", "--local must not spawn a sidecar");
  const b = await ask(["--json", "--idle", "1m", "Turn on the TV"], { env: { ...process.env, LAYA_SIDECAR: "1", LAYA_PORT: String(PORT) } });
  assert.equal(JSON.parse(b.stdout).backend, "remote");
  assert.equal((await discover({ port: PORT })).state, "ready");
  const c = await ask(["--local", "--json", "--lanes", "cpu:8", "Turn on the TV"], { env: { ...process.env, LAYA_SIDECAR: "1", LAYA_PORT: String(PORT) } });
  assert.equal(JSON.parse(c.stdout).backend, "local");
  await stop({ port: PORT });
});

test("foreign service on the port: detected, never stopped, CLI falls back to in-process", async () => {
  const foreign = createServer((req, res) => res.end("hello"));
  await new Promise((r) => foreign.listen(PORT, "127.0.0.1", r));
  try {
    assert.equal((await discover({ port: PORT })).state, "foreign");
    await assert.rejects(stop({ port: PORT }), (e) => e.code === "FOREIGN_PORT");
    const s = await ask(["--status"]);
    assert.match(s.stdout, /used by something else/);
    const a = await ask(["--sidecar", "--json", "--lanes", "cpu:8", "Dim the lights"]);
    assert.match(a.stderr, /falling back to in-process/);
    assert.equal(JSON.parse(a.stdout).backend, "local");
  } finally {
    foreign.close();
  }
});

test("--start makes it ready and prints connection info; a foreground serve on the same port exits with 3", async () => {
  const s = await ask(["--start", "--idle", "30s"]);
  const info = JSON.parse(s.stdout);
  assert.equal(info.url, `http://127.0.0.1:${PORT}`);
  assert.ok(info.spawned);
  assert.deepEqual(info.lanes, ["webgpu:fp16", "cpu:8"]);
  const again = JSON.parse((await ask(["--start"])).stdout);
  assert.equal(again.spawned, false);
  assert.equal(again.pid, info.pid);
  // a second server on the same port must not load anything
  const p = spawn(node, ["serve.mjs", "--port", String(PORT)], { cwd: PROJECT_ROOT, stdio: ["ignore", "pipe", "pipe"] });
  const code = await new Promise((r) => p.once("exit", r));
  assert.equal(code, 3);
  await stop({ port: PORT });
});
