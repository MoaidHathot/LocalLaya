/**
 * Integration test for demo/ (needs the model bundle + GPU; ~1 minute). Port 8797, like the sidecar tests.
 *
 *   npm run test:demo          (= node --test --test-concurrency=1 test/demo.test.mjs)
 *
 * Scenarios: the library in mode auto (sidecar) - decide / decideMany / helpers / overrides / errors; the
 * keep-alive client's per-call cost; mode local gives the same answers as the sidecar; the CLI (one-shot,
 * --json, --file, exit codes); every example runs.
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";
import { createLaya, gate, readInputs, summarize, top } from "../demo/laya.mjs";
import { discover, stop, PROJECT_ROOT } from "../src/sidecar-client.mjs";

const PORT = 8797;
const run = promisify(execFile);
const node = process.execPath;
const cli = (args, opts = {}) => run(node, ["demo/cli.mjs", "--port", String(PORT), ...args], { cwd: PROJECT_ROOT, timeout: 180_000, ...opts });
const quiet = () => {};
let laya;

before(async () => {
  await stop({ port: PORT }).catch(() => {});
});
after(async () => {
  await laya?.close().catch(() => {});
  await stop({ port: PORT }).catch(() => {});
});

test("createLaya auto: spawns the sidecar, decide / decideMany / helpers, per-call overrides, error mapping", async () => {
  const logs = [];
  laya = await createLaya({ mode: "auto", port: PORT, idle: "3m", log: (m) => logs.push(m) });
  assert.equal(laya.mode, "sidecar");
  assert.ok(logs.some((m) => /starting one/.test(m)), logs.join(" | "));
  assert.equal((await discover({ port: PORT })).state, "ready");
  const r = await laya.decide("Charged twice. Refund today or I cancel.", { preset: "triage" });
  assert.equal(r.answers.department.choice, "billing");
  assert.equal(r.preset, "triage");
  assert.deepEqual(r.state, { message: "Charged twice. Refund today or I cancel." });
  assert.deepEqual(top(r.answers.department), { label: "billing", p: r.answers.department.probabilities.billing });
  assert.equal(gate(r.answers.department), "act");
  assert.equal(gate({ type: "noul", noul: 0.6 }), "ask");
  assert.equal(gate({ type: "noul", noul: 0.5 }), "unsure");
  assert.equal(top({ type: "noul", noul: 0.2 }).label, "no");
  const s = top(r.answers.urgency);
  assert.ok(s.score >= 0 && s.max === 3 && typeof s.label === "string");
  assert.match(summarize(r.answers), /department=billing 0\.\d\d, urgency=/);
  // state input + own questions, and the state is returned verbatim
  const q = { needs_reply: { type: "noul", instructions: "Does this need a reply?" } };
  const r2 = await laya.decide({ state: { email: "Can you make it at 3pm?" } }, { questions: q });
  assert.deepEqual(Object.keys(r2.answers), ["needs_reply"]);
  assert.deepEqual(r2.state, { email: "Can you make it at 3pm?" });
  assert.equal(r2.preset, null);
  // decideMany keeps the order
  const inputs = readInputs(path.join(PROJECT_ROOT, "demo", "data", "tickets.json"));
  const many = await laya.decideMany(inputs, { preset: "triage" }, { concurrency: 4 });
  assert.equal(many.length, inputs.length);
  assert.deepEqual(many.map((x) => x.state.message), inputs);
  // overrides: forced lane, policy, deadline, exec are accepted and reflected in routing
  const forced = await laya.decide("Turn on the TV", { lane: "cpu:8" });
  assert.equal(forced.routing.lane, "cpu:8");
  const pol = await laya.decide("Turn on the TV", { policy: "prefer-cpu" });
  assert.equal(pol.routing.lane, "cpu:8");
  assert.match(pol.routing.reason, /prefer-cpu/);
  const dl = await laya.decide("Turn on the TV", { deadlineMs: 5000 });
  assert.match(dl.routing.reason, /deadline/);
  const ex = await laya.decide("Turn on the TV", { exec: { graph: false } });
  assert.ok(ex.answers.intent);
  // errors: unknown preset / bad policy / bad questions -> BAD_REQUEST (HTTP 400)
  for (const [input, opts, re] of [["x", { preset: "nope" }, /unknown preset/], ["x", { policy: "fastest" }, /policy must be one of/], ["x", { questions: { q: { type: "bogus" } } }, /type must be/], ["x", { lane: "cpu:99" }, /not loaded/]]) {
    await assert.rejects(laya.decide(input, opts), (e) => e.code === "BAD_REQUEST" && re.test(e.message), `${JSON.stringify(opts)} -> ${re}`);
  }
  await assert.rejects(laya.decide(42), /input must be/);
});

test("keep-alive client: per-call cost = inference + a few ms; health / stats / presets", async () => {
  for (let i = 0; i < 3; i++) await laya.decide("Turn off the living room lights");
  const t = [];
  const inf = [];
  for (let i = 0; i < 12; i++) {
    const t0 = performance.now();
    const r = await laya.decide(`Turn off the living room lights (${i})`);
    t.push(performance.now() - t0);
    inf.push(r.routing.ms);
  }
  const p50 = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  console.log(`      round trip p50 ${p50(t).toFixed(1)} ms, inference p50 ${p50(inf).toFixed(1)} ms -> HTTP + client overhead ${(p50(t) - p50(inf)).toFixed(1)} ms`);
  assert.ok(p50(t) - p50(inf) < 8, `keep-alive overhead ${(p50(t) - p50(inf)).toFixed(1)} ms`);
  const h = await laya.health();
  assert.equal(h.service, "laya");
  assert.equal(h.status, "ready");
  const st = await laya.stats();
  assert.ok(st.lanes && st.queue);
  const p = await laya.presets();
  assert.ok(p.triage && p["smart-home"]);
});

test("mode local: same request logic in-process, same answers as the sidecar", async () => {
  const local = await createLaya({ mode: "local", lanes: "webgpu:fp16", log: quiet });
  try {
    assert.equal(local.mode, "local");
    const a = await laya.decide("Charged twice. Refund today or I cancel.", { preset: "triage", lane: "webgpu:fp16" });
    const b = await local.decide("Charged twice. Refund today or I cancel.", { preset: "triage" });
    assert.equal(b.routing.lane, "webgpu:fp16");
    assert.equal(a.answers.department.choice, b.answers.department.choice);
    for (const k of Object.keys(a.answers.department.probabilities)) assert.ok(Math.abs(a.answers.department.probabilities[k] - b.answers.department.probabilities[k]) < 1e-3, k);
    assert.equal(a.calibration, b.calibration, "the same per-preset calibration file is applied");
    await assert.rejects(local.decide("x", { preset: "nope" }), (e) => e.status === 400 && /unknown preset/.test(e.message));
    assert.equal((await local.health()).mode, "local");
  } finally {
    await local.close();
  }
});

test("cli: one-shot text / --json, --file table, --health, exit codes", async () => {
  const a = await cli(["--preset", "triage", "Charged twice. Refund today or I cancel."]);
  assert.match(a.stdout, /department\s+billing\s+\d+%/);
  assert.match(a.stderr, /mode sidecar/);
  const j = JSON.parse((await cli(["--json", "--quiet", "--lane", "webgpu:fp16", "Turn on the TV"])).stdout);
  assert.equal(j.routing.lane, "webgpu:fp16");
  assert.equal(j.answers.intent.choice, "control_device");
  const f = await cli(["--preset", "guard", "--file", "demo/data/prompts.json", "--quiet"]);
  assert.equal(f.stdout.trim().split("\n").length, 10);
  assert.match(f.stdout, /jailbreak=yes/);
  const fj = JSON.parse((await cli(["--preset", "guard", "--file", "demo/data/prompts.json", "--quiet", "--json"])).stdout);
  assert.equal(fj.length, 10);
  const h = JSON.parse((await cli(["--health", "--quiet"])).stdout);
  assert.equal(h.service, "laya");
  for (const [args, code, re] of [[["--preset", "nope", "x"], 2, /unknown preset/], [[], 2, /nothing to decide/], [["--questions", '{"q":{"type":"bogus"}}', "x"], 2, /type must be/]]) {
    const e = await cli(args).then(() => null, (err) => err);
    assert.ok(e, `expected failure for ${args.join(" ")}`);
    assert.equal(e.code, code);
    assert.match(e.stderr, re);
  }
});

test("every example runs against the sidecar", async () => {
  // the examples read LAYA_PORT (default 8787 = the user's sidecar); point them at the test port
  const { stdout } = await run(node, ["demo/run-all.mjs", "--n", "12"], { cwd: PROJECT_ROOT, timeout: 300_000, maxBuffer: 8 * 1024 * 1024, env: { ...process.env, LAYA_PORT: String(PORT) } });
  assert.match(stdout, /5\/5 examples ok/);
  assert.match(stdout, /routing: .*->/);
  assert.match(stdout, /allow \d+, BLOCK \d+/);
  assert.match(stdout, /-> json\.parse\(\)/);
  assert.match(stdout, /1 caller, sequential/);
});
