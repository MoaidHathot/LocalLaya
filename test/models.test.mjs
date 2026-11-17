/** Real Python model / CLI / HTTP integration. Requires python:setup and both prepared models. */
import { test, after } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { createLaya } from "../demo/laya.mjs";
import { ensureSidecar, createClient, stop, PROJECT_ROOT } from "../src/sidecar-client.mjs";
import { PythonBackend } from "../src/python-backend.mjs";
import { PYTHON, modelOptions } from "../src/models.mjs";

const PORT = 8798;
const run = promisify(execFile);
const questions = {
  team: { type: "choice", instructions: "Which team handles this request?", criteria: { billing: "invoices, payments and refunds", technical: "bugs and outages" } },
  refund: { type: "noul", instructions: "Does the customer request a refund?" },
  urgency: { type: "score", instructions: "How urgent is this request?", criteria: ["can wait", "soon", "today"] },
};
const text = "I was charged twice. Please refund the duplicate today.";
const valid = (r, model, device) => {
  assert.equal(r.routing.model, model);
  assert.equal(r.routing.device, device);
  assert.deepEqual(Object.keys(r.answers), Object.keys(questions));
  assert.equal(r.answers.team.type, "choice");
  assert.ok(Number.isFinite(r.answers.refund.noul));
  const score = r.answers.urgency;
  const expected = Object.entries(score.probabilities).reduce((s, [k, p]) => s + Number(k) * p, 0);
  assert.ok(Math.abs(score.score - expected) < 0.002, "score keeps LocalLaya's expected-level meaning");
};
after(async () => { await stop({ port: PORT, timeoutMs: 30_000 }).catch(() => {}); });

test("reserved KV storage preserves history and isolates per-question cache branches", { timeout: 30_000 }, async () => {
  await run(PYTHON, ["test/vega_cache_test.py"], { cwd: PROJECT_ROOT, timeout: 25_000 });
});

test("Multilingual graph cache is bounded and optional acceleration falls back cleanly", { timeout: 30_000 }, async () => {
  await run(PYTHON, ["test/multilingual_cache_test.py"], { cwd: PROJECT_ROOT, timeout: 25_000 });
});

test("Python timeout stops the real interpreter as well as the Windows venv launcher", { timeout: 15_000 }, async (t) => {
  const b = await PythonBackend.create(modelOptions({ model: "vega", device: "cpu" }), {
    python: PYTHON, script: fileURLToPath(new URL("./fixtures/python-hang.py", import.meta.url)), timeoutMs: 1500,
  });
  const pid = b.info.pid;
  const cleanup = () => { try { process.kill(pid); } catch { /* already stopped */ } };
  t.signal.addEventListener("abort", cleanup, { once: true });
  try {
    await assert.rejects(b.decide("hang", questions), /timed out/);
    await b.close();
    assert.throws(() => process.kill(pid, 0), "the actual Python interpreter must be gone");
  } finally { cleanup(); await b.close(); }
});

test("Vega local: CPU and CUDA, readout switching, accurate multi-question truncation and score semantics", { timeout: 240_000 }, async () => {
  const d = await createLaya({ mode: "local", model: "vega", device: "cpu", maxLen: 128 });
  try {
    const a = await d.decide({ state: text }, { questions });
    valid(a, "vega", "cpu");
    assert.equal(a.calibration, null);
    assert.equal(a.routing.exec.acceleration.reader, "torch-reference", "CPU stays usable with optional CUDA kernels installed");
    assert.equal(a.usage.truncated, false);
    assert.ok(a.usage.input_tokens > 128, "summing question rows is not a truncation test");
    const b = await d.decide({ state: text }, { questions, device: "cuda", readout: "engine" });
    valid(b, "vega", "cuda");
    assert.equal(b.routing.exec.readout, "engine");
    assert.equal(b.routing.exec.acceleration.requested, "auto");
    assert.equal(b.routing.exec.acceleration.head.mode, "graph");
    const ref = await d.decide({ state: text }, { questions, device: "cuda", acceleration: "off" });
    valid(ref, "vega", "cuda");
    assert.equal(ref.routing.exec.acceleration.reader, "torch-reference");
    assert.equal(ref.routing.exec.acceleration.head.mode, "eager");
    assert.equal(ref.answers.team.choice, b.answers.team.choice);
    const cut = await d.decide({ state: "background ".repeat(300) + text }, { questions, device: "cuda" });
    assert.equal(cut.usage.truncated, true);
    assert.ok(cut.usage.state_tokens_dropped > 0);
  } finally { await d.close(); }
});

test("Multilingual local: CPU/CUDA, Unicode, an input longer than 512 tokens, and switching back to original Laya", { timeout: 240_000 }, async () => {
  const d = await createLaya({ mode: "local", model: "multilingual", device: "cpu", lanes: "cuda:fp16" });
  try {
    const a = await d.decide({ state: "تم خصم المبلغ مرتين من حسابي. أرجو إعادة المبلغ الزائد اليوم." }, { questions });
    valid(a, "multilingual", "cpu");
    assert.equal(a.routing.exec.acceleration.backend, "eager");
    assert.equal(a.answers.team.choice, "billing");
    const b = await d.decide({ state: "background ".repeat(1100) + text }, { questions, device: "cuda" });
    valid(b, "multilingual", "cuda");
    assert.ok(b.usage.state_tokens > 512);
    assert.equal(b.usage.truncated, false);
    assert.equal(b.usage.context_tokens, 8192);
    assert.ok(["eager", "tilelang"].includes(b.routing.exec.acceleration.backend));
    if (b.routing.exec.acceleration.backend === "tilelang") {
      assert.equal(b.routing.exec.acceleration.graphs.mode, "kernels", "first use avoids graph capture cost");
      assert.equal(b.routing.exec.acceleration.graphs.limit, 8);
      const capture = await d.decide({ state: "background ".repeat(1100) + text }, { questions, device: "cuda" });
      assert.equal(capture.routing.exec.acceleration.graphs.mode, "capture");
      const replay = await d.decide({ state: "background ".repeat(1100) + text }, { questions, device: "cuda" });
      assert.equal(replay.routing.exec.acceleration.graphs.mode, "replay");
      assert.equal(replay.answers.team.choice, b.answers.team.choice);
    }
    const ref = await d.decide({ state: "background ".repeat(1100) + text }, { questions, device: "cuda", acceleration: "off" });
    valid(ref, "multilingual", "cuda");
    assert.equal(ref.routing.exec.acceleration.backend, "eager");
    assert.equal(ref.answers.team.choice, b.answers.team.choice);
    const original = await d.decide("Turn off the living room lights", { model: "laya", device: "cuda" });
    assert.equal(original.routing.lane, "cuda:fp16");
    assert.equal(original.routing.contextTokens, 512);
    assert.equal(original.calibration, "calibration/smart-home.json");
    assert.equal(original.answers.intent.choice, "control_device");
    const again = await d.decide({ state: text }, { model: "multilingual", device: "cuda", questions });
    valid(again, "multilingual", "cuda");
  } finally { await d.close(); }
});

test("Vega long input retains all tokens and reports bounded causal prefill through the public API", { timeout: 180_000 }, async () => {
  const d = await createLaya({ mode: "local", model: "vega", device: "cuda" });
  try {
    const r = await d.decide({ state: "background ".repeat(33000) + text }, { questions: { team: questions.team } });
    assert.equal(r.routing.model, "vega");
    assert.equal(r.routing.device, "cuda");
    assert.ok(r.usage.state_tokens > 32768);
    assert.equal(r.usage.truncated, false);
    assert.equal(r.usage.state_tokens_dropped, 0);
    assert.equal(r.routing.exec.acceleration.prefill.used, true);
    assert.equal(r.routing.exec.acceleration.prefill.symbolicMaskUsed, true);
    assert.ok(r.routing.exec.acceleration.prefill.lastChunks > 1);
  } finally { await d.close(); }
});

test("sidecar, HTTP library, both CLIs and skill wrapper preserve model selection and expose metadata", { timeout: 300_000 }, async () => {
  await ensureSidecar({ port: PORT, model: "vega", device: "cuda", idle: "5m" });
  const client = createClient({ port: PORT });
  try {
    const a = await client.decide({ state: text, questions, model: "vega", device: "cuda" });
    valid(a, "vega", "cuda");
    const reference = await client.decide({ state: text, questions, model: "vega", device: "cuda", acceleration: "off" });
    assert.equal(reference.routing.exec.acceleration.requested, "off");
    for (const script of ["ask.mjs", "skills/laya-decisions/scripts/laya.mjs"]) {
      const argv = [script, "--sidecar", "--port", String(PORT), "--model", "vega", "--device", "cuda", "--acceleration", "off", "--questions", JSON.stringify(questions), "--json", text];
      const result = JSON.parse((await run(process.execPath, argv, { cwd: PROJECT_ROOT, timeout: 120_000 })).stdout);
      assert.equal(result.routing.exec.acceleration.requested, "off");
      assert.equal(result.routing.exec.acceleration.reader, "torch-reference");
    }
    assert.ok((await client.models()).available.multilingual);
    const d = await createLaya({ mode: "http", url: `http://127.0.0.1:${PORT}`, model: "vega", device: "cuda" });
    try { valid(await d.decide({ state: text }, { questions }), "vega", "cuda"); }
    finally { await d.close(); }
    const qa = JSON.stringify(questions);
    const argv = ["--port", String(PORT), "--model", "multilingual", "--device", "cuda", "--acceleration", "off", "--questions", qa, "--json", text];
    const cli = JSON.parse((await run(process.execPath, ["ask.mjs", "--mode", "sidecar", ...argv], { cwd: PROJECT_ROOT, timeout: 120_000 })).stdout);
    valid(cli, "multilingual", "cuda");
    assert.equal(cli.routing.exec.acceleration.requested, "off");
    assert.equal(cli.backend, "remote");
    const demo = JSON.parse((await run(process.execPath, ["demo/cli.mjs", "--mode", "sidecar", ...argv], { cwd: PROJECT_ROOT, timeout: 120_000 })).stdout);
    valid(demo, "multilingual", "cuda");
    assert.equal(demo.routing.exec.acceleration.requested, "off");
    const skill = JSON.parse((await run(process.execPath, ["skills/laya-decisions/scripts/laya.mjs", ...argv], { cwd: PROJECT_ROOT, timeout: 120_000 })).stdout);
    valid(skill, "multilingual", "cuda");
    assert.equal(skill.routing.exec.acceleration.requested, "off");
    assert.equal((await client.health()).models.active.model, "multilingual");
    const bad = await client.decide({ text, model: "vega", lane: "cpu:8" }).then(() => null, (e) => e);
    assert.equal(bad.code, "BAD_REQUEST");
  } finally { client.close(); await stop({ port: PORT, timeoutMs: 30_000 }); }
});
