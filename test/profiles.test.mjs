/** Real model checks for the model + local/sidecar shortcut. Requires prepared Python models and CUDA. */
import { test, after } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { ensureSidecar, stop, PROJECT_ROOT } from "../src/sidecar-client.mjs";
import { PYTHON } from "../src/models.mjs";

const run = promisify(execFile);
const PORT = 8796;
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !["LAYA_DEVICE", "LAYA_MODEL", "LAYA_SIDECAR", "LAYA_LANES"].includes(k)));
const text = "Charged twice. Please refund the duplicate.";
const cli = async (args, checkProgress = false) => {
  const { stdout, stderr } = await run(process.execPath, ["ask.mjs", "--json", "--preset", "triage", ...args, text], {
    cwd: PROJECT_ROOT, env, timeout: 180_000, maxBuffer: 8 * 1024 * 1024,
  });
  if (checkProgress) {
    assert.match(stderr, /Python worker started/);
    assert.match(stderr, /Importing PyTorch: done in/);
    assert.match(stderr, /checkpoint on cuda: done in/);
    assert.match(stderr, /First decision completed in/);
  }
  return JSON.parse(stdout); // Startup/progress output must not contaminate the JSON protocol.
};

after(async () => { await stop({ port: PORT, timeoutMs: 30_000 }).catch(() => {}); });

test("automatic device selection preserves explicit CUDA and surfaces model errors", async () => {
  await run(PYTHON, ["test/runtime_profile_test.py"], { cwd: PROJECT_ROOT, timeout: 30_000 });
});

test("model + local automatically selects a one-shot profile for all three models", { timeout: 300_000 }, async () => {
  for (const model of ["laya", "vega", "multilingual"]) {
    const r = await cli(["--model", model, "--local"], model !== "laya");
    assert.equal(r.routing.model, model);
    assert.equal(r.routing.device, "cuda");
    assert.equal(r.routing.profile.lifetime, "one-shot");
    assert.equal(r.routing.profile.graphs, "off");
    assert.equal(r.answers.department.choice, "billing");
    if (model === "laya") {
      assert.equal(r.routing.profile.laneLoading, "first-available");
      assert.equal(r.routing.exec.mode, "dynamic");
      assert.deepEqual(r.routing.alternatives, []);
    } else {
      assert.equal(r.routing.exec.graphReplayEnabled, false);
      assert.ok(r.routing.initialization.processMs >= r.routing.initialization.totalMs);
      assert.ok(r.routing.initialization.phases.every(p => p.status === "done"));
      if (model === "multilingual") {
        assert.equal(r.routing.exec.acceleration.backend, "eager");
        assert.equal(r.routing.exec.acceleration.requested, "auto");
        assert.equal(r.routing.exec.acceleration.selected, "off");
        assert.deepEqual(r.routing.exec.acceleration.versions, {}, "one-shot auto never initializes the optional compiler");
        assert.equal(r.routing.contextTokens, 8192);
      }
    }
  }
});

test("explicit device, acceleration and graph choices override one-shot defaults", { timeout: 180_000 }, async () => {
  const graph = await cli(["--model", "vega", "--local", "--graphs", "on"]);
  assert.equal(graph.routing.profile.graphs, "on");
  assert.equal(graph.routing.exec.acceleration.head.mode, "graph");
  const cpu = await cli(["--model", "multilingual", "--local", "--device", "cpu", "--acceleration", "off", "--max-len", "1024"]);
  assert.equal(cpu.routing.device, "cpu");
  assert.equal(cpu.routing.contextTokens, 1024);
  assert.equal(cpu.routing.exec.acceleration.backend, "eager");
  const fused = await cli(["--model", "multilingual", "--local", "--acceleration", "on"]);
  assert.equal(fused.routing.exec.acceleration.backend, "tilelang");
  assert.equal(fused.routing.exec.acceleration.requested, "on");
  assert.equal(fused.routing.exec.acceleration.selected, "on");
  assert.equal(fused.routing.exec.graphReplayEnabled, false);
});

test("the same minimal requests use reusable profiles through the sidecar", { timeout: 300_000 }, async () => {
  await ensureSidecar({ port: PORT, model: "vega", idle: "5m" });
  try {
    for (const model of ["vega", "multilingual", "laya"]) {
      const r = await cli(["--model", model, "--sidecar", "--port", String(PORT)]);
      assert.equal(r.backend, "remote");
      assert.equal(r.routing.model, model);
      assert.equal(r.routing.profile.lifetime, "persistent");
      assert.equal(r.routing.profile.graphs, "on");
      if (model === "laya") assert.equal(r.routing.profile.laneLoading, "parallel");
      if (model === "multilingual") {
        assert.equal(r.routing.exec.acceleration.backend, "tilelang");
        assert.equal(r.routing.exec.acceleration.requested, "auto");
        assert.equal(r.routing.exec.acceleration.graphs.mode, "kernels");
        const capture = await cli(["--model", model, "--sidecar", "--port", String(PORT)]);
        assert.equal(capture.routing.exec.acceleration.graphs.mode, "capture");
        assert.equal(capture.routing.initialization, undefined);
        const replay = await cli(["--model", model, "--sidecar", "--port", String(PORT)]);
        assert.equal(replay.routing.exec.acceleration.graphs.mode, "replay");
        assert.equal(replay.answers.department.choice, r.answers.department.choice);
      }
    }
  } finally { await stop({ port: PORT, timeoutMs: 30_000 }); }
});
