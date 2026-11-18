import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ModelRouter } from "../src/model-router.mjs";
import { PythonBackend } from "../src/python-backend.mjs";
import { modelOptions, PROJECT_ROOT } from "../src/models.mjs";
import { CalibrationCache, decideRequest } from "../src/decide-core.mjs";
import { decide } from "../src/sidecar-client.mjs";
import { runtimeProfile } from "../src/runtime-profile.mjs";
import { LayaRouter } from "../src/ep-router.mjs";

const questions = { q: { type: "noul", instructions: "Is this a request?" } };
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };

test("model controls refuse unsupported combinations and reset budgets when selecting another model", () => {
  const defaults = modelOptions({ model: "vega-u", device: "cuda", readout: "engine" });
  assert.equal(defaults.maxLen, 73728);
  assert.deepEqual(modelOptions({ model: "laya-multilingual" }, defaults), { model: "multilingual", device: "cuda", maxLen: 8192, readout: "auto", acceleration: "auto", graphs: "auto" });
  assert.equal(modelOptions({ model: "multilingual", acceleration: "off" }).acceleration, "off");
  assert.equal(modelOptions({ model: "multilingual", acceleration: "on" }).acceleration, "on");
  for (const o of [{ model: "missing" }, { model: ["vega"] }, { model: "vega", maxLen: 73729 }, { model: "vega", maxLen: "8192" },
    { model: "laya", maxLen: 8192 }, { model: "vega", lane: "cpu:8" }, { model: "multilingual", exec: { graph: false } },
    { model: "vega", policy: "auto" }, { model: "vega", calibration: { temperature_by_options: {} } },
    { model: "multilingual", readout: "engine" }, { model: "vega", readout: "ttt" }, { device: "webgpu" },
    { model: "vega", acceleration: "unknown" }, { model: "laya", acceleration: "auto" },
    { graphs: "maybe" }, { model: "multilingual", acceleration: "off", graphs: "on" }]) {
    assert.throws(() => modelOptions(o), (e) => e.status === 400, JSON.stringify(o));
  }
});

test("automatic profiles follow lifetime and preserve explicit runtime choices", () => {
  for (const model of ["laya", "vega", "multilingual"]) {
    const selection = modelOptions({ model });
    const once = runtimeProfile(selection, { lifetime: "one-shot" });
    const reusable = runtimeProfile(selection);
    assert.equal(once.graphReplay, false);
    assert.equal(once.sampleLoad, false);
    assert.equal(once.gpuKeepAliveMs, 0);
    assert.equal(reusable.graphReplay, true);
    assert.equal(runtimeProfile(modelOptions({ model, graphs: "on" }), { lifetime: "one-shot" }).graphReplay, true);
    assert.equal(runtimeProfile(modelOptions({ model, graphs: "off" })).graphReplay, false);
    assert.equal(runtimeProfile(modelOptions({ model, device: "cpu" })).graphReplay, false);
  }
  const laya = modelOptions({ model: "laya" });
  assert.deepEqual(runtimeProfile(laya, { lifetime: "one-shot" }).lanes, ["cuda:fp16", "webgpu:fp16", "cpu:8"]);
  assert.equal(runtimeProfile(laya, { lifetime: "one-shot" }).loadStrategy, "first-available");
  assert.equal(runtimeProfile(laya).loadStrategy, "parallel");
  const manual = runtimeProfile(laya, { lifetime: "one-shot", lanes: ["webgpu:fp16", "cpu:8"], sampleLoad: true, gpuKeepAliveMs: 2000 });
  assert.deepEqual(manual.lanes, ["webgpu:fp16", "cpu:8"]);
  assert.equal(manual.loadStrategy, "parallel");
  assert.equal(manual.sampleLoad, true);
  assert.equal(manual.gpuKeepAliveMs, 2000);
  assert.deepEqual(runtimeProfile(laya, { lifetime: "one-shot", lane: "cpu:16" }).lanes, ["cpu:16"]);
  assert.equal(runtimeProfile(laya, { lifetime: "one-shot", policy: "prefer-cpu" }).loadStrategy, "parallel");
  assert.equal(runtimeProfile(laya, { lifetime: "one-shot", deadlineMs: 50 }).loadStrategy, "parallel");
  assert.equal(runtimeProfile(laya, { cudaGraph: false }).graphReplay, false);
  assert.equal(runtimeProfile(modelOptions({ graphs: "on" }), { cudaGraph: false }).graphReplay, true);
  assert.throws(() => runtimeProfile(laya, { lifetime: "unknown" }), /lifetime/);
});

test("Multilingual one-shot auto avoids TileLang startup; explicit acceleration and graphs still work", () => {
  const profile = (o = {}, lifetime = "one-shot") => runtimeProfile(modelOptions({ model: "multilingual", ...o }), { lifetime });
  const automatic = profile();
  assert.equal(automatic.acceleration, "off");
  assert.equal(automatic.requestedAcceleration, "auto");
  assert.match(automatic.reason, /skips TileLang/);
  assert.equal(profile({ graphs: "off" }).acceleration, "off");
  assert.equal(profile({}, "persistent").acceleration, "on");
  assert.equal(profile({ acceleration: "on" }).acceleration, "on");
  assert.equal(profile({ acceleration: "on" }).graphReplay, false, "forcing kernels need not pay for graph capture");
  assert.equal(profile({ graphs: "on" }).acceleration, "on", "explicit graphs also enables its required backend");
  assert.equal(profile({ acceleration: "off" }, "persistent").acceleration, "off");
  assert.equal(profile({ acceleration: "off" }, "persistent").graphReplay, false);
  for (const lifetime of ["one-shot", "persistent"]) {
    const cpu = profile({ device: "cpu", acceleration: "on", graphs: "on" }, lifetime);
    assert.equal(cpu.acceleration, "off", "explicit CPU uses eager even with the optional CUDA kernels installed");
    assert.equal(cpu.graphReplay, false);
    assert.equal(cpu.requestedAcceleration, "on");
  }
  assert.equal(runtimeProfile(modelOptions({ model: "vega" }), { lifetime: "one-shot" }).acceleration, "on");
});

test("one-shot lane loading selects the first usable provider and reaches the CPU fallback", async () => {
  for (const available of ["cuda:fp16", "webgpu:fp16", "cpu:8", null]) {
    const r = Object.create(LayaRouter.prototype);
    r.opts = { lanes: ["cuda:fp16", "webgpu:fp16", "cpu:8"], loadStrategy: "first-available" };
    r.lanes = new Map(); r.loading = new Set(r.opts.lanes);
    const attempts = [];
    r._probeLane = async (lane) => { attempts.push(lane); if (lane === available) { r.lanes.set(lane, {}); return true; } return false; };
    if (available) {
      await r._loadConfiguredLanes(null);
      assert.deepEqual([...r.lanes.keys()], [available]);
      assert.deepEqual(attempts, r.opts.lanes.slice(0, r.opts.lanes.indexOf(available) + 1));
    } else await assert.rejects(r._loadConfiguredLanes(null), /no lane/);
    assert.equal(r.loading.size, 0);
  }
});

class FakeRouter extends ModelRouter {
  async open(selection) {
    this.events ??= [];
    this.events.push(`open:${selection.model}:${selection.device}:${selection.maxLen}`);
    if (selection.device === "cuda" && this.failCuda) throw new Error("CUDA unavailable");
    const lane = selection.model === "laya" ? "cpu:8" : `${selection.model}:${selection.device}`;
    return {
      lanes: new Map([[lane, {}]]), pendingLanes: [], ready: Promise.resolve(),
      decide: async (state, qs, opts) => {
        this.events.push(`start:${state}`);
        if (state === "held") { this.started.resolve(); await this.release.promise; }
        if (state === "fail") throw new Error("inference failed");
        this.events.push(`end:${state}`);
        this.lastOptions = opts;
        this.lastQuestions = qs;
        return { answers: { q: { type: "noul", noul: 0.5 } }, usage: { input_tokens: 10 }, routing: { lane, device: selection.device === "auto" ? "cpu" : selection.device, ms: 1, queueMs: 0 } };
      },
      close: async () => { this.events.push(`close:${selection.model}`); },
    };
  }
}

test("all model selections receive canonical questions and malformed aliases never switch backends", async () => {
  const aliases = {
    refund: { type: "predicate", instructions: "Refund requested?" },
    team: { type: "choice", instructions: "Team?", choices: { billing: "refunds", support: "other" } },
    severity: { type: "score", instructions: "Severity?", levels: ["Low", "High"] },
  };
  const before = structuredClone(aliases);
  const r = await FakeRouter.create({ model: "laya", device: "cpu" });
  try {
    for (const model of ["laya", "vega", "multilingual"]) {
      await r.decide("ticket", aliases, { model });
      assert.equal(r.lastQuestions.refund.type, "noul");
      assert.deepEqual(r.lastQuestions.team.criteria, aliases.team.choices);
      assert.deepEqual(r.lastQuestions.severity.criteria, aliases.severity.levels);
      assert.equal(r.lastQuestions.team.choices, undefined);
    }
    const opens = r.events.filter(e => e.startsWith("open:")).length;
    await assert.rejects(r.decide("bad ticket", { team: { ...aliases.team, criteria: ["different", "options"] } }, { model: "vega" }), (e) => e.status === 400 && /conflict/.test(e.message));
    assert.equal(r.active.selection.model, "multilingual");
    assert.equal(r.events.filter(e => e.startsWith("open:")).length, opens);
    assert.equal(r.pending, 0);
    assert.deepEqual(aliases, before);
  } finally { await r.close(); }
});

test("effective graph policy controls reuse and reports one-shot versus persistent configuration", async () => {
  const r = await FakeRouter.create({ model: "vega", device: "cuda", lifetime: "one-shot" });
  try {
    const a = await r.decide("first", questions);
    assert.equal(a.routing.profile.lifetime, "one-shot");
    assert.equal(a.routing.profile.graphs, "off");
    await r.decide("explicit off", questions, { graphs: "off" });
    assert.equal(r.events.filter(e => e.startsWith("open:")).length, 1, "same effective policy reuses the loaded model");
    const b = await r.decide("explicit on", questions, { graphs: "on" });
    assert.equal(b.routing.profile.graphs, "on");
    assert.equal(r.events.filter(e => e.startsWith("open:")).length, 2);
  } finally { await r.close(); }
});

test("model switching waits for inference, keeps FIFO order, reuses weights across readouts, and drains on close", async () => {
  const r = await FakeRouter.create({ model: "vega", device: "cpu" });
  r.started = deferred(); r.release = deferred();
  const first = r.decide("held", questions);
  await r.started.promise;
  const second = r.decide("other", questions, { model: "multilingual" });
  const third = r.decide("original", questions, { model: "laya" });
  assert.equal(r.events.filter((e) => e.startsWith("close:")).length, 0, "a model cannot close under a running call");
  r.release.resolve();
  const results = await Promise.all([first, second, third]);
  assert.deepEqual(results.map((v) => v.routing.model), ["vega", "multilingual", "laya"]);
  assert.ok(r.events.indexOf("end:held") < r.events.indexOf("close:vega"));
  assert.ok(r.events.indexOf("end:other") < r.events.indexOf("close:multilingual"));
  assert.ok(results[1].routing.queueMs > 0);
  await r.decide("again", questions, { model: "vega" });
  const opens = r.events.filter((e) => e.startsWith("open:")).length;
  await r.decide("engine", questions, { model: "vega", readout: "engine" });
  assert.equal(r.lastOptions.readout, "engine");
  assert.equal(r.events.filter((e) => e.startsWith("open:")).length, opens);
  await r.decide("reference", questions, { model: "vega", acceleration: "off" });
  assert.equal(r.lastOptions.acceleration, "off");
  assert.equal(r.events.filter((e) => e.startsWith("open:")).length, opens + 1, "changing acceleration reloads the Python worker");
  const accepted = r.decide("accepted", questions, { model: "multilingual", maxLen: 1024 });
  const closing = r.close();
  await assert.rejects(r.decide("late", questions), /closed/);
  await accepted; await closing;
  assert.ok(r.events.indexOf("end:accepted") < r.events.lastIndexOf("close:multilingual"));
});

test("model/device failures are surfaced, never answered by another model; a later selection recovers", async () => {
  const r = await FakeRouter.create({ model: "laya", device: "cpu" });
  try {
    r.failCuda = true;
    await assert.rejects(r.decide("request", questions, { model: "vega", device: "cuda" }), /CUDA unavailable/);
    assert.equal(r.backend, null);
    assert.equal(r.events.filter((e) => e.startsWith("start:")).length, 0);
    r.failCuda = false;
    assert.equal((await r.decide("retry", questions, { model: "vega", device: "cuda" })).routing.model, "vega", "the same selection can retry after setup is repaired");
    const recovered = await r.decide("request", questions, { model: "laya" });
    assert.equal(recovered.routing.model, "laya");
    await assert.rejects(r.decide("fail", questions, { model: "vega", device: "cpu" }), /inference failed/);
    assert.equal(r.active.selection.model, "vega");
    await assert.rejects(r.decide("request", questions, { model: "laya", lane: "cpu:99" }), (e) => e.status === 400 && /not loaded/.test(e.message));
  } finally { await r.close(); }
});

test("the original EP router receives a whole burst for its own queue predictions; switching waits for every admitted call", async () => {
  const r = await FakeRouter.create({ model: "laya", device: "cpu" });
  r.started = deferred(); r.release = deferred();
  const held = r.decide("held", questions);
  await r.started.promise;
  // This fake backend deliberately does not serialize: admission of the second request proves
  // the wrapper lets the real EP router see and price its own queue, rather than hiding it.
  await r.decide("burst", questions, { deadlineMs: 200 });
  assert.equal(r.lastOptions.deadlineMs, 200);
  const switchModel = r.decide("next-model", questions, { model: "vega" });
  await Promise.resolve();
  assert.ok(!r.events.includes("close:laya"));
  r.release.resolve();
  await held; await switchModel; await r.close();
  assert.ok(r.events.indexOf("close:laya") > r.events.indexOf("end:held"));
});

test("preset calibration is model-scoped, including server overrides and request null", async () => {
  const base = path.join(PROJECT_ROOT, ".laya");
  await mkdir(base, { recursive: true });
  const root = await mkdtemp(path.join(base, "calibration-test-"));
  const r = await FakeRouter.create({ model: "laya", device: "cpu" });
  try {
    await mkdir(path.join(root, "calibration/multilingual"), { recursive: true });
    await writeFile(path.join(root, "calibration/triage.json"), JSON.stringify({ temperature_by_options: { "noul:2": 4 } }));
    await writeFile(path.join(root, "calibration/multilingual/triage.json"), JSON.stringify({ temperature_by_options: { "noul:2": 2 } }));
    const cache = new CalibrationCache({ root, override: "calibration/triage.json" });
    const presets = { triage: { questions, state: (text) => ({ message: text }) } };
    const body = { text: "test", preset: "triage" };
    const old = await decideRequest(r, body, presets, cache);
    assert.equal(old.calibration, "calibration/triage.json");
    assert.equal(r.lastOptions.calibration.temperature_by_options["noul:2"], 4);
    const vega = await decideRequest(r, { ...body, model: "vega" }, presets, cache);
    assert.equal(vega.calibration, null);
    assert.equal(r.lastOptions.calibration, null);
    const ml = await decideRequest(r, { ...body, model: "multilingual" }, presets, cache);
    assert.equal(ml.calibration, "calibration/multilingual/triage.json");
    assert.equal(r.lastOptions.calibration.temperature_by_options["noul:2"], 2);
    await decideRequest(r, { ...body, calibration: null }, presets, cache);
    assert.equal(r.lastOptions.calibration, null);
    await assert.rejects(decideRequest(r, { ...body, model: "vega", calibration: { temperature_by_options: {} } }, presets, cache), /cannot be applied/);
    await assert.rejects(decideRequest({}, { ...body, model: "vega" }, presets, cache), /requires ModelRouter/);
  } finally { await r.close(); await rm(root, { recursive: true, force: true }); }
});

const fixture = fileURLToPath(new URL("./fixtures/python-worker.mjs", import.meta.url));
const openFixture = (timeoutMs = 5000) => PythonBackend.create(modelOptions({ model: "vega", device: "cpu" }), { python: process.execPath, script: fixture, timeoutMs });

test("Python protocol: valid typed answers, recoverable input errors and malformed-response rejection", async () => {
  const b = await openFixture();
  try {
    const r = await b.decide("ok", questions, { readout: "engine" });
    assert.equal(r.routing.model, "vega");
    assert.equal(r.routing.exec.readout, "engine");
    await assert.rejects(b.decide("bad-input", questions), (e) => e.status === 400);
    assert.equal((await b.decide("ok", questions)).answers.q.noul, 0.75);
    await assert.rejects(b.decide("malformed", questions), /missing answers/);
    assert.equal(b.dead, true);
  } finally { await b.close(); }
});

test("Python startup gets resolved controls and reports requested controls and first-use timings", async () => {
  const b = await PythonBackend.create(modelOptions({ model: "vega" }), {
    python: process.execPath, script: fixture, graphReplay: false, acceleration: "off", accelerationReason: "startup policy",
  });
  try {
    assert.equal(b.info.requestedGraphs, "off");
    assert.equal(b.info.workerAcceleration, "off");
    assert.equal(b.info.acceleration.requested, "auto");
    assert.equal(b.info.acceleration.selected, "off");
    const [first, second] = await Promise.all([b.decide("first", questions), b.decide("second", questions)]);
    assert.equal(first.routing.exec.acceleration.requested, "auto");
    assert.equal(first.routing.exec.acceleration.selected, "off");
    assert.equal(first.routing.exec.acceleration.reason, "startup policy");
    assert.ok(first.routing.initialization.processMs > 0);
    assert.deepEqual(first.routing.initialization.phases, b.info.startup.phases);
    assert.equal(second.routing.initialization, undefined, "startup cost is reported once, even with concurrent callers");
  }
  finally { await b.close(); }
});

test("Python process exit, timeout and missing executable reject pending callers and release the child", async () => {
  for (const state of ["die", "hang"]) {
    const b = await openFixture(state === "hang" ? 1000 : 5000);
    try { await assert.rejects(b.decide(state, questions), /exited|timed out/); }
    finally { await b.close(); }
    assert.equal(b.dead, true);
    assert.equal(b.pending.size, 0);
  }
  await assert.rejects(PythonBackend.create(modelOptions({ model: "vega" }), { python: path.join(PROJECT_ROOT, ".no-python-executable") }), /cannot start/);
  await assert.rejects(PythonBackend.create(modelOptions({ model: "multilingual" }), { python: process.execPath, script: fixture }), /different model/);
});

test("an older sidecar cannot silently ignore model or acceleration selection", async () => {
  let decisions = 0;
  let advertisesModels = false;
  let advertisesVegaAcceleration = false;
  const server = createServer((req, res) => {
    if (req.url === "/decide") decisions++;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ service: "laya", status: "ready", pid: process.pid, lanes: ["cpu:8"], ...(advertisesModels ? { models: { available: { vega: advertisesVegaAcceleration ? { accelerations: ["auto", "off"] } : {}, multilingual: {} } } } : {}) }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    await assert.rejects(decide({ model: "vega", text: "test" }, { port: server.address().port }), (e) => e.code === "MODEL_UNSUPPORTED");
    advertisesModels = true;
    await assert.rejects(decide({ model: "vega", acceleration: "off", text: "test" }, { port: server.address().port }), (e) => e.code === "MODEL_UNSUPPORTED");
    advertisesVegaAcceleration = true;
    await assert.rejects(decide({ model: "vega", acceleration: "on", text: "test" }, { port: server.address().port }), (e) => e.code === "MODEL_UNSUPPORTED");
    await assert.rejects(decide({ model: "multilingual", acceleration: "off", text: "test" }, { port: server.address().port }), (e) => e.code === "MODEL_UNSUPPORTED");
    assert.equal(decisions, 0);
  } finally { await new Promise((resolve) => server.close(resolve)); }
});
