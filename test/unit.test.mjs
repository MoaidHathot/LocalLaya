import { test } from "node:test";
import assert from "node:assert/strict";
import { chooseLane, LatencyModel, LayaRouter, nBucket, thermalState, DEFAULT_PRIORS, estimateWork } from "../src/ep-router.mjs";
import { parseDuration, SidecarError } from "../src/sidecar-client.mjs";
import { bucketKey, fitTemperature, metrics, optionCount, optionLabels, softmax } from "../src/calibration.mjs";
import { presetFromJson, presetToJson, PRESETS } from "../data/presets.mjs";
import { buildSessionConfig, parseWebgpuOptions, pCoreAffinity } from "../src/laya-client.mjs";

test("buildSessionConfig: EP selection, thread pinning, WebGPU provider options", () => {
  // cpu: pinned pool = 16 threads with an affinity string of 15 entries (the caller is the 16th)
  const cpu = buildSessionConfig("cpu", { pinToPCores: true });
  assert.deepEqual(cpu.executionProviders, ["cpu"]);
  assert.equal(cpu.sessionOptions.intraOpNumThreads, 16);
  assert.equal(cpu.sessionOptions.extra.session.intra_op_thread_affinities.split(";").length, 15);
  assert.equal(pCoreAffinity(8, 16), "1;3;5;7;9;11;13"); // 7 workers on distinct physical cores, 1-based ids
  // cpu:8 unpinned: no affinity
  assert.equal(buildSessionConfig("cpu", { threads: 8 }).sessionOptions.extra, undefined);
  // webgpu without options stays the plain string form (what the library used before)
  assert.deepEqual(buildSessionConfig("webgpu").executionProviders, ["webgpu"]);
  // webgpu with options: object form with the options inline, undefined values dropped
  const w = buildSessionConfig("webgpu", { webgpuOptions: { validationMode: "disabled", storageBufferCacheMode: "bucket", enableRobustness: false, preferredLayout: undefined } });
  assert.deepEqual(w.executionProviders, [{ name: "webgpu", validationMode: "disabled", storageBufferCacheMode: "bucket", enableRobustness: false }]);
  // unknown keys are rejected here rather than by the native binding at session creation
  assert.throws(() => buildSessionConfig("webgpu", { webgpuOptions: { enableGraphCapture: true } }), /unknown WebGPU EP option "enableGraphCapture"/);
  assert.throws(() => buildSessionConfig("nope"), /unsupported ep/);
  // the CLI spec form
  assert.deepEqual(parseWebgpuOptions("validationMode=disabled, enableRobustness=false,forceCpuNodeNames=a;b"), { validationMode: "disabled", enableRobustness: false, forceCpuNodeNames: ["a", "b"] });
  assert.deepEqual(parseWebgpuOptions(""), {});
  assert.throws(() => parseWebgpuOptions("bogus=1"), /unknown WebGPU EP option/);
  assert.throws(() => parseWebgpuOptions("novalue"), /key=value/);
});

test("nBucket and thermalState boundaries", () => {
  assert.equal(nBucket(1), "1");
  assert.equal(nBucket(3), "2-3");
  assert.equal(nBucket(4), "4-6");
  assert.equal(nBucket(10), "7-10");
  assert.equal(nBucket(11), "11+");
  assert.equal(thermalState(0), "hot");
  assert.equal(thermalState(399), "hot");
  assert.equal(thermalState(400), "warm");
  assert.equal(thermalState(1999), "warm");
  assert.equal(thermalState(2000), "cold");
  assert.equal(thermalState(Infinity), "cold");
});

test("LatencyModel: prior, EMA update, neighbour scaling, work normalisation", () => {
  const m = new LatencyModel("webgpu");
  // without a work estimate the prior is returned at its reference work
  assert.deepEqual(m.predict(3, "cold"), { ms: DEFAULT_PRIORS.webgpu.ms.cold["2-3"], source: "prior" });
  m.observe(3, "hot", 50);
  assert.ok(Math.abs(m.predict(3, "hot").ms - 50) < 1e-9);
  m.observe(3, "hot", 60); // rate EMA: 0.3*60 + 0.7*50 = 53 at reference work
  assert.ok(Math.abs(m.predict(3, "hot").ms - 53) < 1e-9);
  // twice the work -> twice the predicted time
  assert.ok(Math.abs(m.predict(3, "hot", 2 * 3 * 85).ms - 106) < 1e-9);
  // unseen bucket in the same state: prior scaled by observed/prior ratio of a seen bucket
  const p = m.predict(10, "hot");
  assert.equal(p.source, "ema-scaled(from 2-3)");
  assert.ok(Math.abs(p.ms - (53 / 55) * 136) < 1e-9);
  // cpu lanes ignore the thermal state
  const c = new LatencyModel("cpu");
  c.observe(1, "cold", 100);
  assert.ok(Math.abs(c.predict(1, "hot").ms - 100) < 1e-9);
});

test("estimateWork grows with questions and state length, capped by max_len", () => {
  const q1 = { a: { type: "noul", instructions: "x" } };
  const q3 = { ...q1, b: { type: "choice", instructions: "y", criteria: { p: "1", q: "2" } }, c: { type: "score", instructions: "z", criteria: ["l", "h"] } };
  const short = { m: "hi" };
  const long = { m: "word ".repeat(400) };
  assert.ok(estimateWork(short, q3) > estimateWork(short, q1));
  assert.ok(estimateWork(long, q1) > estimateWork(short, q1));
  assert.ok(estimateWork(long, q1) <= 512);
  assert.ok(estimateWork(long, q3) <= 3 * 512);
});

test("chooseLane: auto picks fastest; exploration stays within 2x of the best lane's own latency", () => {
  const cands = [
    { lane: "cpu", predictedMs: 105, share: 0.65 },
    { lane: "webgpu", predictedMs: 180, share: 0.02 },
  ];
  assert.equal(chooseLane(cands, { policy: "auto", explore: 0 }).lane, "cpu");
  // rng < explore -> explore; the alternative (180) is within 2x of 105 -> chosen
  const seq = [0.01, 0.0];
  const explored = chooseLane(cands, { policy: "auto", explore: 0.5, rng: () => seq.shift() });
  assert.equal(explored.lane, "webgpu");
  assert.equal(explored.explored, true);
  // alternative too slow (>2x) -> never explored
  const far = [
    { lane: "cpu", predictedMs: 100, share: 0.65 },
    { lane: "webgpu", predictedMs: 500, share: 0.02 },
  ];
  assert.equal(chooseLane(far, { policy: "auto", explore: 1, rng: () => 0 }).lane, "cpu");
  // with a shared queue wait the 2x test uses the lanes' own latencies, not wait + own
  const queued = [
    { lane: "webgpu", predictedMs: 600, ownMs: 100, share: 0.02 },
    { lane: "cpu", predictedMs: 800, ownMs: 300, share: 0.65 },
  ];
  assert.equal(chooseLane(queued, { policy: "auto", explore: 1, rng: () => 0 }).lane, "webgpu");
});

test("chooseLane: policies and deadline", () => {
  const cands = [
    { lane: "cpu", predictedMs: 105, share: 0.65 },
    { lane: "cpu:8", predictedMs: 116, share: 0.22 },
    { lane: "webgpu", predictedMs: 180, share: 0.02 },
  ];
  assert.equal(chooseLane(cands, { policy: "prefer-gpu" }).lane, "webgpu");
  assert.equal(chooseLane(cands, { policy: "prefer-cpu" }).lane, "cpu");
  assert.equal(chooseLane(cands, { policy: "min-cpu" }).lane, "webgpu");
  // deadline 150: cpu and cpu:8 meet it; least CPU share among them -> cpu:8
  assert.equal(chooseLane(cands, { deadlineMs: 150 }).lane, "cpu:8");
  // deadline 200: all meet it -> webgpu (least share)
  assert.equal(chooseLane(cands, { deadlineMs: 200 }).lane, "webgpu");
  // deadline nobody meets -> fastest
  assert.equal(chooseLane(cands, { deadlineMs: 50 }).lane, "cpu");
  assert.throws(() => chooseLane([], {}));
});

test("calibration helpers", () => {
  assert.equal(bucketKey("choice", 4), "choice:3-5");
  assert.equal(bucketKey("noul", 2), "noul:2");
  assert.equal(bucketKey("choice", 12), "choice:11+");
  assert.equal(optionCount({ type: "noul", instructions: "" }), 2);
  assert.equal(optionCount({ type: "score", instructions: "", criteria: ["a", "b", "c"] }), 3);
  assert.deepEqual(optionLabels({ type: "choice", instructions: "", criteria: { x: "1", y: "2" } }), ["x", "y"]);
  const p = softmax([0, 0, 0, 0]);
  assert.ok(p.every((v) => Math.abs(v - 0.25) < 1e-12));
  // perfectly calibrated & confident set: ECE ~0, NLL ~0
  const m = metrics([{ probs: [0.99, 0.01], gold: 0 }, { probs: [0.01, 0.99], gold: 1 }]);
  assert.equal(m.accuracy, 1);
  assert.ok(m.ece < 0.02);
  // temperature fit: sharp logits that are right 50% of the time -> fitted T flattens (T > 1)
  const samples = [];
  for (let i = 0; i < 40; i++) samples.push({ logits: [4, 0], gold: i % 2 });
  const { T } = fitTemperature(samples);
  assert.ok(T > 5, `expected a large temperature for uninformative logits, got ${T}`);
});

test("presets: JSON form <-> preset object", () => {
  const q = { kind: { type: "choice", instructions: "?", criteria: { a: "x", b: "y" } } };
  // default template
  let p = presetFromJson({ questions: q });
  assert.deepEqual(p.state("hello"), { text: "hello" });
  // string = single key
  p = presetFromJson({ state: "request", questions: q });
  assert.deepEqual(p.state("hello"), { request: "hello" });
  // object template with literal fields and nesting
  p = presetFromJson({ state: { msg: "$TEXT", app: "x", meta: { src: "cli", body: "$TEXT" } }, questions: q });
  assert.deepEqual(p.state("hi"), { msg: "hi", app: "x", meta: { src: "cli", body: "hi" } });
  // validation
  assert.throws(() => presetFromJson({ questions: {} }));
  assert.throws(() => presetFromJson({ state: { a: "no marker" }, questions: q }));
  // round trip of a built-in preset with extra fields
  const json = presetToJson(PRESETS.triage, PRESETS.triage.questions, { product: "MyApp" }, "desc");
  assert.deepEqual(json.state, { message: "$TEXT", product: "MyApp" });
  assert.equal(json.description, "desc");
  const back = presetFromJson(json);
  assert.deepEqual(back.state("t"), { message: "t", product: "MyApp" });
  assert.deepEqual(back.questions, PRESETS.triage.questions);
});
test("parseDuration", () => {
  assert.equal(parseDuration("0"), 0);
  assert.equal(parseDuration("30s"), 30_000);
  assert.equal(parseDuration("5m"), 300_000);
  assert.equal(parseDuration("1.5h"), 5_400_000);
  assert.equal(parseDuration("250ms"), 250);
  assert.equal(parseDuration("300"), 300_000); // bare number = seconds
  assert.equal(parseDuration(1234), 1234);
  assert.throws(() => parseDuration("soon"), (e) => e instanceof SidecarError && e.code === "BAD_DURATION");
  assert.throws(() => parseDuration(""), SidecarError);
});

test("router predictions: one queue for all lanes - shared wait, choice by own latency, GPU state at start", () => {
  // a router with two fake lanes and no model loaded
  const r = Object.create(LayaRouter.prototype);
  r.lanes = new Map();
  r.inflight = [];
  r.load = { cpuOthers: 0, gpuOthersUtil: 0 };
  r.lastGpuWorkEnd = performance.now(); // GPU hot
  const mk = (lane) => ({ lane, model: new LatencyModel(lane), healthy: true, quarantinedUntil: 0, pending: 0 });
  r.lanes.set("webgpu", mk("webgpu"));
  r.lanes.set("cpu", mk("cpu"));
  const by = (preds) => Object.fromEntries(preds.map((p) => [p.lane, p]));
  const idle = by(r.predictions(3));
  assert.ok(idle.webgpu.predictedMs < idle.cpu.predictedMs, "idle: GPU is the faster lane");
  assert.equal(idle.webgpu.waitMs, 0);
  assert.equal(idle.webgpu.predictedMs, idle.webgpu.ownMs);
  // three GPU calls queued: every lane waits for all of them (one queue); the GPU stays the choice - spilling
  // to the CPU would only insert a 260 ms stall in front of the GPU calls behind it
  r.inflight = [1, 2, 3].map(() => ({ lane: "webgpu", ownMs: 55, startedAt: null }));
  let busy = r.predictions(3);
  const b = by(busy);
  assert.ok(Math.abs(b.webgpu.waitMs - 165) < 1e-9);
  assert.ok(Math.abs(b.cpu.waitMs - 165) < 1e-9);
  assert.ok(Math.abs(b.webgpu.predictedMs - (165 + idle.webgpu.ownMs)) < 1e-9);
  assert.ok(Math.abs(b.cpu.predictedMs - (165 + idle.cpu.ownMs)) < 1e-9);
  assert.equal(b.webgpu.pending, 3);
  assert.equal(chooseLane(busy, { explore: 0 }).lane, "webgpu");
  r.inflight = Array.from({ length: 20 }, () => ({ lane: "webgpu", ownMs: 55, startedAt: null }));
  assert.equal(chooseLane(r.predictions(3), { explore: 0 }).lane, "webgpu", "a deep queue never spills to the CPU");
  // the running call counts only its remaining time
  r.inflight = [{ lane: "webgpu", ownMs: 100, startedAt: performance.now() - 40 }];
  const remaining = r.waitMs();
  assert.ok(remaining > 50 && remaining <= 60, `remaining ${remaining}`);
  r.inflight = [{ lane: "webgpu", ownMs: 100, startedAt: performance.now() - 500 }];
  assert.equal(r.waitMs(), 0, "an overrunning call does not go negative");
  // GPU state when the call will start: GPU work queued ahead -> hot even if the GPU is cold now
  r.lastGpuWorkEnd = -Infinity;
  r.inflight = [{ lane: "webgpu", ownMs: 180, startedAt: null }];
  assert.equal(by(r.predictions(1)).webgpu.state, "hot");
  // only CPU work queued ahead of a cold GPU -> priced "warm" (traffic warms the clocks), not cold
  r.inflight = [{ lane: "cpu", ownMs: 260, startedAt: null }];
  const w = by(r.predictions(1));
  assert.equal(w.webgpu.state, "warm");
  assert.ok(Math.abs(w.webgpu.predictedMs - (260 + DEFAULT_PRIORS.webgpu.ms.warm["1"])) < 1e-9);
  // idle and cold: the cold prior applies and the CPU wins a single question (sporadic traffic)
  r.inflight = [];
  const cold = r.predictions(1);
  assert.equal(by(cold).webgpu.state, "cold");
  assert.equal(chooseLane(cold, { explore: 0 }).lane, "cpu");
  // contention inflation applies to the lane's own latency, not to the shared wait
  r.lastGpuWorkEnd = performance.now();
  r.inflight = [{ lane: "webgpu", ownMs: 55, startedAt: null }];
  r.load.cpuOthers = 0.5;
  const c = by(r.predictions(3)).cpu;
  assert.ok(Math.abs(c.predictedMs - (55 + idle.cpu.ownMs * 2)) < 1e-9);
});