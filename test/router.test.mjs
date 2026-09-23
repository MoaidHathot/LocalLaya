/**
 * Integration test for the router with lanes in worker threads (needs the model bundle + GPU; ~1 minute).
 *
 *   npm run test:router          (= node --test --test-concurrency=1 test/router.test.mjs)
 *
 * Scenarios: worker lanes answer exactly like in-process lanes and keep the main thread responsive during a
 * CPU inference / per-call calibration override reaches the worker and is reset afterwards / a worker that dies
 * mid-call: the call is retried on another lane and the lane is marked gone / waitFor "first" serves before
 * every lane is loaded / a lane that fails to load is dropped, all lanes failing rejects / close() ends the
 * workers.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { LayaRouter } from "../src/ep-router.mjs";
import { STATE, QUESTIONS_3, QUESTIONS_10 } from "../src/questions.mjs";

const LANES = ["webgpu:fp16", "cpu:8"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const quiet = () => {};
const stateFor = (i) => ({ ...STATE, userMessage: `${STATE.userMessage} (${i})` });

/** Largest gap between 5 ms timer ticks while `fn` runs = how long the main thread was blocked. */
async function maxStall(fn) {
  let last = performance.now();
  let gap = 0;
  const t = setInterval(() => {
    const now = performance.now();
    gap = Math.max(gap, now - last);
    last = now;
  }, 5);
  try {
    await fn();
    // a blocking call resolves before the starved timer gets to run: count the time since the last tick too
    gap = Math.max(gap, performance.now() - last);
  } finally {
    clearInterval(t);
  }
  return gap;
}

const closeTo = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
function assertSameAnswers(a, b) {
  assert.deepEqual(Object.keys(a), Object.keys(b));
  for (const id of Object.keys(a)) {
    assert.equal(a[id].type, b[id].type);
    if (a[id].type === "choice") assert.equal(a[id].choice, b[id].choice, `choice for ${id}`);
    if (a[id].type === "noul") closeTo(a[id].noul, b[id].noul, 1e-3, `noul ${id}`);
    if (a[id].type === "score") closeTo(a[id].score, b[id].score, 1e-2, `score ${id}`);
    if (a[id].probabilities) for (const k of Object.keys(a[id].probabilities)) closeTo(a[id].probabilities[k], b[id].probabilities[k], 1e-3, `${id}.${k}`);
  }
}

test("worker lanes: same answers as in-process lanes, main thread stays responsive during a CPU inference", async () => {
  const inProc = await LayaRouter.create({ lanes: LANES, workers: false, sampleLoad: false, log: quiet });
  const workers = await LayaRouter.create({ lanes: LANES, workers: true, sampleLoad: false, log: quiet });
  try {
    assert.deepEqual([...inProc.lanes.keys()].sort(), [...LANES].sort());
    assert.deepEqual([...workers.lanes.keys()].sort(), [...LANES].sort());
    for (const lane of LANES) {
      assert.equal(inProc.stats().lanes[lane].mode, "in-process");
      assert.equal(workers.stats().lanes[lane].mode, "worker");
      const a = await inProc.decide(STATE, QUESTIONS_3, { lane });
      const b = await workers.decide(STATE, QUESTIONS_3, { lane });
      assert.equal(a.routing.lane, lane);
      assert.equal(b.routing.lane, lane);
      assertSameAnswers(a.answers, b.answers);
      assert.equal(a.usage.input_tokens, b.usage.input_tokens);
    }
    // 10 CPU questions take ~1 s: in-process that blocks the event loop for the whole call, in a worker it does not
    const stallInProc = await maxStall(() => inProc.decide(STATE, QUESTIONS_10, { lane: "cpu:8" }));
    const stallWorkers = await maxStall(() => workers.decide(STATE, QUESTIONS_10, { lane: "cpu:8" }));
    console.log(`      main-thread max stall during a 10-question cpu:8 call: in-process ${stallInProc.toFixed(0)} ms, worker ${stallWorkers.toFixed(0)} ms`);
    assert.ok(stallInProc > 200, `in-process inference should block the main thread (${stallInProc.toFixed(0)} ms)`);
    assert.ok(stallWorkers < 100, `worker inference should not block the main thread (${stallWorkers.toFixed(0)} ms)`);
    // the worker round trip costs about nothing
    const inMs = (await inProc.decide(STATE, QUESTIONS_3, { lane: "cpu:8" })).routing.ms;
    const wMs = (await workers.decide(STATE, QUESTIONS_3, { lane: "cpu:8" })).routing.ms;
    console.log(`      cpu:8 3 q: in-process ${inMs.toFixed(0)} ms, worker ${wMs.toFixed(0)} ms`);
    assert.ok(wMs < inMs * 1.5 + 50, `worker call (${wMs.toFixed(0)} ms) should cost about the same as in-process (${inMs.toFixed(0)} ms)`);
  } finally {
    await Promise.all([inProc.close(), workers.close()]);
  }
});

test("per-call calibration override reaches the worker and is reset for the next call", async () => {
  const router = await LayaRouter.create({ lanes: ["cpu:8"], sampleLoad: false, log: quiet });
  try {
    // should_execute (noul) is ~0.9 for the PoC state with the shipped T=1.98: room to flatten and to sharpen
    const p = async (o) => (await router.decide(STATE, QUESTIONS_3, o)).answers.should_execute.noul;
    const base = await p();
    assert.ok(base > 0.6 && base < 0.98, `expected a confident but not saturated P(true), got ${base}`);
    const flat = await p({ calibration: { temperature_by_options: { "noul:2": 50 } } });
    const sharp = await p({ calibration: { temperature_by_options: { "noul:2": 0.5 } } });
    const again = await p();
    assert.ok(Math.abs(flat - 0.5) < Math.abs(base - 0.5) - 0.05, `T=50 flattens P(true): ${base} -> ${flat}`);
    assert.ok(Math.abs(sharp - 0.5) > Math.abs(base - 0.5) + 0.05, `T=0.5 sharpens: ${base} -> ${sharp}`);
    assert.equal(again, base, "without an override the lane is back to its base temperatures");
  } finally {
    await router.close();
  }
});

test("a lane that fails mid-call: the call is retried on another lane and the lane is quarantined", async () => {
  const logs = [];
  const router = await LayaRouter.create({ lanes: LANES, sampleLoad: false, explore: 0, warmup: { state: STATE, sizes: [3] }, log: (m) => logs.push(m) });
  try {
    const gpu = router.lanes.get("webgpu:fp16");
    assert.ok(gpu, "GPU lane loaded");
    // warmed GPU: the router sends a 3-question call there; make that one call fail like a lost session would
    const real = gpu.session.systemOne.bind(gpu.session);
    let failed = 0;
    gpu.session.systemOne = () => {
      failed++;
      gpu.session.systemOne = real;
      return Promise.reject(new Error("simulated lane failure"));
    };
    const r = await router.decide(stateFor(1), QUESTIONS_3);
    assert.equal(failed, 1, "the GPU lane was tried first");
    assert.equal(r.routing.lane, "cpu:8", "retried on the remaining lane");
    assert.equal(r.answers.intent.choice, "control_device");
    const s = router.stats();
    assert.equal(s.lanes["webgpu:fp16"].healthy, false);
    assert.equal(s.lanes["webgpu:fp16"].dead, false);
    assert.equal(s.lanes["webgpu:fp16"].failures, 1);
    assert.ok(logs.some((m) => /webgpu:fp16 failed \(simulated lane failure\); quarantined 60 s/.test(m)), logs.join(" | "));
    assert.deepEqual(router.predictions(3).map((c) => c.lane), ["cpu:8"], "quarantined lane is out of the candidates");
    assert.equal((await router.decide(stateFor(2), QUESTIONS_3)).routing.lane, "cpu:8");
    // a forced lane ignores the quarantine (the caller insists); the real session is back so it works
    assert.equal((await router.decide(stateFor(3), QUESTIONS_3, { lane: "webgpu:fp16" })).routing.lane, "webgpu:fp16");
    assert.equal(router.stats().lanes["webgpu:fp16"].healthy, true, "a successful call ends the quarantine");
    assert.equal(router.inflight.length, 0, "no leaked queue entries");
    assert.equal(gpu.pending, 0);
  } finally {
    await router.close();
  }
});

test("a worker that exits: the lane is marked gone, in-flight calls fail over, later calls avoid it", async () => {
  const logs = [];
  const router = await LayaRouter.create({ lanes: LANES, sampleLoad: false, explore: 0, warmup: { state: STATE, sizes: [3] }, log: (m) => logs.push(m) });
  try {
    const gpu = router.lanes.get("webgpu:fp16");
    // queue two calls behind a CPU call so they sit in the router's queue, then end the GPU worker while it is idle
    // (ending it mid-inference is not survivable: ORT runs native code on that thread - see lane.mjs)
    const cpuCall = router.decide(stateFor(0), QUESTIONS_10, { lane: "cpu:8" });
    const queued = [router.decide(stateFor(1), QUESTIONS_3), router.decide(stateFor(2), QUESTIONS_3)];
    await sleep(20);
    assert.equal(router.inflight.length, 3);
    assert.equal(router.inflight[0].lane, "cpu:8");
    assert.equal(router.inflight[1].lane, "webgpu:fp16", "the queued calls were assigned to the GPU lane");
    await gpu.session.worker.terminate(); // idle worker: safe
    for (let i = 0; i < 100 && !gpu.dead; i++) await sleep(10);
    assert.equal(gpu.dead, true, "the exit event marked the lane gone");
    const results = await Promise.all([cpuCall, ...queued]);
    assert.equal(results[0].routing.lane, "cpu:8");
    assert.equal(results[1].routing.lane, "cpu:8", "queued call failed over from the dead lane");
    assert.equal(results[2].routing.lane, "cpu:8");
    assert.ok(results.every((r) => r.answers.intent.choice === "control_device"));
    const s = router.stats();
    assert.equal(s.lanes["webgpu:fp16"].dead, true);
    assert.equal(s.lanes["webgpu:fp16"].healthy, false);
    assert.ok(logs.some((m) => /webgpu:fp16: gone/.test(m)), `router logged the loss: ${logs.join(" | ")}`);
    assert.deepEqual(router.predictions(3).map((c) => c.lane), ["cpu:8"], "the dead lane is out of the candidates");
    assert.equal((await router.decide(stateFor(4), QUESTIONS_3)).routing.lane, "cpu:8");
    await assert.rejects(router.decide(stateFor(5), QUESTIONS_3, { lane: "webgpu:fp16" }), /gone/);
    assert.equal(router.inflight.length, 0, "no leaked queue entries");
    assert.equal(gpu.pending, 0);
  } finally {
    await router.close();
  }
});

test("waitFor: first - serves as soon as one lane is warm; router.ready resolves when all are", async () => {
  const readyLanes = [];
  const t0 = performance.now();
  const router = await LayaRouter.create({ lanes: LANES, sampleLoad: false, waitFor: "first", warmup: { state: STATE, sizes: [3] }, onLaneReady: (lane, info) => readyLanes.push({ lane, ...info }), log: quiet });
  const firstMs = performance.now() - t0;
  const lanesAtStart = [...router.lanes.keys()];
  const pending = router.pendingLanes;
  try {
    assert.ok(lanesAtStart.length >= 1, "at least one lane serves");
    assert.equal(lanesAtStart.length + pending.length, LANES.length, "every lane is either serving or loading");
    assert.equal(readyLanes.length, lanesAtStart.length);
    const early = await router.decide(STATE, QUESTIONS_3);
    assert.ok(router.lanes.has(early.routing.lane));
    if (pending.length) {
      console.log(`      ${lanesAtStart.join(",")} served after ${(firstMs / 1000).toFixed(1)} s while ${pending.join(",")} was still loading`);
      // forcing the loading lane either fails with a clear message or, if it joined meanwhile, just works
      const r = await router.decide(STATE, QUESTIONS_3, { lane: pending[0] }).then((x) => x, (e) => e);
      assert.ok(r instanceof Error ? /not loaded .*loading/.test(r.message) : r.routing.lane === pending[0], String(r.message ?? r.routing?.lane));
      if (pending[0] === "webgpu:fp16" && router.pendingLanes.includes("webgpu:fp16")) {
        // a burst while only the CPU lane serves: lanes are bound when a call reaches the front of the queue,
        // so once the GPU lane joins the remaining calls move over instead of queueing on the CPU
        const tb = performance.now();
        const burst = await Promise.all(Array.from({ length: 6 }, (_, i) => router.decide(stateFor(10 + i), QUESTIONS_10)));
        const lanes = burst.map((b) => b.routing.lane);
        const moved = burst.filter((b) => b.routing.provisionalLane === "cpu:8" && b.routing.lane === "webgpu:fp16").length;
        console.log(`      6 x 10-question burst during start-up: ${(performance.now() - tb).toFixed(0)} ms, lanes ${lanes.join(" ")}, ${moved} re-bound cpu:8 -> webgpu:fp16`);
        assert.ok(lanes.includes("webgpu:fp16"), "the GPU lane took over part of the burst once it joined");
        assert.ok(moved >= 1, "at least one call provisionally bound to cpu:8 ran on the GPU lane");
        assert.ok(burst.every((b) => b.answers.intent.choice === "control_device"));
      }
    } else console.log(`      both lanes were ready together after ${(firstMs / 1000).toFixed(1)} s (nothing to observe)`);
    await router.ready;
    const allMs = performance.now() - t0;
    assert.deepEqual([...router.lanes.keys()].sort(), [...LANES].sort());
    assert.deepEqual(router.pendingLanes, []);
    assert.equal(readyLanes.length, 2);
    for (const r of readyLanes) {
      assert.equal(r.mode, "worker");
      assert.ok(r.warmup[3].ms > 0, "warm-up recorded");
      assert.ok(r.loadMs > 0);
    }
    console.log(`      all lanes ready after ${(allMs / 1000).toFixed(1)} s`);
  } finally {
    await router.close();
  }
});

test("a lane that fails to load is dropped; when every lane fails create() rejects", async () => {
  const logs = [];
  const router = await LayaRouter.create({ lanes: ["nope:4", "cpu:8"], sampleLoad: false, log: (m) => logs.push(m) });
  try {
    assert.deepEqual([...router.lanes.keys()], ["cpu:8"]);
    assert.ok(logs.some((m) => /nope:4: unavailable - unsupported ep/.test(m)), logs.join(" | "));
    assert.equal((await router.decide(STATE, QUESTIONS_3)).routing.lane, "cpu:8");
  } finally {
    await router.close();
  }
  await assert.rejects(LayaRouter.create({ lanes: ["nope:4"], sampleLoad: false, log: quiet }), /no lane could be loaded/);
});

test("close() ends the worker threads", async () => {
  const router = await LayaRouter.create({ lanes: LANES, sampleLoad: false, log: quiet });
  const sessions = [...router.lanes.values()].map((L) => L.session);
  const exits = sessions.map((s) => new Promise((resolve) => s.worker.once("exit", resolve)));
  await router.close();
  assert.equal(router.lanes.size, 0);
  const codes = await Promise.race([Promise.all(exits), sleep(5000).then(() => null)]);
  assert.ok(codes, "every worker exited within 5 s of close()");
  for (const s of sessions) {
    assert.equal(s.dead, true);
    await assert.rejects(s.systemOne(STATE, QUESTIONS_3), /closed|gone/);
  }
  await assert.rejects(router.decide(STATE, QUESTIONS_3), /closed/);
});
