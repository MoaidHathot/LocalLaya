/**
 * Integration test for the CUDA process lane (needs the model bundle, an NVIDIA GPU and the Python venv from
 * `npm run cuda:setup`; skipped entirely otherwise). ~60 s.
 *
 *   npm run test:cuda          (= node --test --test-concurrency=1 test/cuda-lane.test.mjs)
 *
 * Scenarios: answers agree with the WebGPU lane (arg-max on every eval item, |delta p| within fp16 tolerance)
 * and the temperature override reaches the process / CUDA Graph replay (the default) agrees with the dynamic graph
 * on the whole eval set, exec: { graph: false } forces the dynamic graph per call, the bucket grid has the
 * documented edges, 1-question replays are faster than dynamic calls / cudaGraph: false and the lazy path (no
 * eager buckets: dynamic first, a bucket appears after a shape was seen twice and the lane had an idle gap) /
 * the router picks CUDA for a hot burst, reports routing.exec, the queue stays FIFO / killing the Python process
 * mid-queue: in-flight calls fail over, the lane is gone, later calls avoid it / close() ends the process / a wrong
 * python path is reported as an unavailable lane, not a crash.
 */
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { LayaRouter } from "../src/ep-router.mjs";
import { openLane, LaneDeadError } from "../src/lane.mjs";
import { optionLabels } from "../src/calibration.mjs";
import { STATE, QUESTIONS_1, QUESTIONS_3, QUESTIONS_10 } from "../src/questions.mjs";
import { EVAL_SET, stateFor } from "../data/smart-home-eval.mjs";
import { VARIANTS } from "../data/question-variants.mjs";

const run = promisify(execFile);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const quiet = () => {};
const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};
let cudaOk = false;
before(async () => {
  cudaOk = await run(process.execPath, ["tools/setup-cuda-lane.mjs", "--check"], { timeout: 60_000 }).then(() => true, () => false);
  if (!cudaOk) console.log("      CUDA lane not available on this machine (npm run cuda:setup) - tests skipped");
});
const skipUnlessCuda = (t) => !cudaOk && (t.skip("no CUDA lane"), true);

const probsOf = (ans, q) => (ans.type === "noul" ? [1 - ans.noul, ans.noul] : optionLabels(q).map((l) => ans.probabilities[l] ?? 0));
const pickOf = (ans) => (ans.type === "noul" ? (ans.noul >= 0.5 ? "true" : "false") : ans.type === "score" ? Object.entries(ans.probabilities).sort((a, b) => b[1] - a[1])[0][0] : ans.choice);
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

/** Compare two answer sets question by question: arg-max agreement (flips only on near-ties) and max |delta p|. */
function compareAnswers(a, b, qs, acc) {
  for (const qid of Object.keys(qs)) {
    acc.total++;
    const pa = probsOf(a.answers[qid], qs[qid]);
    const pb = probsOf(b.answers[qid], qs[qid]);
    acc.maxDiff = Math.max(acc.maxDiff, ...pa.map((v, i) => Math.abs(v - pb[i])));
    if (pickOf(a.answers[qid]) === pickOf(b.answers[qid])) acc.agree++;
    else {
      const top = [...pa].sort((x, y) => y - x);
      acc.maxTie = Math.max(acc.maxTie, top[0] - top[1]);
    }
  }
}
const newAcc = () => ({ total: 0, agree: 0, maxDiff: 0, maxTie: 0 });

/**
 * Register the shapes of a workload with the lane (a shape seen twice gets a bucket) and wait until the lane has
 * prepared and captured everything it scheduled (it does that in idle gaps; under back-to-back traffic it would
 * take ~2.5 s per bucket). Returns the stats snapshot.
 */
async function warmShapes(lane, items, timeoutMs = 20_000) {
  for (let pass = 0; pass < 2; pass++) for (const it of items) await lane.systemOne(it.state, it.questions);
  const t0 = performance.now();
  for (;;) {
    const st = await lane.stats();
    const pending = Object.values(st.buckets).filter((b) => b.state === "building" || b.state === "prepared").length + st.queued.length;
    if (!pending) return st;
    if (performance.now() - t0 > timeoutMs) throw new Error(`buckets still building after ${timeoutMs} ms: ${JSON.stringify(st.buckets)} queued ${JSON.stringify(st.queued)}`);
    await sleep(200);
  }
}
function assertClose(acc, what) {
  assert.ok(acc.maxDiff < 0.05, `${what}: max |dp| ${acc.maxDiff}`);
  assert.ok(acc.agree >= acc.total - 2, `${what}: ${acc.total - acc.agree} disagreements`);
  assert.ok(acc.maxTie < 0.05, `${what}: a disagreement with margin ${acc.maxTie} is not a near-tie`);
}

test("cuda lane answers like the webgpu lane (same fp16 bundle) on the whole eval set; temperature override works", async (t) => {
  if (skipUnlessCuda(t)) return;
  const cuda = await openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16" }, { log: quiet });
  const wg = await openLane("webgpu:fp16", { ep: "webgpu", modelDir: "models/laya-onnx-fp16" }, { worker: false, log: quiet });
  try {
    assert.equal(cuda.mode, "process");
    assert.ok(cuda.providers.includes("CUDAExecutionProvider"));
    assert.equal(cuda.graph.enabled, true, "CUDA graphs on by default");
    const qs = VARIANTS.v3;
    const st = await warmShapes(cuda, EVAL_SET.map((ex) => ({ state: stateFor(ex), questions: qs })));
    const built = Object.entries(st.buckets).filter(([, b]) => b.state === "ready").map(([k]) => k);
    assert.equal(st.build_failures, 0, JSON.stringify(st.buckets));
    assert.ok(built.some((k) => k.startsWith("3x") && !["3x96x8"].includes(k)), `the eval set's 3-row shapes got buckets: ${built.join(" ")}`);
    const acc = newAcc();
    const modes = { graph: 0, dynamic: 0 };
    for (const ex of EVAL_SET) {
      const [a, b] = await Promise.all([wg.systemOne(stateFor(ex), qs), cuda.systemOne(stateFor(ex), qs)]);
      assert.equal(a.usage.input_tokens, b.usage.input_tokens);
      modes[b.exec.mode]++;
      compareAnswers(a, b, qs, acc);
    }
    console.log(`      ${acc.agree}/${acc.total} arg-max agreement with webgpu:fp16 over ${EVAL_SET.length} states x ${Object.keys(qs).length} questions; max |dp| ${acc.maxDiff.toFixed(4)}${acc.agree < acc.total ? `; flips were near-ties (largest margin ${acc.maxTie.toFixed(3)})` : ""}; cuda ran ${modes.graph} graph / ${modes.dynamic} dynamic (buckets ${built.join(" ")})`);
    assertClose(acc, "cuda vs webgpu");
    assert.ok(modes.graph >= EVAL_SET.length * 0.9, `eval calls replay a graph once the shapes are known (${modes.graph} graph / ${modes.dynamic} dynamic)`);
    // per-call temperature override reaches the process and is reset afterwards
    const base = (await cuda.systemOne(STATE, QUESTIONS_3)).answers.should_execute.noul;
    const flat = (await cuda.systemOne(STATE, QUESTIONS_3, { "noul:2": 50 })).answers.should_execute.noul;
    const again = (await cuda.systemOne(STATE, QUESTIONS_3)).answers.should_execute.noul;
    assert.ok(Math.abs(flat - 0.5) < Math.abs(base - 0.5) - 0.05, `T=50 flattens: ${base} -> ${flat}`);
    assert.equal(again, base);
    // speed sanity (not a benchmark): 3 questions well under the webgpu lane's 32 ms
    const times = [];
    for (let i = 0; i < 10; i++) {
      const t0 = performance.now();
      await cuda.systemOne(STATE, QUESTIONS_3);
      times.push(performance.now() - t0);
    }
    console.log(`      cuda 3 q p50 ${median(times).toFixed(1)} ms (remote session.run ${cuda.lastInferenceMs.toFixed(1)} ms, ${cuda.remote.lastExec.mode})`);
    assert.ok(median(times) < 30, `cuda 3 q p50 ${median(times).toFixed(1)} ms`);
  } finally {
    await Promise.all([cuda.close(), wg.close()]);
  }
});

test("CUDA Graph replay agrees with the dynamic graph on the eval set; exec.graph=false per call; bucket grid edges; 1 q replay is faster", async (t) => {
  if (skipUnlessCuda(t)) return;
  const cuda = await openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16" }, { log: quiet });
  try {
    const qs = VARIANTS.v3;
    await warmShapes(cuda, EVAL_SET.map((ex) => ({ state: stateFor(ex), questions: qs })));
    const acc = newAcc();
    let graphCalls = 0;
    for (const ex of EVAL_SET) {
      const g = await cuda.systemOne(stateFor(ex), qs);
      const d = await cuda.systemOne(stateFor(ex), qs, undefined, { graph: false });
      assert.equal(d.exec.mode, "dynamic", "exec: { graph: false } runs the dynamic graph");
      assert.equal(d.exec.bucket, null);
      if (g.exec.mode === "graph") {
        graphCalls++;
        assert.ok(Array.isArray(g.exec.bucket) && g.exec.bucket.length === 3, `bucket reported: ${JSON.stringify(g.exec.bucket)}`);
        assert.ok(g.exec.bucket[0] >= 3 && g.exec.bucket[2] >= 8, `bucket covers 3 rows x 8 options: ${g.exec.bucket}`);
        compareAnswers(d, g, qs, acc);
      }
    }
    console.log(`      graph vs dynamic: ${acc.agree}/${acc.total} arg-max agreement on ${graphCalls}/${EVAL_SET.length} replayed calls; max |dp| ${acc.maxDiff.toFixed(4)}${acc.agree < acc.total ? ` (near-tie margin ${acc.maxTie.toFixed(3)})` : ""}`);
    assert.ok(graphCalls >= EVAL_SET.length * 0.9, `${graphCalls}/${EVAL_SET.length} calls replayed a graph`);
    assertClose(acc, "graph vs dynamic");
    // the bucket grid: next size up on every axis, nothing above 512 tokens / 16 rows / 32 options / n x L > 1536
    assert.deepEqual(await cuda.bucketFor(3, 96, 8), [3, 96, 8]);
    assert.deepEqual(await cuda.bucketFor(3, 97, 8), [3, 128, 8]);
    assert.deepEqual(await cuda.bucketFor(1, 40, 3), [1, 64, 8]);
    assert.deepEqual(await cuda.bucketFor(7, 150, 9), [8, 160, 16]);
    assert.deepEqual(await cuda.bucketFor(16, 96, 8), [16, 96, 8]);
    assert.equal(await cuda.bucketFor(16, 97, 8), null, "16 x 128 is above the work limit");
    assert.equal(await cuda.bucketFor(4, 600, 8), null, "longer than 512 tokens");
    assert.equal(await cuda.bucketFor(17, 10, 8), null, "more than 16 rows");
    assert.equal(await cuda.bucketFor(1, 10, 33), null, "more than 32 options");
    // a 10-question call (10 rows) is not on the eager list: dynamic until the shape was seen twice and captured
    const ten = await cuda.systemOne(STATE, QUESTIONS_10);
    assert.equal(ten.exec.mode, "dynamic", "first sighting of a new shape stays dynamic");
    assert.equal((await cuda.stats()).buckets["10x96x8"], undefined, "one sighting schedules nothing");
    // 1 question: replay vs dynamic, interleaved (the launch overhead is half of a dynamic 1 q call)
    const g = [];
    const d = [];
    for (let i = 0; i < 12; i++) {
      let t0 = performance.now();
      const rg = await cuda.systemOne({ ...STATE, userMessage: `one ${i}` }, QUESTIONS_1);
      g.push(performance.now() - t0);
      t0 = performance.now();
      const rd = await cuda.systemOne({ ...STATE, userMessage: `one ${i}` }, QUESTIONS_1, undefined, { graph: false });
      d.push(performance.now() - t0);
      assert.equal(rg.exec.mode, "graph");
      assert.equal(rd.exec.mode, "dynamic");
      assert.equal(pickOf(rg.answers.intent), pickOf(rd.answers.intent));
    }
    console.log(`      1 q round trip p50: graph ${median(g).toFixed(1)} ms vs dynamic ${median(d).toFixed(1)} ms`);
    assert.ok(median(g) < median(d), `1 q graph p50 ${median(g).toFixed(1)} ms should beat dynamic ${median(d).toFixed(1)} ms`);
    const st = await cuda.stats();
    assert.equal(st.graphEnabled, true);
    assert.equal(st.build_failures, 0, JSON.stringify(st.buckets));
    assert.ok(st.buckets["1x96x8"]?.state === "ready" && st.buckets["3x96x8"]?.state === "ready", `eager buckets ready: ${JSON.stringify(st.buckets)}`);
    for (const [k, b] of Object.entries(st.buckets)) if (b.checkDelta != null) assert.ok(b.checkDelta < 0.25, `bucket ${k} replay vs dynamic check ${b.checkDelta}`);
  } finally {
    await cuda.close();
  }
});

test("cudaGraph: false serves dynamically; without eager buckets a shape seen twice gets a bucket in the next idle gap", async (t) => {
  if (skipUnlessCuda(t)) return;
  const off = await openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16", cudaGraph: false }, { log: quiet });
  try {
    assert.equal(off.graph.enabled, false);
    const r = await off.systemOne(STATE, QUESTIONS_3);
    assert.equal(r.exec.mode, "dynamic");
    assert.equal(await off.bucketFor(3, 96, 8), null);
    assert.equal((await off.stats()).graphEnabled, false);
  } finally {
    await off.close();
  }
  const lazy = await openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16", graphBuckets: "" }, { log: quiet });
  try {
    assert.equal(lazy.graph.enabled, true);
    assert.deepEqual(Object.keys((await lazy.stats()).buckets), [], "no eager buckets");
    assert.equal((await lazy.systemOne(STATE, QUESTIONS_3)).exec.mode, "dynamic", "first sighting");
    assert.equal((await lazy.systemOne(STATE, QUESTIONS_3)).exec.mode, "dynamic", "second sighting schedules the bucket");
    await sleep(2500); // prepare (~250 ms) waits for a 100 ms gap, the capture (~100 ms) for the next one
    const st = await lazy.stats();
    assert.equal(st.buckets["3x96x8"]?.state, "ready", JSON.stringify(st.buckets));
    assert.ok(st.buckets["3x96x8"].checkDelta < 0.25, `checked against the dynamic graph on the real inputs: ${st.buckets["3x96x8"].checkDelta}`);
    const r = await lazy.systemOne(STATE, QUESTIONS_3);
    assert.equal(r.exec.mode, "graph");
    assert.deepEqual(r.exec.bucket, [3, 96, 8]);
  } finally {
    await lazy.close();
  }
});

test("router: cuda is the default first lane and takes a hot burst with routing.exec; killing the process mid-queue fails over and marks it gone", async (t) => {
  if (skipUnlessCuda(t)) return;
  const logs = [];
  const router = await LayaRouter.create({ lanes: ["cuda:fp16", "cpu:8"], sampleLoad: false, explore: 0, warmup: { state: STATE, sizes: [3] }, log: (m) => logs.push(m) });
  try {
    const cuda = router.lanes.get("cuda:fp16");
    assert.ok(cuda, "cuda lane loaded");
    assert.equal(cuda.session.mode, "process");
    assert.ok(logs.some((m) => /lane cuda:fp16: ready .*cuda graphs on/.test(m)), logs.join(" | "));
    const burst = await Promise.all(Array.from({ length: 6 }, (_, i) => router.decide({ ...STATE, userMessage: `burst ${i}` }, QUESTIONS_3)));
    assert.ok(burst.every((r) => r.routing.lane === "cuda:fp16"), `hot burst on cuda: ${burst.map((r) => r.routing.lane).join(",")}`);
    assert.ok(burst.every((r) => r.answers.intent.choice === "control_device"));
    assert.ok(burst.every((r) => r.routing.exec?.mode === "graph" && Array.isArray(r.routing.exec.bucket)), `routing.exec reports the replay: ${JSON.stringify(burst.map((r) => r.routing.exec))}`);
    assert.ok(burst.every((r) => r.exec === undefined), "exec is moved from the result into routing");
    const dyn = await router.decide(STATE, QUESTIONS_3, { exec: { graph: false } });
    assert.equal(dyn.routing.exec?.mode, "dynamic");
    const cpu = await router.decide(STATE, QUESTIONS_3, { lane: "cpu:8" });
    assert.equal(cpu.routing.exec, undefined, "in-node lanes report no exec");
    const stats = router.stats();
    assert.equal(stats.lanes["cuda:fp16"].cudaGraph.enabled, true);
    const detailed = await router.detailedStats();
    assert.ok(detailed.lanes["cuda:fp16"].process.buckets["3x96x8"]?.state === "ready", JSON.stringify(detailed.lanes["cuda:fp16"].process));
    // queue three calls behind a 10-question one, then kill the Python process while they wait
    const pid = cuda.session.pid;
    const first = router.decide(STATE, QUESTIONS_10);
    const queued = [1, 2, 3].map((i) => router.decide({ ...STATE, userMessage: `after kill ${i}` }, QUESTIONS_3));
    await sleep(5);
    process.kill(pid);
    const results = await Promise.allSettled([first, ...queued]);
    const ok = results.filter((r) => r.status === "fulfilled").map((r) => r.value);
    // the call that was running inside the process at the kill may fail over or be lost mid-flight; the queued ones must fail over
    assert.ok(ok.length >= 3, `${ok.length}/4 calls answered after the kill: ${results.map((r) => (r.status === "fulfilled" ? r.value.routing.lane : r.reason.message.slice(0, 60))).join(" | ")}`);
    assert.ok(ok.every((r) => r.routing.lane === "cpu:8"), `failed over to cpu:8: ${ok.map((r) => r.routing.lane).join(",")}`);
    assert.equal(cuda.dead, true);
    assert.equal(router.stats().lanes["cuda:fp16"].dead, true);
    assert.ok(logs.some((m) => /cuda:fp16: gone \(process exited/.test(m)), logs.join(" | "));
    assert.deepEqual(router.predictions(3).map((c) => c.lane), ["cpu:8"]);
    assert.equal((await router.decide(STATE, QUESTIONS_3)).routing.lane, "cpu:8");
    await assert.rejects(router.decide(STATE, QUESTIONS_3, { lane: "cuda:fp16" }), /gone/);
    assert.equal(router.inflight.length, 0);
    assert.ok(!alive(pid));
  } finally {
    await router.close();
  }
});

test("close() ends the Python process; a bad python path is an unavailable lane, not a crash", async (t) => {
  if (skipUnlessCuda(t)) return;
  const lane = await openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16" }, { log: quiet });
  const pid = lane.pid;
  assert.ok(alive(pid));
  await lane.close();
  assert.equal(lane.dead, true);
  await sleep(300);
  assert.ok(!alive(pid), "python process exited on close");
  await assert.rejects(lane.systemOne(STATE, QUESTIONS_3), (e) => e instanceof LaneDeadError);
  await assert.rejects(openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16", python: "C:/definitely/not/python.exe" }, { log: quiet }), /could not start/);
  // through the router: the cuda lane is dropped with a log line and the other lanes serve (what happens on a machine without the venv)
  const logs = [];
  const router = await LayaRouter.create({ lanes: ["cuda:fp16", "cpu:8"], sampleLoad: false, python: "C:/definitely/not/python.exe", log: (m) => logs.push(m) });
  try {
    assert.deepEqual([...router.lanes.keys()], ["cpu:8"]);
    assert.ok(logs.some((m) => /cuda:fp16: unavailable - could not start/.test(m)), logs.join(" | "));
    assert.equal((await router.decide(STATE, QUESTIONS_3)).routing.lane, "cpu:8");
  } finally {
    await router.close();
  }
});
