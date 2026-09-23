/**
 * Integration test for the CUDA process lane (needs the model bundle, an NVIDIA GPU and the Python venv from
 * `npm run cuda:setup`; skipped entirely otherwise). ~40 s.
 *
 *   npm run test:cuda          (= node --test --test-concurrency=1 test/cuda-lane.test.mjs)
 *
 * Scenarios: answers agree with the WebGPU lane (arg-max on every eval item, |delta p| within fp16 tolerance)
 * and the temperature override reaches the process / the router picks CUDA for a hot burst and the queue
 * stays FIFO / killing the Python process mid-queue: in-flight calls fail over, the lane is gone, later calls
 * avoid it / close() ends the process / a wrong python path is reported as an unavailable lane, not a crash.
 */
import { test, before } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { LayaRouter } from "../src/ep-router.mjs";
import { openLane, LaneDeadError } from "../src/lane.mjs";
import { optionLabels } from "../src/calibration.mjs";
import { STATE, QUESTIONS_3, QUESTIONS_10 } from "../src/questions.mjs";
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

test("cuda lane answers like the webgpu lane (same fp16 bundle) on the whole eval set; temperature override works", async (t) => {
  if (skipUnlessCuda(t)) return;
  const cuda = await openLane("cuda:fp16", { ep: "cuda", modelDir: "models/laya-onnx-fp16" }, { log: quiet });
  const wg = await openLane("webgpu:fp16", { ep: "webgpu", modelDir: "models/laya-onnx-fp16" }, { worker: false, log: quiet });
  try {
    assert.equal(cuda.mode, "process");
    assert.ok(cuda.providers.includes("CUDAExecutionProvider"));
    const qs = VARIANTS.v3;
    let total = 0;
    let agree = 0;
    let maxDiff = 0;
    let maxTie = 0;
    for (const ex of EVAL_SET) {
      const [a, b] = await Promise.all([wg.systemOne(stateFor(ex), qs), cuda.systemOne(stateFor(ex), qs)]);
      assert.equal(a.usage.input_tokens, b.usage.input_tokens);
      for (const qid of Object.keys(qs)) {
        total++;
        const pa = probsOf(a.answers[qid], qs[qid]);
        const pb = probsOf(b.answers[qid], qs[qid]);
        const d = Math.max(...pa.map((v, i) => Math.abs(v - pb[i])));
        maxDiff = Math.max(maxDiff, d);
        if (pickOf(a.answers[qid]) === pickOf(b.answers[qid])) agree++;
        else {
          // a flip is acceptable only on a near-tie: the two top probabilities within the fp16 tolerance
          const top = [...pa].sort((x, y) => y - x);
          maxTie = Math.max(maxTie, top[0] - top[1]);
        }
      }
    }
    console.log(`      ${agree}/${total} arg-max agreement with webgpu:fp16 over ${EVAL_SET.length} states x ${Object.keys(qs).length} questions; max |dp| ${maxDiff.toFixed(4)}${agree < total ? `; flips were near-ties (largest margin ${maxTie.toFixed(3)})` : ""}`);
    assert.ok(maxDiff < 0.05, `max |dp| ${maxDiff}`);
    assert.ok(agree >= total - 2, `${total - agree} disagreements`);
    assert.ok(maxTie < 0.05, `a disagreement with margin ${maxTie} is not a near-tie`);
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
    times.sort((x, y) => x - y);
    console.log(`      cuda 3 q p50 ${times[5].toFixed(1)} ms (remote session.run ${cuda.lastInferenceMs.toFixed(1)} ms)`);
    assert.ok(times[5] < 30, `cuda 3 q p50 ${times[5].toFixed(1)} ms`);
  } finally {
    await Promise.all([cuda.close(), wg.close()]);
  }
});

test("router: cuda is the default first lane and takes a hot burst; killing the process mid-queue fails over and marks it gone", async (t) => {
  if (skipUnlessCuda(t)) return;
  const logs = [];
  const router = await LayaRouter.create({ lanes: ["cuda:fp16", "cpu:8"], sampleLoad: false, explore: 0, warmup: { state: STATE, sizes: [3] }, log: (m) => logs.push(m) });
  try {
    const cuda = router.lanes.get("cuda:fp16");
    assert.ok(cuda, "cuda lane loaded");
    assert.equal(cuda.session.mode, "process");
    const burst = await Promise.all(Array.from({ length: 6 }, (_, i) => router.decide({ ...STATE, userMessage: `burst ${i}` }, QUESTIONS_3)));
    assert.ok(burst.every((r) => r.routing.lane === "cuda:fp16"), `hot burst on cuda: ${burst.map((r) => r.routing.lane).join(",")}`);
    assert.ok(burst.every((r) => r.answers.intent.choice === "control_device"));
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
