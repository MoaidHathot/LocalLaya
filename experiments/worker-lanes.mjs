/**
 * Experiment: can each lane live in its own worker thread, so that (a) CPU and GPU inferences overlap and
 * (b) the main thread stays responsive while an inference runs?
 *
 * This was the prototype for src/lane.mjs (worker lanes are the router's default since 0.5). Kept as the
 * measurement of what mixing lanes in parallel costs.
 *
 * Background: onnxruntime-node 1.30 runs `session.run()` synchronously on the calling JS thread
 * (dist/backend.js wraps it in setImmediate + Promise). In one thread every inference blocks the event loop
 * for its full duration, so two lanes never run in parallel and an HTTP server cannot even read requests
 * while a 300 ms CPU inference runs (experiments/interference.mjs: GPU 80 -> 470 ms while a CPU session in
 * the same thread is busy; 82 ms when the CPU load is in another process).
 *
 *   node experiments/worker-lanes.mjs
 *   node experiments/worker-lanes.mjs --cpu-threads 16
 */
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { latencyStats } from "../src/metrics.mjs";
import { STATE, QUESTIONS_3 } from "../src/questions.mjs";

// ---- worker: one Laya session, answers { id, state, questions } messages ---------------------------------
if (!isMainThread) {
  const { loadLaya } = await import("../src/laya-client.mjs");
  const { laya } = await loadLaya({ ...workerData, log: () => {}, logSeverityLevel: 3 });
  await laya.systemOne(STATE, QUESTIONS_3);
  parentPort.postMessage({ ready: true });
  parentPort.on("message", async (m) => {
    if (m.close) {
      await laya.close();
      parentPort.postMessage({ closed: true });
      return;
    }
    const t0 = performance.now();
    const r = await laya.systemOne(m.state, m.questions);
    parentPort.postMessage({ id: m.id, ms: performance.now() - t0, answers: r.answers });
  });
} else {
  const { values: args } = parseArgs({ options: { "cpu-threads": { type: "string", default: "8" }, "gpu-threads": { type: "string" }, calls: { type: "string", default: "12" } } });
  const N = Number(args.calls);
  const gpuThreads = args["gpu-threads"] ? Number(args["gpu-threads"]) : undefined;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const stateFor = (i) => ({ ...STATE, userMessage: `${STATE.userMessage} (${i})` });
  const f0 = (x) => x.toFixed(0).padStart(5);
  const here = fileURLToPath(import.meta.url);

  /** Lane = worker thread + request/response plumbing. */
  const lane = (name, loadOpts) => {
    const w = new Worker(here, { workerData: loadOpts });
    const waiting = new Map();
    let seq = 0;
    const ready = new Promise((resolve, reject) => {
      w.once("message", (m) => (m.ready ? resolve() : reject(new Error(`unexpected ${JSON.stringify(m)}`))));
      w.once("error", reject);
    });
    w.on("message", (m) => {
      if (m.id !== undefined) waiting.get(m.id)?.(m), waiting.delete(m.id);
    });
    return {
      name,
      ready,
      call: (state, questions) =>
        new Promise((resolve) => {
          const id = seq++;
          waiting.set(id, resolve);
          w.postMessage({ id, state, questions });
        }),
      close: () =>
        new Promise((resolve) => {
          w.on("message", (m) => m.closed && resolve());
          w.postMessage({ close: true });
        }).then(() => w.terminate()),
    };
  };

  const t0 = performance.now();
  const gpu = lane(`webgpu:fp16${gpuThreads ? ` (cpu pool ${gpuThreads})` : ""}`, { ep: "webgpu", modelDir: "models/laya-onnx-fp16", threads: gpuThreads });
  const cpu = lane(`cpu:${args["cpu-threads"]}`, { ep: "cpu", threads: Number(args["cpu-threads"]), pinToPCores: true });
  await Promise.all([gpu.ready, cpu.ready]);
  console.log(`two worker lanes ready in ${((performance.now() - t0) / 1000).toFixed(1)} s: ${gpu.name}, ${cpu.name}`);

  const measure = async (L, n = N) => {
    for (let i = 0; i < 3; i++) await L.call(stateFor(900 + i), QUESTIONS_3);
    const total = [];
    const inner = [];
    for (let i = 0; i < n; i++) {
      const s = performance.now();
      const r = await L.call(stateFor(i), QUESTIONS_3);
      total.push(performance.now() - s);
      inner.push(r.ms);
    }
    return { total: latencyStats(total), inner: latencyStats(inner) };
  };
  const whileBusy = async (busy, fn) => {
    let stop = false;
    const bg = (async () => {
      let i = 0;
      while (!stop) await busy.call(stateFor(5000 + i++), QUESTIONS_3);
    })();
    await sleep(50);
    try {
      return await fn();
    } finally {
      stop = true;
      await bg;
    }
  };
  /** Main-thread responsiveness: largest gap between 5 ms timer ticks while `fn` runs. */
  const withTicker = async (fn) => {
    let last = performance.now();
    let maxGap = 0;
    const t = setInterval(() => {
      const now = performance.now();
      maxGap = Math.max(maxGap, now - last);
      last = now;
    }, 5);
    try {
      return { ...(await fn()), maxGapMs: maxGap };
    } finally {
      clearInterval(t);
    }
  };
  const row = (label, r) => console.log(`  ${label.padEnd(36)} total p50 ${f0(r.total.p50)}  p95 ${f0(r.total.p95)}   in-worker p50 ${f0(r.inner.p50)}${r.maxGapMs !== undefined ? `   main-thread max stall ${f0(r.maxGapMs)} ms` : ""}`);

  console.log(`\n${N} timed calls per cell, 3 questions ("total" = round trip through the worker, "in-worker" = systemOne inside the worker)`);
  row("gpu alone", await withTicker(() => measure(gpu)));
  row(`gpu while ${cpu.name} busy`, await withTicker(() => whileBusy(cpu, () => measure(gpu))));
  row(`${cpu.name} alone`, await withTicker(() => measure(cpu, Math.max(6, N / 2))));
  row(`${cpu.name} while gpu busy`, await withTicker(() => whileBusy(gpu, () => measure(cpu, Math.max(6, N / 2)))));

  console.log("\nthroughput of a 24-call burst (3 q), GPU warmed before each burst, two repeats:");
  const burst = async (label, pick) => {
    for (let i = 0; i < 3; i++) await gpu.call(stateFor(800 + i), QUESTIONS_3);
    const out = [];
    for (let rep = 0; rep < 2; rep++) {
      const s = performance.now();
      await Promise.all(Array.from({ length: 24 }, (_, i) => pick(i).call(stateFor(i), QUESTIONS_3)));
      const wall = performance.now() - s;
      out.push(`${((24 / wall) * 1000).toFixed(1).padStart(5)} calls/s (${wall.toFixed(0)} ms)`);
    }
    console.log(`  ${label.padEnd(36)} ${out.join("   ")}`);
  };
  await burst("gpu only", () => gpu);
  await burst("gpu + every 7th call on cpu", (i) => (i % 7 === 6 ? cpu : gpu));
  await burst("gpu + every 4th call on cpu", (i) => (i % 4 === 3 ? cpu : gpu));
  await burst("cpu only", () => cpu);

  await Promise.all([gpu.close(), cpu.close()]);
  console.log("\nworkers closed cleanly");
}
