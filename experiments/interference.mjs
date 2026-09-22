/**
 * Experiment: why does the GPU lane slow down when a pinned CPU lane in the same process is busy, and what
 * removes it? (experiments/throughput.mjs scenario C measured webgpu:fp16 3 q: 49 ms alone -> 424 ms p50 /
 * 1775 ms p95 while cpu:8 ran back-to-back.)
 *
 *   node experiments/interference.mjs                              # gpu variants default,1,2 ; cpu lane cpu:8
 *   node experiments/interference.mjs --gpu-threads default,1 --cpu-threads 16
 *   node experiments/interference.mjs --no-child                   # skip the separate-process CPU load
 *
 * For each GPU-session variant (CPU-side intra-op threads of the WebGPU session: ORT default | 1 | 2 | ...):
 *   alone            sequential GPU calls, nothing else running
 *   cpu in-process   the same while a pinned CPU session in this process runs back-to-back
 *   cpu in child     the same while a separate node process runs the pinned CPU session back-to-back
 * plus the CPU lane's own latency alone vs. while the GPU lane is busy.
 *
 * If "in child" is as slow as "in-process", the cause is machine-level (cores / memory bandwidth /
 * scheduler), not shared ORT state. If a 1-thread GPU session removes the slowdown, the culprit is the
 * WebGPU session's default CPU pool (24 unpinned threads) waiting for members that share cores with the
 * pinned CPU lane.
 */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { loadLaya } from "../src/laya-client.mjs";
import { latencyStats } from "../src/metrics.mjs";
import { STATE, QUESTIONS_3 } from "../src/questions.mjs";

const { values: args } = parseArgs({
  options: {
    "gpu-threads": { type: "string", default: "default,1,2" },
    "cpu-threads": { type: "string", default: "8" },
    fp16: { type: "boolean", default: true },
    calls: { type: "string", default: "12" },
    "no-child": { type: "boolean", default: false },
    child: { type: "boolean", default: false }, // internal: run as the CPU-load child process
  },
});
const cpuThreads = Number(args["cpu-threads"]);
const N = Number(args.calls);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stateFor = (i) => ({ ...STATE, userMessage: `${STATE.userMessage} (${i})` });
const f0 = (x) => x.toFixed(0).padStart(5);

// ---- child mode: load a pinned CPU session and run it back-to-back until stdin closes ------------------
if (args.child) {
  const { laya } = await loadLaya({ ep: "cpu", threads: cpuThreads, pinToPCores: true, log: () => {}, logSeverityLevel: 3 });
  await laya.systemOne(STATE, QUESTIONS_3);
  process.stdout.write("ready\n");
  let stop = false;
  process.stdin.on("end", () => (stop = true));
  process.stdin.on("close", () => (stop = true));
  process.stdin.resume();
  let i = 0;
  while (!stop) await laya.systemOne(stateFor(i++), QUESTIONS_3);
  await laya.close();
  process.exit(0);
}

// ---- main -----------------------------------------------------------------------------------------------
const measure = async (laya, n = N) => {
  for (let i = 0; i < 3; i++) await laya.systemOne(stateFor(900 + i), QUESTIONS_3);
  const times = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    await laya.systemOne(stateFor(i), QUESTIONS_3);
    times.push(performance.now() - t0);
  }
  return latencyStats(times);
};

/** Run `fn` while `busyLaya` executes back-to-back calls in this process. */
const whileBusy = async (busyLaya, fn) => {
  let stop = false;
  const bg = (async () => {
    let i = 0;
    while (!stop) await busyLaya.systemOne(stateFor(5000 + i++), QUESTIONS_3);
  })();
  await sleep(50);
  try {
    return await fn();
  } finally {
    stop = true;
    await bg;
  }
};

/** Run `fn` while a child process executes the pinned CPU session back-to-back. */
const whileChildBusy = async (fn) => {
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), "--child", "--cpu-threads", String(cpuThreads)], { stdio: ["pipe", "pipe", "inherit"], windowsHide: true });
  await new Promise((resolve, reject) => {
    child.stdout.on("data", (d) => String(d).includes("ready") && resolve());
    child.on("exit", (c) => reject(new Error(`child exited early (${c})`)));
  });
  await sleep(100);
  try {
    return await fn();
  } finally {
    child.stdin.end();
    await new Promise((r) => child.on("exit", r));
  }
};

const row = (label, s) => console.log(`  ${label.padEnd(34)} p50 ${f0(s.p50)}  p95 ${f0(s.p95)}  max ${f0(s.max)} ms`);

console.log(`CPU lane: cpu:${cpuThreads} pinned; GPU lane: webgpu${args.fp16 ? " fp16" : ""}; ${N} timed calls per cell, 3 questions`);
const { laya: cpu } = await loadLaya({ ep: "cpu", threads: cpuThreads, pinToPCores: true, log: () => {}, logSeverityLevel: 3 });
const results = [];
for (const v of args["gpu-threads"].split(",")) {
  const threads = v === "default" ? undefined : Number(v);
  const { laya: gpu, loadMs } = await loadLaya({ ep: "webgpu", threads, modelDir: args.fp16 ? "models/laya-onnx-fp16" : undefined, log: () => {}, logSeverityLevel: 3 });
  console.log(`\nGPU session, CPU-side intra-op threads = ${v} (loaded in ${(loadMs / 1000).toFixed(1)} s)`);
  const alone = await measure(gpu);
  row("gpu alone", alone);
  const inProc = await whileBusy(cpu, () => measure(gpu));
  row(`gpu while cpu:${cpuThreads} busy (in-process)`, inProc);
  let child = null;
  if (!args["no-child"]) {
    child = await whileChildBusy(() => measure(gpu));
    row(`gpu while cpu:${cpuThreads} busy (child process)`, child);
  }
  const cpuAlone = await measure(cpu, Math.max(6, Math.floor(N / 2)));
  row(`cpu:${cpuThreads} alone`, cpuAlone);
  const cpuBusy = await whileBusy(gpu, () => measure(cpu, Math.max(6, Math.floor(N / 2))));
  row(`cpu:${cpuThreads} while gpu busy (in-process)`, cpuBusy);
  results.push({ gpuThreads: v, alone, inProc, child, cpuAlone, cpuBusy });
  await gpu.close();
}
await cpu.close();

console.log("\nsummary (GPU p50 ms):");
console.log("  gpu threads   alone   cpu in-process   cpu in child   | cpu alone   cpu while gpu busy");
for (const r of results) console.log(`  ${String(r.gpuThreads).padEnd(12)} ${f0(r.alone.p50)}   ${f0(r.inProc.p50).padStart(14)}   ${(r.child ? f0(r.child.p50) : "    -").padStart(12)}   | ${f0(r.cpuAlone.p50).padStart(9)}   ${f0(r.cpuBusy.p50).padStart(18)}`);
