/**
 * Experiment: latency of sporadic calls (idle gaps between calls) vs. back-to-back calls.
 * A desktop assistant issues one decision every few seconds; the GPU drops to idle clocks in between
 * and has to ramp up again. CPUs ramp much faster.
 *
 *   node experiments/sporadic.mjs --ep webgpu
 *   node experiments/sporadic.mjs --ep webgpu --fp16
 *   node experiments/sporadic.mjs --ep cpu
 */
import { parseArgs } from "node:util";
import { loadLaya } from "../src/laya-client.mjs";
import { latencyStats } from "../src/metrics.mjs";
import { STATE, QUESTIONS_1, QUESTIONS_3 } from "../src/questions.mjs";

const { values: args } = parseArgs({ options: { ep: { type: "string", default: "webgpu" }, calls: { type: "string", default: "8" }, fp16: { type: "boolean", default: false } } });
const calls = Number(args.calls);
const { laya } = await loadLaya({ ep: args.ep, modelDir: args.fp16 ? "models/laya-onnx-fp16" : undefined, log: () => {}, logSeverityLevel: 3 });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

for (let i = 0; i < 5; i++) await laya.systemOne(STATE, QUESTIONS_3); // fully warm
console.log(`ep=${args.ep}${args.fp16 ? " fp16" : ""}; ${calls} calls per gap setting, 3 questions`);
console.log("gap(ms)   p50      mean     min      max");
for (const gap of [0, 250, 1000, 3000]) {
  const times = [];
  for (let i = 0; i < calls; i++) {
    if (gap) await sleep(gap);
    const t0 = performance.now();
    await laya.systemOne({ ...STATE, userMessage: `${STATE.userMessage} (${i})` }, QUESTIONS_3);
    times.push(performance.now() - t0);
  }
  const s = latencyStats(times);
  console.log(`${String(gap).padStart(6)}   ${s.p50.toFixed(1).padStart(6)}   ${s.mean.toFixed(1).padStart(6)}   ${s.min.toFixed(1).padStart(6)}   ${s.max.toFixed(1).padStart(6)}`);
}
console.log("\n1 question, gap 3000 ms:");
const t1 = [];
for (let i = 0; i < calls; i++) {
  await sleep(3000);
  const t0 = performance.now();
  await laya.systemOne({ ...STATE, userMessage: `${STATE.userMessage} (${i})` }, QUESTIONS_1);
  t1.push(performance.now() - t0);
}
const s1 = latencyStats(t1);
console.log(`  p50 ${s1.p50.toFixed(1)}  mean ${s1.mean.toFixed(1)}  min ${s1.min.toFixed(1)}  max ${s1.max.toFixed(1)}`);
await laya.close();
