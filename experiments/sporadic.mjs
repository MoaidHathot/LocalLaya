/**
 * Experiment: latency of sporadic calls (idle gaps between calls) vs. back-to-back calls.
 * A desktop assistant issues one decision every few seconds; the GPU drops to idle clocks in between
 * and has to ramp up again. CPUs ramp much faster.
 *
 *   node experiments/sporadic.mjs --ep webgpu
 *   node experiments/sporadic.mjs --ep webgpu --fp16
 *   node experiments/sporadic.mjs --ep cuda --fp16       # the Python process lane (npm run cuda:setup)
 *   node experiments/sporadic.mjs --ep cpu
 *   node experiments/sporadic.mjs --router --lanes cuda:fp16,cpu:8 [--keepalive 5000] [--no-sampling]
 *                                                        # through LayaRouter.decide() as the sidecar runs it:
 *                                                        # lane choice per call, optional GPU keep-alive
 */
import { parseArgs } from "node:util";
import { openLane } from "../src/lane.mjs";
import { LayaRouter } from "../src/ep-router.mjs";
import { latencyStats } from "../src/metrics.mjs";
import { STATE, QUESTIONS_1, QUESTIONS_3 } from "../src/questions.mjs";

const { values: args } = parseArgs({
  options: {
    ep: { type: "string", default: "webgpu" },
    calls: { type: "string", default: "8" },
    fp16: { type: "boolean", default: false },
    router: { type: "boolean", default: false },
    lanes: { type: "string", default: "cuda:fp16,webgpu:fp16,cpu:8" },
    keepalive: { type: "string", default: "0" },
    "no-sampling": { type: "boolean", default: false },
  },
});
const calls = Number(args.calls);
let laya;
let router = null;
if (args.router) {
  router = await LayaRouter.create({ lanes: args.lanes.split(","), gpuKeepAliveMs: Number(args.keepalive), sampleLoad: !args["no-sampling"], warmup: { state: STATE, sizes: [1, 3] }, log: () => {} });
  const lanesOf = [];
  laya = { systemOne: async (st, qs) => { const r = await router.decide(st, qs); lanesOf.push(r.routing.lane); return r; }, close: () => router.close(), lanesOf };
} else {
  laya = await openLane(`${args.ep}${args.fp16 ? ":fp16" : ""}`, { ep: args.ep, modelDir: args.fp16 ? "models/laya-onnx-fp16" : undefined }, { worker: false, log: () => {} });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

for (let i = 0; i < 5; i++) await laya.systemOne(STATE, QUESTIONS_3); // fully warm
console.log(router ? `router lanes=${[...router.lanes.keys()].join(",")} keepalive=${args.keepalive} sampling=${!args["no-sampling"]}; ${calls} calls per gap setting, 3 questions` : `ep=${args.ep}${args.fp16 ? " fp16" : ""}; ${calls} calls per gap setting, 3 questions`);
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
  const lanes = laya.lanesOf ? `   ${laya.lanesOf.splice(0).map((l) => l.split(":")[0]).join(" ")}` : "";
  console.log(`${String(gap).padStart(6)}   ${s.p50.toFixed(1).padStart(6)}   ${s.mean.toFixed(1).padStart(6)}   ${s.min.toFixed(1).padStart(6)}   ${s.max.toFixed(1).padStart(6)}${lanes}`);
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
console.log(`  p50 ${s1.p50.toFixed(1)}  mean ${s1.mean.toFixed(1)}  min ${s1.min.toFixed(1)}  max ${s1.max.toFixed(1)}${laya.lanesOf ? `   ${laya.lanesOf.splice(0).map((l) => l.split(":")[0]).join(" ")}` : ""}`);
await laya.close();
