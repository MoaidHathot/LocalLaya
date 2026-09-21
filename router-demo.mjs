/**
 * Demo of the execution-provider router under the traffic patterns that matter for a desktop assistant.
 *
 *   node router-demo.mjs                              # lanes webgpu + cpu, policy auto
 *   node router-demo.mjs --lanes webgpu,cpu:8         # spare the CPU: 8-thread lane instead of the default
 *   node router-demo.mjs --lanes webgpu,cpu,dml       # dml is probed and dropped (fails at inference here)
 *   node router-demo.mjs --keepalive 4000             # keep GPU clocks up for 4 s after each GPU call
 *   node router-demo.mjs --policy prefer-gpu
 *   node router-demo.mjs --calibration calibration/smart-home-v3.json
 */
import { parseArgs } from "node:util";
import { LayaRouter } from "./src/ep-router.mjs";
import { STATE, QUESTIONS_1, QUESTIONS_3, QUESTIONS_10 } from "./src/questions.mjs";

const { values: args } = parseArgs({
  options: {
    lanes: { type: "string", default: "webgpu,cpu" },
    policy: { type: "string", default: "auto" },
    keepalive: { type: "string", default: "0" },
    calibration: { type: "string" },
    explore: { type: "string", default: "0.05" },
  },
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (r) => `${r.routing.lane.padEnd(7)} ${r.routing.ms.toFixed(0).padStart(5)} ms  (pred ${r.routing.predictedMs.toFixed(0).padStart(4)}, gpu ${r.routing.gpuState.padEnd(4)}${r.routing.explored ? ", explored" : ""})  ${r.routing.reason}`;

const t0 = performance.now();
const router = await LayaRouter.create({
  lanes: args.lanes.split(","),
  policy: args.policy,
  explore: Number(args.explore),
  gpuKeepAliveMs: Number(args.keepalive),
  calibration: args.calibration,
  log: (m) => console.log(`  [router] ${m}`),
});
console.log(`router ready in ${((performance.now() - t0) / 1000).toFixed(1)} s with lanes: ${[...router.lanes.keys()].join(", ")}`);

console.log("\nwarm-up (compiles GPU shaders, primes latency estimates):");
const w = await router.warmup({ state: STATE }); // use a representative state: latency depends on its length
for (const [lane, sizes] of Object.entries(w)) console.log(`  ${lane.padEnd(7)} ${Object.entries(sizes).map(([n, v]) => `${n}q first ${v.firstMs.toFixed(0)} / then ${v.ms.toFixed(0)} ms`).join("   ")}`);

console.log("\nA) burst: 10 back-to-back calls, 3 questions");
for (let i = 0; i < 10; i++) console.log("  " + fmt(await router.decide({ ...STATE, userMessage: `${STATE.userMessage} ${i}` }, QUESTIONS_3)));

console.log("\nB) sporadic: 6 calls with 3 s pauses, 1 question (GPU goes cold in between)");
for (let i = 0; i < 6; i++) {
  await sleep(3000);
  console.log("  " + fmt(await router.decide({ ...STATE, userMessage: `Is the ${["kitchen", "bedroom", "office", "porch", "garage", "hall"][i]} light on?` }, QUESTIONS_1)));
}

console.log("\nC) sporadic: 4 calls with 3 s pauses, 3 questions");
for (let i = 0; i < 4; i++) {
  await sleep(3000);
  console.log("  " + fmt(await router.decide({ ...STATE, userMessage: `Lock the ${["front", "back", "side", "garage"][i]} door` }, QUESTIONS_3)));
}

console.log("\nD) one 10-question call after a 3 s pause");
await sleep(3000);
console.log("  " + fmt(await router.decide(STATE, QUESTIONS_10)));

console.log("\nE) deadline 150 ms, 3 questions, after a 3 s pause (meet the deadline with the least CPU)");
await sleep(3000);
console.log("  " + fmt(await router.decide(STATE, QUESTIONS_3, { deadlineMs: 150 })));

console.log("\nF) forced lane");
for (const lane of router.lanes.keys()) console.log("  " + fmt(await router.decide(STATE, QUESTIONS_3, { lane })));

const s = router.stats();
console.log("\nlatency estimates (EMA, ms) per lane / GPU state / question bucket:");
for (const [lane, L] of Object.entries(s.lanes)) {
  console.log(`  ${lane.padEnd(7)} calls=${L.calls} healthy=${L.healthy} ${Object.entries(L.ema).map(([k, v]) => `${k}: ${v.ms.toFixed(0)} (n=${v.n})`).join("  ")}`);
}
console.log(`load: other processes CPU ${(s.load.cpuOthers * 100).toFixed(0)}%  GPU (others) ${(s.load.gpuOthersUtil * 100).toFixed(0)}%  GPU SM clock ${s.load.gpuSmClockMHz ?? "-"} MHz  VRAM free ${s.load.gpuMemFreeMiB ?? "-"} MiB  keep-alive calls ${s.keepAliveCalls}`);
const byLane = {};
for (const h of router.history) (byLane[h.lane] ??= []).push(h.ms);
console.log(`calls by lane: ${Object.entries(byLane).map(([l, ms]) => `${l} ${ms.length}x (mean ${(ms.reduce((a, b) => a + b, 0) / ms.length).toFixed(0)} ms)`).join(", ")}`);
await router.close();
