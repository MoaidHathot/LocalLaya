/**
 * Experiment: how do the execution providers react to (a) shapes they have not seen before and
 * (b) concurrent calls?
 *
 *   node experiments/shape-and-concurrency.mjs --ep webgpu
 *   node experiments/shape-and-concurrency.mjs --ep cpu
 *
 * (a) Real workloads have a different token count per state, so the [n, L] input shape changes almost
 *     every call. GPU EPs may compile shape-specialised shaders; this measures first-sight vs repeat
 *     latency for 16 distinct lengths.
 * (b) Throughput of k concurrent systemOne calls vs. sequential calls.
 */
import { parseArgs } from "node:util";
import { loadLaya } from "../src/laya-client.mjs";
import { latencyStats } from "../src/metrics.mjs";
import { STATE, QUESTIONS_1, QUESTIONS_3 } from "../src/questions.mjs";

const { values: args } = parseArgs({ options: { ep: { type: "string", default: "webgpu" } } });
const { laya, loadMs } = await loadLaya({ ep: args.ep, log: () => {} });
console.log(`ep=${args.ep} loaded in ${(loadMs / 1000).toFixed(2)} s`);

const filler = "the quick brown fox jumps over the lazy dog and keeps running through the garden ";
const stateOfLength = (words) => ({ ...STATE, userMessage: `Turn off the living room lights. ${filler.repeat(6).split(" ").slice(0, words).join(" ")}` });

// ---- (a) shape sensitivity ------------------------------------------------------------------------
console.log("\n(a) first-sight vs repeat latency for distinct sequence lengths (1 question)");
await laya.systemOne(STATE, QUESTIONS_1); // generic warm-up
const lengths = Array.from({ length: 16 }, (_, i) => 2 + i * 7); // 2 .. 107 filler words
const first = [];
const repeat = [];
for (const w of lengths) {
  const s = stateOfLength(w);
  const t0 = performance.now();
  const r = await laya.systemOne(s, QUESTIONS_1);
  const a = performance.now() - t0;
  const t1 = performance.now();
  await laya.systemOne(s, QUESTIONS_1);
  const b = performance.now() - t1;
  first.push(a);
  repeat.push(b);
  console.log(`  tokens=${String(r.usage.input_tokens).padStart(3)}  first ${a.toFixed(1).padStart(7)} ms   repeat ${b.toFixed(1).padStart(6)} ms`);
}
const fs = latencyStats(first);
const rs = latencyStats(repeat);
console.log(`  first-sight: p50 ${fs.p50.toFixed(1)}  mean ${fs.mean.toFixed(1)}  max ${fs.max.toFixed(1)} | repeat: p50 ${rs.p50.toFixed(1)}  mean ${rs.mean.toFixed(1)}`);

// Second pass over the same lengths, in a different order: are shapes cached across calls?
console.log("\n    second pass over the same 16 lengths (shuffled):");
const shuffled = [...lengths].sort(() => Math.random() - 0.5);
const second = [];
for (const w of shuffled) {
  const t0 = performance.now();
  await laya.systemOne(stateOfLength(w), QUESTIONS_1);
  second.push(performance.now() - t0);
}
const ss = latencyStats(second);
console.log(`    p50 ${ss.p50.toFixed(1)}  mean ${ss.mean.toFixed(1)}  max ${ss.max.toFixed(1)} ms`);

// Unseen lengths after all that: does the cost keep recurring?
console.log("\n    16 never-seen lengths after warm-up:");
const unseen = Array.from({ length: 16 }, (_, i) => 5 + i * 7);
const un = [];
for (const w of unseen) {
  const t0 = performance.now();
  await laya.systemOne(stateOfLength(w), QUESTIONS_1);
  un.push(performance.now() - t0);
}
const us = latencyStats(un);
console.log(`    p50 ${us.p50.toFixed(1)}  mean ${us.mean.toFixed(1)}  max ${us.max.toFixed(1)} ms`);

// ---- (b) concurrency --------------------------------------------------------------------------------
console.log("\n(b) throughput: sequential vs concurrent (3 questions, 24 calls total)");
const N = 24;
for (const k of [1, 2, 4, 8]) {
  const t0 = performance.now();
  for (let i = 0; i < N; i += k) {
    await Promise.all(Array.from({ length: Math.min(k, N - i) }, (_, j) => laya.systemOne(stateOfLength(10 + ((i + j) % 5)), QUESTIONS_3)));
  }
  const wall = performance.now() - t0;
  console.log(`  concurrency ${k}: ${((N / wall) * 1000).toFixed(2)} calls/s   (${(wall / N).toFixed(1)} ms per call amortised)`);
}

await laya.close();
