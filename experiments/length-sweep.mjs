/**
 * Experiment: is CPU latency sensitive to the exact sequence length? Sweep the state length token by
 * token and record latency per resulting input length.
 *
 *   node experiments/length-sweep.mjs --ep cpu
 *   node experiments/length-sweep.mjs --ep webgpu
 */
import { parseArgs } from "node:util";
import { loadLaya } from "../src/laya-client.mjs";
import { STATE, QUESTIONS_3, QUESTIONS_1 } from "../src/questions.mjs";

const { values: args } = parseArgs({
  options: { ep: { type: "string", default: "cpu" }, threads: { type: "string" }, reps: { type: "string", default: "3" }, q: { type: "string", default: "3" } },
});
const reps = Number(args.reps);
const questions = args.q === "1" ? QUESTIONS_1 : QUESTIONS_3;
const { laya } = await loadLaya({ ep: args.ep, threads: args.threads ? Number(args.threads) : undefined, log: () => {} });
for (let i = 0; i < 3; i++) await laya.systemOne(STATE, questions);

console.log(`ep=${args.ep}${args.threads ? ` threads=${args.threads}` : ""}  questions=${args.q}  (min of ${reps} reps per length)`);
console.log("suffix_words  input_tokens  per_q_tokens  min_ms   max_ms");
const words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu one two three four five six seven eight nine ten eleven twelve".split(" ");
for (let w = 0; w <= 36; w += 1) {
  const s = { ...STATE, userMessage: `${STATE.userMessage} ${words.slice(0, w).join(" ")}` };
  let min = Infinity;
  let max = 0;
  let tokens = 0;
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    const res = await laya.systemOne(s, questions);
    const ms = performance.now() - t0;
    min = Math.min(min, ms);
    max = Math.max(max, ms);
    tokens = res.usage.input_tokens;
  }
  const n = Object.keys(questions).length;
  console.log(`${String(w).padStart(12)}  ${String(tokens).padStart(12)}  ${String(Math.round(tokens / n)).padStart(12)}  ${min.toFixed(1).padStart(7)}  ${max.toFixed(1).padStart(7)}`);
}
await laya.close();
