/**
 * Fidelity + speed of the fp16 bundle (tools/convert_fp16.py) against the fp32 reference.
 *
 *   node experiments/fp16-fidelity.mjs --ep webgpu
 *   node experiments/fp16-fidelity.mjs --ep cpu
 *
 * Runs the labelled eval set (65 states x 3 questions) on both bundles with the same EP and reports:
 * arg-max agreement, max / mean absolute probability difference, accuracy of both, and latency.
 */
import { parseArgs } from "node:util";
import { loadLaya } from "../src/laya-client.mjs";
import { optionLabels } from "../src/calibration.mjs";
import { latencyStats } from "../src/metrics.mjs";
import { VARIANTS } from "../data/question-variants.mjs";
import { EVAL_SET, stateFor } from "../data/smart-home-eval.mjs";
import { STATE, QUESTIONS_1, QUESTIONS_3, QUESTIONS_10 } from "../src/questions.mjs";

const { values: args } = parseArgs({ options: { ep: { type: "string", default: "webgpu" }, fp16: { type: "string", default: "models/laya-onnx-fp16" } } });
const QUESTIONS = VARIANTS.v3;

async function run(label, opts) {
  const { laya, loadMs } = await loadLaya({ ep: args.ep, log: () => {}, ...opts });
  console.log(`${label}: loaded in ${(loadMs / 1000).toFixed(1)} s`);
  for (let i = 0; i < 3; i++) await laya.systemOne(STATE, QUESTIONS_3);
  const out = [];
  for (const ex of EVAL_SET) out.push((await laya.systemOne(stateFor(ex), QUESTIONS)).answers);
  const lat = {};
  for (const [n, qs] of [[1, QUESTIONS_1], [3, QUESTIONS_3], [10, QUESTIONS_10]]) {
    for (let i = 0; i < 3; i++) await laya.systemOne(STATE, qs);
    const t = [];
    for (let i = 0; i < 15; i++) {
      const t0 = performance.now();
      await laya.systemOne(STATE, qs);
      t.push(performance.now() - t0);
    }
    lat[n] = latencyStats(t);
  }
  await laya.close();
  return { out, lat };
}

const probsOf = (ans, q) => (ans.type === "noul" ? [1 - ans.noul, ans.noul] : optionLabels(q).map((l) => ans.probabilities[l]));
const gold = (qid, ex) => (qid === "should_execute" ? String(ex.should_execute) : ex[qid]);
const pick = (ans, q) => (ans.type === "noul" ? (ans.noul >= 0.5 ? "true" : "false") : ans.choice);

const ref = await run("fp32", {});
const f16 = await run("fp16", { modelDir: args.fp16 });

let agree = 0;
let total = 0;
let maxDiff = 0;
let sumDiff = 0;
let nDiff = 0;
const acc = { fp32: 0, fp16: 0 };
for (let i = 0; i < EVAL_SET.length; i++) {
  for (const qid of Object.keys(QUESTIONS)) {
    const q = QUESTIONS[qid];
    const a = ref.out[i][qid];
    const b = f16.out[i][qid];
    total++;
    if (pick(a, q) === pick(b, q)) agree++;
    if (pick(a, q) === gold(qid, EVAL_SET[i])) acc.fp32++;
    if (pick(b, q) === gold(qid, EVAL_SET[i])) acc.fp16++;
    const pa = probsOf(a, q);
    const pb = probsOf(b, q);
    pa.forEach((v, j) => {
      const d = Math.abs(v - pb[j]);
      maxDiff = Math.max(maxDiff, d);
      sumDiff += d;
      nDiff++;
    });
  }
}
console.log(`\nfidelity over ${total} answers (${EVAL_SET.length} states x ${Object.keys(QUESTIONS).length} questions):`);
console.log(`  arg-max agreement fp16 vs fp32: ${agree}/${total} (${((100 * agree) / total).toFixed(1)}%)`);
console.log(`  |delta probability|: max ${maxDiff.toFixed(4)}  mean ${(sumDiff / nDiff).toFixed(5)}`);
console.log(`  accuracy vs gold: fp32 ${(acc.fp32 / total).toFixed(3)}  fp16 ${(acc.fp16 / total).toFixed(3)}`);
console.log(`\nlatency p50 (ms) on ${args.ep}, back-to-back:`);
console.log(`          1 q      3 q      10 q`);
console.log(`  fp32  ${ref.lat[1].p50.toFixed(1).padStart(6)}   ${ref.lat[3].p50.toFixed(1).padStart(6)}   ${ref.lat[10].p50.toFixed(1).padStart(6)}`);
console.log(`  fp16  ${f16.lat[1].p50.toFixed(1).padStart(6)}   ${f16.lat[3].p50.toFixed(1).padStart(6)}   ${f16.lat[10].p50.toFixed(1).padStart(6)}`);
