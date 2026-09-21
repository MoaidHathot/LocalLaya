/**
 * Evaluate accuracy + calibration of Laya on the labelled smart-home set, then refit one temperature per
 * (question type, option-count) bucket - pooled over every question that falls in the bucket - and report
 * the honest (leave-one-out) effect.
 *
 *   node calibrate.mjs                          # CPU, v1 wording, writes calibration/smart-home-v1.json
 *   node calibrate.mjs --ep webgpu --variant v2 # improved criteria wording, writes calibration/smart-home-v2.json
 *   node calibrate.mjs --out calibration/my-domain.json
 *
 * The written table can be applied at load time: loadLaya({ calibration: "calibration/smart-home-v2.json" }).
 */
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { loadLaya } from "./src/laya-client.mjs";
import { bucketKey, captureLogits, currentTemperature, fitTemperature, metrics, optionCount, optionLabels, softmax, systemOneWithLogits } from "./src/calibration.mjs";
import { VARIANTS } from "./data/question-variants.mjs";
import { EVAL_SET, stateFor } from "./data/smart-home-eval.mjs";

const { values: args } = parseArgs({
  options: { ep: { type: "string", default: "cpu" }, variant: { type: "string", default: "v1" }, out: { type: "string" } },
});
const QUESTIONS = VARIANTS[args.variant];
if (!QUESTIONS) throw new Error(`unknown variant ${args.variant}; have ${Object.keys(VARIANTS).join(", ")}`);
const outPath = args.out ?? `calibration/smart-home-${args.variant}.json`;

const goldIndex = (qid, ex) => {
  const labels = optionLabels(QUESTIONS[qid]);
  const g = qid === "should_execute" ? String(ex.should_execute) : ex[qid];
  const i = labels.indexOf(g);
  if (i < 0) throw new Error(`gold label ${g} not an option of ${qid}`);
  return i;
};

const { laya, loadMs } = await loadLaya({ ep: args.ep, log: () => {} });
const capture = captureLogits(laya);
console.log(`Loaded (ep=${args.ep}) in ${(loadMs / 1000).toFixed(1)} s. Variant ${args.variant}. Evaluating ${EVAL_SET.length} examples x ${Object.keys(QUESTIONS).length} questions...`);

// ---- collect logits -----------------------------------------------------------------------------------
const samples = Object.fromEntries(Object.keys(QUESTIONS).map((q) => [q, []])); // qid -> [{logits, gold, ex}]
const t0 = performance.now();
for (const ex of EVAL_SET) {
  const { raw } = await systemOneWithLogits(laya, capture, stateFor(ex), QUESTIONS);
  for (const qid of Object.keys(QUESTIONS)) samples[qid].push({ logits: raw[qid], gold: goldIndex(qid, ex), ex });
}
console.log(`Inference done in ${((performance.now() - t0) / 1000).toFixed(1)} s.\n`);

// ---- fit one temperature per bucket on pooled samples ------------------------------------------------
const bucketOf = Object.fromEntries(Object.keys(QUESTIONS).map((qid) => [qid, bucketKey(QUESTIONS[qid].type, optionCount(QUESTIONS[qid]))]));
const pooled = {};
for (const qid of Object.keys(QUESTIONS)) (pooled[bucketOf[qid]] ??= []).push(...samples[qid].map((s, i) => ({ ...s, qid, i })));
const fitted = Object.fromEntries(Object.entries(pooled).map(([b, S]) => [b, fitTemperature(S).T]));

// Leave-one-out at the bucket level: drop the example (all questions of that example in the bucket), refit, score.
const looProbs = Object.fromEntries(Object.keys(QUESTIONS).map((q) => [q, []]));
for (const [b, S] of Object.entries(pooled)) {
  for (let ex = 0; ex < EVAL_SET.length; ex++) {
    const rest = S.filter((s) => s.i !== ex);
    const { T } = fitTemperature(rest);
    for (const s of S.filter((s) => s.i === ex)) looProbs[s.qid][ex] = { probs: softmax(s.logits.map((v) => v / T)), gold: s.gold, T };
  }
  void b;
}

// ---- report --------------------------------------------------------------------------------------------
const f = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : "-");
const table = {
  source: "calibrate.mjs",
  dataset: "data/smart-home-eval.mjs",
  variant: args.variant,
  examples: EVAL_SET.length,
  ep: args.ep,
  createdAt: new Date().toISOString(),
  temperature_by_options: Object.fromEntries(Object.entries(fitted).map(([b, T]) => [b, Number(T.toFixed(4))])),
  report: {},
};

console.log("Fitted temperatures (pooled per bucket):");
for (const [b, T] of Object.entries(fitted)) {
  const shippedT = laya.config.temperature_by_options[b];
  console.log(`  ${b.padEnd(12)} shipped ${f(shippedT)} -> fitted ${f(T)}   (questions: ${Object.keys(QUESTIONS).filter((q) => bucketOf[q] === b).join(", ")}, n=${pooled[b].length})`);
}
console.log("");

for (const qid of Object.keys(QUESTIONS)) {
  const q = QUESTIONS[qid];
  const k = optionCount(q);
  const Tship = currentTemperature(laya, q);
  const S = samples[qid];
  const rawM = metrics(S.map((s) => ({ probs: softmax(s.logits), gold: s.gold })));
  const shipped = metrics(S.map((s) => ({ probs: softmax(s.logits.map((v) => v / Tship)), gold: s.gold })));
  const looM = metrics(looProbs[qid]);

  console.log(`=== ${qid}  (${q.type}, ${k} options, bucket ${bucketOf[qid]}, n=${S.length}) ===`);
  console.log(`                    accuracy   NLL     Brier   ECE     mean-conf`);
  console.log(`  raw (T=1)         ${f(rawM.accuracy)}      ${f(rawM.nll)}   ${f(rawM.brier)}   ${f(rawM.ece)}   ${f(rawM.meanConfidence)}`);
  console.log(`  shipped T=${f(Tship, 2)}    ${f(shipped.accuracy)}      ${f(shipped.nll)}   ${f(shipped.brier)}   ${f(shipped.ece)}   ${f(shipped.meanConfidence)}`);
  console.log(`  refit (LOO)       ${f(looM.accuracy)}      ${f(looM.nll)}   ${f(looM.brier)}   ${f(looM.ece)}   ${f(looM.meanConfidence)}`);

  const relShip = metrics(S.map((s) => ({ probs: softmax(s.logits.map((v) => v / Tship)), gold: s.gold })), 5);
  const relLoo = metrics(looProbs[qid], 5);
  console.log(`  reliability (confidence bin -> n, avg confidence, actual accuracy):`);
  console.log(`      bin        shipped                      refit (LOO)`);
  for (let i = 0; i < 5; i++) {
    const s = relShip.reliability[i];
    const l = relLoo.reliability[i];
    const cell = (b) => (b.n ? `n=${String(b.n).padStart(2)} conf ${f(b.avgConf, 2)} acc ${f(b.accuracy, 2)}` : "".padEnd(27));
    console.log(`      ${s.bin}    ${cell(s)}    ${cell(l)}`);
  }

  const labels = optionLabels(q);
  const mistakes = S.map((s) => ({ s, p: softmax(s.logits.map((v) => v / Tship)) }))
    .filter(({ s, p }) => p.indexOf(Math.max(...p)) !== s.gold)
    .sort((x, y) => Math.max(...y.p) - Math.max(...x.p));
  if (mistakes.length) {
    console.log(`  ${mistakes.length} mistakes; most confident (shipped T):`);
    for (const { s, p } of mistakes.slice(0, 4)) {
      const pred = p.indexOf(Math.max(...p));
      console.log(`    "${s.ex.message}"  ->  ${labels[pred]} p=${f(p[pred], 2)}   (gold ${labels[s.gold]} p=${f(p[s.gold], 2)})`);
    }
  }
  // confusion summary (gold -> predicted counts)
  const conf = {};
  for (const s of S) {
    const p = softmax(s.logits);
    const pred = labels[p.indexOf(Math.max(...p))];
    const key = `${labels[s.gold]} -> ${pred}`;
    if (pred !== labels[s.gold]) conf[key] = (conf[key] ?? 0) + 1;
  }
  const confusions = Object.entries(conf).sort((a, b) => b[1] - a[1]);
  if (confusions.length) console.log(`  confusions: ${confusions.map(([k, v]) => `${k} (${v})`).join(", ")}`);
  console.log("");

  table.report[qid] = { bucket: bucketOf[qid], n: S.length, shippedT: Tship, fittedT: fitted[bucketOf[qid]], raw: rawM, shipped, refitLOO: looM, confusions: conf };
}

await mkdir(path.dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify(table, null, 2) + "\n");
console.log(`Wrote ${outPath}  temperature_by_options=${JSON.stringify(table.temperature_by_options)}`);
await laya.close();
