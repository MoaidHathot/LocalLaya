/**
 * Evaluate accuracy + calibration of Laya on labelled examples, then refit one temperature per
 * (question type, option-count) bucket - pooled over every question in the bucket - and report the honest
 * (leave-one-out) effect. Writes a table that ask.mjs / serve.mjs pick up automatically.
 *
 * Your domain:
 *   node calibrate.mjs --preset dev-request --eval presets/dev-request.eval.json      # -> calibration/dev-request.json
 *
 *   eval file = JSON array (or { "items": [...] }) of
 *     { "text": "the message",  "gold": { "<question id>": <gold> } }        // text is wrapped by the preset
 *     { "state": { ... },       "gold": { ... } }                             // or an explicit state
 *   gold per question: choice -> option key; noul -> true/false; score -> level index or level text.
 *   Questions with fewer than --min labelled items (default 10) are reported but not fitted.
 *
 * The PoC's smart-home set (data/smart-home-eval.mjs) with a question wording variant:
 *   node calibrate.mjs --variant v3                                                    # -> calibration/smart-home-v3.json
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { loadLaya } from "./src/laya-client.mjs";
import { bucketKey, captureLogits, currentTemperature, fitTemperature, metrics, optionCount, optionLabels, softmax, systemOneWithLogits } from "./src/calibration.mjs";
import { loadPresets } from "./data/presets.mjs";

const { values: args } = parseArgs({
  options: {
    ep: { type: "string", default: "webgpu" },
    preset: { type: "string" },
    eval: { type: "string" },
    variant: { type: "string" },
    min: { type: "string", default: "10" },
    out: { type: "string" },
  },
});
const minItems = Number(args.min);

// ---- inputs: questions, items, state builder ---------------------------------------------------------------
let QUESTIONS;
let items; // [{ text?, state?, gold: {qid: value} }]
let stateOf;
let label;
let outPath;
if (args.eval) {
  if (!args.preset) throw new Error("--eval needs --preset <name>");
  const presets = await loadPresets();
  const preset = presets[args.preset];
  if (!preset || preset.invalid) throw new Error(`preset ${args.preset} not found or invalid (have ${Object.keys(presets).join(", ")})`);
  QUESTIONS = preset.questions;
  const raw = JSON.parse(await readFile(args.eval, "utf8"));
  items = Array.isArray(raw) ? raw : raw.items;
  if (!Array.isArray(items) || !items.length) throw new Error("eval file: expected a non-empty array (or { items: [...] })");
  stateOf = (it) => (it.state !== undefined ? it.state : preset.state(it.text));
  label = `preset ${args.preset}, ${args.eval}`;
  outPath = args.out ?? `calibration/${args.preset}.json`;
} else {
  const variant = args.variant ?? "v1";
  const { VARIANTS } = await import("./data/question-variants.mjs");
  const { EVAL_SET, stateFor } = await import("./data/smart-home-eval.mjs");
  QUESTIONS = VARIANTS[variant];
  if (!QUESTIONS) throw new Error(`unknown variant ${variant}; have ${Object.keys(VARIANTS).join(", ")}`);
  items = EVAL_SET.map((ex) => ({ text: ex.message, gold: { intent: ex.intent, should_execute: ex.should_execute, target_device: ex.target_device }, _ex: ex }));
  stateOf = (it) => stateFor(it._ex);
  label = `smart-home variant ${variant}`;
  outPath = args.out ?? `calibration/smart-home-${variant}.json`;
}

/** gold value -> option index for a question, or -1 when missing / not an option */
function goldIndex(qid, gold) {
  if (gold === undefined || gold === null) return -1;
  const q = QUESTIONS[qid];
  const labels = optionLabels(q);
  if (q.type === "noul") return labels.indexOf(String(gold).toLowerCase() === "true" || gold === true ? "true" : "false");
  if (q.type === "score") {
    if (typeof gold === "number") return gold >= 0 && gold < labels.length ? gold : -1;
    const i = q.criteria.findIndex((c) => c === gold || c.split(":")[0].trim() === String(gold).trim());
    return i;
  }
  return labels.indexOf(String(gold));
}

// ---- inference: collect raw logits ----------------------------------------------------------------------------
const { laya, loadMs } = await loadLaya({ ep: args.ep, log: () => {} });
const capture = captureLogits(laya);
console.log(`Loaded (ep=${args.ep}) in ${(loadMs / 1000).toFixed(1)} s. ${label}: ${items.length} items x ${Object.keys(QUESTIONS).length} questions...`);
const samples = Object.fromEntries(Object.keys(QUESTIONS).map((q) => [q, []])); // qid -> [{logits, gold, i, shown}]
const badGold = [];
const t0 = performance.now();
for (let i = 0; i < items.length; i++) {
  const it = items[i];
  const { raw } = await systemOneWithLogits(laya, capture, stateOf(it), QUESTIONS);
  const shown = it.text ?? JSON.stringify(it.state);
  for (const qid of Object.keys(QUESTIONS)) {
    const g = goldIndex(qid, it.gold?.[qid]);
    if (g < 0) {
      if (it.gold?.[qid] !== undefined) badGold.push(`item ${i} (${shown.slice(0, 40)}): ${qid} = ${JSON.stringify(it.gold[qid])} is not an option`);
      continue;
    }
    samples[qid].push({ logits: raw[qid], gold: g, i, shown });
  }
}
console.log(`Inference done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
for (const b of badGold) console.log(`  WARNING ${b}`);
const fitted_qids = Object.keys(QUESTIONS).filter((q) => samples[q].length >= minItems);
const skipped = Object.keys(QUESTIONS).filter((q) => !fitted_qids.includes(q));
if (skipped.length) console.log(`  not fitted (fewer than ${minItems} labelled items): ${skipped.map((q) => `${q} (${samples[q].length})`).join(", ")}`);
console.log("");

// ---- fit one temperature per bucket on pooled samples ---------------------------------------------------------
const bucketOf = Object.fromEntries(fitted_qids.map((qid) => [qid, bucketKey(QUESTIONS[qid].type, optionCount(QUESTIONS[qid]))]));
const pooled = {};
for (const qid of fitted_qids) (pooled[bucketOf[qid]] ??= []).push(...samples[qid].map((s) => ({ ...s, qid })));
const fitted = Object.fromEntries(Object.entries(pooled).map(([b, S]) => [b, fitTemperature(S).T]));

// leave-one-out per item (all of an item's questions in the bucket are held out together)
const looProbs = Object.fromEntries(fitted_qids.map((q) => [q, []]));
for (const S of Object.values(pooled)) {
  const byItem = new Map();
  for (const s of S) (byItem.get(s.i) ?? byItem.set(s.i, []).get(s.i)).push(s);
  for (const [i, held] of byItem) {
    const { T } = fitTemperature(S.filter((s) => s.i !== i));
    for (const s of held) looProbs[s.qid].push({ probs: softmax(s.logits.map((v) => v / T)), gold: s.gold });
  }
}

// ---- report --------------------------------------------------------------------------------------------------
const f = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : "-");
const table = {
  source: "calibrate.mjs",
  label,
  preset: args.preset ?? "smart-home",
  eval: args.eval ?? "data/smart-home-eval.mjs",
  items: items.length,
  ep: args.ep,
  createdAt: new Date().toISOString(),
  temperature_by_options: Object.fromEntries(Object.entries(fitted).map(([b, T]) => [b, Number(T.toFixed(4))])),
  report: {},
};

console.log("Fitted temperatures (pooled per bucket):");
for (const [b, T] of Object.entries(fitted)) {
  console.log(`  ${b.padEnd(12)} shipped ${f(laya.config.temperature_by_options[b])} -> fitted ${f(T)}   (questions: ${fitted_qids.filter((q) => bucketOf[q] === b).join(", ")}, n=${pooled[b].length})`);
}
console.log("");

for (const qid of Object.keys(QUESTIONS)) {
  const S = samples[qid];
  if (!S.length) continue;
  const q = QUESTIONS[qid];
  const k = optionCount(q);
  const Tship = currentTemperature(laya, q);
  const labels = optionLabels(q);
  const rawM = metrics(S.map((s) => ({ probs: softmax(s.logits), gold: s.gold })));
  const shipped = metrics(S.map((s) => ({ probs: softmax(s.logits.map((v) => v / Tship)), gold: s.gold })));
  const fittedHere = fitted_qids.includes(qid);
  const looM = fittedHere ? metrics(looProbs[qid]) : null;

  // baselines: always guessing the most frequent gold label, and uniform chance
  const counts = {};
  for (const s of S) counts[s.gold] = (counts[s.gold] ?? 0) + 1;
  const majority = Math.max(...Object.values(counts)) / S.length;
  const chance = 1 / k;
  const verdict = rawM.accuracy <= Math.max(majority, chance) + 0.05 ? "NOT USABLE zero-shot: no better than guessing -> reword the question/options, or answer it with code instead" : rawM.accuracy < 0.7 ? "weak: usable only with confidence gating; reword or fine-tune" : "usable";
  console.log(`=== ${qid}  (${q.type}, ${k} options, bucket ${bucketKey(q.type, k)}, n=${S.length}${fittedHere ? "" : ", NOT FITTED"}) ===`);
  console.log(`  baselines: majority class ${f(majority, 2)}, chance ${f(chance, 2)}   ->  ${verdict}`);
  console.log(`                    accuracy   NLL     Brier   ECE     mean-conf`);
  console.log(`  raw (T=1)         ${f(rawM.accuracy)}      ${f(rawM.nll)}   ${f(rawM.brier)}   ${f(rawM.ece)}   ${f(rawM.meanConfidence)}`);
  console.log(`  shipped T=${f(Tship, 2)}    ${f(shipped.accuracy)}      ${f(shipped.nll)}   ${f(shipped.brier)}   ${f(shipped.ece)}   ${f(shipped.meanConfidence)}`);
  if (looM) console.log(`  refit (LOO)       ${f(looM.accuracy)}      ${f(looM.nll)}   ${f(looM.brier)}   ${f(looM.ece)}   ${f(looM.meanConfidence)}`);

  const relShip = metrics(S.map((s) => ({ probs: softmax(s.logits.map((v) => v / Tship)), gold: s.gold })), 5);
  const relLoo = looM ? metrics(looProbs[qid], 5) : null;
  console.log(`  reliability (confidence bin -> n, avg confidence, actual accuracy):`);
  console.log(`      bin        shipped                      ${relLoo ? "refit (LOO)" : ""}`);
  for (let i = 0; i < 5; i++) {
    const s = relShip.reliability[i];
    const l = relLoo?.reliability[i];
    const cell = (b) => (b?.n ? `n=${String(b.n).padStart(2)} conf ${f(b.avgConf, 2)} acc ${f(b.accuracy, 2)}` : "".padEnd(27));
    console.log(`      ${s.bin}    ${cell(s)}    ${cell(l)}`);
  }

  const mistakes = S.map((s) => ({ s, p: softmax(s.logits.map((v) => v / Tship)) }))
    .filter(({ s, p }) => p.indexOf(Math.max(...p)) !== s.gold)
    .sort((x, y) => Math.max(...y.p) - Math.max(...x.p));
  if (mistakes.length) {
    console.log(`  ${mistakes.length} mistakes; most confident (shipped T):`);
    for (const { s, p } of mistakes.slice(0, 5)) {
      const pred = p.indexOf(Math.max(...p));
      console.log(`    "${s.shown.slice(0, 70)}"  ->  ${labels[pred]} p=${f(p[pred], 2)}   (gold ${labels[s.gold]} p=${f(p[s.gold], 2)})`);
    }
  }
  const conf = {};
  for (const s of S) {
    const p = softmax(s.logits);
    const pred = labels[p.indexOf(Math.max(...p))];
    if (pred !== labels[s.gold]) conf[`${labels[s.gold]} -> ${pred}`] = (conf[`${labels[s.gold]} -> ${pred}`] ?? 0) + 1;
  }
  const confusions = Object.entries(conf).sort((a, b) => b[1] - a[1]);
  if (confusions.length) console.log(`  confusions: ${confusions.map(([k2, v]) => `${k2} (${v})`).join(", ")}`);
  console.log("");
  table.report[qid] = { bucket: bucketKey(q.type, k), n: S.length, fitted: fittedHere, majorityBaseline: majority, chanceBaseline: chance, verdict, shippedT: Tship, fittedT: fitted[bucketOf[qid]] ?? null, raw: rawM, shipped, refitLOO: looM, confusions: conf };
}

await mkdir(path.dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify(table, null, 2) + "\n");
console.log(`Wrote ${outPath}  temperature_by_options=${JSON.stringify(table.temperature_by_options)}`);
if (args.preset) console.log(`ask.mjs / serve.mjs apply it automatically for preset ${args.preset}.`);
await laya.close();
