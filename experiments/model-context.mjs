#!/usr/bin/env node
/**
 * Measured context/latency smoke test, one model resident at a time (not an accuracy benchmark).
 *   node experiments/model-context.mjs --models vega,multilingual --device cuda --words 0,1024,4096,7800
 *   node experiments/model-context.mjs --models vega --device cuda --words 16384,32768,70000
 *   node experiments/model-context.mjs --models vega,multilingual --device cpu --words 0
 * Raw results go to results/; use the actual state_tokens in each record, not the filler word count.
 */
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { ModelRouter } from "../src/model-router.mjs";
import { PROJECT_ROOT, MODELS, modelName, PYTHON_MANIFEST } from "../src/models.mjs";

const { values: a } = parseArgs({ options: {
  models: { type: "string", default: "vega,multilingual" }, device: { type: "string", default: "cuda" },
  words: { type: "string", default: "0,1024,4096,7800" }, repeats: { type: "string", default: "3" }, out: { type: "string" },
} });
const models = a.models.split(",").map(modelName);
const words = a.words.split(",").map(Number);
const repeats = Number(a.repeats);
if (words.some((n) => !Number.isSafeInteger(n) || n < 0) || !Number.isSafeInteger(repeats) || repeats < 1) throw new Error("--words needs nonnegative integers; --repeats needs a positive integer");
const question = { task: { type: "choice", instructions: "What does the customer request in the document?", criteria: {
  refund: "refund a duplicate charge", password: "reset a forgotten password", plan: "upgrade a subscription plan",
} } };
const ending = "The customer was charged twice and requests a refund of the duplicate payment.";
const out = path.resolve(PROJECT_ROOT, a.out ?? `results/model-context-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
await mkdir(path.dirname(out), { recursive: true });
const report = { date: new Date().toISOString(), device: a.device, runtime: PYTHON_MANIFEST.runtime, profiles: {}, records: [], note: "Synthetic repeated filler + a final refund request. Capacity/latency smoke test, not a document-accuracy benchmark." };
const save = () => writeFile(out, JSON.stringify(report, null, 2) + "\n");

for (const model of models) {
  let router;
  try {
    const t0 = performance.now();
    router = await ModelRouter.create({ model, device: a.device, sampleLoad: false, gpuKeepAliveMs: 0 });
    report.profiles[model] = { maxLen: MODELS[model].maxLen, loadMs: performance.now() - t0, info: router.backend.info ?? null };
    // Warm up outside the measurements, even if the requested sweep starts with a long document.
    await router.decide(ending, question);
    for (const n of words) {
      if (n > MODELS[model].maxLen - 256) continue;
      const state = "background ".repeat(n) + ending;
      const times = [];
      let r;
      try {
        for (let i = 0; i < (n === 0 ? repeats : 1); i++) {
          r = await router.decide(state, question);
          times.push(r.routing.ms);
        }
        const row = { model, fillerWords: n, timesMs: times, medianMs: [...times].sort((x, y) => x - y)[Math.floor(times.length / 2)],
          usage: r.usage, answer: r.answers.task, exec: r.routing.exec };
        report.records.push(row);
        console.log(`${model} ${a.device}: ${r.usage.state_tokens ?? "?"} state tokens, ${row.medianMs.toFixed(1)} ms, ${r.routing.exec?.peakVramMiB ?? "?"} MiB, ${r.answers.task.choice} p=${r.answers.task.probabilities.refund}, truncated=${r.usage.truncated ?? "unknown"}`);
      } catch (e) {
        report.records.push({ model, fillerWords: n, error: e.message });
        console.error(`${model} at ${n} filler words: ${e.message}`);
        await save();
        break;
      }
      await save();
    }
  } finally { await router?.close(); }
}
await save();
console.log(`Saved ${out}`);
