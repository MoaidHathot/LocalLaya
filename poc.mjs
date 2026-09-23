/**
 * PoC: load Laya once, answer typed questions about a state in ONE forward pass, print the result.
 *
 *   node poc.mjs                         # CPU, 16 threads pinned to P-cores (stable); --nopin for ORT defaults
 *   node poc.mjs --ep webgpu             # NVIDIA GPU via the WebGPU EP (3-5x faster back-to-back)
 *   node poc.mjs --ep webgpu --fp16      # optimised half-precision bundle (tools/optimize_graph.py): half the VRAM
 *   node poc.mjs --ep dml                # DirectML: loads, but fails at inference on this graph (ORT 1.30)
 *   node poc.mjs --calibration calibration/smart-home-v3.json   # refit temperatures for this domain
 *   node poc.mjs --runs 10               # timed runs after warm-up (default 10)
 */
import { parseArgs } from "node:util";
import { loadLaya, MODEL_REVISION } from "./src/laya-client.mjs";
import { latencyStats, snapshotSystem, queryGpu } from "./src/metrics.mjs";
import { STATE, QUESTIONS_3 } from "./src/questions.mjs";

const { values: args } = parseArgs({
  options: {
    ep: { type: "string", default: "cpu" },
    runs: { type: "string", default: "10" },
    threads: { type: "string" },
    nopin: { type: "boolean", default: false },
    fp16: { type: "boolean", default: false },
    calibration: { type: "string" },
  },
});
const runs = Number(args.runs);
const threads = args.threads ? Number(args.threads) : undefined;
const pinToPCores = args.ep === "cpu" && !args.nopin;
const modelDir = args.fp16 ? "models/laya-onnx-fp16" : undefined;

const log = (m) => console.log(`[laya] ${m}`);
let lastLine = "";
const onProgress = ({ file, received, total }) => {
  const line = total
    ? `downloading ${file}: ${((received / total) * 100).toFixed(1)}% (${(received / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB)`
    : `downloading ${file}: ${(received / 1e6).toFixed(0)} MB`;
  if (line !== lastLine) {
    process.stdout.write(`\r${line.padEnd(90)}`);
    lastLine = line;
  }
};

console.log(`Loading Laya (ep=${args.ep}${threads ? `, threads=${threads}` : ""}${pinToPCores ? ", pinned to P-cores" : ""}${args.fp16 ? ", fp16" : ""}, revision=${MODEL_REVISION.slice(0, 12)})...`);
const before = snapshotSystem();
const gpuBefore = await queryGpu();
const { laya, loadMs, source, modelDir: loadedFrom } = await loadLaya({ ep: args.ep, threads, pinToPCores, modelDir, calibration: args.calibration, log, onProgress });
if (lastLine) process.stdout.write("\n");
const after = snapshotSystem();
const gpuAfter = await queryGpu();
console.log(`Model loaded in ${(loadMs / 1000).toFixed(2)} s from ${source} (${loadedFrom})`);
console.log(`Process RSS: ${before.rssMiB.toFixed(0)} -> ${after.rssMiB.toFixed(0)} MiB (+${(after.rssMiB - before.rssMiB).toFixed(0)} MiB)`);
if (gpuBefore && gpuAfter) {
  console.log(`GPU memory used: ${gpuBefore.memUsedMiB} -> ${gpuAfter.memUsedMiB} MiB (+${gpuAfter.memUsedMiB - gpuBefore.memUsedMiB} MiB) on ${gpuAfter.name}`);
}

// Warm-up (first run includes lazy kernel/graph initialisation and is not representative).
console.log("\nWarming up...");
const tw = performance.now();
await laya.systemOne(STATE, QUESTIONS_3);
console.log(`Warm-up call: ${(performance.now() - tw).toFixed(1)} ms`);

// Timed runs
const times = [];
let result;
for (let i = 0; i < runs; i++) {
  const t0 = performance.now();
  result = await laya.systemOne(STATE, QUESTIONS_3);
  const ms = performance.now() - t0;
  times.push(ms);
  console.log(`Run ${String(i + 1).padStart(2)}: ${ms.toFixed(2)} ms`);
}
const st = latencyStats(times);
console.log(`\nAverage: ${st.mean.toFixed(2)} ms   p50: ${st.p50.toFixed(2)}   p95: ${st.p95.toFixed(2)}   min: ${st.min.toFixed(2)}   max: ${st.max.toFixed(2)}`);

console.log("\nResult (3 questions, one forward pass):");
console.dir(result, { depth: null, colors: true });

// Sanity check: does the model actually discriminate between different states?
console.log("\nSanity check on contrasting states (intent choice + P(should_execute)):");
const contrast = [
  { ...STATE, userMessage: "What's the weather like tomorrow?" },
  { ...STATE, userMessage: "Hey, how are you doing today?" },
  { ...STATE, userMessage: "Lock the front door right now, someone is trying to get in!" },
];
for (const s of contrast) {
  const r = await laya.systemOne(s, QUESTIONS_3);
  const a = r.answers;
  console.log(
    `  "${s.userMessage}"\n     -> intent=${a.intent.choice} (conf ${a.intent.confidence}), should_execute=${a.should_execute.noul}, urgency=${a.urgency.score}/3, tokens=${r.usage.input_tokens}`,
  );
}

await laya.close();
