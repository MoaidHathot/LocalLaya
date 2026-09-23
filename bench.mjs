/**
 * Benchmark one execution provider across question-set sizes, with resource metrics.
 *
 *   node bench.mjs --ep cpu                     # default sizes 1,3,10 ; 20 timed runs each
 *   node bench.mjs --ep cpu --threads 8         # tune intra-op threads (P-cores only on hybrid CPUs)
 *   node bench.mjs --ep dml                     # DirectML on the RTX 4070
 *   node bench.mjs --ep webgpu --verbose        # ORT info-level logs (shows CPU-fallback node partitioning)
 *   node bench.mjs --ep cpu --pin                # 16 threads pinned to the P-cores (stable latency on hybrid CPUs)
 *   node bench.mjs --ep webgpu --fp16            # optimised half-precision bundle from tools/optimize_graph.py
 *   node bench.mjs --ep dml --out results/x.json
 *
 * Runs in its own process so RSS / GPU-memory deltas are attributable to one configuration.
 */
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { loadLaya, MODEL_REVISION } from "./src/laya-client.mjs";
import { Sampler, cpuInfo, latencyStats, queryGpu, snapshotSystem } from "./src/metrics.mjs";
import { STATE, QUESTION_SETS } from "./src/questions.mjs";

const { values: args } = parseArgs({
  options: {
    ep: { type: "string", default: "cpu" },
    threads: { type: "string" },
    sizes: { type: "string", default: "1,3,10" },
    runs: { type: "string", default: "20" },
    warmup: { type: "string", default: "3" },
    out: { type: "string" },
    verbose: { type: "boolean", default: false },
    opt: { type: "string" },
    pin: { type: "boolean", default: false },
    fp16: { type: "boolean", default: false },
    calibration: { type: "string" },
    "no-gpu-poll": { type: "boolean", default: false },
  },
});

const ep = args.ep;
const threads = args.threads ? Number(args.threads) : undefined;
const sizes = args.sizes.split(",").map(Number);
const runs = Number(args.runs);
const warmup = Number(args.warmup);
const label = `${ep}${threads ? `-t${threads}` : ""}${args.pin ? "-pin" : ""}${args.fp16 ? "-fp16" : ""}${args.opt ? `-opt-${args.opt}` : ""}`;
const gpuPoll = !args["no-gpu-poll"];

const log = (m) => console.error(`[bench:${label}] ${m}`);

const result = {
  label,
  ep,
  threads: threads ?? null,
  modelRevision: MODEL_REVISION,
  node: process.version,
  ort: (await import("onnxruntime-node/package.json", { with: { type: "json" } })).default.version,
  machine: { ...cpuInfo(), gpu: (await queryGpu())?.name ?? null },
  timestamp: new Date().toISOString(),
  load: null,
  cases: [],
  error: null,
};

// ---- load ----------------------------------------------------------------------------------------
const sysBefore = snapshotSystem();
const gpuBefore = await queryGpu();
let laya;
try {
  const loaded = await loadLaya({ ep, threads, log, logSeverityLevel: args.verbose ? 1 : undefined, optLevel: args.opt, pinToPCores: args.pin, modelDir: args.fp16 ? "models/laya-onnx-fp16" : undefined, calibration: args.calibration });
  laya = loaded.laya;
  const sysAfter = snapshotSystem();
  const gpuAfter = await queryGpu();
  result.load = {
    ms: loaded.loadMs,
    source: loaded.source,
    rssBeforeMiB: sysBefore.rssMiB,
    rssAfterMiB: sysAfter.rssMiB,
    gpuMemBeforeMiB: gpuBefore?.memUsedMiB ?? null,
    gpuMemAfterMiB: gpuAfter?.memUsedMiB ?? null,
  };
  log(`loaded in ${(loaded.loadMs / 1000).toFixed(2)} s; RSS +${(sysAfter.rssMiB - sysBefore.rssMiB).toFixed(0)} MiB; GPU mem +${(gpuAfter?.memUsedMiB ?? 0) - (gpuBefore?.memUsedMiB ?? 0)} MiB`);
} catch (e) {
  result.error = `load failed: ${e?.message ?? e}`;
  log(result.error);
  await emit();
  process.exit(2);
}

// ---- cases ---------------------------------------------------------------------------------------
for (const n of sizes) {
  const questions = QUESTION_SETS[n];
  if (!questions) {
    log(`no question set of size ${n}; skipping`);
    continue;
  }
  const c = { questions: n, warmupMs: [], latency: null, metrics: null, sample: null, error: null };
  try {
    for (let i = 0; i < warmup; i++) {
      const t0 = performance.now();
      await laya.systemOne(STATE, questions);
      c.warmupMs.push(performance.now() - t0);
    }
    const sampler = new Sampler({ intervalMs: 200, gpu: gpuPoll });
    await sampler.start();
    const times = [];
    let last;
    const tStart = performance.now();
    for (let i = 0; i < runs; i++) {
      const t0 = performance.now();
      last = await laya.systemOne(STATE, questions);
      times.push(performance.now() - t0);
    }
    const wallMs = performance.now() - tStart;
    c.metrics = await sampler.stop();
    c.latency = latencyStats(times);
    c.times = times;
    c.throughputCallsPerSec = (runs / wallMs) * 1000;
    c.inputTokens = last.usage.input_tokens;
    c.sample = last.answers;
    log(
      `${String(n).padStart(2)} q: p50 ${c.latency.p50.toFixed(1)} ms  mean ${c.latency.mean.toFixed(1)}  p95 ${c.latency.p95.toFixed(1)}  ` +
        `| cpu ${c.metrics.cpuCorePct?.avg.toFixed(0)}% core  rss ${c.metrics.rssMiB?.max.toFixed(0)} MiB` +
        (c.metrics.gpu?.utilPct ? `  gpu ${c.metrics.gpu.utilPct.avg.toFixed(0)}% (max ${c.metrics.gpu.utilPct.max}) mem ${c.metrics.gpu.memUsedMiB.max} MiB ${c.metrics.gpu.powerW.avg.toFixed(0)} W` : ""),
    );
  } catch (e) {
    c.error = e?.message ?? String(e);
    log(`${n} q failed: ${c.error}`);
  }
  result.cases.push(c);
}

await laya.close();
result.maxRssMiB = process.resourceUsage().maxRSS / 1024;
await emit();

async function emit() {
  const out = args.out ?? path.join("results", `bench-${label}-${result.timestamp.replace(/[:.]/g, "-")}.json`);
  await mkdir(path.dirname(out), { recursive: true });
  await writeFile(out, JSON.stringify(result, null, 2) + "\n");
  log(`wrote ${out}`);
  // machine-readable line for the orchestrator
  console.log(JSON.stringify({ out }));
}
