/**
 * Experiment: throughput (calls/s, questions/s) and latency of the router's lanes under load.
 *
 *   node experiments/throughput.mjs                                   # lanes webgpu:fp16,cpu:8 ; 3 questions
 *   node experiments/throughput.mjs --questions 5                     # 5 questions per call
 *   node experiments/throughput.mjs --lanes webgpu:fp16,cpu --quick   # 16-thread CPU lane, shorter runs
 *   node experiments/throughput.mjs --scenarios A,C                   # subset of the scenarios below
 *
 * Laya is not generative: the unit is one systemOne call (state + N questions, one forward pass), so
 * throughput is calls/s and questions/s, not tokens/s. Everything goes through LayaRouter.decide() so the
 * numbers include the per-lane queue exactly as serve.mjs / the sidecar see it.
 *
 * Scenarios
 *   A) single lane, closed loop: k = 1,2,4,8 callers always in flight, lane forced. The router serialises
 *      calls per lane, so k > 1 should not raise throughput; this measures it for every lane and shows the
 *      queueing a caller sees.
 *   B) all lanes, closed loop: policy auto vs prefer-gpu at k = 1,2,4,8. Does spreading a burst over
 *      CPU + GPU beat GPU-only?
 *   C) cross-lane interference: latency of one lane while the other lane is kept busy in the same process
 *      (the default deployment loads both). Isolated vs. under load, no queueing involved (distinct lanes).
 *   D) open loop: calls arrive at a fixed rate r (0.33 .. 20 / s) whether or not the previous one finished;
 *      policy auto vs. each lane forced. Achieved rate, latency and lane split per arrival rate; the low rates
 *      are the sporadic / cold-GPU regime. Issuing stops when the backlog exceeds --max-backlog (saturated).
 *
 * Output: results/throughput-<stamp>.json (raw, git-ignored) and results/throughput-<stamp>-summary.md.
 */
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { LayaRouter, isGpuLane } from "../src/ep-router.mjs";
import { cpuInfo, latencyStats, queryGpu } from "../src/metrics.mjs";
import { STATE, QUESTIONS_10 } from "../src/questions.mjs";

const { values: args } = parseArgs({
  options: {
    lanes: { type: "string", default: "webgpu:fp16,cpu:8" },
    questions: { type: "string", default: "3" },
    scenarios: { type: "string", default: "A,B,C,D" },
    concurrency: { type: "string", default: "1,2,4,8" },
    rates: { type: "string", default: "0.33,1,2,5,10,20" },
    calls: { type: "string" }, // closed-loop calls per run (default 24, --quick 12)
    duration: { type: "string" }, // open-loop seconds per rate (default 8, --quick 5)
    "max-backlog": { type: "string", default: "16" },
    "rate-gap": { type: "string", default: "2500" }, // idle ms before each open-loop rate (cold GPU); 0 = start warm
    "no-sampling": { type: "boolean", default: false }, // disable the router's background CPU / nvidia-smi sampling
    "no-pin-process": { type: "boolean", default: false }, // leave the process affinity alone (default: P-cores, see pinProcessToPCores)
    quick: { type: "boolean", default: false },
    out: { type: "string" },
  },
});

const lanes = args.lanes.split(",").map((s) => s.trim()).filter(Boolean);
const nQ = Number(args.questions);
const questions = Object.fromEntries(Object.entries(QUESTIONS_10).slice(0, nQ));
if (Object.keys(questions).length !== nQ) throw new Error(`--questions must be 1..${Object.keys(QUESTIONS_10).length}`);
const scenarios = new Set(args.scenarios.toUpperCase().split(","));
const concurrency = args.concurrency.split(",").map(Number);
const rates = args.rates.split(",").map(Number);
const CALLS = Number(args.calls ?? (args.quick ? 12 : 24));
const DURATION_S = Number(args.duration ?? (args.quick ? 5 : 8));
const MAX_BACKLOG = Number(args["max-backlog"]);
const RATE_GAP_MS = Number(args["rate-gap"]);
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const outJson = args.out ?? path.join("results", `throughput-${stamp}.json`);
const outMd = outJson.replace(/\.json$/, "-summary.md");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stateFor = (i) => ({ ...STATE, userMessage: `${STATE.userMessage} (${i})` });
const f0 = (x) => (Number.isFinite(x) ? x.toFixed(0) : "-");
const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : "-");
const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : "-");

// ---- router -------------------------------------------------------------------------------------------
const t0 = performance.now();
const router = await LayaRouter.create({ lanes, sampleLoad: !args["no-sampling"], pinProcess: !args["no-pin-process"], log: (m) => console.log(`  [router] ${m}`) });
const loaded = [...router.lanes.keys()];
console.log(`router ready in ${((performance.now() - t0) / 1000).toFixed(1)} s; lanes ${loaded.join(", ")}; ${nQ} question(s) per call`);
const gpuLanes = loaded.filter(isGpuLane);
const cpuLanes = loaded.filter((l) => !isGpuLane(l));
const call = (i, o) => router.decide(stateFor(i), questions, o);

const warm = async (o, n = 3) => {
  for (let i = 0; i < n; i++) await call(1000 + i, o);
};

// ---- measurement helpers -----------------------------------------------------------------------------
const record = (r, tStart, tOrigin, extra = {}) => ({
  lane: r.routing.lane,
  ms: r.routing.ms,
  queueMs: r.routing.queueMs,
  totalMs: performance.now() - tStart,
  predictedMs: r.routing.predictedMs,
  pendingAtChoice: r.routing.pendingAtChoice,
  gpuState: r.routing.gpuState,
  at: tStart - tOrigin,
  ...extra,
});

function summarise(records, wallMs) {
  const total = latencyStats(records.map((r) => r.totalMs));
  const inference = latencyStats(records.map((r) => r.ms));
  const queue = latencyStats(records.map((r) => r.queueMs));
  const byLane = {};
  for (const r of records) byLane[r.lane] = (byLane[r.lane] ?? 0) + 1;
  // prediction quality: median of actual inference ms / predicted ms (the router's prediction includes queueing, so compare with total)
  const ratios = records.filter((r) => r.predictedMs > 0).map((r) => r.totalMs / r.predictedMs);
  const predRatio = ratios.length ? latencyStats(ratios).p50 : null;
  return {
    calls: records.length,
    wallMs,
    callsPerSec: (records.length / wallMs) * 1000,
    questionsPerSec: (records.length * nQ / wallMs) * 1000,
    total: { p50: total.p50, p95: total.p95, max: total.max, mean: total.mean },
    inference: { p50: inference.p50, p95: inference.p95, max: inference.max },
    queue: { p50: queue.p50, p95: queue.p95, max: queue.max },
    byLane,
    predRatio,
  };
}

const laneMix = (byLane) => Object.entries(byLane).map(([l, c]) => `${l} ${c}`).join(", ");

/** k callers, each issuing the next call as soon as its previous one returned, until `calls` are done. */
async function closedLoop({ k, calls, opts }) {
  const records = [];
  let next = 0;
  const origin = performance.now();
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= calls) return;
      const s = performance.now();
      const r = await call(i, opts);
      records.push(record(r, s, origin));
    }
  };
  await Promise.all(Array.from({ length: k }, worker));
  return { records, summary: summarise(records, performance.now() - origin) };
}

/** Calls issued at fixed intervals regardless of completion; stops issuing when the backlog exceeds maxBacklog. */
async function openLoop({ ratePerSec, calls, opts, maxBacklog = MAX_BACKLOG }) {
  const interval = 1000 / ratePerSec;
  const records = [];
  const inflight = new Set();
  let issued = 0;
  let saturated = false;
  const origin = performance.now();
  for (let i = 0; i < calls; i++) {
    const due = origin + i * interval;
    const wait = due - performance.now();
    if (wait > 0) await sleep(wait);
    if (inflight.size >= maxBacklog) {
      saturated = true;
      break;
    }
    const s = performance.now();
    issued++;
    const p = call(i, opts)
      .then((r) => records.push(record(r, s, origin, { lateMs: s - due })))
      .finally(() => inflight.delete(p));
    inflight.add(p);
  }
  const issuingMs = performance.now() - origin;
  await Promise.all([...inflight]);
  // achieved rate over the issuing window (the drain after the last issue is not offered load)
  const summary = summarise(records, Math.max(issuingMs, records.length ? Math.max(...records.map((r) => r.at + r.totalMs)) : issuingMs));
  summary.offeredPerSec = ratePerSec;
  summary.issued = issued;
  summary.saturated = saturated;
  summary.inflightMax = Math.max(...records.map((r) => records.filter((o) => o.at <= r.at && o.at + o.totalMs > r.at).length));
  return { records, summary };
}

// ---- run ----------------------------------------------------------------------------------------------
const result = {
  timestamp: new Date().toISOString(),
  node: process.version,
  ort: (await import("onnxruntime-node/package.json", { with: { type: "json" } })).default.version,
  machine: { ...cpuInfo(), gpu: (await queryGpu())?.name ?? null },
  lanes: loaded,
  processAffinity: router.processAffinity,
  questions: nQ,
  calls: CALLS,
  durationS: DURATION_S,
  scenarios: {},
};

console.log("\nwarm-up (shaders, latency estimates):");
const w = await router.warmup({ state: STATE, sizes: [nQ] });
for (const [lane, sizes] of Object.entries(w)) console.log(`  ${lane.padEnd(12)} ${Object.entries(sizes).map(([n, v]) => `${n}q first ${f0(v.firstMs)} / then ${f0(v.ms)} ms`).join("   ")}`);

if (scenarios.has("A")) {
  console.log(`\nA) single lane, closed loop (${CALLS} calls, lane forced)`);
  console.log("  lane          k   calls/s     q/s   total p50   p95     inference p50   queue p50   max");
  const A = [];
  for (const lane of loaded) {
    for (const k of concurrency) {
      await warm({ lane });
      const { summary, records } = await closedLoop({ k, calls: CALLS, opts: { lane } });
      A.push({ lane, k, ...summary, records });
      console.log(`  ${lane.padEnd(12)} ${String(k).padStart(2)}   ${f2(summary.callsPerSec).padStart(6)}   ${f1(summary.questionsPerSec).padStart(6)}   ${f0(summary.total.p50).padStart(6)}   ${f0(summary.total.p95).padStart(6)}     ${f0(summary.inference.p50).padStart(6)}          ${f0(summary.queue.p50).padStart(6)}   ${f0(summary.queue.max).padStart(6)}`);
    }
  }
  result.scenarios.A = A;
}

if (scenarios.has("B") && loaded.length > 1) {
  console.log(`\nB) all lanes, closed loop (${CALLS} calls): policy auto vs prefer-gpu`);
  console.log("  policy       k   calls/s     q/s   total p50   p95     max    lanes                       total/predicted");
  const B = [];
  for (const policy of ["auto", "prefer-gpu"]) {
    for (const k of concurrency) {
      await warm({ policy: "prefer-gpu" });
      if (cpuLanes.length) await warm({ lane: cpuLanes[0] }, 1);
      const { summary, records } = await closedLoop({ k, calls: CALLS, opts: { policy } });
      B.push({ policy, k, ...summary, records });
      console.log(`  ${policy.padEnd(11)} ${String(k).padStart(2)}   ${f2(summary.callsPerSec).padStart(6)}   ${f1(summary.questionsPerSec).padStart(6)}   ${f0(summary.total.p50).padStart(6)}   ${f0(summary.total.p95).padStart(6)}   ${f0(summary.total.max).padStart(6)}    ${laneMix(summary.byLane).padEnd(28)} ${f2(summary.predRatio)}`);
    }
  }
  result.scenarios.B = B;
}

if (scenarios.has("C") && gpuLanes.length && cpuLanes.length) {
  const N = args.quick ? 8 : 12;
  console.log(`\nC) cross-lane interference (${N} sequential calls on one lane while the other lane runs back-to-back)`);
  console.log("  lane          isolated p50   p95   | other lane busy p50   p95   | slowdown");
  const C = [];
  const measure = async (lane) => {
    await warm({ lane });
    const { summary, records } = await closedLoop({ k: 1, calls: N, opts: { lane } });
    return { ...summary, records };
  };
  const withBackground = async (lane, bgLane) => {
    let stop = false;
    const bg = (async () => {
      let i = 0;
      while (!stop) await call(5000 + i++, { lane: bgLane });
    })();
    await sleep(50); // let the background lane start its first inference
    const s = await measure(lane);
    stop = true;
    await bg;
    return s;
  };
  for (const [lane, other] of [
    [gpuLanes[0], cpuLanes[0]],
    [cpuLanes[0], gpuLanes[0]],
  ]) {
    const alone = await measure(lane);
    const busy = await withBackground(lane, other);
    const row = { lane, otherLane: other, alone: alone.inference, busy: busy.inference, slowdown: busy.inference.p50 / alone.inference.p50, aloneMs: alone.records.map((r) => r.ms), busyMs: busy.records.map((r) => r.ms) };
    C.push(row);
    console.log(`  ${lane.padEnd(12)}  ${f0(alone.inference.p50).padStart(8)}   ${f0(alone.inference.p95).padStart(5)}   |  ${f0(busy.inference.p50).padStart(11)}   ${f0(busy.inference.p95).padStart(5)}   |  ${f2(row.slowdown)}x  (other = ${other})`);
  }
  result.scenarios.C = C;
}

if (scenarios.has("D")) {
  console.log(`\nD) open loop: fixed arrival rate for ~${DURATION_S} s (min 8 calls, max 60), policy auto vs lanes forced; backlog cap ${MAX_BACKLOG}; ${RATE_GAP_MS} ms idle before each rate`);
  console.log("  set           rate   achieved   total p50   p95      max    inference p50   lanes                        note");
  const D = [];
  const sets = [...(loaded.length > 1 ? [{ name: "auto", opts: { policy: "auto" } }] : []), ...loaded.map((lane) => ({ name: lane, opts: { lane } }))];
  for (const set of sets) {
    for (const rate of rates) {
      const calls = Math.min(60, Math.max(8, Math.round(rate * DURATION_S)));
      if (RATE_GAP_MS > 0) await sleep(RATE_GAP_MS); // start every rate from the same (cold) GPU state so the low rates are comparable
      const { summary, records } = await openLoop({ ratePerSec: rate, calls, opts: set.opts });
      D.push({ set: set.name, rate, ...summary, records });
      const note = summary.saturated ? `saturated after ${summary.issued} calls` : summary.total.p50 > 1000 / rate ? "latency > interval (queue builds)" : "";
      console.log(`  ${set.name.padEnd(12)} ${f2(rate).padStart(5)}   ${f2(summary.callsPerSec).padStart(6)}/s   ${f0(summary.total.p50).padStart(6)}   ${f0(summary.total.p95).padStart(6)}   ${f0(summary.total.max).padStart(6)}    ${f0(summary.inference.p50).padStart(6)}         ${laneMix(summary.byLane).padEnd(28)} ${note}`);
    }
  }
  result.scenarios.D = D;
  result.rateGapMs = RATE_GAP_MS;
}

result.stats = router.stats();
await router.close();

// ---- write ----------------------------------------------------------------------------------------------
await mkdir(path.dirname(outJson), { recursive: true });
await writeFile(outJson, JSON.stringify(result, null, 2));
await writeFile(outMd, markdown(result));
console.log(`\nwrote ${outJson}\n      ${outMd}`);

function markdown(r) {
  const L = [];
  L.push(`# Laya throughput experiment (${r.timestamp})`, "");
  L.push(`Machine: ${r.machine.model} (${r.machine.logical} threads), ${r.machine.totalMemGiB.toFixed(0)} GiB RAM, GPU: ${r.machine.gpu ?? "-"}; Node ${r.node}, onnxruntime-node ${r.ort}`, "");
  L.push(`Lanes: ${r.lanes.map((l) => `\`${l}\``).join(", ")}. ${r.questions} question(s) per call (~${r.questions * 85} input tokens). Process affinity: ${r.processAffinity?.applied ? `P-cores (${r.processAffinity.mask})` : `not set (${r.processAffinity?.reason ?? "-"})`}. All calls via \`LayaRouter.decide()\`; "total" = wall time seen by the caller incl. the queue, "inference" = \`routing.ms\`.`, "");
  if (r.scenarios.A) {
    L.push(`## A) Single lane, closed loop (${r.calls} calls, k callers in flight, lane forced)`, "");
    L.push("| lane | k | calls/s | q/s | total p50 (ms) | total p95 | inference p50 | queue p50 | queue max |", "|---|---|---|---|---|---|---|---|---|");
    for (const a of r.scenarios.A) L.push(`| \`${a.lane}\` | ${a.k} | ${f2(a.callsPerSec)} | ${f1(a.questionsPerSec)} | ${f0(a.total.p50)} | ${f0(a.total.p95)} | ${f0(a.inference.p50)} | ${f0(a.queue.p50)} | ${f0(a.queue.max)} |`);
    L.push("");
  }
  if (r.scenarios.B) {
    L.push(`## B) All lanes, closed loop (${r.calls} calls): policy auto vs prefer-gpu`, "");
    L.push("| policy | k | calls/s | q/s | total p50 (ms) | total p95 | total max | lanes used | median total/predicted |", "|---|---|---|---|---|---|---|---|---|");
    for (const b of r.scenarios.B) L.push(`| ${b.policy} | ${b.k} | ${f2(b.callsPerSec)} | ${f1(b.questionsPerSec)} | ${f0(b.total.p50)} | ${f0(b.total.p95)} | ${f0(b.total.max)} | ${laneMix(b.byLane)} | ${f2(b.predRatio)} |`);
    L.push("");
  }
  if (r.scenarios.C) {
    L.push("## C) Cross-lane interference (sequential calls on one lane while the other lane runs back-to-back in the same process)", "");
    L.push("| lane | isolated p50 (ms) | isolated p95 | other lane busy p50 | busy p95 | slowdown | other lane |", "|---|---|---|---|---|---|---|");
    for (const c of r.scenarios.C) L.push(`| \`${c.lane}\` | ${f0(c.alone.p50)} | ${f0(c.alone.p95)} | ${f0(c.busy.p50)} | ${f0(c.busy.p95)} | ${f2(c.slowdown)}x | \`${c.otherLane}\` |`);
    L.push("");
  }
  if (r.scenarios.D) {
    L.push(`## D) Open loop: fixed arrival rate (~${r.durationS} s per rate, min 8 / max 60 calls, ${r.rateGapMs} ms idle before each rate)`, "");
    L.push("| set | offered calls/s | achieved calls/s | total p50 (ms) | total p95 | total max | inference p50 | lanes used | note |", "|---|---|---|---|---|---|---|---|---|");
    for (const d of r.scenarios.D) L.push(`| ${d.set === "auto" ? "auto" : `\`${d.set}\``} | ${f2(d.rate)} | ${f2(d.callsPerSec)} | ${f0(d.total.p50)} | ${f0(d.total.p95)} | ${f0(d.total.max)} | ${f0(d.inference.p50)} | ${laneMix(d.byLane)} | ${d.saturated ? `saturated after ${d.issued} calls` : d.total.p50 > 1000 / d.rate ? "latency > interval" : ""} |`);
    L.push("");
  }
  return L.join("\n") + "\n";
}
