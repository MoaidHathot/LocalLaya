/**
 * Execution-provider router for Laya: keeps one Laya session per "lane" (cpu, cpu:8, webgpu, dml) and
 * picks a lane per systemOne call.
 *
 * Why a router at all (measured on i9-14900KF + RTX 4070, see results/ and experiments/):
 *  - Back-to-back traffic:   WebGPU is 3.4x (1 q) .. 5.5x (10 q) faster than the CPU.
 *  - Sporadic traffic:       the GPU drops to ~225 MHz between calls; a 1-question call after a 3 s pause
 *                            takes ~180 ms on WebGPU but ~105 ms on the CPU. The CPU does not care about gaps.
 *  - Availability:           DirectML loads this graph but fails at inference; WebGPU is "experimental".
 *                            A lane that throws is quarantined and the call is retried on another lane.
 *  - Contention:             other processes using the CPU (or the GPU at high clocks) slow that lane down.
 *
 * One queue for all lanes. onnxruntime-node (1.30) runs `session.run()` synchronously on the JS thread
 * (dist/backend.js: setImmediate + blocking run), so inferences of different lanes never overlap in one
 * thread: a CPU call inserted into a GPU burst stalls every GPU call behind it for its full duration
 * (webgpu:fp16 3 q: 49 ms alone, 321-479 ms p50 while cpu:8 ran in the same thread; policy auto at 8 parallel
 * callers fell from 20 to 6.6 calls/s - experiments/throughput.mjs, experiments/interference.mjs). Lanes in
 * worker threads (the default, see lane.mjs) do overlap, but mixing still loses: the pinned CPU pool slows the
 * GPU lane 1.5x while adding 4 calls/s (experiments/worker-lanes.mjs). Hence a single FIFO: every lane waits
 * for the same queue, the choice is by the lane's own predicted latency at the moment it will run, and `ms` is
 * pure inference. The workers buy a responsive main thread (a server keeps accepting requests during a 1 s
 * CPU inference) and lanes that load side by side instead of one after the other.
 *
 * Decision = argmin over lanes of  wait_ms + own_ms(lane, gpu_state_at_start) * contention_factor(lane), where
 * own_ms comes from an exponentially-weighted average of observed latencies keyed by (lane, GPU thermal state,
 * question-count bucket), seeded with priors measured on this machine. Optional epsilon-exploration (only
 * while idle) keeps the estimates of non-chosen lanes fresh. Policies: "auto" | "prefer-gpu" | "prefer-cpu" |
 * "min-cpu"; per-call `lane` override and `deadlineMs` (meet the deadline with the least CPU share).
 */
import { createCpuLoadMeter, queryGpu } from "./metrics.mjs";
import { pinProcessToPCores } from "./laya-client.mjs";
import { openLane } from "./lane.mjs";

export const N_BUCKETS = ["1", "2-3", "4-6", "7-10", "11+"];
export const nBucket = (n) => (n <= 1 ? "1" : n <= 3 ? "2-3" : n <= 6 ? "4-6" : n <= 10 ? "7-10" : "11+");

/** GPU thermal state from the time since this process last finished GPU work. */
export const thermalState = (msSinceGpuWork) => (msSinceGpuWork < 400 ? "hot" : msSinceGpuWork < 2000 ? "warm" : "cold");
/** Interval of the GPU keep-alive calls (see LayaRouter#_scheduleKeepAlive): 500 ms keeps the CUDA lane at ~50 ms cold-start, 1000 ms does not. */
export const KEEP_ALIVE_INTERVAL_MS = 500;
/** A CPU / GPU load sample older than this no longer inflates predictions (sampling pauses while a sidecar idles, and a
 * one-shot sample at start-up must not decide the routing minutes later). */
export const LOAD_STALE_MS = 10_000;

/**
 * Priors (ms per systemOne call). Measured values from this repo's results for 1 / 2-3 / 7-10; the 4-6 and
 * 11+ entries are interpolations. They only matter until the EMA has seen a few real calls.
 * "share" = fraction of the machine's CPU the lane occupies while running (used by deadline / min-cpu).
 * GPU numbers are for the optimised fp16 bundle (tools/optimize_graph.py, 2026-09-23): hot 21 / 32 / 83 ms,
 * after a 1 s pause ~50 / 81, after 3 s ~101 / 156 (experiments/sporadic.mjs --fp16).
 */
export const DEFAULT_PRIORS = {
  // "cpu" = 16 intra-op threads pinned to the P-cores (stable: no 5x slow mode from E-core stragglers)
  cpu: { share: 0.5, ms: { any: { "1": 110, "2-3": 260, "4-6": 480, "7-10": 950, "11+": 1400 } } },
  "cpu:8": { share: 0.25, ms: { any: { "1": 100, "2-3": 330, "4-6": 560, "7-10": 900, "11+": 1350 } } },
  // ORT default thread count (24), unpinned: fastest when Windows schedules well (213 ms / 3 q) but p95 > 1.1 s
  "cpu:auto": { share: 0.65, ms: { any: { "1": 105, "2-3": 235, "4-6": 430, "7-10": 750, "11+": 1100 } } },
  webgpu: {
    share: 0.02,
    ms: {
      hot: { "1": 22, "2-3": 33, "4-6": 55, "7-10": 84, "11+": 130 },
      warm: { "1": 50, "2-3": 80, "4-6": 105, "7-10": 140, "11+": 200 },
      cold: { "1": 105, "2-3": 160, "4-6": 190, "7-10": 230, "11+": 320 },
    },
  },
  // DirectML runs the optimised graph (allowzero=0) but is 6-10x slower than WebGPU for batch > 1 (measured 2026-09-23)
  dml: { share: 0.03, ms: { hot: { "1": 19, "2-3": 217, "4-6": 240, "7-10": 264, "11+": 330 }, warm: { "1": 50, "2-3": 250, "4-6": 280, "7-10": 300, "11+": 380 }, cold: { "1": 110, "2-3": 320, "4-6": 350, "7-10": 380, "11+": 470 } } },
  // CUDA EP in a Python process (tools/cuda_lane.py): launch-bound, almost flat in batch size (hot 9 / 12 / 24 ms for
  // 1 / 3 / 10 q). Cold is bimodal without keep-alive (experiments/sporadic.mjs --ep cuda: 3 s gap 50-300 ms, p50 130-215)
  // and ~45-55 ms with the keep-alive on; the priors below are the no-keep-alive medians, the EMA learns the rest.
  cuda: { share: 0.03, ms: { hot: { "1": 10, "2-3": 13, "4-6": 18, "7-10": 25, "11+": 34 }, warm: { "1": 35, "2-3": 45, "4-6": 55, "7-10": 65, "11+": 85 }, cold: { "1": 150, "2-3": 170, "4-6": 190, "7-10": 210, "11+": 250 } } },
};

export const isGpuLane = (lane) => lane.startsWith("webgpu") || lane.startsWith("dml") || lane.startsWith("cuda");
/** Bundle directory used by the ":fp16" lane variant (output of tools/optimize_graph.py). */
export const FP16_MODEL_DIR = process.env.LAYA_FP16_DIR ?? "models/laya-onnx-fp16";
/**
 * Lane syntax:
 *   "webgpu" | "webgpu:fp16" (half-precision bundle: half the VRAM, ~5-13 % faster) | "dml"
 *   "cpu" (16 threads pinned to P-cores) | "cpu:8" (8 threads, pinned) | "cpu:24:nopin" (unpinned) |
 *   "cpu:auto" (ORT default thread count, unpinned)
 */
export const parseLane = (lane) => {
  const [ep, t, flag] = lane.split(":");
  if (ep !== "cpu") return { ep, threads: undefined, pin: false, modelDir: t === "fp16" ? FP16_MODEL_DIR : undefined };
  if (t === "auto") return { ep, threads: undefined, pin: false };
  return { ep, threads: t ? Number(t) : undefined, pin: flag !== "nopin" };
};

/**
 * Pure decision function (unit-tested).
 * @param {Array<{lane:string, predictedMs:number, ownMs?:number, share:number}>} candidates  healthy lanes;
 *   predictedMs = queue wait + own inflated latency, ownMs = the lane's own part (defaults to predictedMs)
 * @param {{policy?:string, deadlineMs?:number, explore?:number, rng?:()=>number}} opts
 */
export function chooseLane(candidates, opts = {}) {
  const { policy = "auto", deadlineMs, explore = 0, rng = Math.random } = opts;
  if (!candidates.length) throw new Error("chooseLane: no healthy lanes");
  const own = (c) => c.ownMs ?? c.predictedMs;
  const byLatency = [...candidates].sort((a, b) => a.predictedMs - b.predictedMs);
  const gpu = candidates.filter((c) => isGpuLane(c.lane)).sort((a, b) => a.predictedMs - b.predictedMs)[0];
  const cpu = candidates.filter((c) => !isGpuLane(c.lane)).sort((a, b) => a.predictedMs - b.predictedMs)[0];

  if (deadlineMs) {
    const meeting = candidates.filter((c) => c.predictedMs <= deadlineMs).sort((a, b) => a.share - b.share || a.predictedMs - b.predictedMs);
    if (meeting.length) return { ...meeting[0], reason: `meets ${deadlineMs} ms deadline with least CPU share` };
    return { ...byLatency[0], reason: `no lane predicted to meet ${deadlineMs} ms; fastest chosen` };
  }
  if (policy === "prefer-gpu" && gpu) return { ...gpu, reason: "policy prefer-gpu" };
  if (policy === "prefer-cpu" && cpu) return { ...cpu, reason: "policy prefer-cpu" };
  if (policy === "min-cpu") {
    const least = [...candidates].sort((a, b) => a.share - b.share || a.predictedMs - b.predictedMs)[0];
    return { ...least, reason: "policy min-cpu" };
  }
  // auto: fastest, with epsilon-exploration among lanes whose own latency is within 2x of the best
  const best = byLatency[0];
  if (explore > 0 && byLatency.length > 1 && rng() < explore) {
    const alt = byLatency.slice(1).filter((c) => own(c) <= own(best) * 2);
    if (alt.length) return { ...alt[Math.floor(rng() * alt.length)], reason: "exploration", explored: true };
  }
  return { ...best, reason: `fastest predicted (${byLatency.map((c) => `${c.lane} ${c.predictedMs.toFixed(0)}`).join(", ")})` };
}

/** Rough token count of a string (ModernBERT tokenizer on English text: ~4 chars per token). */
const estTokens = (s) => Math.ceil(s.length / 4);

/**
 * Work estimate for one systemOne call = questions x padded sequence length (the ONNX batch is [n, L]).
 * L per question = header + option texts (capped at head_max_len 192) + state (rest of max_len 512).
 * Only used to make latency observations transferable between short and long states.
 */
export function estimateWork(state, questions) {
  const stateTokens = estTokens(typeof state === "string" ? state : JSON.stringify(state));
  let L = 0;
  for (const q of Object.values(questions)) {
    const ins = typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions);
    const opts = q.type === "noul" ? "false: no, the statement does not hold true: yes, the statement holds" : Array.isArray(q.criteria) ? q.criteria.join(" ") : Object.entries(q.criteria).map(([k, v]) => `${k}: ${v ?? ""}`).join(" ");
    const head = Math.min(192, 8 + estTokens(ins) + estTokens(opts) + 2 * (q.type === "noul" ? 2 : Array.isArray(q.criteria) ? q.criteria.length : Object.keys(q.criteria).length));
    L = Math.max(L, Math.min(512, head + 3 + Math.min(stateTokens, 512 - head - 3)));
  }
  return Object.keys(questions).length * L;
}

/** Work at which the DEFAULT_PRIORS were measured (the PoC state: ~85 padded tokens per question). */
const PRIOR_REF_WORK = { "1": 85, "2-3": 3 * 85, "4-6": 5 * 90, "7-10": 10 * 95, "11+": 15 * 95 };

/** EMA of ms-per-work-unit per (GPU state, question bucket), with prior fallback. */
export class LatencyModel {
  constructor(lane, priors = DEFAULT_PRIORS[lane] ?? DEFAULT_PRIORS[lane.split(":")[0]] ?? DEFAULT_PRIORS.cpu, alpha = 0.3) {
    this.lane = lane;
    this.priors = priors.ms;
    this.share = priors.share;
    this.alpha = alpha;
    this.ema = {}; // `${state}|${bucket}` -> { rate, ms, n }   (rate = ms per work unit; ms = last observed, for reporting)
  }
  stateKey(state) {
    return this.priors.any ? "any" : state;
  }
  priorRate(s, b) {
    const ms = this.priors[s]?.[b] ?? this.priors.any?.[b] ?? 200;
    return ms / PRIOR_REF_WORK[b];
  }
  /** @param work  estimateWork(...) result; defaults to the prior reference work of the bucket */
  predict(n, state, work) {
    const b = nBucket(n);
    const s = this.stateKey(state);
    const w = work ?? PRIOR_REF_WORK[b];
    const e = this.ema[`${s}|${b}`];
    if (e) return { ms: e.rate * w, source: `ema(n=${e.n})` };
    // neighbouring bucket's observed/prior ratio corrects this bucket's prior
    for (const nb of N_BUCKETS) {
      const ne = this.ema[`${s}|${nb}`];
      if (ne) return { ms: (ne.rate / this.priorRate(s, nb)) * this.priorRate(s, b) * w, source: `ema-scaled(from ${nb})` };
    }
    return { ms: this.priorRate(s, b) * w, source: "prior" };
  }
  observe(n, state, ms, work) {
    const b = nBucket(n);
    const key = `${this.stateKey(state)}|${b}`;
    const rate = ms / (work ?? PRIOR_REF_WORK[b]);
    const e = this.ema[key];
    this.ema[key] = e ? { rate: this.alpha * rate + (1 - this.alpha) * e.rate, ms, n: e.n + 1 } : { rate, ms, n: 1 };
  }
}

/** Representative question set of size n for warm-ups (choice / noul / score mix). */
const warmupQuestions = (n) => Object.fromEntries(Array.from({ length: n }, (_, i) => [`q${i}`, i % 3 === 0 ? { type: "choice", instructions: `question ${i}`, criteria: { a: "option a", b: "option b", c: "option c", d: "option d" } } : i % 3 === 1 ? { type: "noul", instructions: `statement ${i}` } : { type: "score", instructions: `level ${i}`, criteria: ["low", "mid", "high", "max"] }]));
const WARMUP_STATE = { warmup: "the quick brown fox jumps over the lazy dog" };

export class LayaRouter {
  /** Use LayaRouter.create(). */
  constructor(opts) {
    this.opts = opts;
    this.lanes = new Map(); // lane -> { lane, session, model, healthy, dead, failures, quarantinedUntil, calls, pending, loadMs, probeMs }
    this.loading = new Set(); // lanes still being loaded / probed / warmed
    this.queue = Promise.resolve(); // one FIFO for every lane (see the header: mixing lanes loses)
    this.inflight = []; // queued + running calls: { lane, ownMs, startedAt }
    this.direct = new Set(); // start-up probe / warm-up calls running outside the FIFO (see _direct)
    this.lastGpuWorkEnd = -Infinity;
    this.gpuBusy = 0;
    this.load = { cpuOthers: 0, gpuOthersUtil: 0, gpuSmClockMHz: null, gpuMemFreeMiB: null, sampledAt: null, cpuSampledAt: null };
    this._timers = [];
    this._keepAliveUntil = 0;
    this.log = opts.log ?? (() => {});
    this.history = [];
    this.closed = false;
    this.ready = Promise.resolve(this);
  }

  /**
   * @param {object} o
   * @param {string[]} [o.lanes=["cuda:fp16","webgpu:fp16","cpu:8"]] lanes to load; a lane whose load or probe fails is dropped (e.g. cuda without the Python venv)
   * @param {"auto"|"prefer-gpu"|"prefer-cpu"|"min-cpu"} [o.policy="auto"]
   * @param {number} [o.explore=0.05]                exploration probability (auto policy, idle only)
   * @param {string|object} [o.calibration]          applied to every lane (see loadLaya)
   * @param {boolean} [o.sampleLoad=true]            background CPU / nvidia-smi sampling for contention
   * @param {number} [o.gpuKeepAliveMs=0]            after a GPU call keep the GPU awake for this long with a tiny call every 500 ms (serve.mjs: 30 s)
   * @param {number} [o.minGpuFreeMiB=2200]          skip GPU lanes when less VRAM than this is free
   * @param {boolean} [o.pinProcess=true]            Windows hybrid CPUs: restrict the process to the P-cores (see pinProcessToPCores)
   * @param {boolean} [o.workers=true]               run each lane's session in a worker thread (lane.mjs); false = in this thread
   * @param {string} [o.python]                      Python with onnxruntime-gpu for cuda lanes (default LAYA_PYTHON or .venv/Scripts/python.exe)
   * @param {number} [o.deviceId]                    GPU index for cuda / dml lanes (default 0)
   * @param {boolean} [o.cudaGraph=true]             cuda lanes: CUDA Graph replay on static bucket graphs (tools/cuda_lane.py); false = dynamic graph only
   * @param {string} [o.graphBuckets]                cuda lanes: buckets built right after start ("1x96x8,3x96x8,..."); the rest come from traffic
   * @param {string} [o.graphBucketsFile]            cuda lanes: JSON memory of the shapes real traffic used (written by the lane); its top
   *                                                 graphMaxEager shapes (default 6) are built right after start instead of the built-in defaults
   * @param {number} [o.graphMaxSessions]            cuda lanes: LRU bound on captured buckets (default 16)
   * @param {number} [o.graphMaxVramMiB]             cuda lanes: VRAM budget for the buckets' activations (default 1536)
   * @param {number} [o.graphMaxWork]                cuda lanes: rows x length above which calls stay dynamic (default 1536)
   * @param {{state?:object, sizes?:number[]}} [o.warmup]  warm every lane (shader compile, EMA priming) before it serves
   * @param {"all"|"first"} [o.waitFor="all"]        resolve when every lane is done, or as soon as the first lane serves;
   *                                                 `router.ready` resolves when all lanes are done either way
   * @param {(lane:string, info:{loadMs:number, probeMs:number, warmup:object|null, readyMs:number})=>void} [o.onLaneReady]
   * @param {(m:string)=>void} [o.log]
   */
  static async create(o = {}) {
    const r = new LayaRouter({ lanes: ["cuda:fp16", "webgpu:fp16", "cpu:8"], policy: "auto", explore: 0.05, sampleLoad: true, gpuKeepAliveMs: 0, minGpuFreeMiB: 2200, pinProcess: true, workers: true, warmup: null, waitFor: "all", onLaneReady: null, ...o });
    // in the background: overlaps with the model loads, done before the first call
    const pinned = r.opts.pinProcess ? pinProcessToPCores({ log: r.log }) : Promise.resolve({ applied: false, reason: "disabled" });
    const gpu = await queryGpu();
    // clocks and free VRAM are always useful; the utilisation only when sampling keeps it fresh (a one-shot value
    // would inflate GPU predictions for the router's whole life - e.g. a test suite's previous run still showing)
    if (gpu) r._ingestGpu(gpu, { util: r.opts.sampleLoad });
    let firstReady;
    const first = new Promise((resolve) => (firstReady = resolve));
    for (const lane of r.opts.lanes) r.loading.add(lane);
    r.ready = Promise.all(
      r.opts.lanes.map((lane) =>
        r._probeLane(lane, gpu).then((ok) => {
          r.loading.delete(lane);
          if (ok) firstReady();
        }),
      ),
    ).then(() => {
      if (!r.lanes.size) throw new Error("LayaRouter: no lane could be loaded");
      return r;
    });
    r.ready.catch(() => {}); // awaited below and/or by the caller; never an unhandled rejection
    if (r.opts.waitFor === "first") await Promise.race([first, r.ready]);
    else await r.ready;
    r.processAffinity = await pinned;
    if (r.opts.sampleLoad) r._startSampling(!!gpu);
    return r;
  }

  /** Lanes that are still loading (waitFor: "first"). */
  get pendingLanes() {
    return [...this.loading];
  }

  /** Load, probe (a real inference: DML loads fine but throws on run) and optionally warm one lane. */
  async _probeLane(lane, gpu) {
    const { ep, threads, pin, modelDir } = parseLane(lane);
    if (isGpuLane(lane) && gpu && gpu.memTotalMiB - gpu.memUsedMiB < this.opts.minGpuFreeMiB) {
      this.log(`lane ${lane}: skipped, only ${gpu.memTotalMiB - gpu.memUsedMiB} MiB VRAM free`);
      return false;
    }
    const t0 = performance.now();
    let session = null;
    try {
      session = await openLane(lane, { ep, threads, pinToPCores: pin, modelDir, calibration: this.opts.calibration, python: this.opts.python, deviceId: this.opts.deviceId, cudaGraph: this.opts.cudaGraph, graphBuckets: this.opts.graphBuckets, graphBucketsFile: this.opts.graphBucketsFile, graphMaxEager: this.opts.graphMaxEager, graphMaxSessions: this.opts.graphMaxSessions, graphMaxVramMiB: this.opts.graphMaxVramMiB, graphMaxWork: this.opts.graphMaxWork }, { worker: this.opts.workers, log: () => {} });
      if (this.closed) throw new Error("router closed while loading");
      const L = { lane, session, model: new LatencyModel(lane), healthy: true, dead: false, failures: 0, quarantinedUntil: 0, calls: 0, pending: 0, loadMs: session.loadMs, probeMs: 0 };
      session.onDeath((cause) => this._laneDied(L, cause));
      const probe = await this._direct(L, () => session.systemOne({ probe: "ok" }, { q: { type: "noul", instructions: "Is this a probe?" } }));
      L.probeMs = probe.ms;
      const warm = this.opts.warmup ? await this._warmLane(L, this.opts.warmup) : null;
      if (this.closed) throw new Error("router closed while loading");
      this.lanes.set(lane, L);
      const readyMs = performance.now() - t0;
      this.log(`lane ${lane}: ready (load ${(session.loadMs / 1000).toFixed(1)} s, probe ${L.probeMs.toFixed(0)} ms${warm ? `, warm-up ${Object.entries(warm).map(([n, v]) => `${n}q ${v.ms.toFixed(0)}`).join(" / ")} ms` : ""}, ${session.mode}${session.graph ? `, cuda graphs ${session.graph.enabled ? "on" : "off"}` : ""})`);
      this.opts.onLaneReady?.(lane, { loadMs: session.loadMs, probeMs: L.probeMs, warmup: warm, readyMs, mode: session.mode });
      return true;
    } catch (e) {
      this.log(`lane ${lane}: unavailable - ${String(e?.message ?? e).split("\n")[0].slice(0, 140)}`);
      await session?.close().catch(() => {});
      return false;
    }
  }

  _laneDied(L, cause) {
    L.healthy = false;
    L.dead = true;
    L.quarantinedUntil = Infinity;
    if (!this.closed) this.log(`lane ${L.lane}: gone (${cause}); calls go to the remaining lanes`);
  }

  /**
   * Run `fn` on a lane right away, outside the FIFO, with the GPU bookkeeping the queue would do. For the
   * start-up probe and warm-up: they must not wait behind (or hold up) the calls of the lanes already
   * serving - a GPU lane that joins 2 s late costs a start-up burst 3-5x more than the moment of overlap.
   * @returns {Promise<{result:any, ms:number, stateAtStart:string}>}
   */
  async _direct(L, fn) {
    const gpu = isGpuLane(L.lane);
    const tStart = performance.now();
    const stateAtStart = gpu ? thermalState(tStart - this.lastGpuWorkEnd) : "any";
    if (gpu) this.gpuBusy++;
    const p = (async () => {
      try {
        const result = await fn();
        return { result, ms: performance.now() - tStart, stateAtStart };
      } finally {
        if (gpu) {
          this.gpuBusy--;
          this.lastGpuWorkEnd = performance.now();
        }
      }
    })();
    this.direct.add(p);
    p.finally(() => this.direct.delete(p)).catch(() => {});
    return p;
  }

  /**
   * Run a call on the FIFO shared by every lane. `L` is the lane it is provisionally bound to and `ownMs` its
   * predicted duration (both feed the wait estimates of the callers behind it). When the call reaches the front
   * `rebind()` may return another lane record to run on instead (lanes that joined, died or got quarantined
   * while the call waited). Times the call from when it actually starts; tracks the GPU busy state and the
   * per-lane pending counts.
   * @returns {Promise<{L:object, result:any, ms:number, queueMs:number, stateAtStart:string}>}
   */
  _enqueue(L, ownMs, run, rebind = null) {
    const t0 = performance.now();
    const entry = { lane: L.lane, ownMs, startedAt: null };
    this.inflight.push(entry);
    L.pending++;
    return (this.queue = this.queue.catch(() => {}).then(async () => {
      const other = rebind?.();
      if (other && other !== L) {
        L.pending--;
        L = other;
        L.pending++;
        entry.lane = L.lane;
      }
      const gpu = isGpuLane(L.lane);
      const tStart = performance.now();
      entry.startedAt = tStart;
      const stateAtStart = gpu ? thermalState(tStart - this.lastGpuWorkEnd) : "any";
      if (gpu) this.gpuBusy++;
      try {
        const result = await run(L);
        return { L, result, ms: performance.now() - tStart, queueMs: tStart - t0, stateAtStart };
      } catch (e) {
        if (e && typeof e === "object" && !e.lane) e.lane = L.lane; // tell the caller which lane the call ran on
        throw e;
      } finally {
        if (gpu) {
          this.gpuBusy--;
          this.lastGpuWorkEnd = performance.now();
        }
        L.pending--;
        const i = this.inflight.indexOf(entry);
        if (i >= 0) this.inflight.splice(i, 1);
      }
    }));
  }

  /**
   * Warm one lane, outside the FIFO (see _direct): GPU lanes get two calls per size (the first compiles shaders
   * for that shape and is not recorded, the second primes the EMA); CPU lanes have nothing to compile, so one
   * call per size does both.
   */
  async _warmLane(L, { state = WARMUP_STATE, sizes = [1, 3, 10] } = {}) {
    const report = {};
    const gpu = isGpuLane(L.lane);
    for (const n of sizes) {
      const qs = warmupQuestions(n);
      const work = estimateWork(state, qs);
      // exec.graph = false: the cuda lane answers on its dynamic graph and does not build a bucket for the warm-up
      // shapes (real traffic decides which shapes deserve one); the EMA starts from the dynamic times, which is
      // conservative and corrected by the first replays
      const first = await this._direct(L, () => L.session.systemOne(state, qs, undefined, { graph: false }));
      const timed = gpu ? await this._direct(L, () => L.session.systemOne(state, qs, undefined, { graph: false })) : first;
      L.model.observe(n, timed.stateAtStart, timed.ms, work);
      report[n] = { firstMs: first.ms, ms: timed.ms };
    }
    return report;
  }

  _ingestGpu(g, { util = true } = {}) {
    const idleForMs = performance.now() - this.lastGpuWorkEnd;
    // util measured while we were not using the GPU is external load; at idle clocks the compositor's
    // "utilisation" is harmless, so weight it by clock level.
    if (util && !this.gpuBusy && idleForMs > 500) {
      const clockFactor = g.smClockMaxMHz ? Math.min(1, g.smClockMHz / g.smClockMaxMHz) : 0.5;
      this.load.gpuOthersUtil = (g.utilPct / 100) * (clockFactor > 0.5 ? 1 : 0.15);
    }
    this.load.gpuSmClockMHz = g.smClockMHz;
    this.load.gpuMemFreeMiB = g.memTotalMiB - g.memUsedMiB;
    this.load.sampledAt = Date.now();
  }

  _startSampling(hasGpu) {
    this._hasGpu = hasGpu;
    this.sampling = true;
    const meter = createCpuLoadMeter();
    meter.tick();
    const cpuTimer = setInterval(() => {
      const m = meter.tick();
      if (m) {
        this.load.cpuOthers = m.others;
        this.load.cpuSampledAt = Date.now();
      }
    }, 1000);
    cpuTimer.unref();
    this._timers.push(cpuTimer);
    if (hasGpu) {
      let inFlight = false;
      const gpuTimer = setInterval(async () => {
        if (inFlight) return;
        inFlight = true;
        try {
          const g = await queryGpu();
          if (g) this._ingestGpu(g);
        } finally {
          inFlight = false;
        }
      }, 2000);
      gpuTimer.unref();
      this._timers.push(gpuTimer);
    }
  }

  /** Stop the background CPU / nvidia-smi sampling (e.g. while a sidecar is parked idle). Idempotent. */
  pauseSampling() {
    for (const t of this._timers) clearInterval(t);
    this._timers = [];
    this.sampling = false;
  }

  /** Restart sampling after pauseSampling(). Idempotent; no-op when sampling was disabled at creation. */
  resumeSampling() {
    if (this.sampling || !this.opts.sampleLoad) return;
    this._startSampling(!!this._hasGpu);
  }

  /** Contention multiplier for a lane's predicted latency; a sample older than LOAD_STALE_MS counts as no load. */
  _inflation(lane) {
    const now = Date.now();
    if (isGpuLane(lane)) return 1 / Math.max(0.2, 1 - (this.load.sampledAt && now - this.load.sampledAt < LOAD_STALE_MS ? this.load.gpuOthersUtil : 0));
    return 1 / Math.max(0.15, 1 - (this.load.cpuSampledAt && now - this.load.cpuSampledAt < LOAD_STALE_MS ? this.load.cpuOthers : 0));
  }

  /** Predicted time until the queue is empty: full estimates for queued calls, the remainder for the running one. */
  waitMs(now = performance.now()) {
    let wait = 0;
    for (const e of this.inflight) wait += e.startedAt === null ? e.ownMs : Math.max(0, e.ownMs - (now - e.startedAt));
    return wait;
  }

  /**
   * Current predictions for n questions (and optional work estimate), for every healthy lane:
   * predictedMs = waitMs (same for every lane: one queue) + ownMs (the lane's own latency x contention).
   * The GPU thermal state is the one expected when the call starts, not now: "hot" when GPU work is queued
   * ahead of it; otherwise the projected idle time - but never "cold" while other calls are queued, because a
   * queue means traffic and the GPU's first call warms the clocks for everything behind it.
   * `atFront`: the call is about to run (nothing ahead of it): no wait, the GPU state as it is right now.
   */
  predictions(n, work, { atFront = false } = {}) {
    const now = performance.now();
    const waitMs = atFront ? 0 : this.waitMs(now);
    const pending = atFront ? 0 : this.inflight.length;
    let state = !atFront && this.inflight.some((e) => isGpuLane(e.lane)) ? "hot" : thermalState(now + waitMs - this.lastGpuWorkEnd);
    if (pending && state === "cold") state = "warm";
    const out = [];
    for (const L of this.lanes.values()) {
      if (L.dead || (!L.healthy && now < L.quarantinedUntil)) continue;
      const p = L.model.predict(n, state, work);
      const ownMs = p.ms * this._inflation(L.lane);
      out.push({ lane: L.lane, rawMs: p.ms, ownMs, waitMs, predictedMs: waitMs + ownMs, pending, source: p.source, share: L.model.share, state: isGpuLane(L.lane) ? state : "any" });
    }
    return out;
  }

  /** Candidates for one call: healthy lanes not yet tried, or the forced lane (which ignores a quarantine). */
  _candidates(n, work, o, tried, atFront) {
    let candidates = this.predictions(n, work, { atFront }).filter((c) => !tried.includes(c.lane));
    if (!o.lane) return candidates;
    candidates = candidates.filter((c) => c.lane === o.lane);
    if (candidates.length) return candidates;
    const L = this.lanes.get(o.lane);
    if (!L) throw new Error(`lane ${o.lane} not loaded (have ${[...this.lanes.keys()].join(", ")}${this.loading.size ? `; loading ${[...this.loading].join(", ")}` : ""})`);
    if (L.dead) throw new Error(`lane ${o.lane} is gone`);
    if (tried.includes(o.lane)) return [];
    const waitMs = atFront ? 0 : this.waitMs();
    const ownMs = L.model.predict(n, thermalState(performance.now() - this.lastGpuWorkEnd), work).ms;
    return [{ lane: o.lane, ownMs, waitMs, predictedMs: waitMs + ownMs, pending: atFront ? 0 : this.inflight.length, share: L.model.share, state: "forced" }];
  }

  /**
   * Answer `questions` about `state`, choosing the lane automatically.
   * @param {object} [o] { lane, policy, deadlineMs, calibration, exec }
   *   calibration: a table from calibrate.mjs ({ temperature_by_options }) applied for this call only; without
   *   it the lane's temperatures are the ones it was created with (LayaRouter.create({ calibration }) or shipped).
   *   exec: execution options passed to the lane, e.g. { graph: false } to skip CUDA graph replay on a cuda lane
   *   (lanes ignore what they do not understand); reported back in routing.exec.
   * @returns result with an extra `routing` field
   */
  async decide(state, questions, o = {}) {
    if (this.closed) throw new Error("router is closed");
    if (!this.lanes.size) throw new Error(this.loading.size ? `no lane ready yet (loading ${[...this.loading].join(", ")})` : "no lanes loaded");
    const n = Object.keys(questions).length;
    const work = estimateWork(state, questions);
    const policy = o.policy ?? this.opts.policy;
    const tried = [];
    for (let attempt = 0; attempt < this.lanes.size; attempt++) {
      // Provisional choice now (fastest predicted, no exploration): it sets the wait estimate for the callers
      // behind this one. The binding choice happens when the call reaches the front of the queue, because
      // lanes may have joined (start-up), died or been quarantined meanwhile - and exploration only makes
      // sense there, when nobody is queued behind to be delayed by an exploratory slow call.
      const provisional = this._candidates(n, work, o, tried, false);
      if (!provisional.length) break;
      let choice = chooseLane(provisional, { policy, deadlineMs: o.deadlineMs, explore: 0 });
      const provisionalLane = choice.lane;
      let alternatives = provisional;
      const rebind = () => {
        const fresh = this._candidates(n, work, o, tried, true);
        if (!fresh.length) return null; // keep the provisional lane; if it is gone the call fails and the retry loop moves on
        alternatives = fresh;
        const c = chooseLane(fresh, { policy, deadlineMs: o.deadlineMs, explore: o.lane || this.inflight.length > 1 ? 0 : this.opts.explore });
        choice = { ...c, waitMs: choice.waitMs, predictedMs: choice.waitMs + c.ownMs, pending: choice.pending };
        return this.lanes.get(c.lane);
      };
      let L = this.lanes.get(provisionalLane);
      try {
        // Timing and the GPU thermal state are taken when the call actually starts, not when it was queued;
        // nothing else runs between start and completion, so `ms` is the inference alone. The per-call
        // temperature override travels with the call and is applied by the lane right before it runs.
        const run = await this._enqueue(L, choice.ownMs, (lane) => lane.session.systemOne(state, questions, o.calibration?.temperature_by_options, o.exec), rebind);
        L = run.L;
        const exec = run.result.exec ?? null; // what the lane did (cuda: { mode: "graph"|"dynamic", bucket, remoteMs, stallMs? })
        if (exec) delete run.result.exec;
        // a call that waited for a CUDA-graph build step inside the process (exec.stallMs) is not evidence about the
        // lane's latency: learning it would send the next calls of a burst to a slower lane (measured: 8 calls 1.5 s
        // instead of 0.2 s, half of them on webgpu)
        L.model.observe(n, run.stateAtStart, exec?.stallMs ? Math.max(run.ms - exec.stallMs, exec.remoteMs ?? 1) : run.ms, work);
        L.calls++;
        L.healthy = true;
        L.failures = 0;
        const gpu = isGpuLane(L.lane);
        const routing = { lane: L.lane, ms: run.ms, queueMs: run.queueMs, n, work, gpuState: run.stateAtStart, predictedMs: choice.predictedMs, ownMs: choice.ownMs, waitMs: choice.waitMs ?? 0, pendingAtChoice: choice.pending ?? 0, reason: choice.reason, explored: !!choice.explored, ...(exec ? { exec } : {}), ...(L.lane !== provisionalLane ? { provisionalLane } : {}), alternatives: alternatives.filter((c) => c.lane !== L.lane).map((c) => ({ lane: c.lane, predictedMs: c.predictedMs })), load: { cpuOthers: this.load.cpuOthers, gpuOthersUtil: this.load.gpuOthersUtil } };
        this.history.push(routing);
        if (this.history.length > 1000) this.history.shift();
        if (gpu && this.opts.gpuKeepAliveMs > 0) this._scheduleKeepAlive(L);
        return { ...run.result, routing };
      } catch (e) {
        L = (e?.lane && this.lanes.get(e.lane)) || L; // the lane the call actually ran on (rebind may have moved it)
        L.failures++;
        L.healthy = false;
        L.quarantinedUntil = L.dead ? Infinity : performance.now() + 60_000;
        tried.push(L.lane);
        if (!this.closed) this.log(`lane ${L.lane} failed (${String(e?.message ?? e).split("\n")[0].slice(0, 120)}); ${L.dead ? "gone" : "quarantined 60 s"}, retrying on another lane`);
      }
    }
    throw new Error(`all lanes failed for this call (tried ${tried.join(", ")})`);
  }

  /**
   * Keep the GPU awake after real GPU work by issuing a tiny call every KEEP_ALIVE_INTERVAL_MS on the GPU lane
   * that served last, until gpuKeepAliveMs have passed since the last real call. Measured 2026-09-23 on the CUDA
   * lane (results/sidecar-modes-2026-09-23-summary.md): a 3-question call after a 3 s pause takes ~200 ms cold
   * (bimodal 50-300) and ~45-55 ms with keep-alive, for 1-4 W of GPU power while the keep-alive runs; a 1 s
   * interval no longer helps. On the WebGPU lane the same trick did not help (2026-09-21).
   */
  _scheduleKeepAlive(lane) {
    this._keepAliveUntil = performance.now() + this.opts.gpuKeepAliveMs;
    if (lane) this._keepAliveLane = lane;
    if (this._keepAliveTimer) return;
    const tick = async () => {
      this._keepAliveTimer = null;
      if (performance.now() >= this._keepAliveUntil || this.closed) return;
      const L = this._keepAliveLane && this._keepAliveLane.healthy && !this._keepAliveLane.dead ? this._keepAliveLane : [...this.lanes.values()].find((x) => isGpuLane(x.lane) && x.healthy && !x.dead);
      if (!L) return;
      if (!this.inflight.length) {
        try {
          // exec.graph = false: a synthetic shape must not earn a CUDA-graph bucket (buckets follow real traffic)
          await this._enqueue(L, 30, () => L.session.systemOne("x", { k: { type: "noul", instructions: "keep-alive" } }, undefined, { graph: false }));
          this.keepAliveCalls = (this.keepAliveCalls ?? 0) + 1;
        } catch {
          /* ignore */
        }
      }
      this._keepAliveTimer = setTimeout(tick, KEEP_ALIVE_INTERVAL_MS);
      this._keepAliveTimer.unref();
    };
    this._keepAliveTimer = setTimeout(tick, KEEP_ALIVE_INTERVAL_MS);
    this._keepAliveTimer.unref();
  }

  /** Compile GPU shaders / prime the EMA with representative calls on every lane (also available per lane via create({ warmup })). */
  async warmup({ state = WARMUP_STATE, sizes = [1, 3, 10] } = {}) {
    const report = {};
    for (const L of this.lanes.values()) report[L.lane] = await this._warmLane(L, { state, sizes });
    return report;
  }

  /** stats() plus what the process lanes report about themselves (cuda: captured buckets, hits, VRAM). */
  async detailedStats() {
    const s = this.stats();
    for (const L of this.lanes.values()) {
      if (typeof L.session.stats === "function") s.lanes[L.lane].process = await L.session.stats().catch(() => null);
    }
    return s;
  }

  stats() {
    const lanes = {};
    for (const L of this.lanes.values()) lanes[L.lane] = { healthy: L.healthy, dead: L.dead, mode: L.session.mode, calls: L.calls, pending: L.pending, failures: L.failures, loadMs: L.loadMs, probeMs: L.probeMs, share: L.model.share, ema: L.model.ema, ...(L.session.graph ? { cudaGraph: L.session.graph } : {}) };
    return { lanes, loading: [...this.loading], queue: { pending: this.inflight.length, waitMs: this.waitMs() }, load: this.load, sampling: !!this.sampling, processAffinity: this.processAffinity ?? null, keepAliveCalls: this.keepAliveCalls ?? 0, gpuState: thermalState(performance.now() - this.lastGpuWorkEnd) };
  }

  /**
   * Release every session (and end the worker threads). New calls are rejected immediately; calls already
   * queued finish first (a session must not be released under a running inference). Lanes still loading are
   * closed as they finish.
   */
  async close() {
    this.closed = true;
    this.pauseSampling();
    if (this._keepAliveTimer) clearTimeout(this._keepAliveTimer);
    this._keepAliveUntil = 0;
    await this.queue.catch(() => {});
    await Promise.allSettled([...this.direct]);
    await Promise.all([...this.lanes.values()].map((L) => L.session.close().catch(() => {})));
    this.lanes.clear();
  }
}
