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
 * Decision = argmin over lanes of  predicted_ms(lane) * contention_factor(lane), where predicted_ms comes
 * from an exponentially-weighted average of observed latencies keyed by (lane, GPU thermal state,
 * question-count bucket), seeded with priors measured on this machine. Optional epsilon-exploration keeps the
 * estimates of non-chosen lanes fresh. Policies: "auto" | "prefer-gpu" | "prefer-cpu" | "min-cpu"; per-call
 * `lane` override and `deadlineMs` (meet the deadline with the least CPU share).
 */
import { createCpuLoadMeter, queryGpu } from "./metrics.mjs";
import { loadLaya } from "./laya-client.mjs";

export const N_BUCKETS = ["1", "2-3", "4-6", "7-10", "11+"];
export const nBucket = (n) => (n <= 1 ? "1" : n <= 3 ? "2-3" : n <= 6 ? "4-6" : n <= 10 ? "7-10" : "11+");

/** GPU thermal state from the time since this process last finished GPU work. */
export const thermalState = (msSinceGpuWork) => (msSinceGpuWork < 400 ? "hot" : msSinceGpuWork < 2000 ? "warm" : "cold");

/**
 * Priors (ms per systemOne call). Measured values from this repo's results for 1 / 2-3 / 7-10; the 4-6 and
 * 11+ entries are interpolations. They only matter until the EMA has seen a few real calls.
 * "share" = fraction of the machine's CPU the lane occupies while running (used by deadline / min-cpu).
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
      hot: { "1": 30, "2-3": 55, "4-6": 90, "7-10": 136, "11+": 220 },
      warm: { "1": 60, "2-3": 90, "4-6": 130, "7-10": 190, "11+": 300 },
      cold: { "1": 180, "2-3": 200, "4-6": 240, "7-10": 300, "11+": 420 },
    },
  },
  dml: { share: 0.03, ms: { hot: { "1": 40, "2-3": 70, "4-6": 110, "7-10": 170, "11+": 260 }, warm: { "1": 80, "2-3": 110, "4-6": 160, "7-10": 230, "11+": 350 }, cold: { "1": 200, "2-3": 230, "4-6": 280, "7-10": 350, "11+": 480 } } },
};

export const isGpuLane = (lane) => lane.startsWith("webgpu") || lane.startsWith("dml");
/** Bundle directory used by the ":fp16" lane variant (output of tools/convert_fp16.py). */
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
 * @param {Array<{lane:string, predictedMs:number, share:number}>} candidates  healthy lanes with inflated predictions
 * @param {{policy?:string, deadlineMs?:number, explore?:number, rng?:()=>number}} opts
 */
export function chooseLane(candidates, opts = {}) {
  const { policy = "auto", deadlineMs, explore = 0, rng = Math.random } = opts;
  if (!candidates.length) throw new Error("chooseLane: no healthy lanes");
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
  // auto: fastest, with epsilon-exploration among lanes within 2x of the best
  const best = byLatency[0];
  if (explore > 0 && byLatency.length > 1 && rng() < explore) {
    const alt = byLatency.slice(1).filter((c) => c.predictedMs <= best.predictedMs * 2);
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

export class LayaRouter {
  /** Use LayaRouter.create(). */
  constructor(opts) {
    this.opts = opts;
    this.lanes = new Map(); // lane -> { laya, model, healthy, failures, quarantinedUntil, calls }
    this.lastGpuWorkEnd = -Infinity;
    this.gpuBusy = 0;
    this.load = { cpuOthers: 0, gpuOthersUtil: 0, gpuSmClockMHz: null, gpuMemFreeMiB: null, sampledAt: null };
    this._timers = [];
    this._keepAliveUntil = 0;
    this.log = opts.log ?? (() => {});
    this.history = [];
  }

  /**
   * @param {object} o
   * @param {string[]} [o.lanes=["webgpu","cpu"]]   lanes to load; the first GPU lane whose probe fails is dropped
   * @param {"auto"|"prefer-gpu"|"prefer-cpu"|"min-cpu"} [o.policy="auto"]
   * @param {number} [o.explore=0.05]                exploration probability (auto policy)
   * @param {string|object} [o.calibration]          applied to every lane (see loadLaya)
   * @param {boolean} [o.sampleLoad=true]            background CPU / nvidia-smi sampling for contention
   * @param {number} [o.gpuKeepAliveMs=0]            after a GPU call keep the GPU clocks up for this long with tiny dummy calls
   * @param {number} [o.minGpuFreeMiB=2200]          skip GPU lanes when less VRAM than this is free
   * @param {(m:string)=>void} [o.log]
   */
  static async create(o = {}) {
    const r = new LayaRouter({ lanes: ["webgpu", "cpu"], policy: "auto", explore: 0.05, sampleLoad: true, gpuKeepAliveMs: 0, minGpuFreeMiB: 2200, ...o });
    const gpu = await queryGpu();
    if (gpu) r._ingestGpu(gpu);
    await Promise.all(r.opts.lanes.map((lane) => r._probeLane(lane, gpu)));
    if (!r.lanes.size) throw new Error("LayaRouter: no lane could be loaded");
    if (r.opts.sampleLoad) r._startSampling(!!gpu);
    return r;
  }

  async _probeLane(lane, gpu) {
    const { ep, threads, pin, modelDir } = parseLane(lane);
    if (isGpuLane(lane) && gpu && gpu.memTotalMiB - gpu.memUsedMiB < this.opts.minGpuFreeMiB) {
      this.log(`lane ${lane}: skipped, only ${gpu.memTotalMiB - gpu.memUsedMiB} MiB VRAM free`);
      return;
    }
    const t0 = performance.now();
    try {
      const { laya } = await loadLaya({ ep, threads, pinToPCores: pin, modelDir, calibration: this.opts.calibration, log: () => {}, logSeverityLevel: 3 });
      // functional probe: a real inference (DML loads fine here but throws on run)
      const p0 = performance.now();
      await laya.systemOne({ probe: "ok" }, { q: { type: "noul", instructions: "Is this a probe?" } });
      const probeMs = performance.now() - p0;
      this.lanes.set(lane, { lane, laya, model: new LatencyModel(lane), healthy: true, failures: 0, quarantinedUntil: 0, calls: 0, loadMs: performance.now() - t0, probeMs, shippedTemps: { ...laya.config.temperature_by_options }, queue: Promise.resolve() });
      this.log(`lane ${lane}: ready (load ${((performance.now() - t0) / 1000).toFixed(1)} s, probe ${probeMs.toFixed(0)} ms)`);
    } catch (e) {
      this.log(`lane ${lane}: unavailable - ${String(e?.message ?? e).split("\n")[0].slice(0, 140)}`);
    }
  }

  _ingestGpu(g) {
    const idleForMs = performance.now() - this.lastGpuWorkEnd;
    // util measured while we were not using the GPU is external load; at idle clocks the compositor's
    // "utilisation" is harmless, so weight it by clock level.
    if (!this.gpuBusy && idleForMs > 500) {
      const clockFactor = g.smClockMaxMHz ? Math.min(1, g.smClockMHz / g.smClockMaxMHz) : 0.5;
      this.load.gpuOthersUtil = (g.utilPct / 100) * (clockFactor > 0.5 ? 1 : 0.15);
    }
    this.load.gpuSmClockMHz = g.smClockMHz;
    this.load.gpuMemFreeMiB = g.memTotalMiB - g.memUsedMiB;
    this.load.sampledAt = Date.now();
  }

  _startSampling(hasGpu) {
    const meter = createCpuLoadMeter();
    meter.tick();
    const cpuTimer = setInterval(() => {
      const m = meter.tick();
      if (m) this.load.cpuOthers = m.others;
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

  /** Contention multiplier for a lane's predicted latency. */
  _inflation(lane) {
    if (isGpuLane(lane)) return 1 / Math.max(0.2, 1 - this.load.gpuOthersUtil);
    return 1 / Math.max(0.15, 1 - this.load.cpuOthers);
  }

  /** Current predictions for n questions (and optional work estimate), for every healthy lane. */
  predictions(n, work) {
    const now = performance.now();
    const state = thermalState(now - this.lastGpuWorkEnd);
    const out = [];
    for (const L of this.lanes.values()) {
      if (!L.healthy && now < L.quarantinedUntil) continue;
      const p = L.model.predict(n, state, work);
      out.push({ lane: L.lane, rawMs: p.ms, predictedMs: p.ms * this._inflation(L.lane), source: p.source, share: L.model.share, state: isGpuLane(L.lane) ? state : "any" });
    }
    return out;
  }

  /**
   * Answer `questions` about `state`, choosing the lane automatically.
   * @param {object} [o] { lane, policy, deadlineMs, calibration }
   *   calibration: a table from calibrate.mjs ({ temperature_by_options }) applied for this call only; without
   *   it the lane's temperatures are the ones it was created with (LayaRouter.create({ calibration }) or shipped).
   * @returns result with an extra `routing` field
   */
  async decide(state, questions, o = {}) {
    const n = Object.keys(questions).length;
    const work = estimateWork(state, questions);
    const tried = [];
    for (let attempt = 0; attempt < this.lanes.size; attempt++) {
      let candidates = this.predictions(n, work).filter((c) => !tried.includes(c.lane));
      if (o.lane) {
        candidates = candidates.filter((c) => c.lane === o.lane);
        if (!candidates.length) {
          const L = this.lanes.get(o.lane);
          if (!L) throw new Error(`lane ${o.lane} not loaded (have ${[...this.lanes.keys()].join(", ")})`);
          candidates = [{ lane: o.lane, predictedMs: L.model.predict(n, thermalState(performance.now() - this.lastGpuWorkEnd), work).ms, share: L.model.share, state: "forced" }];
        }
      }
      if (!candidates.length) break;
      const choice = chooseLane(candidates, { policy: o.policy ?? this.opts.policy, deadlineMs: o.deadlineMs, explore: o.lane ? 0 : this.opts.explore });
      const L = this.lanes.get(choice.lane);
      const gpu = isGpuLane(choice.lane);
      const stateAtStart = gpu ? thermalState(performance.now() - this.lastGpuWorkEnd) : "any";
      const t0 = performance.now();
      try {
        if (gpu) this.gpuBusy++;
        // One call at a time per lane: the session serialises the work anyway (measured: no throughput gain
        // from concurrency) and it lets us switch the per-call temperatures without racing another call.
        const result = await (L.queue = L.queue.catch(() => {}).then(() => {
          const temps = L.laya.config.temperature_by_options;
          for (const k of Object.keys(temps)) delete temps[k];
          Object.assign(temps, L.shippedTemps, o.calibration?.temperature_by_options ?? {});
          return L.laya.systemOne(state, questions);
        }));
        const ms = performance.now() - t0;
        if (gpu) this.lastGpuWorkEnd = performance.now();
        L.model.observe(n, stateAtStart, ms, work);
        L.calls++;
        L.healthy = true;
        L.failures = 0;
        const routing = { lane: choice.lane, ms, n, work, gpuState: stateAtStart, predictedMs: choice.predictedMs, reason: choice.reason, explored: !!choice.explored, alternatives: candidates.filter((c) => c.lane !== choice.lane).map((c) => ({ lane: c.lane, predictedMs: c.predictedMs })), load: { cpuOthers: this.load.cpuOthers, gpuOthersUtil: this.load.gpuOthersUtil } };
        this.history.push(routing);
        if (this.history.length > 1000) this.history.shift();
        if (gpu && this.opts.gpuKeepAliveMs > 0) this._scheduleKeepAlive();
        return { ...result, routing };
      } catch (e) {
        L.failures++;
        L.healthy = false;
        L.quarantinedUntil = performance.now() + 60_000;
        tried.push(choice.lane);
        this.log(`lane ${choice.lane} failed (${String(e?.message ?? e).split("\n")[0].slice(0, 120)}); quarantined 60 s, retrying on another lane`);
      } finally {
        if (gpu) this.gpuBusy--;
      }
    }
    throw new Error(`all lanes failed for this call (tried ${tried.join(", ")})`);
  }

  /** Keep the GPU clocks up after real GPU work by issuing tiny calls until gpuKeepAliveMs has elapsed. */
  _scheduleKeepAlive() {
    this._keepAliveUntil = performance.now() + this.opts.gpuKeepAliveMs;
    if (this._keepAliveTimer) return;
    const gpuLane = [...this.lanes.values()].find((L) => isGpuLane(L.lane) && L.healthy);
    if (!gpuLane) return;
    const tick = async () => {
      this._keepAliveTimer = null;
      if (performance.now() >= this._keepAliveUntil) return;
      if (!this.gpuBusy) {
        try {
          this.gpuBusy++;
          await gpuLane.laya.systemOne("x", { k: { type: "noul", instructions: "keep-alive" } });
          this.lastGpuWorkEnd = performance.now();
          this.keepAliveCalls = (this.keepAliveCalls ?? 0) + 1;
        } catch {
          /* ignore */
        } finally {
          this.gpuBusy--;
        }
      }
      this._keepAliveTimer = setTimeout(tick, 250);
      this._keepAliveTimer.unref();
    };
    this._keepAliveTimer = setTimeout(tick, 250);
    this._keepAliveTimer.unref();
  }

  /** Compile GPU shaders / prime the EMA with representative calls on every lane. */
  async warmup({ state = { warmup: "the quick brown fox jumps over the lazy dog" }, sizes = [1, 3, 10] } = {}) {
    const mk = (n) => Object.fromEntries(Array.from({ length: n }, (_, i) => [`q${i}`, i % 3 === 0 ? { type: "choice", instructions: `question ${i}`, criteria: { a: "option a", b: "option b", c: "option c", d: "option d" } } : i % 3 === 1 ? { type: "noul", instructions: `statement ${i}` } : { type: "score", instructions: `level ${i}`, criteria: ["low", "mid", "high", "max"] }]));
    const report = {};
    for (const L of this.lanes.values()) {
      report[L.lane] = {};
      for (const n of sizes) {
        const qs = mk(n);
        const first = performance.now();
        await L.laya.systemOne(state, qs); // first call: may include shader compilation -> not recorded
        const firstMs = performance.now() - first;
        const t1 = performance.now();
        await L.laya.systemOne(state, qs);
        const ms = performance.now() - t1;
        L.model.observe(n, isGpuLane(L.lane) ? "hot" : "any", ms, estimateWork(state, qs));
        if (isGpuLane(L.lane)) this.lastGpuWorkEnd = performance.now();
        report[L.lane][n] = { firstMs, ms };
      }
    }
    return report;
  }

  stats() {
    const lanes = {};
    for (const L of this.lanes.values()) lanes[L.lane] = { healthy: L.healthy, calls: L.calls, failures: L.failures, loadMs: L.loadMs, share: L.model.share, ema: L.model.ema };
    return { lanes, load: this.load, keepAliveCalls: this.keepAliveCalls ?? 0, gpuState: thermalState(performance.now() - this.lastGpuWorkEnd) };
  }

  async close() {
    for (const t of this._timers) clearInterval(t);
    if (this._keepAliveTimer) clearTimeout(this._keepAliveTimer);
    this._keepAliveUntil = 0;
    await Promise.all([...this.lanes.values()].map((L) => L.laya.close().catch(() => {})));
    this.lanes.clear();
  }
}
