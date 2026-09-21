/**
 * Lightweight resource sampler for benchmarks.
 *
 *  - Process CPU: process.cpuUsage() deltas -> % of one logical core (can exceed 100 with threads)
 *                 and normalised to the whole machine.
 *  - Process RAM: RSS from process.memoryUsage(), sampled; plus maxRSS from process.resourceUsage().
 *  - System RAM:  os.totalmem() - os.freemem().
 *  - GPU:         nvidia-smi polled in the background (utilisation %, memory used MiB, power W, temp).
 *                 DirectML work is not visible under --query-compute-apps (it is D3D12, not CUDA),
 *                 so GPU memory is reported as whole-adapter usage; the delta vs. the pre-load baseline
 *                 approximates what this process holds.
 *
 * nvidia-smi is spawned asynchronously every `intervalMs`; it is a separate process and does not block
 * the Node event loop, but it does consume ~1-2 % of one core while polling.
 */
import { execFile } from "node:child_process";
import os from "node:os";
import { promisify } from "node:util";

const execFileP = promisify(execFile);
const NCPU = os.cpus().length;

export async function queryGpu() {
  try {
    const { stdout } = await execFileP(
      "nvidia-smi",
      ["--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.max.sm", "--format=csv,noheader,nounits"],
      { windowsHide: true, timeout: 5000 },
    );
    const [name, util, memUsed, memTotal, power, temp, smClock, smClockMax] = stdout.trim().split("\n")[0].split(",").map((s) => s.trim());
    return {
      name,
      utilPct: Number(util),
      memUsedMiB: Number(memUsed),
      memTotalMiB: Number(memTotal),
      powerW: Number(power),
      tempC: Number(temp),
      smClockMHz: Number(smClock),
      smClockMaxMHz: Number(smClockMax),
    };
  } catch {
    return null;
  }
}

/**
 * Machine-wide CPU utilisation from os.cpus() deltas, split into "ours" (this process) and "others".
 * Call tick() periodically; the first call returns null.
 */
export function createCpuLoadMeter() {
  let lastCpus = os.cpus();
  let lastProc = process.cpuUsage();
  let lastT = performance.now();
  return {
    tick() {
      const cpus = os.cpus();
      const proc = process.cpuUsage(lastProc);
      const now = performance.now();
      let busy = 0;
      let total = 0;
      cpus.forEach((c, i) => {
        const p = lastCpus[i]?.times ?? c.times;
        const dBusy = c.times.user - p.user + (c.times.nice - p.nice) + (c.times.sys - p.sys) + (c.times.irq - p.irq);
        const dIdle = c.times.idle - p.idle;
        busy += dBusy;
        total += dBusy + dIdle;
      });
      const dtMs = now - lastT;
      const ours = Math.min(1, (proc.user + proc.system) / 1000 / Math.max(1, dtMs) / cpus.length);
      lastCpus = cpus;
      lastProc = process.cpuUsage();
      lastT = now;
      if (total <= 0) return null;
      const machine = busy / total;
      return { machine, ours, others: Math.max(0, machine - ours) };
    },
  };
}

export function snapshotSystem() {
  const mem = process.memoryUsage();
  return {
    rssMiB: mem.rss / 1048576,
    heapUsedMiB: mem.heapUsed / 1048576,
    externalMiB: mem.external / 1048576,
    systemUsedMiB: (os.totalmem() - os.freemem()) / 1048576,
    systemTotalMiB: os.totalmem() / 1048576,
  };
}

export class Sampler {
  /** @param {{intervalMs?: number, gpu?: boolean}} opts */
  constructor({ intervalMs = 250, gpu = true } = {}) {
    this.intervalMs = intervalMs;
    this.gpu = gpu;
    this.samples = [];
    this._timer = null;
    this._lastCpu = null;
    this._lastT = null;
  }

  async start() {
    this.samples = [];
    this._lastCpu = process.cpuUsage();
    this._lastT = performance.now();
    this._startT = this._lastT;
    this._inFlight = false;
    const tick = async () => {
      if (this._inFlight) return;
      this._inFlight = true;
      try {
        const now = performance.now();
        const cpu = process.cpuUsage(this._lastCpu);
        const dtMs = now - this._lastT;
        this._lastCpu = process.cpuUsage();
        this._lastT = now;
        const cpuCorePct = dtMs > 50 ? ((cpu.user + cpu.system) / 1000 / dtMs) * 100 : null;
        const sys = snapshotSystem();
        const gpu = this.gpu ? await queryGpu() : null;
        this.samples.push({ tMs: now - this._startT, cpuCorePct, cpuMachinePct: cpuCorePct === null ? null : cpuCorePct / NCPU, ...sys, gpu });
      } finally {
        this._inFlight = false;
      }
    };
    await tick();
    this._timer = setInterval(tick, this.intervalMs);
  }

  async stop() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    // one final sample so short windows still have >= 2 points
    await new Promise((r) => setTimeout(r, 0));
    return this.summary();
  }

  summary() {
    const s = this.samples;
    const pick = (f) => s.map(f).filter((v) => typeof v === "number" && Number.isFinite(v));
    const stats = (arr) =>
      arr.length ? { min: Math.min(...arr), max: Math.max(...arr), avg: arr.reduce((a, b) => a + b, 0) / arr.length } : null;
    const ru = process.resourceUsage();
    return {
      samples: s.length,
      cpuCorePct: stats(pick((x) => x.cpuCorePct)),
      cpuMachinePct: stats(pick((x) => x.cpuMachinePct)),
      rssMiB: stats(pick((x) => x.rssMiB)),
      maxRssMiB: ru.maxRSS / 1024,
      systemUsedMiB: stats(pick((x) => x.systemUsedMiB)),
      gpu: this.gpu
        ? {
            name: s.find((x) => x.gpu)?.gpu?.name ?? null,
            utilPct: stats(pick((x) => x.gpu?.utilPct)),
            memUsedMiB: stats(pick((x) => x.gpu?.memUsedMiB)),
            powerW: stats(pick((x) => x.gpu?.powerW)),
            tempC: stats(pick((x) => x.gpu?.tempC)),
          }
        : null,
    };
  }
}

export const cpuInfo = () => ({ model: os.cpus()[0]?.model?.trim(), logical: NCPU, totalMemGiB: os.totalmem() / 1073741824 });

/** Percentile helper for latency arrays (ms). */
export function latencyStats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))];
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const sd = Math.sqrt(sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / sorted.length);
  return { n: sorted.length, min: sorted[0], p50: q(50), p90: q(90), p95: q(95), max: sorted[sorted.length - 1], mean, sd };
}
