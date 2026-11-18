/** Text decisions through the pinned Python Laya API. Stdout is a strict NDJSON protocol. */
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import path from "node:path";
import { PROJECT_ROOT, PYTHON, MODEL_TIMEOUT_MS } from "./models.mjs";
import { normalizeQuestions } from "./question-schema.mjs";

export class PythonBackend {
  static async create(selection, { python = PYTHON, log = () => {}, timeoutMs = MODEL_TIMEOUT_MS, graphReplay = selection.graphs !== "off",
    acceleration = selection.acceleration ?? "off", accelerationReason, script = path.join(PROJECT_ROOT, "tools/laya_worker.py") } = {}) {
    const b = new PythonBackend();
    b.selection = selection;
    b.acceleration = acceleration;
    b.accelerationReason = accelerationReason;
    b.timeoutMs = timeoutMs;
    b.pending = new Map();
    b.seq = 0;
    b.calls = 0;
    b.dead = false;
    b.closed = false;
    b.log = log;
    const started = performance.now();
    log(`starting Python worker for ${selection.model} ...`);
    const child = spawn(python, [script, "--model", selection.model, "--device", selection.device, "--max-len", String(selection.maxLen), "--acceleration", acceleration, "--graphs", graphReplay ? "on" : "off"], {
      cwd: PROJECT_ROOT, stdio: ["pipe", "pipe", "pipe"], windowsHide: true,
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", USE_TF: "0" },
    });
    b.child = child;
    let resolveReady, rejectReady;
    const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
    const timer = setTimeout(() => b.fail(new Error(`Python ${selection.model} did not load within ${timeoutMs} ms`)), timeoutMs);
    const heartbeat = setInterval(() => log(`still loading ${selection.model} (${((performance.now() - started) / 1000).toFixed(1)}s elapsed)`), 5000);
    b.rejectReady = rejectReady;
    // close (rather than exit) also covers a failed spawn and waits until the pipes have closed.
    b.exited = new Promise((resolve) => child.once("close", (code, signal) => {
      b.fail(new Error(`Python ${selection.model} exited (${code ?? signal})`));
      resolve();
    }));
    child.on("error", (e) => b.fail(new Error(`cannot start ${python}: ${e.message}; run npm run python:setup`)));
    child.stdin.on("error", (e) => b.fail(e));
    createInterface({ input: child.stderr, crlfDelay: Infinity }).on("line", log);
    createInterface({ input: child.stdout, crlfDelay: Infinity }).on("line", (line) => {
      try {
        const m = JSON.parse(line);
        if (!b.info) {
          if (!m.ready) throw new Error(m.error ?? "Python worker did not report ready");
          if (m.model !== selection.model || m.maxLen !== selection.maxLen || !["cpu", "cuda"].includes(m.device)) throw new Error("Python worker reported a different model, context or unsupported device");
          b.info = m;
          if (m.acceleration) b.info.acceleration = b.accelerationInfo(m.acceleration);
          b.loadMs = performance.now() - started;
          b.lane = `${selection.model}:${m.device}`;
          b.rejectReady = null;
          resolveReady(b);
        } else {
          const p = b.pending.get(m.id);
          if (!p) throw new Error(`unexpected Python response id ${m.id}`);
          if (!m.error && (!m.result?.answers || !m.result?.usage || !m.exec?.device)) throw new Error("Python response is missing answers, usage or execution device");
          b.pending.delete(m.id);
          clearTimeout(p.timer);
          if (m.error) p.reject(Object.assign(new Error(m.error), { status: m.status ?? 500 }));
          else p.resolve(m);
        }
      } catch (e) { b.fail(e); }
    });
    try {
      await ready;
      return b;
    } catch (e) {
      b.stopProcess();
      await b.exited;
      throw e;
    } finally { clearTimeout(timer); clearInterval(heartbeat); }
  }

  fail(error) {
    if (this.dead) return;
    this.dead = true;
    this.rejectReady?.(error);
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(error); }
    this.pending.clear();
    if (!this.closed) this.stopProcess();
  }

  stopProcess() {
    if (this._killing) return;
    this._killing = true;
    this.child.stdin?.destroy();
    if (!this.child.pid || this.child.exitCode !== null) return;
    const kill = () => {
      // Windows venv python.exe is a launcher; the interpreter reports its own PID.
      if (Number.isSafeInteger(this.info?.pid) && this.info.pid > 0 && this.info.pid !== this.child.pid && this.info.pid !== process.pid) {
        try { process.kill(this.info.pid); } catch { /* interpreter already exited */ }
      }
      this.child.kill();
    };
    if (process.platform !== "win32") return kill();
    // Include a venv launcher's descendants, including during loading before ready/PID.
    const killer = spawn("taskkill.exe", ["/PID", String(this.child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
    killer.on("error", kill);
    killer.on("exit", (code) => { if (code !== 0 && this.child.exitCode === null) kill(); });
  }

  request(body) {
    if (this.dead || this.closed) return Promise.reject(new Error("Python model process is closed"));
    return new Promise((resolve, reject) => {
      const id = this.seq++;
      const payload = JSON.stringify({ id, ...body }) + "\n";
      const timer = setTimeout(() => this.fail(new Error(`Python decision timed out after ${this.timeoutMs} ms; process stopped`)), this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(payload, (e) => { if (e) this.fail(e); });
    });
  }

  async decide(state, questions, options = {}) {
    questions = normalizeQuestions(questions);
    const t0 = performance.now();
    const r = await this.request({ state, questions, readout: options.readout, calibration: options.calibration });
    const ms = performance.now() - t0;
    const firstUse = this.calls === 0;
    this.calls++;
    this.lastMs = ms;
    const exec = { ...r.exec, remoteMs: r.ms };
    if (r.exec.acceleration) exec.acceleration = this.accelerationInfo(r.exec.acceleration, options.acceleration);
    this.lastExec = exec;
    return { ...r.result, routing: {
      lane: `${this.selection.model}:${r.exec.device}`, model: this.selection.model, device: r.exec.device,
      backend: "laya-python", ms, queueMs: 0, n: Object.keys(questions).length,
      predictedMs: null, reason: "explicit model selection", exec,
      ...(firstUse ? { initialization: { processMs: this.loadMs, ...(this.info.startup ?? {}) } } : {}),
    } };
  }

  accelerationInfo(info, requested = this.selection.acceleration ?? "off") {
    return { ...info, requested, selected: this.acceleration === "off" ? "off" : "on",
      ...(this.acceleration === "off" && this.accelerationReason ? { reason: this.accelerationReason } : {}) };
  }

  get lanes() { return new Map([[this.lane, { dead: this.dead }]]); }
  get pendingLanes() { return []; }
  get ready() { return Promise.resolve(this); }
  stats() {
    return { lanes: { [this.lane]: { mode: "process", healthy: !this.dead, dead: this.dead, calls: this.calls,
      pending: this.pending.size, loadMs: this.info.loadMs, ema: {}, process: { ...this.info, lastMs: this.lastMs, exec: this.lastExec } } },
      loading: [], queue: { pending: this.pending.size, waitMs: 0 }, load: { cpuOthers: 0, gpuOthersUtil: 0 }, sampling: false, gpuState: "unknown" };
  }
  async detailedStats() { return this.stats(); }
  async close() {
    if (this.closed) return this.exited;
    this.closed = true;
    if (!this.dead) this.child.stdin.end(JSON.stringify({ op: "close" }) + "\n");
    const timer = setTimeout(() => this.stopProcess(), 5000);
    try { await this.exited; } finally { clearTimeout(timer); }
  }
}
