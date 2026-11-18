/**
 * Explicit model selection above the existing EP router. One resident model at a time:
 * switching closes the previous backend after its accepted calls finish. Model choices
 * never compete on latency and an inference error never substitutes a different model.
 */
import { MODELS, GRAPH_MODES, modelOptions } from "./models.mjs";
import { PythonBackend } from "./python-backend.mjs";
import { runtimeProfile } from "./runtime-profile.mjs";
import { normalizeQuestions } from "./question-schema.mjs";

const bad = (message) => Object.assign(new Error(message), { status: 400 });

export class ModelRouter {
  constructor(options = {}) {
    this.opts = options;
    this.defaults = modelOptions(options);
    const profile = runtimeProfile(this.defaults, options);
    this.lifetime = profile.lifetime;
    this.log = options.log ?? (() => {});
    this.backend = null;
    this.active = null;
    this.loadingModel = null;
    this.queue = Promise.resolve();
    this.group = null;
    this.pending = 0;
    this.closed = false;
    this._sampling = profile.sampleLoad;
  }

  static async create(options = {}) {
    const r = new this(options);
    await r.activate(r.defaults);
    return r;
  }

  selection(options = {}) { return modelOptions(options, this.defaults); }
  selectionKey(selection) {
    const profile = runtimeProfile(selection, this.opts);
    return JSON.stringify([selection.model, selection.device, selection.maxLen, selection.acceleration ?? "off", profile.acceleration ?? "off", profile.graphReplay]);
  }
  get lanes() { return this.backend?.lanes ?? new Map(); }
  get pendingLanes() { return this.loadingModel ? [this.loadingModel] : this.backend?.pendingLanes ?? []; }
  get ready() { return this.backend?.ready ?? Promise.resolve(this); }
  get sampling() { return !!this.backend?.sampling; }
  pauseSampling() { this._sampling = false; this.backend?.pauseSampling?.(); }
  resumeSampling() { this._sampling = true; this.backend?.resumeSampling?.(); }
  async warmup(options) { return this.backend?.warmup?.(options); }

  async open(selection) {
    const profile = runtimeProfile(selection, this.opts);
    this.log(`runtime profile: ${profile.lifetime}${profile.acceleration ? `; acceleration ${profile.acceleration} (${profile.accelerationReason})` : ""}; graphs ${profile.graphReplay ? "on when supported" : "off"}`);
    if (selection.model !== "laya") return PythonBackend.create(selection, { python: this.opts.modelPython, log: this.log,
      timeoutMs: this.opts.timeoutMs, graphReplay: profile.graphReplay, acceleration: profile.acceleration, accelerationReason: profile.accelerationReason });
    const { LayaRouter } = await import("./ep-router.mjs");
    return LayaRouter.create({ ...this.opts, lanes: profile.lanes, loadStrategy: profile.loadStrategy,
      cudaGraph: profile.graphReplay, gpuKeepAliveMs: profile.gpuKeepAliveMs, warmup: profile.warmup, sampleLoad: this._sampling });
  }

  async activate(selection) {
    const key = this.selectionKey(selection);
    if (key === this.active?.key && !this.backend?.dead) return 0;
    this.loadingModel = selection.model;
    const t0 = performance.now();
    try {
      if (this.backend) {
        this.log(`switching ${this.active.selection.model} -> ${selection.model}; releasing previous model`);
        // The legacy router may still be bringing up additional lanes after early readiness.
        const ready = this.backend.ready;
        await this.backend.close();
        await ready.catch(() => {});
      }
      this.backend = null;
      this.active = null;
      this.backend = await this.open(selection);
      this.active = { key, selection };
      return performance.now() - t0;
    } finally { this.loadingModel = null; }
  }

  decide(state, questions, options = {}) {
    if (this.closed) return Promise.reject(new Error("model router is closed"));
    let selection;
    try { selection = this.selection(options); questions = normalizeQuestions(questions); } catch (e) { return Promise.reject(e); }
    const queuedAt = performance.now();
    const key = this.selectionKey(selection);
    let group = this.group;
    if (!group || group.failed || group.key !== key || (this.active?.key === key && this.backend?.dead)) {
      // Consecutive requests for one selection share an activation barrier. Let the original
      // EP router see its entire burst so queue predictions, deadlines and exploration keep
      // their existing semantics. Only a model switch waits for that group's whole drain.
      group = { key, drain: this.queue, serial: Promise.resolve(), loadMs: 0 };
      group.activation = this.queue.catch(() => {}).then(async () => {
        group.loadMs = await this.activate(selection);
        return this.backend;
      });
      group.activation.catch(() => { group.failed = true; });
      this.group = group;
    }
    this.pending++;
    const run = async () => {
      const backend = await group.activation;
      const loadMs = group.loadMs;
      group.loadMs = 0;
      const queueMs = Math.max(0, performance.now() - queuedAt - loadMs);
      const o = { ...options, ...selection };
      if (selection.model === "laya") {
        if (selection.device !== "auto") {
          await backend.ready;
          const matches = [...backend.lanes.keys()].filter((l) => l.split(":")[0] === selection.device);
          if (o.lane && o.lane.split(":")[0] !== selection.device) throw bad(`lane ${o.lane} does not match device ${selection.device}`);
          o.lane ??= matches[0];
          if (!o.lane) throw bad(`no ${selection.device} lane loaded; configure --lanes for that device`);
        }
        if (o.lane && !backend.lanes.has(o.lane)) {
          if (backend.pendingLanes.includes(o.lane)) await backend.ready;
          if (!backend.lanes.has(o.lane)) throw bad(`lane ${o.lane} not loaded; have ${[...backend.lanes.keys()].join(", ")}`);
        }
      }
      const r = await backend.decide(state, questions, o);
      const profile = runtimeProfile(selection, this.opts);
      const actualDevice = r.routing.device ?? r.routing.lane.split(":")[0];
      const graphEnabled = r.routing.exec?.graphReplayEnabled ?? (actualDevice === "cuda" && profile.graphReplay);
      r.routing = { ...r.routing, model: selection.model, contextTokens: selection.maxLen,
        backend: r.routing.backend ?? "laya-onnx", device: actualDevice,
        profile: { lifetime: profile.lifetime, graphs: graphEnabled ? "on" : "off", requestedGraphs: profile.requestedGraphs,
          reason: profile.reason, ...(selection.model === "laya" ? { laneLoading: profile.loadStrategy }
            : { acceleration: profile.acceleration, requestedAcceleration: profile.requestedAcceleration }) },
        queueMs: (r.routing.queueMs ?? 0) + queueMs, ...(loadMs ? { loadMs } : {}) };
      return r;
    };
    const task = (selection.model === "laya" ? run() : group.serial.then(run)).finally(() => { this.pending--; });
    const settled = task.then(() => {}, () => {});
    group.serial = settled;
    group.drain = Promise.all([group.drain, settled]).then(() => {});
    this.queue = group.drain;
    return task;
  }

  modelInfo() {
    return { default: this.defaults, active: this.active?.selection ?? null, loading: this.loadingModel, lifetime: this.lifetime,
      available: Object.fromEntries(Object.entries(MODELS).map(([id, p]) => [id, { name: p.name, backend: p.backend, contextTokens: p.maxLen, devices: ["auto", "cpu", "cuda"], graphs: GRAPH_MODES, ...(p.accelerations ? { accelerations: p.accelerations } : {}) }])) };
  }
  stats() {
    const s = this.backend?.stats() ?? { lanes: {}, load: { cpuOthers: 0, gpuOthersUtil: 0 }, gpuState: "unknown" };
    return { ...s, models: this.modelInfo(), queue: { ...(s.queue ?? {}), pending: this.pending }, loading: this.pendingLanes };
  }
  async detailedStats() {
    const backend = this.backend;
    const s = await backend?.detailedStats();
    if (backend !== this.backend) return this.stats();
    return { ...s, ...this.stats(), ...(s ? { lanes: s.lanes } : {}) };
  }
  async close() {
    if (this._closing) return this._closing;
    this.closed = true;
    this._closing = (async () => {
      await this.queue.catch(() => {});
      const ready = this.ready;
      await this.backend?.close();
      await ready.catch(() => {});
      this.backend = null;
      this.active = null;
    })();
    return this._closing;
  }
}
