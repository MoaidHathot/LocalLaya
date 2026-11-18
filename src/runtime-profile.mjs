/** Benchmark-informed defaults. Explicit settings take precedence; model choice stays explicit. */
export const LIFETIMES = ["one-shot", "persistent"];
export const DEFAULT_LAYA_LANES = ["cuda:fp16", "webgpu:fp16", "cpu:8"];

export function runtimeProfile(selection, options = {}) {
  const lifetime = options.lifetime ?? "persistent";
  if (!LIFETIMES.includes(lifetime)) throw Object.assign(new Error("lifetime must be one-shot or persistent"), { status: 400 });
  const once = lifetime === "one-shot";
  const requestedGraphs = selection.graphs ?? "auto";
  const requestedAcceleration = selection.acceleration ?? "auto";
  // TileLang's process initialization costs seconds, even with its disk cache warm.
  // For one Multilingual decision, use eager; explicit on/graphs opt into that cost.
  const eagerStartup = once && selection.model === "multilingual"
    && requestedAcceleration === "auto" && requestedGraphs !== "on";
  const acceleration = requestedAcceleration === "off" || selection.device === "cpu" || eagerStartup ? "off" : "on";
  const accelerationReason = requestedAcceleration === "off" ? "explicit acceleration=off"
    : selection.device === "cpu" ? "CPU uses eager/reference execution"
    : eagerStartup ? "one-shot auto skips TileLang initialization"
    : requestedAcceleration === "on" ? "explicit acceleration=on"
    : "use available accelerated kernels";
  const graphDefault = selection.model === "laya" ? options.cudaGraph ?? !once : !once;
  const graphReplay = selection.device !== "cpu" && acceleration !== "off"
    && (requestedGraphs === "auto" ? graphDefault : requestedGraphs === "on");
  const manualLanes = options.lanes ?? (options.lane ? [options.lane] : undefined);
  const lanes = manualLanes ?? (selection.device === "cpu" ? ["cpu:8"] : selection.device === "cuda" ? ["cuda:fp16"] : options.legacyDefaultLanes ?? DEFAULT_LAYA_LANES);
  // A soft routing policy/deadline needs its candidate lanes. A bare one-shot only needs
  // the first validated provider in benchmark order, including a CPU fallback.
  const chooseOne = once && !manualLanes && selection.device === "auto"
    && (!options.policy || options.policy === "auto") && options.deadlineMs === undefined;
  return {
    lifetime,
    graphReplay,
    requestedGraphs,
    ...(selection.model !== "laya" ? { acceleration, requestedAcceleration, accelerationReason } : {}),
    reason: (once ? "single decision: minimize startup work" : "reused model: amortize initialization across calls")
      + (selection.model !== "laya" ? `; ${accelerationReason}` : "")
      + (requestedGraphs === "auto" ? "; automatic graph policy" : `; explicit graphs=${requestedGraphs}`),
    lanes: [...lanes],
    loadStrategy: options.loadStrategy ?? (chooseOne ? "first-available" : "parallel"),
    sampleLoad: options.sampleLoad ?? !once,
    gpuKeepAliveMs: options.gpuKeepAliveMs ?? (once ? 0 : 30_000),
    warmup: options.warmup ?? null,
  };
}
